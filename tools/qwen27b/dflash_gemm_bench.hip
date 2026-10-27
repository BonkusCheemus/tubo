// DFlash2 exact small-batch GEMM ablations, using the production templates.
// Build inside Nix with tools/bench/build.sh and this source path.
// Run: /tmp/dflash_gemm_bench
//   bf16|q8|q4|q5|iq4|q6|q3|iq4-nl|iq3-s|iq3-xxs|iq2-s|iq2-xs|iq2-xxs|q2
//   M K [iterations]
// [batch] Scalar Q8/Q4/Q5/IQ4/Q6 use batch=1; append swiglu [up-quant] for
// gate/up. Defaults: 24 iterations, batch 7 for Q6 and 8 otherwise; widths 2–8
// supported. BF16/Q8/Q4/Q5/IQ4 also accept widths 12/16 for shared
// verification. Weights use anonymous huge-page advice and read-only host
// registration, matching production. Quantized projections rotate matrices
// beyond cache capacity.
// Wide BF16 repeats one matrix across chunks, including the production K/V
// cache hint. Every geometry must be bit-exact with the reference before timing
// counts. Q3_K/IQ4_NL wide ablations cover their native widths 14 and 16.
//
// Every format that DecodeQuantSub16 handles but that has no tuned
// small-batch dispatch yet (Q6_K, Q3_K, IQ4_NL, IQ3_S, IQ3_XXS, IQ2_S, IQ2_XS,
// IQ2_XXS, Q2_K) runs widths 1–8 through one shared geometry family, the same
// template family the Q4_K/Q5_K winners came from. Q4_K/Q5_K/IQ4_XS keep their
// own measured gating so their earlier numbers stay comparable. Geometry
// "tile1" is the production wave32 fallback for every untuned format, so it is
// the baseline each candidate must beat; it is also the bit-exact reference.
//
// iq3-s also gets the snb_ arms, which read a host-repacked copy of
// the weights (sn_iq3.hpp) in a row-interleaved layout with a sign-folded LDS
// codebook (sn_iq3.hip). They are approximate arms (int8 activations with one
// q8_K scale per 256 weights), judged by rel_error and top1_agreement like the
// other dot4 arms. Only the snb_ arms and their repacked buffer are affected;
// every other geometry still reads the original buffer.
#define ENGINE_ENABLE_HIP 1
#include <sys/mman.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/qwen/hip/kernels/small_batch_dot4.hpp"
#include "src/models/qwen/hip/kernels/small_batch_gemm.hpp"
#include "src/models/qwen/hip/kernels/sn_iq3.hip"
#include "src/models/qwen/hip/kernels/sn_iq3.hpp"
#include "src/models/qwen/hip/kernels/sn_q4k.hpp"
#include "src/models/qwen/hip/kernels/sn_q4k_k4.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"
// Embed the scalar implementation so ablations use its actual templates and
// arithmetic without linking the complete model runtime.
#include "src/models/qwen/hip/kernels/gemv_quant.hip"
#include "src/models/qwen/hip/kernels/swiglu.hip"

namespace gufo::hip::detail {
// small_batch_dot4_v3_wave64.hip (-mwavefrontsize64).
bool TryLaunchDot4V3Wave64(core::GgmlType type, std::size_t batch,
                           std::uint32_t rows, std::uint32_t waves,
                           std::uint32_t per_lane, bool act_k, const void* w,
                           const float* x, float* y, std::int8_t* xq,
                           float2* xds, std::uint32_t m, std::uint32_t k,
                           hipStream_t stream);
}  // namespace gufo::hip::detail

namespace {

void Check(hipError_t error) {
  if (error != hipSuccess)
    throw std::runtime_error(hipGetErrorString(error));
}

template<typename T>
class Buffer {
public:
  explicit Buffer(std::size_t count) {
    Check(hipMalloc(&data_, count * sizeof(T)));
  }
  ~Buffer() { (void)hipFree(data_); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  T* Get() const { return data_; }

private:
  T* data_{};
};

class MappedWeights {
public:
  MappedWeights(const void* source, std::size_t bytes) : bytes_(bytes) {
    host_ = mmap(nullptr, bytes_, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (host_ == MAP_FAILED)
      throw std::runtime_error("weight mmap failed");
    try {
      (void)madvise(host_, bytes_, MADV_HUGEPAGE);
      Check(hipMemcpy(host_, source, bytes_, hipMemcpyDeviceToHost));
      if (mprotect(host_, bytes_, PROT_READ) != 0)
        throw std::runtime_error("weight mprotect failed");
      Check(hipHostRegister(host_, bytes_,
                            hipHostRegisterMapped | hipHostRegisterReadOnly));
      registered_ = true;
      Check(hipHostGetDevicePointer(&device_, host_, 0));
    } catch (...) {
      Release();
      throw;
    }
  }
  ~MappedWeights() { Release(); }
  MappedWeights(const MappedWeights&) = delete;
  MappedWeights& operator=(const MappedWeights&) = delete;
  const unsigned char* Get() const {
    return static_cast<const unsigned char*>(device_);
  }

private:
  void Release() {
    if (registered_)
      (void)hipHostUnregister(host_);
    (void)munmap(host_, bytes_);
  }
  void* host_{};
  void* device_{};
  std::size_t bytes_{};
  bool registered_{};
};

class Event {
public:
  Event() { Check(hipEventCreate(&event_)); }
  ~Event() { (void)hipEventDestroy(event_); }
  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;
  hipEvent_t Get() const { return event_; }

private:
  hipEvent_t event_{};
};

__device__ unsigned Hash(unsigned value) {
  value ^= value >> 16;
  value *= 0x7feb352dU;
  value ^= value >> 15;
  value *= 0x846ca68bU;
  return value ^ (value >> 16);
}

__global__ void InitBytes(unsigned char* data, std::size_t size) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size)
    data[index] = Hash(index);
}

__global__ void InitScales(unsigned char* data, std::size_t blocks,
                           std::size_t stride, std::size_t offset, bool affine,
                           std::size_t blocks_per_copy = 0,
                           std::size_t copy_bytes = 0) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= blocks)
    return;
  const auto copy = blocks_per_copy ? index / blocks_per_copy : 0;
  const auto block = blocks_per_copy ? index % blocks_per_copy : index;
  auto* scale = reinterpret_cast<__half*>(data + copy * copy_bytes +
                                          block * stride + offset);
  scale[0] = __float2half(0.007F + (Hash(index) % 1000) * 0.00001F);
  if (affine)
    scale[1] = __float2half(0.004F + (Hash(index + 19) % 1000) * 0.00001F);
}

template<typename T>
__global__ void InitValues(T* data, std::size_t size) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size)
    data[index] = T((static_cast<int>(Hash(index) % 20001) - 10000) * 0.00027F);
}

__global__ void Compare(const float* expected, const float* actual,
                        std::size_t count, unsigned* mismatches) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count &&
      (!isfinite(actual[index]) ||
       __float_as_uint(expected[index]) != __float_as_uint(actual[index]))) {
    atomicAdd(mismatches, 1U);
  }
}

// 24 back-to-back iterations of one GEMM keep the queue full, so
// per-launch cost and serialization are hidden. Real decode runs each GEMM as a
// link in a dependency chain of ~300 per round, where they are not. With
// GUFO_DFLASH_CHAIN=1 this kernel creates that dependency and nothing else.
//
// x[0] * (1.0F + 0.0F * y[0]) is x[0] bitwise for every finite y: 0*y is +-0,
// 1.0 + -+0 is exactly 1.0, and multiplying by 1.0 preserves -0 and NaN. It
// cannot be folded away either, because 0*inf is NaN, so the read of y stays
// and the next GEMM's read of x must wait for the GEMM that wrote y. The
// bit-exact compare each round is what proves it harmless: if this ever
// perturbed the input, every geometry would start mismatching at once.
__global__ void ChainDependency(const float* y, float* x) {
  x[0] = x[0] * (1.0F + 0.0F * y[0]);
}

// GUFO_DFLASH_PREQUANT=1: the IQ3_S snb_/opus_ arms quantize the activations
// once before their correctness launch and once before their timed loop (both
// outside the events), and time only the GEMM launches: the served path's
// cost, where one quantize per input is shared by every GEMM that reads it.
// Unset or 0 keeps the default timing (quantize + GEMM per launch).
bool PrequantMode() {
  const char* flag = std::getenv("GUFO_DFLASH_PREQUANT");
  if (flag != nullptr &&
      !((flag[0] == '0' || flag[0] == '1') && flag[1] == '\0'))
    throw std::invalid_argument("GUFO_DFLASH_PREQUANT must be unset, 0, or 1");
  return flag != nullptr && flag[0] == '1';
}

struct Geometry {
  const char* name;
  std::function<void(const void*)> launch;
  // Nominal full-matrix traversals per iteration, not DRAM transactions. A
  // geometry that splits the batch over two launches reads every weight twice,
  // so without this its GB/s reads half of what it actually moves. Same field
  // and meaning as the earlier GEMM bench, so the CSV columns line up.
  std::size_t weight_passes = 1;
  // dot4: int8 activations, not bit-exact. Judged by relative error
  // (max |ref - out| / max |ref|) and per-token top-1 row agreement instead.
  bool approximate = false;
  // IQ3 x2 must additionally match its corresponding R1 V3 geometry bitwise.
  std::function<void(const void*)> pair_reference;
  // Untimed setup run before the correctness launch and before the timed loop
  // (GUFO_DFLASH_PREQUANT=1: the snb_/opus_ activation quantize).
  std::function<void()> prepare;
  // 0: pair_reference must match bitwise. >0: it may differ by at most this
  // relative error. SN Q4_K recomputes V3's integer sub-block sums but flushes
  // one full super-block per lane instead of two half-super-blocks, so only
  // the fp32 summation order differs.
  float pair_rel_error = 0.0F;
};

template<std::size_t Batch>
void Run(const std::string& type, std::size_t rows, std::size_t columns,
         int iterations, bool fused = false, const std::string& up_type = "") {
  using namespace gufo;
  using namespace gufo::hip;
  const std::size_t column_block = type == "bf16"                       ? 8
                                   : (type == "q8" || type == "iq4-nl") ? 32
                                                                        : 256;
  // Screening formats: no tuned small-batch dispatch yet, so they share
  // one geometry family at widths 1-8.
  const bool screened = type == "q6" || type == "q3" || type == "iq4-nl" ||
                        type == "iq3-s" || type == "iq3-xxs" ||
                        type == "iq2-s" || type == "iq2-xs" ||
                        type == "iq2-xxs" || type == "q2";
  if (rows == 0 || columns == 0 || columns % column_block != 0 ||
      iterations < 1 ||
      (type != "bf16" && type != "q8" && type != "q4" && type != "q5" &&
       type != "iq4" && !screened)) {
    throw std::invalid_argument(
        "require bf16|q8|q4|q5|iq4|q6|q3|iq4-nl|iq3-s|iq3-xxs|iq2-s|iq2-xs|"
        "iq2-xxs|q2, positive M, complete weight blocks, positive iterations");
  }
  // Above width 8 only Q3_K/IQ4_NL (at 14/16) and Q6_K have a typed wide
  // branch. Every other screened format would silently be benched as IQ4_XS,
  // so refuse instead of reporting a number for the wrong template.
  if (screened && Batch > 8 && type != "q6" &&
      !((type == "q3" || type == "iq4-nl") && (Batch == 14 || Batch == 16)))
    throw std::invalid_argument(
        "this format is benched at widths 1-8; q3/iq4-nl add 14/16, q6 adds "
        "9-16");
  if constexpr (Batch > 16) {
    if (type != "bf16")
      throw std::invalid_argument("24/32-row ablations require bf16");
  }
  if constexpr (Batch == 1) {
    if (type == "bf16")
      throw std::invalid_argument("scalar ablations require q8|q4|q5|iq4|q6");
  }
  const bool mixed = !up_type.empty() && up_type != type;
  if (mixed && (!fused || Batch != 1 ||
                !((type == "q4" && (up_type == "q5" || up_type == "iq4")) ||
                  (type == "iq4" && (up_type == "q4" || up_type == "q5"))))) {
    throw std::invalid_argument(
        "mixed swiglu supports q4/q5, q4/iq4, iq4/q4 and iq4/q5");
  }
  constexpr std::size_t batch = Batch;
  // Byte sizes are the ggml_dequant.hpp block structs (their static_asserts).
  const auto block_size = [](const std::string& quant) -> std::size_t {
    return quant == "q8"        ? 34
           : quant == "q4"      ? 144
           : quant == "q5"      ? 176
           : quant == "iq4"     ? 136
           : quant == "q3"      ? 110
           : quant == "iq4-nl"  ? 18
           : quant == "iq3-s"   ? 110
           : quant == "iq3-xxs" ? 98
           : quant == "iq2-s"   ? 82
           : quant == "iq2-xs"  ? 74
           : quant == "iq2-xxs" ? 66
           : quant == "q2"      ? 84
                                : 210;
  };
  const std::size_t block_bytes = block_size(type);
  const std::size_t matrix_bytes =
      type == "bf16"
          ? rows * columns * 2
          : rows * columns / ((type == "q8" || type == "iq4-nl") ? 32 : 256) *
                block_bytes;
  const std::size_t up_block_bytes = mixed ? block_size(up_type) : block_bytes;
  const std::size_t up_matrix_bytes =
      mixed ? rows * columns / 256 * up_block_bytes : matrix_bytes;
  const std::size_t bytes = matrix_bytes + (fused ? up_matrix_bytes : 0);
  if (Batch != 1 && type != "bf16" && (type != "q8" || Batch > 8) &&
      (!FitsSmallBatch32BitIndices(std::min<std::size_t>(batch, 8), rows,
                                   columns, matrix_bytes / rows) ||
       rows > std::numeric_limits<std::uint32_t>::max() / batch ||
       columns > std::numeric_limits<std::uint32_t>::max() / batch)) {
    throw std::invalid_argument("K-quant ablations require 32-bit offsets");
  }
  const std::size_t copies =
      Batch > 8 && type == "bf16"
          ? 1
          : std::max<std::size_t>(1,
                                  (128ULL * 1024 * 1024 + bytes - 1) / bytes);
  const std::size_t physical = bytes * copies;
  Buffer<unsigned char> weights(physical);
  Buffer<float> input(batch * columns), output(batch * rows),
      reference(batch * rows);
  Buffer<unsigned> mismatch(1);
  if (type == "bf16") {
    InitValues<<<(physical / 2 + 255) / 256, 256>>>(
        reinterpret_cast<hip_bfloat16*>(weights.Get()), physical / 2);
  } else {
    InitBytes<<<(physical + 255) / 256, 256>>>(weights.Get(), physical);
    if (mixed) {
      const auto gate_blocks = matrix_bytes / block_bytes;
      const auto up_blocks = up_matrix_bytes / up_block_bytes;
      InitScales<<<(copies * gate_blocks + 255) / 256, 256>>>(
          weights.Get(), copies * gate_blocks, block_bytes, 0,
          type == "q4" || type == "q5", gate_blocks, bytes);
      InitScales<<<(copies * up_blocks + 255) / 256, 256>>>(
          weights.Get() + matrix_bytes, copies * up_blocks, up_block_bytes, 0,
          up_type == "q4" || up_type == "q5", up_blocks, bytes);
    } else {
      // Byte offset of the block's fp16 super-scale, and whether a second
      // fp16 (dmin) follows it: Q6_K d at 208, Q3_K d at 108, Q2_K d/dmin at
      // 80/82, every other format carries d (and Q4_K/Q5_K dmin) at offset 0.
      InitScales<<<(physical / block_bytes + 255) / 256, 256>>>(
          weights.Get(), physical / block_bytes, block_bytes,
          type == "q6"   ? 208
          : type == "q3" ? 108
          : type == "q2" ? 80
                         : 0,
          type == "q4" || type == "q5" || type == "q2");
    }
  }
  InitValues<<<(batch * columns + 255) / 256, 256>>>(input.Get(),
                                                     batch * columns);
  std::vector<Geometry> geometries;
  if constexpr (Batch == 1) {
    const auto scalar = [&]<core::GgmlType Type,
                            core::GgmlType UpType = Type>() {
      const auto launch = [&]<core::GgmlType StaticType,
                              core::GgmlType StaticUpType = StaticType,
                              bool FastIq = false>(const void* w) {
        // Only the formats production fuses gate/up for get a fused kernel
        // instantiation; the screened low-bit formats are GEMV only, so
        // instantiating the fused template for them would be dead code.
        constexpr bool kFusable = StaticType == core::GgmlType::kF32 ||
                                  StaticType == core::GgmlType::kQ8_0 ||
                                  StaticType == core::GgmlType::kQ4_K ||
                                  StaticType == core::GgmlType::kQ5_K ||
                                  StaticType == core::GgmlType::kIQ4_XS ||
                                  StaticType == core::GgmlType::kQ6_K;
        if (fused && !kFusable)
          throw std::invalid_argument("swiglu requires q8|q4|q5|iq4|q6");
        if constexpr (kFusable) {
          if (fused) {
            Wave32FusedQuantSwiGLUGEMVKernel_2Rows<4, StaticType, StaticUpType>
                <<<(rows + 7) / 8, 128>>>(
                    w, Type,
                    static_cast<const unsigned char*>(w) + matrix_bytes, UpType,
                    input.Get(), output.Get(), rows, columns);
            return;
          }
        }
        {
          Q8KBlockGEMVKernel_2Rows<4, StaticType, FastIq>
              <<<(rows + 7) / 8, 128>>>(w, Type, input.Get(), output.Get(),
                                        rows, columns);
        }
      };
      geometries = {{"generic",
                     [&, launch](const void* w) {
                       launch.template operator()<core::GgmlType::kF32>(w);
                     }},
                    {"typed", [&, launch](const void* w) {
                       launch.template operator()<Type, UpType>(w);
                     }}};
      if (GufoDevIqFastEnabled()) {
        if constexpr (Type == core::GgmlType::kIQ2_S ||
                      Type == core::GgmlType::kIQ3_S ||
                      Type == core::GgmlType::kIQ3_XXS) {
          geometries.push_back({"iq_fast", [&, launch](const void* w) {
                                  launch.template operator()<Type, UpType, true>(
                                      w);
                                }});
        }
      }
    };
    if (type == "q8") {
      scalar.template operator()<core::GgmlType::kQ8_0>();
    } else if (type == "q4") {
      if (up_type == "q5")
        scalar.template
        operator()<core::GgmlType::kQ4_K, core::GgmlType::kQ5_K>();
      else if (up_type == "iq4")
        scalar.template
        operator()<core::GgmlType::kQ4_K, core::GgmlType::kIQ4_XS>();
      else
        scalar.template operator()<core::GgmlType::kQ4_K>();
    } else if (type == "q5") {
      scalar.template operator()<core::GgmlType::kQ5_K>();
    } else if (type == "q3") {
      scalar.template operator()<core::GgmlType::kQ3_K>();
    } else if (type == "iq4-nl") {
      scalar.template operator()<core::GgmlType::kIQ4_NL>();
    } else if (type == "iq3-s") {
      scalar.template operator()<core::GgmlType::kIQ3_S>();
    } else if (type == "iq3-xxs") {
      scalar.template operator()<core::GgmlType::kIQ3_XXS>();
    } else if (type == "iq2-s") {
      scalar.template operator()<core::GgmlType::kIQ2_S>();
    } else if (type == "iq2-xs") {
      scalar.template operator()<core::GgmlType::kIQ2_XS>();
    } else if (type == "iq2-xxs") {
      scalar.template operator()<core::GgmlType::kIQ2_XXS>();
    } else if (type == "q2") {
      scalar.template operator()<core::GgmlType::kQ2_K>();
    } else if (type == "iq4") {
      if (up_type == "q4")
        scalar.template
        operator()<core::GgmlType::kIQ4_XS, core::GgmlType::kQ4_K>();
      else if (up_type == "q5")
        scalar.template
        operator()<core::GgmlType::kIQ4_XS, core::GgmlType::kQ5_K>();
      else
        scalar.template operator()<core::GgmlType::kIQ4_XS>();
    } else {
      scalar.template operator()<core::GgmlType::kQ6_K>();
    }
  } else if (type == "bf16") {
    if constexpr (Batch > 8) {
      geometries = {
          {"split8",
           [&](const void* w) {
             const auto launch = [&]<std::size_t Width>(std::size_t offset) {
               const auto* weights = static_cast<const hip_bfloat16*>(w);
               const float* x = input.Get() + offset * columns;
               float* y = output.Get() + offset * rows;
               if constexpr (Width == 8) {
                 if ((rows == 5120 && columns == 4096) ||
                     (columns == 5120 && (rows == 12288 || rows == 17408))) {
                   BatchedExactBf16GEMMFp32VecKernel<16, Width, 1>
                       <<<(rows + 15) / 16, 512>>>(weights, x, y, rows,
                                                   columns);
                   return;
                 }
               }
               BatchedExactBf16GEMMFp32VecKernel<16, Width, 2>
                   <<<(rows + 31) / 32, 512>>>(weights, x, y, rows, columns);
             };
             for (std::size_t offset = 0; offset + 8 <= Batch; offset += 8)
               launch.template operator()<8>(offset);
             if constexpr (Batch % 8 != 0)
               launch.template operator()<Batch % 8>((Batch / 8) * 8);
           }},
          {"wide", [&](const void* w) {
             if constexpr (Batch == 32) {
               if (detail::TryLaunchBf16SmallBatchWave64(w, input.Get(),
                                                         output.Get(), 16, rows,
                                                         columns, nullptr, 2))
                 return;
               for (std::size_t offset = 0; offset < Batch; offset += 16)
                 BatchedExactBf16GEMMFp32VecKernel<16, 16, 2>
                     <<<(rows + 31) / 32, 512>>>(
                         static_cast<const hip_bfloat16*>(w),
                         input.Get() + offset * columns,
                         output.Get() + offset * rows, rows, columns);
               return;
             } else {
               if constexpr (Batch >= 9 && Batch <= 15 && Batch != 12) {
                 if (rows == 5120 && columns == 6144) {
                   BatchedExactBf16GEMMFp32VecKernel<16, Batch, 1>
                       <<<(rows + 15) / 16, 512>>>(
                           static_cast<const hip_bfloat16*>(w), input.Get(),
                           output.Get(), rows, columns);
                   return;
                 }
               }
               if constexpr ((Batch >= 9 && Batch <= 15 && Batch != 12) ||
                             Batch > 16) {
                 if (!detail::TryLaunchBf16SmallBatchWave64(
                         w, input.Get(), output.Get(), Batch, rows, columns,
                         nullptr))
                   throw std::runtime_error("unsupported wide BF16 batch");
               } else {
                 const auto launch = [&]<unsigned Waves, std::size_t Rows,
                                         bool Stream = true>() {
                   BatchedExactBf16GEMMFp32VecKernel<Waves, Batch, Rows, Stream>
                       <<<(rows + Waves * Rows - 1) / (Waves * Rows),
                          Waves * 32>>>(static_cast<const hip_bfloat16*>(w),
                                        input.Get(), output.Get(), rows,
                                        columns);
                 };
                 if (rows == 5120 && columns == 6144) {
                   launch.template operator()<16, 1>();
                   return;
                 }
                 if constexpr (Batch == 12) {
                   if (columns == 5120 &&
                       (rows == 12288 || rows == 17408 || rows == 248320)) {
                     launch.template operator()<8, 2>();
                     return;
                   }
                 } else {
                   if (rows == 1024 && columns == 5120) {
                     launch.template operator()<16, 2, false>();
                     return;
                   }
                 }
                 launch.template operator()<16, 2>();
               }
             }
           }}};
    } else {
      geometries = {{"rows2",
                     [&](const void* w) {
                       BatchedExactBf16GEMMFp32VecKernel<16, Batch, 2>
                           <<<(rows + 31) / 32, 512>>>(
                               static_cast<const hip_bfloat16*>(w), input.Get(),
                               output.Get(), rows, columns);
                     }},
                    {"rows1", [&](const void* w) {
                       BatchedExactBf16GEMMFp32VecKernel<16, Batch, 1>
                           <<<(rows + 15) / 16, 512>>>(
                               static_cast<const hip_bfloat16*>(w), input.Get(),
                               output.Get(), rows, columns);
                     }}};
    }
  } else if constexpr (Batch <= 8) {
    if (type == "q8") {
      geometries = {{"rows2",
                     [&](const void* w) {
                       SmallBatchQ8_0ExactFp32VecGEMMKernel<16, Batch, 2>
                           <<<(rows + 31) / 32, 512>>>(
                               w, input.Get(), output.Get(), rows, columns);
                     }},
                    {"token_rows",
                     [&](const void* w) {
                       SmallBatchQ8_0ExactFp32VecGEMMKernel<8, 1, 1>
                           <<<dim3((rows + 7) / 8, Batch), 256>>>(
                               w, input.Get(), output.Get(), rows, columns);
                     }},
                    {"rows3", [&](const void* w) {
                       SmallBatchQ8_0ExactFp32VecGEMMKernel<16, Batch, 3>
                           <<<(rows + 47) / 48, 512>>>(
                               w, input.Get(), output.Get(), rows, columns);
                     }}};
      const auto waves = [&]<std::uint32_t Waves, std::size_t Rows>(
                             const char* name) {
        geometries.push_back(
            {name, [&](const void* w) {
               SmallBatchQ8_0ExactFp32VecGEMMKernel<Waves, Batch, Rows>
                   <<<(rows + Waves * Rows - 1) / (Waves * Rows), Waves * 32>>>(
                       w, input.Get(), output.Get(), rows, columns);
             }});
      };
      waves.template operator()<32, 2>("waves32_rows2");
    } else if (type == "q4" || type == "q5" || type == "iq4") {
      const auto kquant = [&]<core::GgmlType Type>() {
        geometries = {
            {"tile1",
             [&](const void* w) {
               SmallBatchKQuantExactFp32GEMMKernel<16, Batch, 3, Type>
                   <<<(rows + 47) / 48, 512>>>(w, input.Get(), output.Get(),
                                               rows, columns);
             }},
            {"tile2", [&](const void* w) {
               SmallBatchKQuantExactFp32GEMMKernel<16, Batch, 3, Type, 2>
                   <<<(rows + 47) / 48, 512>>>(w, input.Get(), output.Get(),
                                               rows, columns);
             }}};
        const auto narrow = [&]<std::size_t Tiles>(const char* name) {
          geometries.push_back(
              {name, [&](const void* w) {
                 SmallBatchKQuantExactFp32GEMMKernel<16, Batch, 3, Type, Tiles,
                                                     12, 1, true>
                     <<<(rows + 47) / 48, 512>>>(w, input.Get(), output.Get(),
                                                 rows, columns);
               }});
        };
        narrow.template operator()<1>("narrow_tile1");
        narrow.template operator()<2>("narrow_tile2");
        if constexpr ((Type == core::GgmlType::kQ4_K && Batch >= 2) ||
                      (Type == core::GgmlType::kQ5_K && Batch >= 2 &&
                       Batch <= 6) ||
                      (Type == core::GgmlType::kIQ4_XS &&
                       (Batch == 2 || Batch == 3 || Batch == 4 || Batch == 5 ||
                        Batch == 7))) {
          const auto rows4 =
              [&]<std::size_t Tiles, std::size_t TokensPerStep = Batch>(
                  const char* name) {
                geometries.push_back(
                    {name, [&](const void* w) {
                       SmallBatchKQuantExactFp32GEMMKernel<
                           16, Batch, 4, Type, Tiles, 4, 1, true, TokensPerStep>
                           <<<(rows + 63) / 64, 512>>>(
                               w, input.Get(), output.Get(), rows, columns);
                     }});
              };
          if constexpr (Type == core::GgmlType::kQ5_K && Batch == 4) {
            rows4.template operator()<1>("rows4_tile1");
            rows4.template operator()<1, 2>("rows4_tile1_step2");
          } else {
            rows4.template operator()<2>("rows4_tile2");
            if constexpr ((Type == core::GgmlType::kQ5_K && Batch == 5) ||
                          (Type == core::GgmlType::kIQ4_XS && Batch == 3))
              rows4.template operator()<1>("rows4_tile1");
          }
        }
        if constexpr ((Batch >= 7 && Type == core::GgmlType::kQ5_K) ||
                      (Batch == 8 && Type == core::GgmlType::kIQ4_XS)) {
          geometries.push_back(
              {"narrow_rows4", [&](const void* w) {
                 SmallBatchKQuantExactFp32GEMMKernel<8, Batch, 4, Type, 1, 4, 1,
                                                     true>
                     <<<(rows + 31) / 32, 256>>>(w, input.Get(), output.Get(),
                                                 rows, columns);
               }});
        }
        if constexpr (Batch == 8 && Type == core::GgmlType::kQ5_K) {
          geometries.push_back(
              {"waves16_rows4", [&](const void* w) {
                 SmallBatchKQuantExactFp32GEMMKernel<16, Batch, 4, Type, 1, 4,
                                                     1, true>
                     <<<(rows + 63) / 64, 512>>>(w, input.Get(), output.Get(),
                                                 rows, columns);
               }});
          const auto token_major = [&]<std::uint32_t Waves>(const char* name) {
            geometries.push_back(
                {name, [&](const void* w) {
                   SmallBatchKQuantExactFp32GEMMKernel<Waves, Batch, 4, Type, 1,
                                                       4, 1, true, 1>
                       <<<(rows + Waves * 4 - 1) / (Waves * 4), Waves * 32>>>(
                           w, input.Get(), output.Get(), rows, columns);
                 }});
          };
          token_major.template operator()<8>("token_major_rows4");
          token_major.template operator()<16>("token_major_waves16_rows4");
        }
        if constexpr (Batch == 8 && Type == core::GgmlType::kQ4_K) {
          geometries.push_back(
              {"groups4", [&](const void* w) {
                 SmallBatchKQuantExactFp32GEMMKernel<16, 4, 3, Type, 1, 12, 2>
                     <<<((rows + 47) / 48) * 2, 512>>>(
                         w, input.Get(), output.Get(), rows, columns);
               }});
          geometries.push_back(
              {"narrow_groups4", [&](const void* w) {
                 SmallBatchKQuantExactFp32GEMMKernel<16, 4, 3, Type, 1, 12, 2,
                                                     true>
                     <<<((rows + 47) / 48) * 2, 512>>>(
                         w, input.Get(), output.Get(), rows, columns);
               }});
        }
      };
      if (type == "q4")
        kquant.template operator()<core::GgmlType::kQ4_K>();
      else if (type == "q5")
        kquant.template operator()<core::GgmlType::kQ5_K>();
      else
        kquant.template operator()<core::GgmlType::kIQ4_XS>();
    } else {
      // Screening family. One geometry per lever, off the production
      // wave32 fallback, for every format without a tuned dispatch. "tile1"
      // stays first so it is both the production baseline and the bit-exact
      // reference every candidate is checked against.
      //
      // Levers, and why each one is here:
      //   NarrowIndex   32-bit offsets instead of 64-bit; the bench already
      //                 guarantees the shapes fit, so this is free address
      //                 arithmetic. Every Q4_K/Q5_K winner used it.
      //   RowsPerWave   2/3/4 output rows share one staged activation tile.
      //                 More reuse, more registers: the trade turns over at a
      //                 different point for each format's decode cost.
      //   TilesPerStage 1 or 2 sub-block tiles per staging barrier.
      //   WavesPerBlock 8/16/32 waves. llama.cpp discussions/23881 measured
      //                 +54% pp on gfx906 from nwarps 4 -> 16 for MMQ and
      //                 concluded the warp count needs per-format treatment;
      //                 our fallback is already 16, so 8 and 32 are the open
      //                 directions.
      //   TokensPerStep all tokens per decode, or one at a time
      //                 (token_major): trades activation reuse for live
      //                 registers, which is the lever that wins once a
      //                 format's decode needs many registers.
      //   TokenGroups   two groups of four tokens in one launch (groups4);
      //                 this is what won Q4_K at width 8.
      const auto screen = [&]<core::GgmlType Type>() {
        const auto add = [&]<std::uint32_t Waves, std::size_t Rows,
                             std::size_t Tiles, std::uint32_t MinWaves,
                             bool Narrow, std::size_t Step = Batch>(
                             const char* name) {
          geometries.push_back(
              {name, [&](const void* w) {
                 SmallBatchKQuantExactFp32GEMMKernel<Waves, Batch, Rows, Type,
                                                     Tiles, MinWaves, 1, Narrow,
                                                     Step>
                     <<<(rows + Waves * Rows - 1) / (Waves * Rows),
                        Waves * 32>>>(w, input.Get(), output.Get(), rows,
                                      columns);
               }});
        };
        add.template operator()<16, 3, 1, 12, false>("tile1");
        add.template operator()<16, 3, 1, 12, true>("narrow_tile1");
        add.template operator()<16, 3, 2, 12, true>("narrow_tile2");
        // MinWaves 1 with two tiles is Q6_K's production FFN geometry.
        add.template operator()<16, 3, 2, 1, true>("narrow_rows3_tile2");
        add.template operator()<16, 2, 1, 12, true>("narrow_rows2_tile1");
        add.template operator()<16, 4, 1, 4, true>("rows4_tile1");
        add.template operator()<16, 4, 2, 4, true>("rows4_tile2");
        add.template operator()<8, 4, 1, 4, true>("narrow_rows4");
        add.template operator()<8, 4, 1, 4, true, 1>("token_major_rows4");
        add.template operator()<32, 4, 1, 1, true>("waves32_rows4");
        if (GufoDevIqFastEnabled()) {
          if constexpr (Type == core::GgmlType::kIQ2_S ||
                        Type == core::GgmlType::kIQ3_S ||
                        Type == core::GgmlType::kIQ3_XXS) {
            geometries.push_back(
                {"iq_fast_tile1", [&](const void* w) {
                   SmallBatchKQuantExactFp32GEMMKernel<
                       16, Batch, 3, Type, 1, 12, 1, false, Batch, 32, 32,
                       false, false, true>
                       <<<(rows + 47) / 48, 512>>>(
                           w, input.Get(), output.Get(), rows, columns);
                 }});
          }
        }
        if constexpr (Batch == 8) {
          // Q4_K's width-8 winner: two token groups of four in one launch.
          geometries.push_back(
              {"narrow_groups4", [&](const void* w) {
                 SmallBatchKQuantExactFp32GEMMKernel<16, 4, 3, Type, 1, 12, 2,
                                                     true>
                     <<<((rows + 47) / 48) * 2, 512>>>(
                         w, input.Get(), output.Get(), rows, columns);
               },
               2});
        }
        if constexpr (Batch == 5) {
          // Q4_K/Q5_K's width-5 winner: rows 0-1 then rows 2-4, each its own
          // launch, so neither partial group pays for unused width.
          geometries.push_back(
              {"b5_split2+3", [&](const void* w) {
                 const auto launch = [&]<std::size_t Width>(std::size_t off) {
                   SmallBatchKQuantExactFp32GEMMKernel<16, Width, 4, Type, 2, 4,
                                                       1, true>
                       <<<(rows + 63) / 64, 512>>>(w, input.Get() + off * columns,
                                                   output.Get() + off * rows,
                                                   rows, columns);
                 };
                 launch.template operator()<2>(0);
                 launch.template operator()<3>(2);
               },
               2});
        }
      };
      if (type == "q6")
        screen.template operator()<core::GgmlType::kQ6_K>();
      else if (type == "q3")
        screen.template operator()<core::GgmlType::kQ3_K>();
      else if (type == "iq4-nl")
        screen.template operator()<core::GgmlType::kIQ4_NL>();
      else if (type == "iq3-s")
        screen.template operator()<core::GgmlType::kIQ3_S>();
      else if (type == "iq3-xxs")
        screen.template operator()<core::GgmlType::kIQ3_XXS>();
      else if (type == "iq2-s")
        screen.template operator()<core::GgmlType::kIQ2_S>();
      else if (type == "iq2-xs")
        screen.template operator()<core::GgmlType::kIQ2_XS>();
      else if (type == "iq2-xxs")
        screen.template operator()<core::GgmlType::kIQ2_XXS>();
      else
        screen.template operator()<core::GgmlType::kQ2_K>();
    }
  } else if constexpr (Batch <= 16) {
    if (type == "q8") {
      geometries = {
          {"split8",
           [&](const void* w) {
             const auto launch = [&]<std::size_t Width>(std::size_t offset) {
               const float* x = input.Get() + offset * columns;
               float* y = output.Get() + offset * rows;
               if ((Width == 2 || Width == 4 || Width == 8) && rows == 34816 &&
                   columns == 5120) {
                 SmallBatchQ8_0ExactFp32VecGEMMKernel<16, Width, 3, true>
                     <<<(rows + 47) / 48, 512>>>(w, x, y, rows, columns);
                 return;
               }
               if constexpr (Width == 8) {
                 if (rows == 5120 && columns == 17408) {
                   SmallBatchQ8_0ExactFp32VecGEMMKernel<16, Width, 3>
                       <<<(rows + 47) / 48, 512>>>(w, x, y, rows, columns);
                   return;
                 }
               }
               SmallBatchQ8_0ExactFp32VecGEMMKernel<16, Width, 2>
                   <<<(rows + 31) / 32, 512>>>(w, x, y, rows, columns);
             };
             launch.template operator()<8>(0);
             launch.template operator()<Batch - 8>(8);
           }},
          {"wide_xor", [&](const void* w) {
             constexpr std::size_t Rows =
                 Batch == 11 || Batch == 13 || Batch == 14 ? 4
                 : Batch == 12                             ? 2
                                                           : 3;
             SmallBatchQ8_0ExactFp32VecGEMMKernel<16, Batch, Rows, true, true>
                 <<<(rows + 16 * Rows - 1) / (16 * Rows), 512>>>(
                     w, input.Get(), output.Get(), rows, columns);
           }}};
    } else {
      const auto quant = [&]<core::GgmlType Type>() {
        geometries = {
            // This generic split is an arithmetic oracle. Its geometry does
            // not represent every shape-specific production dispatch.
            {"generic_split8",
             [&](const void* w) {
               const auto launch = [&]<std::size_t Width>(std::size_t offset) {
                 SmallBatchKQuantExactFp32GEMMKernel<16, Width, 3, Type>
                     <<<(rows + 47) / 48, 512>>>(
                         w, input.Get() + offset * columns,
                         output.Get() + offset * rows, rows, columns);
               };
               launch.template operator()<8>(0);
               launch.template operator()<Batch - 8>(8);
             }},
            {"wide_wave32", [&](const void* w) {
               constexpr std::size_t Step =
                   (Type == core::GgmlType::kIQ4_XS && Batch % 2 == 0) ||
                           (Type == core::GgmlType::kQ4_K && Batch == 12)
                       ? 2
                       : 1;
               SmallBatchKQuantExactFp32GEMMKernel<8, Batch, 4, Type, 1, 4, 1,
                                                   true, Step>
                   <<<(rows + 31) / 32, 256>>>(w, input.Get(), output.Get(),
                                               rows, columns);
             }}};
      };
      if (type == "q4")
        quant.template operator()<core::GgmlType::kQ4_K>();
      else if (type == "q5")
        quant.template operator()<core::GgmlType::kQ5_K>();
      else if (type == "q6")
        quant.template operator()<core::GgmlType::kQ6_K>();
      else if constexpr (Batch == 14 || Batch == 16) {
        if (type == "q3")
          quant.template operator()<core::GgmlType::kQ3_K>();
        else if (type == "iq4-nl")
          quant.template operator()<core::GgmlType::kIQ4_NL>();
        else
          quant.template operator()<core::GgmlType::kIQ4_XS>();
      } else
        quant.template operator()<core::GgmlType::kIQ4_XS>();
    }
  }
  MappedWeights mapped_weights(weights.Get(), physical);
  // small_batch_dot4.hpp, appended after every
  // exact geometry so it is never the reference. Timing includes the
  // activation quantize launch.
  Buffer<std::int8_t> dot4_q(batch * columns);
  Buffer<float2> dot4_ds(batch * columns / 16);
  // The same int8 activations as f16 (opus_b8_vopd_* arms).
  Buffer<std::uint16_t> dot4_h(batch * columns);
  // The snb_ arms read a host-repacked copy of the same matrix in
  // the row-interleaved layout of sn_iq3.hpp. Every other arm and the
  // reference keep reading the original buffer, so nothing else changes. The
  // repacked copies keep their own rotation stride (snb_bytes).
  [[maybe_unused]] std::unique_ptr<MappedWeights> snb_mapped;
  [[maybe_unused]] std::size_t snb_bytes = 0;
  // Per-workgroup clock stamps of opus_b8_k4clk (4 u64 each).
  [[maybe_unused]] std::unique_ptr<Buffer<std::uint64_t>> clk_stamps;
  [[maybe_unused]] std::unique_ptr<Buffer<float>> sk_scratch;
  [[maybe_unused]] std::unique_ptr<Buffer<std::uint32_t>> sk_counter;
  if constexpr (Batch >= 1 && Batch <= 8) {
    if (type == "iq3-s" && !fused) {
      snb_bytes = SnIq3PackedBytes(rows, columns);
      const std::size_t snb_physical = snb_bytes * copies;
      std::vector<unsigned char> host_packed(snb_physical);
      {
        std::vector<unsigned char> host_source(physical);
        Check(hipMemcpy(host_source.data(), weights.Get(), physical,
                        hipMemcpyDeviceToHost));
        for (std::size_t copy = 0; copy < copies; ++copy)
          SnIq3Repack(host_source.data() + (copy * bytes),
                      host_packed.data() + (copy * snb_bytes), rows, columns);
      }
      Buffer<unsigned char> snb_device(snb_physical);
      Check(hipMemcpy(snb_device.Get(), host_packed.data(), snb_physical,
                      hipMemcpyHostToDevice));
      std::vector<unsigned char>().swap(host_packed);
      snb_mapped =
          std::make_unique<MappedWeights>(snb_device.Get(), snb_physical);
      clk_stamps = std::make_unique<Buffer<std::uint64_t>>(
          4 * ((rows + 127) / 128));
      // opus_b8_k4sk split-K: partials for up to 5 splits and one ticket per
      // 128-row group (zeroed here; the kernel resets each ticket it uses).
      sk_scratch = std::make_unique<Buffer<float>>(5 * Batch * rows);
      sk_counter = std::make_unique<Buffer<std::uint32_t>>((rows + 127) / 128);
      Check(hipMemset(sk_counter->Get(), 0,
                      ((rows + 127) / 128) * sizeof(std::uint32_t)));
      // Every snb_/opus_ arm reads dot4_q/dot4_ds through LaunchSnbIq3s, so
      // GUFO_DFLASH_PREQUANT covers all of them (the dot4_* arms below share
      // the buffers but keep quantizing inside their own launch).
      const bool prequant = PrequantMode();
      const auto add_snb = [&, prequant](SnbIq3Arm arm, const char* name) {
        Geometry geometry{
            name,
            [&, arm, prequant, packed = snb_mapped.get()](const void* w) {
              const auto* const original =
                  static_cast<const unsigned char*>(w);
              const std::size_t copy =
                  (original - mapped_weights.Get()) / bytes;
              LaunchSnbIq3s<Batch>(
                  arm, packed->Get() + (copy * snb_bytes), input.Get(),
                  output.Get(), dot4_q.Get(), dot4_ds.Get(),
                  static_cast<std::uint32_t>(rows),
                  static_cast<std::uint32_t>(columns), nullptr, !prequant,
                  clk_stamps->Get(), dot4_h.Get(), sk_scratch->Get(),
                  sk_counter->Get(), 5 * Batch * rows);
            },
            1, true};
        if (prequant)
          geometry.prepare = [&, arm] {  // arm by value: [&] dangled after add_snb returned (rel_error 1)
            const auto subs = static_cast<std::uint32_t>(Batch * columns / 16);
            if (arm == SnbIq3Arm::kOpusB8K4Act)
              QuantizeActivationsQ8KInterleavedKernel<<<(subs + 255U) / 256U,
                                                        256>>>(
                  input.Get(), dot4_q.Get(), dot4_ds.Get(), subs,
                  static_cast<std::uint32_t>(columns));
            else if (SnbArmWantsHalfActs(arm))
              QuantizeActivationsQ8KInterleavedKernel<<<(subs + 255U) / 256U,
                                                        256>>>(
                  input.Get(), dot4_q.Get(), dot4_ds.Get(), subs,
                  static_cast<std::uint32_t>(columns), dot4_h.Get());
            else
              QuantizeActivationsQ8KDot4Kernel<<<(subs + 255U) / 256U, 256>>>(
                  input.Get(), dot4_q.Get(), dot4_ds.Get(), subs);
          };
        geometries.push_back(std::move(geometry));
      };
      add_snb(SnbIq3Arm::kFoldR8S1W16, "snb_r8_fold");
      add_snb(SnbIq3Arm::kFoldR8S1W4, "snb_r8_fold_w4");
      add_snb(SnbIq3Arm::kFoldR8S4W16, "snb_r8_fold_s4");
      add_snb(SnbIq3Arm::kGridR8S1W16, "snb_r8_grid");
      add_snb(SnbIq3Arm::kOpusB8Sgpr, "opus_b8_sgpr");
      add_snb(SnbIq3Arm::kOpusB8SgprK4, "opus_b8_sgpr_k4");
      // k4 variants, see OpusIq3sB8Kernel.
      add_snb(SnbIq3Arm::kOpusB8K4Pad, "opus_b8_k4pad");
      add_snb(SnbIq3Arm::kOpusB8K4Slope, "opus_b8_k4slope");
      add_snb(SnbIq3Arm::kOpusB8K4Clk, "opus_b8_k4clk");
      add_snb(SnbIq3Arm::kOpusB8K4Act, "opus_b8_k4act");
      // k4cut: k4pad minus VALU, output bit-identical to k4.
      // _r = prefetch copies removed only; opus_b8_k4cut = + address shifts
      // folded (32 KiB table); k4cut64 = 64 KiB table, no extra decode op.
      add_snb(SnbIq3Arm::kOpusB8K4CutR, "opus_b8_k4cut_r");
      add_snb(SnbIq3Arm::kOpusB8K4Cut, "opus_b8_k4cut");
      add_snb(SnbIq3Arm::kOpusB8K4Cut64, "opus_b8_k4cut64");
      // Attribution: k4pad + only the address fold / only the 3-token tile.
      add_snb(SnbIq3Arm::kOpusB8K4Fold, "opus_b8_k4fold");
      add_snb(SnbIq3Arm::kOpusB8K4Pad3, "opus_b8_k4pad3");
      // Ceiling probes on k4pad (outputs meaningless; read the ms column):
      // activation s_loads stuck on block 0 / each v_dot4 replaced by v_add3_u32.
      add_snb(SnbIq3Arm::kOpusB8PActFix, "opus_b8_p_actfix");
      add_snb(SnbIq3Arm::kOpusB8PNoDot, "opus_b8_p_nodot");
      // Little's-law hypothesis: next-block loads first / raised priority.
      add_snb(SnbIq3Arm::kOpusB8K4Early, "opus_b8_k4early");
      add_snb(SnbIq3Arm::kOpusB8K4Prio, "opus_b8_k4prio");
      add_snb(SnbIq3Arm::kOpusB8K4Ep, "opus_b8_k4ep");
      // Stack / priority variants around the k4prio win.
      add_snb(SnbIq3Arm::kOpusB8K4Pf, "opus_b8_k4pf");
      add_snb(SnbIq3Arm::kOpusB8K4Prio2, "opus_b8_k4prio2");
      add_snb(SnbIq3Arm::kOpusB8K4PrioW, "opus_b8_k4prioW");
      add_snb(SnbIq3Arm::kOpusB8VopdCvp, "opus_b8_vopd_cvp");
      // k4prio + split-K across workgroups for the mid shapes.
      add_snb(SnbIq3Arm::kOpusB8K4Sk, "opus_b8_k4sk");
      // Forced S on both kernels, vopd_cv + split-K (+ setprio pair).
      add_snb(SnbIq3Arm::kOpusB8K4Sk2, "opus_b8_k4sk2");
      add_snb(SnbIq3Arm::kOpusB8K4Sk4, "opus_b8_k4sk4");
      add_snb(SnbIq3Arm::kOpusB8CvSk, "opus_b8_cvsk");
      add_snb(SnbIq3Arm::kOpusB8CvSk2, "opus_b8_cvsk2");
      add_snb(SnbIq3Arm::kOpusB8CvSk4, "opus_b8_cvsk4");
      add_snb(SnbIq3Arm::kOpusB8CvpSk, "opus_b8_cvpsk");
      // Persistent stream-K (40 WGs x 1024 threads, S via GUFO_PS_SPLITS).
      add_snb(SnbIq3Arm::kOpusB8K4Ps, "opus_b8_k4ps");
      add_snb(SnbIq3Arm::kOpusB8CvPs, "opus_b8_cvps");
      // VOPD arms: 512 v_dual_dot2acc per 256 weights x 8 tokens.
      add_snb(SnbIq3Arm::kOpusB8VopdT16, "opus_b8_vopd_t16");
      add_snb(SnbIq3Arm::kOpusB8VopdCv, "opus_b8_vopd_cv");
      add_snb(SnbIq3Arm::kOpusB8K4TwoRow, "opus_b8_k4_2row");
      add_snb(SnbIq3Arm::kOpusB8KN, "opus_b8_kN");
      add_snb(SnbIq3Arm::kOpusB8Stream, "opus_b8_stream");
      add_snb(SnbIq3Arm::kOpusB8StreamD1, "opus_b8_stream_d1");
      // Ceiling probe, reads no weights: its rel_error/top1 are meaningless
      // and it is never a quality or speed winner, only the compute +
      // activation roof of opus_b8_sgpr_k4 (nominal GB/s).
      add_snb(SnbIq3Arm::kOpusB8Alu, "opus_b8_alu");
      // Floor probes (k4 grid, no weight stream): fixed cost
      // per launch, + codebook staging, + k4's dot4 count on registers.
      // Outputs meaningless; read the ms column.
      add_snb(SnbIq3Arm::kOpusB8Null, "opus_b8_null");
      add_snb(SnbIq3Arm::kOpusB8NullStage, "opus_b8_null_cb");
      add_snb(SnbIq3Arm::kOpusB8Dot4, "opus_b8_dot4");
      // Round 5 rate arms: one instruction kind each.
      add_snb(SnbIq3Arm::kOpusB8RDot4S, "opus_b8_r_dot4s");
      add_snb(SnbIq3Arm::kOpusB8RDot4V, "opus_b8_r_dot4v");
      add_snb(SnbIq3Arm::kOpusB8RDot4U8, "opus_b8_r_dot4u8");
      add_snb(SnbIq3Arm::kOpusB8RDot8, "opus_b8_r_dot8");
      add_snb(SnbIq3Arm::kOpusB8RDot2, "opus_b8_r_dot2");
      add_snb(SnbIq3Arm::kOpusB8RDot2Acc, "opus_b8_r_dot2acc");
      add_snb(SnbIq3Arm::kOpusB8RFma, "opus_b8_r_fma");
      add_snb(SnbIq3Arm::kOpusB8RFmac, "opus_b8_r_fmac");
      add_snb(SnbIq3Arm::kOpusB8RWmma8, "opus_b8_r_wmma8");
      add_snb(SnbIq3Arm::kOpusB8RWmma4, "opus_b8_r_wmma4");
      add_snb(SnbIq3Arm::kOpusB8RDualVV, "opus_b8_r_dual_vv");
      add_snb(SnbIq3Arm::kOpusB8RDualSV, "opus_b8_r_dual_sv");
    }
  }
  // Host repack of the Q4_K weights into the SN layout. Only
  // the sn_ arms read this buffer; every existing arm keeps the original.
  const bool kquant_enabled =
      (type == "q4" || type == "q5" || type == "q6" || type == "iq4") &&
      !fused && Batch <= 8;
  const bool sn_enabled = type == "q4" && kquant_enabled;
  Buffer<unsigned char> sn_weights(sn_enabled ? physical : 1U);
  if (sn_enabled) {
    std::vector<unsigned char> host_weights(physical);
    Check(hipMemcpy(host_weights.data(), weights.Get(), physical,
                    hipMemcpyDeviceToHost));
    std::vector<unsigned char> packed_weights(physical);
    gufo::hip::sn_q4k::RepackTensor(host_weights.data(), packed_weights.data(),
                                    rows * copies, columns);
    Check(hipMemcpy(sn_weights.Get(), packed_weights.data(), physical,
                    hipMemcpyHostToDevice));
  }
  if (kquant_enabled) {
    // Shared, function-scope split-K scratch (up to S = 4)
    // and one ticket per 128-row
    // group for the q4_k4s2 / q4_k4s4 arms; reset here, the kernel resets each
    // ticket it uses.
    // S up to 8, and one ticket per 32-row group so RowWaves = 1
    // or 2 split arms (gridDim.x = m / (32*RowWaves)) stay in bounds.
    sk_scratch = std::make_unique<Buffer<float>>(8 * Batch * rows);
    sk_counter = std::make_unique<Buffer<std::uint32_t>>((rows + 31) / 32);
    Check(hipMemset(sk_counter->Get(), 0,
                    ((rows + 31) / 32) * sizeof(std::uint32_t)));
  }
  // Q4_K/Q5_K chunk interleave, host-mapped like the original weights.
  // Function scope: registered lambdas outlive the setup blocks below.
  [[maybe_unused]] std::unique_ptr<MappedWeights> kquant_il_mapped;
  // Packed min-term sums (SnQ4KPackSumsKernel), function scope.
  Buffer<std::int32_t> il_xsum(kquant_enabled && type != "iq4"
                                   ? (columns / 256) *
                                         (type == "q6" ? 9U : 5U) * Batch
                                   : 1U);
  if (kquant_enabled && rows % 32 == 0) {
    std::vector<unsigned char> host_weights(physical);
    Check(hipMemcpy(host_weights.data(), weights.Get(), physical,
                    hipMemcpyDeviceToHost));
    std::vector<unsigned char> il_weights(physical);
    // Each rotated Q6/IQ4_XS matrix owns its trailing metadata plane.
    for (std::size_t copy = 0; copy < copies; ++copy)
      InterleaveKQuantRows32(host_weights.data() + copy * bytes,
                             il_weights.data() + copy * bytes, rows, columns,
                             block_bytes);
    Buffer<unsigned char> il_device(physical);
    Check(hipMemcpy(il_device.Get(), il_weights.data(), physical,
                    hipMemcpyHostToDevice));
    kquant_il_mapped =
        std::make_unique<MappedWeights>(il_device.Get(), physical);
  }
  if constexpr (Batch >= 4 && Batch <= 8) {
    if (type != "bf16" && type != "q8" && type != "iq4-nl" && !fused) {
      const auto dot4 = [&]<core::GgmlType Type>() {
        constexpr bool kFast = Type == core::GgmlType::kIQ2_S ||
                               Type == core::GgmlType::kIQ3_S ||
                               Type == core::GgmlType::kIQ3_XXS;
        const auto add = [&]<std::size_t Rows, std::uint32_t Waves>(
                             const char* name) {
          geometries.push_back(
              {name,
               [&](const void* w) {
                 LaunchSmallBatchDot4<Type, Batch, Rows, Waves, kFast>(
                     w, input.Get(), output.Get(), dot4_q.Get(),
                     dot4_ds.Get(), static_cast<std::uint32_t>(rows),
                     static_cast<std::uint32_t>(columns), nullptr);
               },
               1, true});
        };
        add.template operator()<2, 8>("dot4_r2w8");
        add.template operator()<4, 8>("dot4_r4w8");
        add.template operator()<4, 4>("dot4_r4w4");
        add.template operator()<8, 4>("dot4_r8w4");
        // v2: LDS-staged activations, column reduction (small_batch_dot4.hpp).
        const auto add_lds = [&]<std::size_t Rows, std::uint32_t Waves,
                                 std::uint32_t Subs>(const char* name) {
          geometries.push_back(
              {name,
               [&](const void* w) {
                 LaunchSmallBatchDot4Lds<Type, Batch, Rows, Waves, Subs, kFast>(
                     w, input.Get(), output.Get(), dot4_q.Get(),
                     dot4_ds.Get(), static_cast<std::uint32_t>(rows),
                     static_cast<std::uint32_t>(columns), nullptr);
               },
               1, true});
        };
        add_lds.template operator()<2, 8, 1>("dot4s_r2w8");
        add_lds.template operator()<4, 8, 1>("dot4s_r4w8");
        add_lds.template operator()<4, 8, 2>("dot4s_r4w8s2");
        add_lds.template operator()<8, 4, 1>("dot4s_r8w4");
      };
      if (type == "q4")
        dot4.template operator()<core::GgmlType::kQ4_K>();
      else if (type == "q5")
        dot4.template operator()<core::GgmlType::kQ5_K>();
      else if (type == "q6")
        dot4.template operator()<core::GgmlType::kQ6_K>();
      else if (type == "iq4")
        dot4.template operator()<core::GgmlType::kIQ4_XS>();
      else if (type == "q3")
        dot4.template operator()<core::GgmlType::kQ3_K>();
      else if (type == "iq3-s")
        dot4.template operator()<core::GgmlType::kIQ3_S>();
      else if (type == "iq3-xxs")
        dot4.template operator()<core::GgmlType::kIQ3_XXS>();
      else if (type == "iq2-s")
        dot4.template operator()<core::GgmlType::kIQ2_S>();
    }
  }
  // v3: half-super-block lanes, Q4_K/Q5_K/Q6_K/IQ4_XS, widths 1-8.
  // Name = dot4v3_r<rows per wave>w<waves>.
  if constexpr (Batch >= 1 && Batch <= 8) {
    if ((type == "q4" || type == "q5" || type == "q6" || type == "iq4" ||
         type == "iq4-nl" || type == "q3" || type == "iq3-s" ||
         type == "iq3-xxs" || type == "iq2-s") &&
        !fused) {
      const auto v3 = [&]<core::GgmlType Type>() {
        const auto add = [&]<std::uint32_t Rows, std::uint32_t Waves,
                             std::uint32_t PerLane = 1, bool ActK = false,
                             std::uint32_t RowGroup = 0>(const char* name) {
          geometries.push_back(
              {name,
               [&](const void* w) {
                 LaunchSmallBatchDot4V3<Type, Batch, Rows, Waves, PerLane,
                                        ActK, 32, RowGroup>(
                     w, input.Get(), output.Get(), dot4_q.Get(),
                     dot4_ds.Get(), static_cast<std::uint32_t>(rows),
                     static_cast<std::uint32_t>(columns), nullptr);
               },
               1, true});
          // headq6: a RowGroup arm's twin is the SAME geometry with
          // RowGroup 0, i.e. the served kernel. Bitwise, always (grouping
          // never moves a value between rows).
          if constexpr (RowGroup != 0U) {
            geometries.back().pair_reference = [&](const void* w) {
              LaunchSmallBatchDot4V3<Type, Batch, Rows, Waves, PerLane, ActK,
                                     32, 0>(
                  w, input.Get(), output.Get(), dot4_q.Get(), dot4_ds.Get(),
                  static_cast<std::uint32_t>(rows),
                  static_cast<std::uint32_t>(columns), nullptr);
            };
          }
          // pair_reference = the R1 launch with the same ActK (same
          // activation quantizer), so RowsPerLane 2/4 must match it bitwise.
          if constexpr ((Type == core::GgmlType::kIQ3_S ||
                         Type == core::GgmlType::kIQ3_XXS ||
                         Type == core::GgmlType::kIQ2_S ||
                         Type == core::GgmlType::kQ4_K ||
                         Type == core::GgmlType::kQ5_K ||
                         Type == core::GgmlType::kQ6_K ||
                         Type == core::GgmlType::kQ3_K ||
                         Type == core::GgmlType::kIQ4_XS ||
                         Type == core::GgmlType::kIQ4_NL) &&
                        PerLane >= 2 && RowGroup == 0U) {
            geometries.back().pair_reference = [&](const void* w) {
              LaunchSmallBatchDot4V3<Type, Batch, Rows, Waves, 1, ActK>(
                  w, input.Get(), output.Get(), dot4_q.Get(), dot4_ds.Get(),
                  static_cast<std::uint32_t>(rows),
                  static_cast<std::uint32_t>(columns), nullptr);
            };
          }
          // The same kernel compiled wave64; it must match the
          // wave32 twin just pushed bitwise. Name = <twin>_w64.
          if constexpr (Type == core::GgmlType::kIQ3_S ||
                        Type == core::GgmlType::kIQ3_XXS ||
                        Type == core::GgmlType::kIQ4_XS) {
            static std::deque<std::string> w64_names;  // stable c_str()
            w64_names.push_back(std::string(name) + "_w64");
            auto twin = geometries.back().launch;
            geometries.push_back(
                {w64_names.back().c_str(),
                 [&](const void* w) {
                   if (!detail::TryLaunchDot4V3Wave64(
                           Type, Batch, Rows, Waves, PerLane, ActK, w,
                           input.Get(), output.Get(), dot4_q.Get(),
                           dot4_ds.Get(), static_cast<std::uint32_t>(rows),
                           static_cast<std::uint32_t>(columns), nullptr))
                     throw std::runtime_error("no wave64 V3 instantiation");
                 },
                 1, true});
            geometries.back().pair_reference = std::move(twin);
          }
        };
        add.template operator()<4, 4>("dot4v3_r4w4");
        add.template operator()<4, 8>("dot4v3_r4w8");
        add.template operator()<8, 4>("dot4v3_r8w4");
        add.template operator()<2, 4>("dot4v3_r2w4");
        add.template operator()<4, 2>("dot4v3_r4w2");
        // x2 = two adjacent rows per lane, Q4_K/Q5_K/Q6_K/IQ4_XS.
        if constexpr (Type == core::GgmlType::kQ4_K ||
                      Type == core::GgmlType::kQ5_K ||
                      Type == core::GgmlType::kQ6_K ||
                      Type == core::GgmlType::kIQ4_XS ||
                      Type == core::GgmlType::kIQ4_NL ||
                      Type == core::GgmlType::kQ3_K ||
                      ((Type == core::GgmlType::kIQ3_S ||
                        Type == core::GgmlType::kIQ3_XXS ||
                        Type == core::GgmlType::kIQ2_S) &&
                       Batch >= 4)) {
          add.template operator()<4, 4, 2>("dot4v3_r4w4x2");
          add.template operator()<4, 8, 2>("dot4v3_r4w8x2");
          add.template operator()<8, 4, 2>("dot4v3_r8w4x2");
          add.template operator()<2, 4, 2>("dot4v3_r2w4x2");
        }
        // x4 = four adjacent rows per lane (kernel allows it at Batch >= 6);
        // pair_reference = the R1 launch (same ActK), as x2.
        if constexpr ((Type == core::GgmlType::kQ4_K ||
                       Type == core::GgmlType::kQ5_K ||
                       Type == core::GgmlType::kQ6_K ||
                       Type == core::GgmlType::kIQ4_XS ||
                       Type == core::GgmlType::kIQ4_NL ||
                       Type == core::GgmlType::kQ3_K ||
                       Type == core::GgmlType::kIQ3_S ||
                       Type == core::GgmlType::kIQ3_XXS ||
                       Type == core::GgmlType::kIQ2_S) &&
                      Batch >= 6) {
          add.template operator()<4, 4, 4>("dot4v3_r4w4x4");
          add.template operator()<4, 8, 4>("dot4v3_r4w8x4");
          add.template operator()<8, 4, 4>("dot4v3_r8w4x4");
          add.template operator()<2, 4, 4>("dot4v3_r2w4x4");
        }
        // k = ActK: q8_K-style activations (one scale per 256), integer
        // sub-block sums (ISA: -14% VALU at Q5_K B8 x2).
        if constexpr (Type == core::GgmlType::kQ4_K ||
                      Type == core::GgmlType::kQ5_K ||
                      Type == core::GgmlType::kQ6_K ||
                      Type == core::GgmlType::kIQ4_XS) {
          add.template operator()<4, 4, 1, true>("dot4v3_r4w4k");
          add.template operator()<4, 8, 1, true>("dot4v3_r4w8k");
          add.template operator()<8, 4, 1, true>("dot4v3_r8w4k");
          add.template operator()<2, 4, 1, true>("dot4v3_r2w4k");
          add.template operator()<4, 4, 2, true>("dot4v3_r4w4x2k");
          add.template operator()<4, 8, 2, true>("dot4v3_r4w8x2k");
          add.template operator()<8, 4, 2, true>("dot4v3_r8w4x2k");
          add.template operator()<2, 4, 2, true>("dot4v3_r2w4x2k");
          if constexpr (Batch >= 6) {
            add.template operator()<4, 4, 4, true>("dot4v3_r4w4x4k");
            add.template operator()<4, 8, 4, true>("dot4v3_r4w8x4k");
            add.template operator()<8, 4, 4, true>("dot4v3_r8w4x4k");
            add.template operator()<2, 4, 4, true>("dot4v3_r2w4x4k");
            // headq6: the SERVED output-head arm (r4w8x4k ActK at
            // B7/B8, {Q6_K, 248320, 5120}) with the lane's 4 rows consumed 2
            // or 1 at a time: VGPR 252/256 -> 174/170 (_g2) or 116/122 (_g1),
            // the B8 arm's 72 B/lane of scratch -> 0, occupancy 5 -> 8 / 10,8
            // waves/SIMD. Each carries the served arm as pair_reference.
            if constexpr (Type == core::GgmlType::kQ6_K) {
              add.template operator()<4, 8, 4, true, 2>("dot4v3_r4w8x4k_g2");
              add.template operator()<4, 8, 4, true, 1>("dot4v3_r4w8x4k_g1");
            }
          }
        }
      };
      if (type == "q4")
        v3.template operator()<core::GgmlType::kQ4_K>();
      else if (type == "q5")
        v3.template operator()<core::GgmlType::kQ5_K>();
      else if (type == "q6")
        v3.template operator()<core::GgmlType::kQ6_K>();
      else if (type == "iq4")
        v3.template operator()<core::GgmlType::kIQ4_XS>();
      else if (type == "iq4-nl")
        v3.template operator()<core::GgmlType::kIQ4_NL>();
      else if (type == "q3")
        v3.template operator()<core::GgmlType::kQ3_K>();
      else if (type == "iq3-s")
        v3.template operator()<core::GgmlType::kIQ3_S>();
      else if (type == "iq3-xxs")
        v3.template operator()<core::GgmlType::kIQ3_XXS>();
      else if (type == "iq2-s")
        v3.template operator()<core::GgmlType::kIQ2_S>();
      else
        v3.template operator()<core::GgmlType::kQ6_K>();
    }
    // The repacked layout. RowsPerLane is 1 or 2; the name is
    // sn_r<RowsPerWave>w<Waves>x<RowsPerLane>, the v3 convention.
    if (sn_enabled) {
      const auto sn = [&]<std::size_t RowsPerLane, std::uint32_t SnRowsPerWave,
                         std::uint32_t SnWaves>(const char* name) {
        geometries.push_back(
            {name,
             [&](const void* w) {
               const auto offset = static_cast<std::size_t>(
                   static_cast<const unsigned char*>(w) -
                   mapped_weights.Get());
               LaunchSnQ4K<Batch, RowsPerLane, SnRowsPerWave, SnWaves>(
                   sn_weights.Get() + offset, input.Get(), output.Get(),
                   dot4_q.Get(), dot4_ds.Get(),
                   static_cast<std::uint32_t>(rows),
                   static_cast<std::uint32_t>(columns), nullptr);
             },
             1, true});
        // Anchor to the verified V3 ActK R1 decode of the same Q4_K weights:
        // the same q8_K quantizer and integer sub-block sums, so only the
        // fp32 flush order differs. Bound it instead of leaving the new
        // decode path's result as a printed number that cannot fail.
        geometries.back().pair_reference = [&](const void* w) {
          LaunchSmallBatchDot4V3<core::GgmlType::kQ4_K, Batch, SnRowsPerWave,
                                 SnWaves, 1, true>(
              w, input.Get(), output.Get(), dot4_q.Get(), dot4_ds.Get(),
              static_cast<std::uint32_t>(rows),
              static_cast<std::uint32_t>(columns), nullptr);
        };
        geometries.back().pair_rel_error = 1e-3F;
      };
      sn.template operator()<1, 2, 8>("sn_r2w8x1");
      sn.template operator()<1, 4, 8>("sn_r4w8x1");
      sn.template operator()<1, 8, 8>("sn_r8w8x1");
      sn.template operator()<2, 8, 8>("sn_r8w8x2");
      sn.template operator()<2, 4, 8>("sn_r4w8x2");
      sn.template operator()<1, 8, 4>("sn_r8w4x1");
      // q4_k4: the k4 structure (4 row-waves x 4 K-phases, 512
      // threads, in-body weight loads, one int32 256-weight sum) on the
      // ORIGINAL ggml Q4_K layout, no repack. q4_k4 = no split; q4_k4s2/s4 add
      // the cross-WG split-K (SnQ4KSplitKCombine). Anchored to the verified V3
      // ActK R1 decode (same q8_K quantizer, same integer sub-block sums), as
      // the sn_ arms.
      const bool prequant4 = PrequantMode();
      const auto add_q4k4 = [&, prequant4]<bool SplitK>(const char* name,
                                                       std::uint32_t splits) {
        Geometry geometry{
            name,
            [&, splits, prequant4](const void* wp) {
              const auto offset = static_cast<std::size_t>(
                  static_cast<const unsigned char*>(wp) - mapped_weights.Get());
              const std::uint32_t s = SnQ4KPickSplit(
                  static_cast<std::uint32_t>(columns) / 256U, splits);
              LaunchSnQ4KK4<Batch, 4, 4, SplitK>(
                  mapped_weights.Get() + offset, input.Get(), output.Get(),
                  dot4_q.Get(), dot4_ds.Get(),
                  static_cast<std::uint32_t>(rows),
                  static_cast<std::uint32_t>(columns), nullptr, s,
                  sk_scratch->Get(), sk_counter->Get(), !prequant4);
            },
            1, true};
        geometry.pair_reference = [&](const void* wp) {
          LaunchSmallBatchDot4V3<core::GgmlType::kQ4_K, Batch, 4, 4, 1, true>(
              wp, input.Get(), output.Get(), dot4_q.Get(), dot4_ds.Get(),
              static_cast<std::uint32_t>(rows),
              static_cast<std::uint32_t>(columns), nullptr);
        };
        geometry.pair_rel_error = 1e-3F;
        if (prequant4) {
          geometry.prepare = [&] {
            const auto subs =
                static_cast<std::uint32_t>(Batch * columns / 16);
            QuantizeActivationsQ8KDot4Kernel<<<(subs + 255U) / 256U, 256>>>(
                input.Get(), dot4_q.Get(), dot4_ds.Get(), subs);
          };
        }
        geometries.push_back(std::move(geometry));
      };
      add_q4k4.template operator()<false>("q4_k4", 1U);
      add_q4k4.template operator()<true>("q4_k4s2", 2U);
      add_q4k4.template operator()<true>("q4_k4s4", 4U);
    }
    if (kquant_il_mapped) {
      const bool prequant_il = PrequantMode();
      const auto add_il =
          [&, prequant_il]<std::uint32_t RowWaves, bool SplitK, bool Q5 = false,
                           std::uint32_t KPhases = 4, bool Hoist = false,
                           bool Packed = false, int Probe = 0,
                           bool Dynamic = false, bool Q6 = false,
                           bool IQ4 = false>(const char* name,
                                             std::uint32_t splits) {
            Geometry geometry{
                name,
                [&, splits, prequant_il](const void* wp) {
                  const auto offset = static_cast<std::size_t>(
                      static_cast<const unsigned char*>(wp) -
                      mapped_weights.Get());
                  const std::uint32_t s = SnQ4KPickSplit(
                      static_cast<std::uint32_t>(columns) / 256U, splits);
                  LaunchSnQ4KK4<Batch, RowWaves, KPhases, SplitK, true, Q5,
                                Hoist, Packed, Probe, Dynamic, Q6, IQ4>(
                      kquant_il_mapped->Get() + offset, input.Get(),
                      output.Get(), dot4_q.Get(), dot4_ds.Get(),
                      static_cast<std::uint32_t>(rows),
                      static_cast<std::uint32_t>(columns), nullptr, s,
                      sk_scratch->Get(), sk_counter->Get(), !prequant_il,
                      il_xsum.Get());
                },
                1, true};
            if (Probe == 0)
              geometry.pair_reference = [&](const void* wp) {
                LaunchSmallBatchDot4V3<
                    IQ4 ? core::GgmlType::kIQ4_XS
                        : (Q6 ? core::GgmlType::kQ6_K
                              : (Q5 ? core::GgmlType::kQ5_K
                                    : core::GgmlType::kQ4_K)),
                    Batch, 4, 4, 1, true>(
                    wp, input.Get(), output.Get(), dot4_q.Get(), dot4_ds.Get(),
                    static_cast<std::uint32_t>(rows),
                    static_cast<std::uint32_t>(columns), nullptr);
              };
            geometry.pair_rel_error = 1e-3F;
            if (prequant_il) {
              geometry.prepare = [&] {
                const auto subs =
                    static_cast<std::uint32_t>(Batch * columns / 16);
                QuantizeActivationsQ8KDot4Kernel<<<(subs + 255U) / 256U, 256>>>(
                    input.Get(), dot4_q.Get(), dot4_ds.Get(), subs);
                if constexpr (Packed)
                  LaunchSnQ4KPackSums<Batch, Q6>(
                      dot4_ds.Get(), il_xsum.Get(),
                      static_cast<std::uint32_t>(columns), nullptr);
              };
            }
            geometries.push_back(std::move(geometry));
          };
      if (type == "iq4") {
        // hp keeps the requested family label; IQ4_XS has no min term,
        // so Packed=false and there is no sums-packing launch.
        add_il.template operator()<2, false, false, 4, true, false, 0, false,
                                   false, true>("iq4il_r2k4hp", 1U);
        add_il.template operator()<4, false, false, 4, true, false, 0, false,
                                   false, true>("iq4il_r4k4hp", 1U);
        add_il.template operator()<1, false, false, 4, true, false, 0, false,
                                   false, true>("iq4il_r1k4hp", 1U);
        add_il.template operator()<2, false, false, 8, true, false, 0, false,
                                   false, true>("iq4il_r2k8hp", 1U);
      } else if (type == "q6") {
        add_il.template operator()<4, false, false, 4, false, false, 0, false,
                                   true>("q6il_k4", 1U);
        add_il.template operator()<2, false, false, 8, true, true, 0, false,
                                   true>("q6il_r2k8hp", 1U);
        add_il.template operator()<1, false, false, 8, true, true, 0, false,
                                   true>("q6il_r1k8hp", 1U);
      } else if (type == "q5") {
        add_il.template operator()<4, false, true>("q5il_k4", 1U);
        add_il.template operator()<2, false, true>("q5il_r2k4", 1U);
        // Grid fill (m5120 k6144: 40 WGs at RW4 vs 80 resident).
        add_il.template operator()<4, true, true>("q5il_k4s2", 2U);
        add_il.template operator()<4, true, true>("q5il_k4s3", 3U);
        add_il.template operator()<4, true, true>("q5il_k4s6", 6U);
        add_il.template operator()<2, false, true, 8>("q5il_r2k8", 1U);
        add_il.template operator()<2, true, true>("q5il_r2k4s2", 2U);
        add_il.template operator()<1, false, true, 8>("q5il_r1k8", 1U);
        // _h: all 11 chunk loads of a block issued together (Hoist).
        add_il.template operator()<4, false, true, 4, true>("q5il_k4h", 1U);
        add_il.template operator()<4, true, true, 4, true>("q5il_k4s2h", 2U);
        add_il.template operator()<4, true, true, 4, true>("q5il_k4s3h", 3U);
        add_il.template operator()<4, true, true, 4, true>("q5il_k4s6h", 6U);
        add_il.template operator()<2, false, true, 8, true>("q5il_r2k8h", 1U);
        add_il.template operator()<2, true, true, 4, true>("q5il_r2k4s2h", 2U);
        add_il.template operator()<1, false, true, 8, true>("q5il_r1k8h", 1U);
        // Round 2: K-phase count (blocks per wave = 24 / KP at k6144).
        add_il.template operator()<2, false, true, 12, true>("q5il_r2k12h", 1U);
        add_il.template operator()<1, false, true, 12, true>("q5il_r1k12h", 1U);
        add_il.template operator()<1, false, true, 24, true>("q5il_r1k24h", 1U);
        add_il.template operator()<2, false, true, 6, true>("q5il_r2k6h", 1U);
        add_il.template operator()<1, false, true, 6, true>("q5il_r1k6h", 1U);
        add_il.template operator()<4, false, true, 8, true>("q5il_r4k8h", 1U);
        add_il.template operator()<1, false, true, 16, true>("q5il_r1k16h", 1U);
        // _hp: Hoist + packed min-term sums (one scalar wait per block).
        add_il.template operator()<2, false, true, 8, true, true>("q5il_r2k8hp", 1U);
        add_il.template operator()<1, false, true, 8, true, true>("q5il_r1k8hp", 1U);
        add_il.template operator()<2, false, true, 6, true, true>("q5il_r2k6hp", 1U);
        add_il.template operator()<1, false, true, 6, true, true>("q5il_r1k6hp", 1U);
        add_il.template operator()<4, false, true, 4, true, true>("q5il_k4hp", 1U);
        add_il.template operator()<2, true, true, 4, true, true>("q5il_r2k4s2hp", 2U);
        // Timing probes, WRONG y (no pair gate): hot activations / loads only.
        add_il.template operator()<2, false, true, 8, true, true, 1>("probe_r2k8_hot", 1U);
        add_il.template operator()<2, false, true, 8, true, true, 2>("probe_r2k8_load", 1U);
        add_il.template operator()<1, false, true, 8, true, true, 2>("probe_r1k8_load", 1U);
        // Per-wave trace (stamps in sk_scratch; dumped after the rounds).
        add_il.template operator()<2, false, true, 8, true, true, 3>("trace_r2k8hp", 1U);
        add_il.template operator()<1, false, true, 8, true, true, 3>("trace_r1k8hp", 1U);
        add_il.template operator()<2, false, true, 8, true, true, 4>("trace_r2k8_load", 1U);
        // _hpd: + dynamic in-WG block claim (LDS counter per row-wave).
        add_il.template operator()<2, false, true, 8, true, true, 0, true>("q5il_r2k8hpd", 1U);
        add_il.template operator()<1, false, true, 8, true, true, 0, true>("q5il_r1k8hpd", 1U);
        add_il.template operator()<4, false, true, 4, true, true, 0, true>("q5il_k4hpd", 1U);
        add_il.template operator()<2, false, true, 6, true, true, 0, true>("q5il_r2k6hpd", 1U);
        add_il.template operator()<4, false, true, 8, true, true, 0, true>("q5il_r4k8hpd", 1U);
        add_il.template operator()<1, false, true, 16, true, true, 0, true>("q5il_r1k16hpd", 1U);
        add_il.template operator()<2, false, true, 8, true, true, 3, true>("trace_r2k8hpd", 1U);
        add_il.template operator()<2, false, true, 8, true, true, 4, true>("trace_r2k8_loadd", 1U);
      } else {
        add_il.template operator()<4, false>("q4il_k4", 1U);
        add_il.template operator()<4, true>("q4il_k4s2", 2U);
        add_il.template operator()<2, false, false, 8, true, true>(
            "q4il_r2k8hp", 1U);
        add_il.template operator()<1, false, false, 8, true, true>(
            "q4il_r1k8hp", 1U);
        // k5120 = 20 blocks: KP 8 splits them 3,3,3,3,2,2,2,2 (m6144 B8 lost
        // to q4il_k4 in an earlier sweep). KP 4/5/10 split them evenly.
        add_il.template operator()<2, false, false, 4, true, true>(
            "q4il_r2k4hp", 1U);
        add_il.template operator()<4, false, false, 4, true, true>(
            "q4il_r4k4hp", 1U);
        add_il.template operator()<2, false, false, 5, true, true>(
            "q4il_r2k5hp", 1U);
        add_il.template operator()<4, false, false, 5, true, true>(
            "q4il_r4k5hp", 1U);
        add_il.template operator()<2, false, false, 10, true, true>(
            "q4il_r2k10hp", 1U);
      }
    }
  }
  const auto quant_type = type == "q4"        ? core::GgmlType::kQ4_K
                          : type == "q5"      ? core::GgmlType::kQ5_K
                          : type == "iq4"     ? core::GgmlType::kIQ4_XS
                          : type == "q6"      ? core::GgmlType::kQ6_K
                          : type == "q3"      ? core::GgmlType::kQ3_K
                          : type == "iq4-nl"  ? core::GgmlType::kIQ4_NL
                          : type == "iq3-s"   ? core::GgmlType::kIQ3_S
                          : type == "iq3-xxs" ? core::GgmlType::kIQ3_XXS
                          : type == "iq2-s"   ? core::GgmlType::kIQ2_S
                          : type == "iq2-xs"  ? core::GgmlType::kIQ2_XS
                          : type == "iq2-xxs" ? core::GgmlType::kIQ2_XXS
                          : type == "q2"      ? core::GgmlType::kQ2_K
                                              : core::GgmlType::kF32;
  const auto launch_native = [&](const void* w) {
    return type == "q8" ? detail::TryLaunchQ8SmallBatchWave64(
                              w, input.Get(), output.Get(), batch, rows,
                              columns, nullptr)
                        : detail::TryLaunchKQuantSmallBatchWave64(
                              quant_type, w, input.Get(), output.Get(), batch,
                              rows, columns, nullptr);
  };
  if (launch_native(mapped_weights.Get())) {
    geometries.push_back({"production_wave64", [&](const void* w) {
                            if (!launch_native(w))
                              throw std::runtime_error(
                                  "wave64 shape changed during benchmark");
                          }});
  }
  // GUFO_DFLASH_ONLY=<geometry name> keeps the reference
  // (geometries.front()) plus that one candidate, so a rocprofv3 --pmc pass,
  // whose CSV drops template args, sees exactly one candidate kernel.
  if (const char* only = std::getenv("GUFO_DFLASH_ONLY")) {
    // A comma list keeps several candidates (one A/B session).
    const std::string list = std::string(",") + only + ",";
    std::vector<Geometry> keep{geometries.front()};
    for (std::size_t i = 1; i < geometries.size(); ++i)
      if (list.find("," + std::string(geometries[i].name) + ",") !=
          std::string::npos)
        keep.push_back(geometries[i]);
    if (keep.size() < 2)
      throw std::invalid_argument("GUFO_DFLASH_ONLY names no candidate geometry");
    geometries = std::move(keep);
  }
  // The strongest available oracle: run the production scalar B1 GEMV once per
  // input row. It shares no geometry with any candidate, so a candidate that
  // matches it bitwise cannot be matching a shared bug. Only the formats
  // LaunchQ8KBlockGEMV decodes (every DecodeQuantSub16 format) qualify, and
  // only unfused widths 2-8, where one row of the batch is one GEMV.
  //
  // Whether the batched kernels agree with it bitwise is only proven for some
  // formats, so the oracle is checked against the production geometry first
  // and dropped if they disagree. A format whose decode order differs from the
  // GEMV's must still be benchable; otherwise one unproven format would waste a
  // whole queued GPU run. The chosen reference is printed on its own line above
  // the CSV so no row is ever read as a stronger check than it was.
  bool scalar_b1_reference =
      Batch > 1 && Batch <= 8 && !fused && type != "bf16" && type != "q8";
  if (scalar_b1_reference) {
    for (std::size_t token = 0; token < batch; ++token) {
      LaunchQ8KBlockGEMV(mapped_weights.Get(), quant_type,
                         input.Get() + token * columns,
                         reference.Get() + token * rows, rows, columns,
                         nullptr);
    }
    Check(hipGetLastError());
    Check(hipMemset(mismatch.Get(), 0, sizeof(unsigned)));
    geometries.front().launch(mapped_weights.Get());
    Check(hipGetLastError());
    Compare<<<(batch * rows + 255) / 256, 256>>>(
        reference.Get(), output.Get(), batch * rows, mismatch.Get());
    unsigned bad = 0;
    Check(hipMemcpy(&bad, mismatch.Get(), sizeof(bad), hipMemcpyDeviceToHost));
    if (bad != 0) {
      std::fprintf(stderr,
                   "note: production B1 GEMV differs from %s in %u of %zu "
                   "outputs; using %s as the reference instead\n",
                   geometries.front().name, bad, batch * rows,
                   geometries.front().name);
      scalar_b1_reference = false;
    }
  }
  if (!scalar_b1_reference) {
    geometries.front().launch(mapped_weights.Get());
    Check(hipGetLastError());
    Check(hipMemcpy(reference.Get(), output.Get(), batch * rows * sizeof(float),
                    hipMemcpyDeviceToDevice));
  }
  Event start, stop;
  const std::string label = mixed ? type + "/" + up_type : type;
  // GUFO_DFLASH_CHAIN=1 makes consecutive timed launches dependent, so the
  // ranking is taken under decode-like serialization instead of a full queue.
  // Chained and unchained numbers are not comparable to each other; the mode is
  // printed so a log can never be read as the wrong one.
  const char* chain_flag = std::getenv("GUFO_DFLASH_CHAIN");
  if (chain_flag != nullptr &&
      !((chain_flag[0] == '0' || chain_flag[0] == '1') && chain_flag[1] == '\0'))
    throw std::invalid_argument("GUFO_DFLASH_CHAIN must be unset, 0, or 1");
  const bool chain = chain_flag != nullptr && chain_flag[0] == '1';
  std::printf("chain,%d\n", chain ? 1 : 0);
  // Which timing the snb_/opus_ rows use: 1 = GEMM only (activations quantized
  // once before the loop), 0 = quantize + GEMM per launch (the default).
  std::printf("prequant,%d\n", PrequantMode() ? 1 : 0);
  std::printf("reference,%s\n", scalar_b1_reference
                                    ? "production_b1_gemv_per_row"
                                    : geometries.front().name);
  std::puts(
      "type,m,k,batch,round,geometry,weight_passes,ms,nominal_weight_GBps,"
      "mismatches,quantize_ms,total_ms,rel_error,top1_agreement");
  for (int round = 0; round < 4; ++round) {
    for (std::size_t index = 0; index < geometries.size(); ++index) {
      const auto& geometry =
          geometries[round % 2 == 0 ? index : geometries.size() - index - 1];
      std::vector<float> pair_expected;
      if (geometry.pair_reference) {
        pair_expected.resize(batch * rows);
        geometry.pair_reference(mapped_weights.Get());
        Check(hipGetLastError());
        Check(hipMemcpy(pair_expected.data(), output.Get(),
                        batch * rows * sizeof(float), hipMemcpyDeviceToHost));
      }
      Check(hipMemset(mismatch.Get(), 0, sizeof(unsigned)));
      // Do not let R1's output conceal missing R2 stores, including odd-M
      // tails.
      // NaN-fill before EVERY geometry, else skipped stores keep reference values.
      Check(hipMemset(output.Get(), 0xFF, batch * rows * sizeof(float)));
      if (geometry.prepare)
        geometry.prepare();
      geometry.launch(mapped_weights.Get());
      Check(hipGetLastError());
      Compare<<<(batch * rows + 255) / 256, 256>>>(
          reference.Get(), output.Get(), batch * rows, mismatch.Get());
      unsigned bad = 0;
      Check(
          hipMemcpy(&bad, mismatch.Get(), sizeof(bad), hipMemcpyDeviceToHost));
      if (bad && !geometry.approximate)
        throw std::runtime_error("geometry changed " + std::to_string(bad) +
                                 " outputs");
      if (!pair_expected.empty()) {
        std::vector<float> pair_actual(batch * rows);
        Check(hipMemcpy(pair_actual.data(), output.Get(),
                        batch * rows * sizeof(float), hipMemcpyDeviceToHost));
        const bool nonfinite =
            std::any_of(pair_actual.begin(), pair_actual.end(),
                        [](float value) { return !std::isfinite(value); });
        const bool bitwise =
            std::memcmp(pair_expected.data(), pair_actual.data(),
                        batch * rows * sizeof(float)) == 0;
        if (nonfinite || (!bitwise && geometry.pair_rel_error <= 0.0F))
          throw std::runtime_error(
              std::string(geometry.name) +
              " is nonfinite or differs bitwise from corresponding R1 V3");
        if (!bitwise) {
          float max_error = 0.0F;
          float max_ref = 0.0F;
          for (std::size_t i = 0; i < batch * rows; ++i) {
            const float error = std::fabs(pair_expected[i] - pair_actual[i]);
            max_error = std::isfinite(error)
                            ? std::max(max_error, error)
                            : std::numeric_limits<float>::infinity();
            max_ref = std::max(max_ref, std::fabs(pair_expected[i]));
          }
          const float pair_error =
              max_ref > 0.0F ? max_error / max_ref : max_error;
          if (!(pair_error <= geometry.pair_rel_error))
            throw std::runtime_error(
                std::string(geometry.name) +
                " differs from corresponding R1 V3 by rel_error " +
                std::to_string(pair_error));
        }
      }
      float rel_error = 0.0F;
      float top1 = 1.0F;
      if (geometry.approximate) {
        std::vector<float> expected(batch * rows), actual(batch * rows);
        Check(hipMemcpy(expected.data(), reference.Get(),
                        batch * rows * sizeof(float), hipMemcpyDeviceToHost));
        Check(hipMemcpy(actual.data(), output.Get(),
                        batch * rows * sizeof(float), hipMemcpyDeviceToHost));
        float max_error = 0.0F, max_ref = 0.0F;
        std::size_t matches = 0;
        for (std::size_t token = 0; token < batch; ++token) {
          std::size_t best_expected = token * rows, best_actual = token * rows;
          for (std::size_t i = token * rows; i < (token + 1) * rows; ++i) {
            const float error = std::fabs(expected[i] - actual[i]);
            max_error = std::isfinite(error)
                            ? std::max(max_error, error)
                            : std::numeric_limits<float>::infinity();
            max_ref = std::max(max_ref, std::fabs(expected[i]));
            best_expected = expected[i] > expected[best_expected] ? i : best_expected;
            best_actual = actual[i] > actual[best_actual] ? i : best_actual;
          }
          matches += best_expected == best_actual;
        }
        rel_error = max_ref > 0.0F ? max_error / max_ref : max_error;
        top1 = static_cast<float>(matches) / static_cast<float>(batch);
      }
      if (geometry.prepare) {  // untimed: finished before the start event
        geometry.prepare();
        Check(hipGetLastError());
        Check(hipDeviceSynchronize());
      }
      Check(hipEventRecord(start.Get()));
      for (int iteration = 0; iteration < iterations; ++iteration) {
        geometry.launch(mapped_weights.Get() + (iteration % copies) * bytes);
        if (chain)
          ChainDependency<<<1, 1>>>(output.Get(), input.Get());
      }
      Check(hipEventRecord(stop.Get()));
      Check(hipEventSynchronize(stop.Get()));
      float milliseconds = 0;
      Check(hipEventElapsedTime(&milliseconds, start.Get(), stop.Get()));
      milliseconds /= iterations;
      // quantize_ms is always zero here: with GUFO_DFLASH_PREQUANT=1 the
      // snb_/opus_ quantize is untimed (see the prequant header line), else
      // it is inside ms. The column exists so the CSV matches the earlier GEMM
      // bench's and both logs can be read by one parser.
      std::printf("%s,%zu,%zu,%zu,%d,%s,%zu,%.6f,%.2f,%u,%.6f,%.6f,%.3g,%.3f\n",
                  label.c_str(), rows, columns, batch, round, geometry.name,
                  geometry.weight_passes, milliseconds,
                  bytes * geometry.weight_passes / (milliseconds * 1e6), bad,
                  0.0F, milliseconds, rel_error, top1);
    }
  }
  // trace_* arms: 24 back-to-back launches (like a timed round),
  // then dump the stamps of the last one; scratch zeroed first, a wave's row
  // is u64[8] = start, block ends 0..3, after reduce, end, smid.
  if (const char* out = std::getenv("GUFO_TRACE_OUT")) {
    for (const auto& geometry : geometries) {
      const std::string name = geometry.name;
      if (name.rfind("trace_", 0) != 0 || !sk_scratch)
        continue;
      const std::size_t words = 8 * Batch * rows / 2;  // u64 in the scratch
      if (geometry.prepare)
        geometry.prepare();
      for (int iteration = 0; iteration < iterations; ++iteration) {
        if (iteration == iterations - 1) {
          Check(hipDeviceSynchronize());
          Check(hipMemset(sk_scratch->Get(), 0, words * 8));
        }
        geometry.launch(mapped_weights.Get() + (iteration % copies) * bytes);
      }
      Check(hipGetLastError());
      Check(hipDeviceSynchronize());
      std::vector<std::uint64_t> stamps(words);
      Check(hipMemcpy(stamps.data(), sk_scratch->Get(), words * 8,
                      hipMemcpyDeviceToHost));
      FILE* f = std::fopen((std::string(out) + "." + name + ".csv").c_str(), "w");
      if (f == nullptr)
        throw std::runtime_error("GUFO_TRACE_OUT not writable");
      std::fputs("wave,start,b0,b1,b2,b3,red,end,smid\n", f);
      for (std::size_t w = 0; w < words / 8; ++w) {
        const std::uint64_t* r = &stamps[8 * w];
        if (r[0] == 0)
          continue;
        std::fprintf(f, "%zu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu\n", w,
                     (unsigned long long)r[0], (unsigned long long)r[1],
                     (unsigned long long)r[2], (unsigned long long)r[3],
                     (unsigned long long)r[4], (unsigned long long)r[5],
                     (unsigned long long)r[6], (unsigned long long)r[7]);
      }
      std::fclose(f);
    }
  }
  // One more launch of opus_b8_k4clk (the timed rounds may end on
  // another arm), then the run's median over workgroups, once, on stderr.
  // Ticks are the 100 MHz steady counter; mhz = shader cycles / ticks * 100
  // (assumes HW_REG_SHADER_CYCLES counts one per shader clock: unverified).
  for (const auto& geometry : geometries) {
    if (std::string(geometry.name) != "opus_b8_k4clk")
      continue;
    const std::size_t blocks = (rows + 127) / 128;
    if (geometry.prepare)
      geometry.prepare();
    geometry.launch(mapped_weights.Get());
    Check(hipGetLastError());
    Check(hipDeviceSynchronize());
    std::vector<std::uint64_t> stamps(4 * blocks);
    Check(hipMemcpy(stamps.data(), clk_stamps->Get(),
                    stamps.size() * sizeof(std::uint64_t),
                    hipMemcpyDeviceToHost));
    std::vector<std::uint64_t> ticks;
    std::vector<double> mhz;
    for (std::size_t b = 0; b < blocks; ++b) {
      const std::uint64_t rt = stamps[(4 * b) + 1] - stamps[4 * b];
      // HW_REG_SHADER_CYCLES is 20 bits wide: mask the wrapped delta.
      const std::uint64_t mt =
          (stamps[(4 * b) + 3] - stamps[(4 * b) + 2]) & 0xFFFFFULL;
      if (rt == 0)
        continue;
      ticks.push_back(rt);
      mhz.push_back(static_cast<double>(mt) / static_cast<double>(rt) * 100.0);
    }
    if (ticks.empty())
      throw std::runtime_error("opus_b8_k4clk: no workgroup stamps");
    const auto median = [](auto values) {
      const auto middle = values.begin() + (values.size() / 2);
      std::nth_element(values.begin(), middle, values.end());
      return *middle;
    };
    std::fprintf(stderr, "K4CLK median_ticks=%llu mhz=%.1f\n",
                 static_cast<unsigned long long>(median(ticks)),
                 median(mhz));
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 4 || argc > 8)
      throw std::invalid_argument(
          "usage: dflash_gemm_bench "
          "bf16|q8|q4|q5|iq4|q6|q3|iq4-nl|iq3-s|iq3-xxs|iq2-s|iq2-xs|iq2-xxs|q2"
          " M K [iterations] [batch=1..16|24|32] [swiglu [up-quant]]");
    const bool fused = argc >= 7;
    const int batch = argc >= 6                      ? std::stoi(argv[5])
                      : std::string(argv[1]) == "q6" ? 7
                                                     : 8;
    const int iterations = argc >= 5 ? std::stoi(argv[4]) : 24;
    if (fused && (batch != 1 || std::string(argv[6]) != "swiglu"))
      throw std::invalid_argument("swiglu requires batch=1");
    if (batch == 1) {
      Run<1>(argv[1], std::stoull(argv[2]), std::stoull(argv[3]), iterations,
             fused, argc == 8 ? argv[7] : "");
    } else if (batch == 2) {
      Run<2>(argv[1], std::stoull(argv[2]), std::stoull(argv[3]), iterations);
    } else if (batch == 3) {
      Run<3>(argv[1], std::stoull(argv[2]), std::stoull(argv[3]), iterations);
    } else if (batch == 4) {
      Run<4>(argv[1], std::stoull(argv[2]), std::stoull(argv[3]), iterations);
    } else if (batch == 5) {
      Run<5>(argv[1], std::stoull(argv[2]), std::stoull(argv[3]), iterations);
    } else if (batch == 6) {
      Run<6>(argv[1], std::stoull(argv[2]), std::stoull(argv[3]), iterations);
    } else if (batch == 7) {
      Run<7>(argv[1], std::stoull(argv[2]), std::stoull(argv[3]), iterations);
    } else if (batch == 8) {
      Run<8>(argv[1], std::stoull(argv[2]), std::stoull(argv[3]), iterations);
    } else if (batch == 9) {
      Run<9>(argv[1], std::stoull(argv[2]), std::stoull(argv[3]), iterations);
    } else if (batch == 10) {
      Run<10>(argv[1], std::stoull(argv[2]), std::stoull(argv[3]), iterations);
    } else if (batch == 11) {
      Run<11>(argv[1], std::stoull(argv[2]), std::stoull(argv[3]), iterations);
    } else if (batch == 12) {
      Run<12>(argv[1], std::stoull(argv[2]), std::stoull(argv[3]), iterations);
    } else if (batch == 13) {
      Run<13>(argv[1], std::stoull(argv[2]), std::stoull(argv[3]), iterations);
    } else if (batch == 14) {
      Run<14>(argv[1], std::stoull(argv[2]), std::stoull(argv[3]), iterations);
    } else if (batch == 15) {
      Run<15>(argv[1], std::stoull(argv[2]), std::stoull(argv[3]), iterations);
    } else if (batch == 16) {
      Run<16>(argv[1], std::stoull(argv[2]), std::stoull(argv[3]), iterations);
    } else if (batch == 24) {
      Run<24>(argv[1], std::stoull(argv[2]), std::stoull(argv[3]), iterations);
    } else if (batch == 32) {
      Run<32>(argv[1], std::stoull(argv[2]), std::stoull(argv[3]), iterations);
    } else {
      throw std::invalid_argument(
          "batch must be one scalar row, 2–8 verification rows, or 9–16 "
          "shared projection rows; bf16 also supports 24/32 rows");
    }
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
