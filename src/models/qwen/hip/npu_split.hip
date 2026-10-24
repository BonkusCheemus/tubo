// DEV: GPU + NPU split of f16 prefill GEMMs (ffn_down, ffn gate/up).
// GPU computes rows [0, m_gpu) with the usual kernel; the NPU computes rows
// [m_gpu, m) with the IRON flm bf16 GEMM (AIE2P, B packed bfp16ebs8) on the
// same activations. All NPU buffers are GPU memory shared by dma-buf.
// Per GEMM, on the stream: x f16 -> A bf16 (rows padded to 256), Q rows ->
// B bfp16 (packed byte-exact vs IRON), event. A worker thread waits on
// the event, runs the NPU, then writes a host flag; the stream waits on the
// flag (hipStreamWaitValue32) and merges C (bf16): ffn_down adds it into the
// residual rows; gate/up (gate rows in C columns [0, n), up rows in [n, 2n))
// applies the same SwiGLU as the GPU epilogue and stores f16.
// Env: GUFO_NPU_SPLIT=<FLM_GEMM_..._npu2.xclbin>; instruction files
// <xclbin stem>_M<M>_K<K>_N<N>.bin beside it (IRON op.compile() names).
// GUFO_NPU_ROWS (ffn_down rows, default 1536, multiple of 256: whole GPU row tiles; 0 = off),
// GUFO_NPU_GU_ROWS (gate/up rows each, default 0 = off, multiple of 128; NPU N = 2x),
// GUFO_NPU_OUT_ROWS (attn_output/ssm_out rows, k 6144 -> m 5120 residual, default 0 = off, multiple of 256),
// GUFO_NPU_STORE_PCT (store-epilogue GEMMs k 5120, 4096 <= m < 17408: attn_qkv/attn_gate/attn_q; NPU takes
// pct% of the rows rounded down to 256; default 0 = off),
// GUFO_NPU_MAX_M (default 8192), GUFO_NPU_MIN_M (default 2048),
// GUFO_NPU_BO_SYNC (default 1), GUFO_NPU_LOG_JOBS (default 4), GUFO_NPU_FLAG=signal (flag in
// hipMallocSignalMemory instead of coherent host memory). No .bin for a shape = GPU-only for that call.
// GUFO_NPU_BCACHE=<file>: packed-B cache on SSD. First cycle (one pass over every split
// GEMM) packs on the GPU as usual and appends each job's B to the file; after that an I/O thread
// streams the next jobs' B from the file (O_DIRECT) into a ring of host buffers the NPU reads,
// and the GPU pack is skipped. GUFO_NPU_BCACHE_SLOTS (ring size, default 4).
// Tagged IQ3 gate/up (scratch copies, SetScratchAliases) are keyed and fingerprinted by
// their logical tensor (npu_bcache_key.hpp); prefill admits them with GUFO_NPU_BCACHE_TAGGED=1.
#if defined(ENGINE_ENABLE_HIP) && defined(GUFO_ENABLE_NPU)
#include "src/models/qwen/hip/npu_split.hpp"

#include <hip/hip_bf16.h>
#include <xrt/experimental/xrt_xclbin.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <cerrno>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/hip/npu_bfp16_pack.hpp"
#include "src/models/qwen/hip/quant_ops.hpp"

namespace gufo::hip::npu {
namespace {
// ffn_down: k kFfn -> m kHidden (residual). gate/up: k kHidden -> m kFfn each.
// attn_output / ssm_out: k kOut -> m kHidden (residual).
constexpr std::size_t kFfn = 17408, kHidden = 5120, kOut = 6144;
// store ops: k kHidden, m in [kStoreMinM, kFfn) (below 4096 DirectGemm's grid ignores m_grid)
constexpr std::size_t kStoreMinM = 4096;
// ffn_down m < 4096 would take DirectGemm kernels whose grid ignores m_grid (double add).
static_assert(kHidden >= 4096);

Bfp16PackShape Shape(std::size_t k, std::size_t n) {
  return {std::uint32_t(k), std::uint32_t(n), 512, 64, 8, 8, 128};  // tn64 kt512 ck128
}

__global__ void HalfToBf16Pad(const __half* x, __hip_bfloat16* a,
                              std::size_t live, std::size_t total) {
  for (std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
       i < total; i += std::size_t(gridDim.x) * blockDim.x)
    a[i] = __float2bfloat16(i < live ? __half2float(x[i]) : 0.0F);
}

// Both packs: weight rows [row0, row0 + n_count) -> B columns [n_off, n_off + n_count).
// One thread per 16-weight sub-block = two 9-byte bfp16 groups. Weight value
// as the f16 GEMM decodes it before its f16 rounding: scale * q - offset.
__global__ void PackRows(core::GgmlType type, const std::uint8_t* w, std::size_t row0,
                         std::uint32_t n_off, std::uint32_t n_count, std::uint8_t* b,
                         Bfp16PackShape p) {
  const std::size_t per_row = p.K / 16, subs = std::size_t(n_count) * per_row;
  const std::size_t row_bytes = QuantRowBytes(type, p.K);
  for (std::size_t s = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
       s < subs; s += std::size_t(gridDim.x) * blockDim.x) {
    const auto n = std::uint32_t(s / per_row), k0 = std::uint32_t(s % per_row) * 16;
    QuantSub16 q;
    DecodeQuantSub16(type, w + (row0 + n) * row_bytes, k0 / 16, q);
    float v[16];
#pragma unroll
    for (int j = 0; j < 16; ++j)
      v[j] = __fsub_rn(__fmul_rn(q.scale, static_cast<float>(q.q[j])), q.offset);
    bfp16_pack_group(v, b + bfp16_group_index(p, n_off + n, k0) * 9);
    bfp16_pack_group(v + 8, b + bfp16_group_index(p, n_off + n, k0 + 8) * 9);
  }
}

// Measured: the old PackRows took ~5 ms per call at M 2048 (scattered byte stores, 72-byte stride).
// Here one thread = 8 rows (one t_in run) x 16 k (groups i, i+1): 16 consecutive
// groups = 144 contiguous bytes, 8-byte aligned (group index of t_in 0 is a
// multiple of 8, x 9 bytes; n_off % 8 == 0), stored as 18 words. Same bytes as
// PackRows (GUFO_NPU_PACK_CHECK=1 compares both on the first calls).
__global__ void PackRows8(core::GgmlType type, const std::uint8_t* w, std::size_t row0,
                          std::uint32_t n_off, std::uint32_t n_count, std::uint8_t* b,
                          Bfp16PackShape p) {
  const std::size_t per_row = p.K / 16, items = std::size_t(n_count / 8) * per_row;
  const std::size_t row_bytes = QuantRowBytes(type, p.K);
  for (std::size_t s = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
       s < items; s += std::size_t(gridDim.x) * blockDim.x) {
    const auto n8 = std::uint32_t(s / per_row) * 8, k0 = std::uint32_t(s % per_row) * 16;
    std::uint64_t out[18] = {};
#pragma unroll
    for (int r = 0; r < 8; ++r) {
      QuantSub16 q;
      DecodeQuantSub16(type, w + (row0 + n8 + r) * row_bytes, k0 / 16, q);
      float v[16];
#pragma unroll
      for (int j = 0; j < 16; ++j)
        v[j] = __fsub_rn(__fmul_rn(q.scale, static_cast<float>(q.q[j])), q.offset);
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        std::uint8_t g[9];
        bfp16_pack_group(v + 8 * h, g);
#pragma unroll
        for (int j = 0; j < 9; ++j) {
          const int pos = (h * 8 + r) * 9 + j;  // group order: i (h) outer, t_in (r) inner
          out[pos >> 3] |= std::uint64_t(g[j]) << (8 * (pos & 7));
        }
      }
    }
    auto* dst = reinterpret_cast<std::uint64_t*>(b + bfp16_group_index(p, n_off + n8, k0) * 9);
#pragma unroll
    for (int j = 0; j < 18; ++j) dst[j] = out[j];
  }
}

__global__ void StoreNpuRows(float* y, const __hip_bfloat16* c, std::size_t batch,
                             std::size_t m, std::size_t m_gpu, std::size_t n) {
  for (std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
       i < batch * n; i += std::size_t(gridDim.x) * blockDim.x)
    y[(i / n) * m + m_gpu + i % n] = __bfloat162float(c[i]);
}

__global__ void AddNpuRows(float* y, const __hip_bfloat16* c, std::size_t batch,
                           std::size_t m, std::size_t m_gpu, std::size_t n) {
  for (std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
       i < batch * n; i += std::size_t(gridDim.x) * blockDim.x) {
    float& out = y[(i / n) * m + m_gpu + i % n];
    out = __fadd_rn(out, __bfloat162float(c[i]));
  }
}

// C row t = [gate rows (n) | up rows (n)]. Same math as the kGateUp epilogue
// (prefill_fp16.hip), on bf16-rounded g and u.
__global__ void GateUpNpuRows(__half* y, const __hip_bfloat16* c, std::size_t batch,
                              std::size_t m, std::size_t m_gpu, std::size_t n) {
  for (std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
       i < batch * n; i += std::size_t(gridDim.x) * blockDim.x) {
    const std::size_t t = i / n, j = i % n;
    const float g = __bfloat162float(c[t * 2 * n + j]);
    const float u = __bfloat162float(c[t * 2 * n + n + j]);
    const float sig = 1.0F / (1.0F + expf(-g));
    float value = __fmul_rn(g * sig, u);
    asm volatile("" : "+v"(value));  // RoundActivation's FP32 boundary
    y[t * m + m_gpu + j] = __float2half_rn(value);
  }
}

std::size_t PageUp(std::size_t bytes) { return (bytes + 4095) / 4096 * 4096; }

std::size_t EnvSize(const char* name, std::size_t fallback) {
  const char* e = std::getenv(name);
  return e != nullptr ? std::strtoull(e, nullptr, 10) : fallback;
}

xrt::bo ImportGpuBuffer(xrt::device& dev, void* ptr, std::size_t bytes) {
  int fd = -1;
  HIP_CHECK(hipMemGetHandleForAddressRange(
      &fd, reinterpret_cast<hipDeviceptr_t>(ptr), bytes,
      hipMemRangeHandleTypeDmaBufFd, 0));
  return xrt::bo(dev, xrt::bo::export_handle(fd));
}


// FNV-1a of the first 256 bytes of weight row row0: identifies the packed slice (review 0d49307: tensor start is not packed).
__global__ void RowFingerprint(core::GgmlType type, const std::uint8_t* w, std::size_t row0, std::size_t k,
                               std::uint64_t* out) {
  const std::uint8_t* p = w + row0 * QuantRowBytes(type, k);
  std::uint64_t h = 1469598103934665603ULL;
  for (int i = 0; i < 256; ++i) h = (h ^ p[i]) * 1099511628211ULL;
  *out = h;
}

// Same hash for a tagged scratch copy, read from the logical root (weights never change,
// so no wait on the unpack that fills the scratch on the prefill stream).
__global__ void RowFingerprintLogical(LogicalWeight src, std::size_t row0, std::uint64_t* out) {
  const auto* root = static_cast<const std::uint8_t*>(src.root);
  *out = LogicalRowFnv(src, row0, [root](std::size_t off) { return root[off]; });
}

// SetScratchAliases: {scratch pointer, logical source} for the tagged gate/up of this layer.
std::array<std::pair<const void*, LogicalWeight>, 2> g_alias{};
const LogicalWeight* AliasOf(const void* w) {
  for (const auto& a : g_alias)
    if (a.first != nullptr && a.first == w) return &a.second;
  return nullptr;
}

std::size_t Page4k(std::size_t bytes) { return (bytes + 4095) / 4096 * 4096; }

// One weight-row range -> B columns, as Pack() takes it.
struct PackSpec {
  std::int32_t type;
  std::uint32_t K, N, row0, n_off, n_count;
  bool operator==(const PackSpec&) const = default;
};

// File: page 0 = FileHead; records from page 1, each Page4k-aligned (O_DIRECT); index
// (RecHead[nrec]) at index_off. Records are in call order, so reading on wraps the cycle.
struct RecHead {
  std::uint32_t npacks;
  PackSpec packs[2];
  std::uint64_t fp[2];  // FNV-1a of the first 256 bytes of each weight tensor
  std::uint64_t bytes, data_off;
};
struct FileHead {
  char magic[8];  // "GNPB0003"
  std::uint64_t down_rows, gu_rows, out_rows, store_pct, nrec, index_off;
};

std::uint64_t Fnv(const std::uint8_t* p, std::size_t n) {
  std::uint64_t h = 1469598103934665603ULL;
  for (std::size_t i = 0; i < n; ++i) h = (h ^ p[i]) * 1099511628211ULL;
  return h;
}

// Arm A. Prefill thread: Acquire (replay) or Record (first cycle). I/O thread:
// fills free slots in record order. NPU worker: WaitReady before the run, Release after.
struct BCache {
  enum class St { kFree, kLoading, kReady };
  struct Slot {
    xrt::bo bo;
    std::uint8_t* host = nullptr;
    St st = St::kFree;
    bool claimed = false;
    std::size_t rec = 0;
    std::uint64_t gen = 0;
  };
  std::string path;
  FileHead want{};  // env this file must match
  std::size_t slot_bytes = 0;
  // record mode (fd_w >= 0)
  int fd_w = -1;
  std::uint8_t* staging = nullptr;  // pinned, page-aligned
  std::uint64_t w_off = 4096;
  std::vector<RecHead> recs;
  WeightKey first_key{};
  PackSpec first_spec{};
  // replay mode (fd_r >= 0)
  int fd_r = -1;
  std::vector<Slot> slots;
  std::size_t fill = 0, consume = 0;
  std::uint64_t gen = 0;
  std::mutex mu;
  std::condition_variable cv;
  hipStream_t fp_stream = nullptr;
  std::uint64_t* fp_host = nullptr;  // pinned, RowFingerprint output
  std::map<WeightKey, std::uint64_t> fp_of;  // KeyOf(w, row0), prefill thread only
  std::size_t hits = 0, resyncs = 0, gpu_packs = 0;
  std::size_t check = 0;  // GUFO_NPU_BCACHE_CHECK: replayed jobs still to byte-compare against a GPU pack
  std::uint8_t* bounce = nullptr;  // set when O_DIRECT cannot read into the BO mapping (I/O thread only)

  std::uint64_t Fingerprint(const PackSpec& sp, const void* w) {
    const LogicalWeight* src = AliasOf(w);
    const WeightKey key = KeyOf(w, sp.row0, src);
    auto it = fp_of.find(key);
    if (it != fp_of.end()) return it->second;
    // weights never change: a private stream need not wait for the prefill stream
    if (src != nullptr) {
      const auto want = src->layout == models::QwenWeightLayout::kIq3SRows8     ? core::GgmlType::kIQ3_S
                        : src->layout == models::QwenWeightLayout::kIq3XxsRows4 ? core::GgmlType::kIQ3_XXS
                                                                               : core::GgmlType(-1);
      if (core::GgmlType(sp.type) != want || src->columns != sp.K ||
          src->columns / 256 * models::Iq3BlockBytes(src->layout) < 256 ||
          src->row_begin + sp.row0 >= src->full_rows)
        throw std::runtime_error("npu bcache: scratch alias does not match the packed weight");
      RowFingerprintLogical<<<1, 1, 0, fp_stream>>>(*src, sp.row0, fp_host);
    } else {
      RowFingerprint<<<1, 1, 0, fp_stream>>>(core::GgmlType(sp.type), static_cast<const std::uint8_t*>(w),
                                             sp.row0, sp.K, fp_host);
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipStreamSynchronize(fp_stream));
    return fp_of[key] = *fp_host;
  }

  bool Matches(const RecHead& rec, const PackSpec* specs, const void* const* ws, int n) {
    if (rec.npacks != std::uint32_t(n)) return false;
    for (int i = 0; i < n; ++i)
      if (!(rec.packs[i] == specs[i]) || rec.fp[i] != Fingerprint(specs[i], ws[i])) return false;
    return true;
  }

  // Open an existing file for replay; false = absent or another env/version.
  // start = first record the I/O thread loads (set before the thread starts).
  bool OpenReplay(xrt::device& dev, xrt::kernel& kern, std::size_t nslots, std::size_t start = 0) {
    fd_r = ::open(path.c_str(), O_RDONLY | O_DIRECT);
    if (fd_r < 0) return false;
    alignas(4096) static std::uint8_t page[4096];
    FileHead h{};
    if (::pread(fd_r, page, 4096, 0) != 4096) return CloseReplay();
    std::memcpy(&h, page, sizeof h);
    if (std::memcmp(h.magic, want.magic, 8) || h.down_rows != want.down_rows ||
        h.gu_rows != want.gu_rows || h.out_rows != want.out_rows ||
        h.store_pct != want.store_pct || h.nrec == 0) {
      std::fprintf(stderr, "npu bcache: %s is for another env, re-recording\n", path.c_str());
      return CloseReplay();
    }
    recs.resize(h.nrec);
    const std::size_t idx_bytes = Page4k(h.nrec * sizeof(RecHead));
    std::vector<std::uint8_t> idx(idx_bytes + 4096);
    auto* aligned = reinterpret_cast<std::uint8_t*>((reinterpret_cast<std::uintptr_t>(idx.data()) + 4095) / 4096 * 4096);
    if (::pread(fd_r, aligned, idx_bytes, off_t(h.index_off)) != ssize_t(idx_bytes)) return CloseReplay();
    std::memcpy(recs.data(), aligned, h.nrec * sizeof(RecHead));
    struct stat st {};
    if (::fstat(fd_r, &st) != 0) return CloseReplay();
    for (const auto& rec : recs)
      if (Page4k(rec.bytes) > slot_bytes || rec.data_off + Page4k(rec.bytes) > std::uint64_t(st.st_size)) {
        std::fprintf(stderr, "npu bcache: %s has a record past EOF or larger than a slot, re-recording\n", path.c_str());
        return CloseReplay();
      }
    consume = fill = start % recs.size();
    StartRing(dev, kern, nslots);
    std::fprintf(stderr, "npu bcache: replay %s, %zu records, %zu slots x %.0f MB\n", path.c_str(),
                 recs.size(), slots.size(), slot_bytes / 1e6);
    return true;
  }
  bool CloseReplay() {
    if (fd_r >= 0) ::close(fd_r);
    fd_r = -1;
    recs.clear();
    return false;
  }

  void StartRing(xrt::device& dev, xrt::kernel& kern, std::size_t nslots) {
    slots.resize(nslots);
    for (auto& s : slots) {
      // B is kernel argument 4 (opcode, insts, n, A, B, C)
      s.bo = xrt::bo(dev, slot_bytes, xrt::bo::flags::host_only, kern.group_id(4));
      s.host = s.bo.map<std::uint8_t*>();
    }
    std::thread([this] { IoLoop(); }).detach();
  }

  void IoLoop() try {
    for (;;) {
      std::size_t i, rec;
      std::uint64_t g;
      {
        std::unique_lock<std::mutex> lock(mu);
        cv.wait(lock, [&] {
          return std::any_of(slots.begin(), slots.end(), [](const Slot& s) { return s.st == St::kFree; });
        });
        i = std::size_t(std::find_if(slots.begin(), slots.end(), [](const Slot& s) { return s.st == St::kFree; }) - slots.begin());
        rec = fill;
        fill = (fill + 1) % recs.size();
        g = gen;
        slots[i].st = St::kLoading;
        slots[i].claimed = false;
        slots[i].rec = rec;
        slots[i].gen = g;
      }
      const RecHead& h = recs[rec];
      const std::size_t want_bytes = Page4k(h.bytes);
      std::uint8_t* dst = bounce != nullptr ? bounce : slots[i].host;
      for (std::size_t done = 0; done < want_bytes;) {
        const ssize_t got = ::pread(fd_r, dst + done, want_bytes - done, off_t(h.data_off + done));
        const int err = got < 0 ? errno : 0;
        if (err == EINTR) continue;
        if (bounce == nullptr && (err == EFAULT || err == EINVAL)) {
          // O_DIRECT could not pin the BO mapping: read via a page-aligned bounce buffer
          bounce = static_cast<std::uint8_t*>(std::aligned_alloc(4096, slot_bytes));
          if (bounce == nullptr) std::abort();
          std::fprintf(stderr, "npu bcache: direct read into the NPU buffer failed (errno %d), using a bounce buffer\n", err);
          dst = bounce;
          done = 0;
          continue;
        }
        if (got <= 0) {
          std::fprintf(stderr, "npu bcache: read failed at record %zu (errno %d, 0 = EOF), aborting\n", rec, err);
          std::abort();  // a slot that never fills would hang the NPU worker and the GPU stream
        }
        done += std::size_t(got);
      }
      if (dst == bounce) std::memcpy(slots[i].host, bounce, want_bytes);
      slots[i].bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      {
        std::lock_guard<std::mutex> lock(mu);
        // a resync while loading made this slot stale: free it unless a job claimed it
        slots[i].st = (slots[i].gen == gen || slots[i].claimed) ? St::kReady : St::kFree;
      }
      cv.notify_all();
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "npu bcache: io thread failed: %s\n", e.what());
    std::abort();
  }

  // Replay: slot holding this job's B (the worker waits until it is loaded), or -1 = pack on the GPU.
  int Acquire(const PackSpec* specs, const void* const* ws, int n) {
    if (fd_r < 0) return -1;
    if (!Matches(recs[consume], specs, ws, n)) {
      // Out of step (a skipped op, another M): find the job, restart the ring after it.
      std::size_t j = 0;
      while (j < recs.size() && !Matches(recs[(consume + j) % recs.size()], specs, ws, n)) ++j;
      ++gpu_packs;
      if (j == recs.size()) return -1;  // not recorded: GPU pack, cursor unchanged
      if (++resyncs <= 20 || resyncs % 1000 == 0)
        std::fprintf(stderr, "npu bcache: resync %zu (hits %zu, gpu packs %zu), skipped %zu records\n",
                     resyncs, hits, gpu_packs, j);
      std::lock_guard<std::mutex> lock(mu);
      consume = fill = (consume + j + 1) % recs.size();
      ++gen;
      for (auto& s : slots)
        if (!s.claimed && s.st == St::kReady) s.st = St::kFree;
      cv.notify_all();
      return -1;
    }
    std::unique_lock<std::mutex> lock(mu);
    const auto mine = [&] {
      return std::find_if(slots.begin(), slots.end(), [&](const Slot& s) {
        return s.st != St::kFree && !s.claimed && s.gen == gen && s.rec == consume;
      });
    };
    // The I/O thread assigns records in order; blocking here only bounds how far the host
    // runs ahead of the GPU (every claimed slot's job is already submitted).
    cv.wait(lock, [&] { return mine() != slots.end(); });
    auto it = mine();
    it->claimed = true;
    consume = (consume + 1) % recs.size();
    if (++hits % 1000 == 0)
      std::fprintf(stderr, "npu bcache: hits %zu, resyncs %zu, gpu packs %zu\n", hits, resyncs, gpu_packs);
    return int(it - slots.begin());
  }

  xrt::bo& WaitReady(int i) {
    std::unique_lock<std::mutex> lock(mu);
    cv.wait(lock, [&] { return slots[i].st == St::kReady; });
    return slots[i].bo;
  }

  void Release(int i) {
    {
      std::lock_guard<std::mutex> lock(mu);
      slots[i].claimed = false;
      slots[i].st = St::kFree;
    }
    cv.notify_all();
  }

  // Record mode: append this job's GPU-packed B (stalls the stream: first cycle only).
  // True when the cycle closed and the file switched to replay.
  bool Record(xrt::device& dev, xrt::kernel& kern, std::size_t nslots, const void* b_gpu,
              std::size_t bytes, hipStream_t ps, const PackSpec* specs, const void* const* ws, int n) {
    if (fd_w < 0) return false;
    const WeightKey key0 = KeyOf(ws[0], specs[0].row0, AliasOf(ws[0]));
    if (!recs.empty() && key0 == first_key && specs[0] == first_spec) {
      FileHead h = want;
      h.nrec = recs.size();
      h.index_off = w_off;
      const std::size_t idx_bytes = Page4k(recs.size() * sizeof(RecHead));
      std::uint8_t* idx = nullptr;
      idx = static_cast<std::uint8_t*>(std::aligned_alloc(4096, idx_bytes + 4096));  // O_DIRECT: page-aligned
      if (idx == nullptr) return false;
      std::memset(idx, 0, idx_bytes + 4096);
      std::memcpy(idx, recs.data(), recs.size() * sizeof(RecHead));
      std::memcpy(idx + idx_bytes, &h, sizeof h);
      const bool ok = ::pwrite(fd_w, idx, idx_bytes, off_t(w_off)) == ssize_t(idx_bytes) &&
                      ::pwrite(fd_w, idx + idx_bytes, 4096, 0) == 4096 && ::fsync(fd_w) == 0;
      std::free(idx);
      ::close(fd_w);
      fd_w = -1;
      const std::string tmp = path + ".tmp";
      if (!ok || ::rename(tmp.c_str(), path.c_str()) != 0) {
        std::fprintf(stderr, "npu bcache: finishing %s failed, GPU pack stays\n", path.c_str());
        ::unlink(tmp.c_str());
        return false;
      }
      std::fprintf(stderr, "npu bcache: recorded %zu jobs, %.2f GB\n", recs.size(), w_off / 1e9);
      recs.clear();
      return OpenReplay(dev, kern, nslots, 1);  // this job is record 0, packed on the GPU already
    }
    if (recs.empty()) {
      first_key = key0;
      first_spec = specs[0];
    }
    RecHead h{};
    h.npacks = std::uint32_t(n);
    for (int i = 0; i < n; ++i) {
      h.packs[i] = specs[i];
      h.fp[i] = Fingerprint(specs[i], ws[i]);
    }
    h.bytes = bytes;
    h.data_off = w_off;
    HIP_CHECK(hipMemcpyAsync(staging, b_gpu, bytes, hipMemcpyDeviceToHost, ps));
    HIP_CHECK(hipStreamSynchronize(ps));
    const std::size_t padded = Page4k(bytes);
    std::memset(staging + bytes, 0, padded - bytes);
    if (::pwrite(fd_w, staging, padded, off_t(w_off)) != ssize_t(padded)) {
      std::fprintf(stderr, "npu bcache: write failed, recording stopped\n");
      ::close(fd_w);
      fd_w = -1;
      return false;
    }
    w_off += padded;
    recs.push_back(h);
    return false;
  }
};

using Insts = std::pair<xrt::bo, unsigned>;

struct Job {
  hipEvent_t ready;
  const Insts* insts;
  std::uint32_t seq;
  std::size_t M, K, N;
  int slot;  // BCache ring slot holding B, or -1 = bo_b (GPU-packed)
};

// GPU-side phases of a logged job: e0 before convert, e1 after pack, e2 after
// the GPU rows (before the flag wait), e3 after the flag wait.
struct Timing {
  std::uint32_t seq;
  std::size_t M, N;
  hipEvent_t e[4];
};

struct Runtime {
  std::string stem;
  std::size_t down_rows, gu_rows, out_rows, store_pct, max_m, min_m, log_jobs;
  bool bo_sync;
  xrt::device dev;
  xrt::hw_context ctx;
  xrt::kernel kern;
  std::map<std::tuple<std::size_t, std::size_t, std::size_t>, std::unique_ptr<Insts>> insts;  // by M, K, N
  // One A/B/C set sized for the largest op; jobs run one at a time in stream order.
  void *a = nullptr, *b = nullptr, *c = nullptr;
  xrt::bo bo_a, bo_b, bo_c;
  std::uint32_t* flag = nullptr;
  std::uint32_t seq = 0;
  hipStream_t stream = nullptr;  // single A/B/C set: one prefill stream only
  bool stream_set = false;
  // Measured: convert+pack (2.1 ms at M 2048) ran on the prefill stream before the GPU rows.
  // GUFO_NPU_SIDE=1 runs it on a high-priority side stream instead, beside the GPU rows.
  bool use_side = false;
  hipStream_t side = nullptr;
  std::mutex mu;
  std::condition_variable cv;
  std::deque<Job> jobs;
  std::deque<Timing> timings;  // host thread only
  std::unique_ptr<BCache> bcache;  // GUFO_NPU_BCACHE, null = GPU pack every call
  std::size_t bcache_slots = 0;

  // NPU rows of a store op with m rows: whole 256-row GPU tiles.
  std::size_t StoreRows(std::size_t m) const { return m * store_pct / 100 / 256 * 256; }

  // Logging only: blocks the host on an old job's events (perturbs speed).
  void DrainTimings(std::uint32_t upto) {
    while (!timings.empty() && timings.front().seq <= upto) {
      Timing t = timings.front();
      timings.pop_front();
      HIP_CHECK(hipEventSynchronize(t.e[3]));
      float pack = 0, gpu = 0, stall = 0;
      HIP_CHECK(hipEventElapsedTime(&pack, t.e[0], t.e[1]));
      HIP_CHECK(hipEventElapsedTime(&gpu, t.e[1], t.e[2]));
      HIP_CHECK(hipEventElapsedTime(&stall, t.e[2], t.e[3]));
      std::fprintf(stderr, "npu split: gpu job %u M %zu N %zu convert+pack %.3f gpu_rows %.3f stall %.3f ms\n",
                   t.seq, t.M, t.N, pack, gpu, stall);
      for (auto& e : t.e) HIP_CHECK(hipEventDestroy(e));
    }
  }

  const Insts* FindInsts(std::size_t M, std::size_t K, std::size_t N) {
    const auto key = std::make_tuple(M, K, N);
    auto it = insts.find(key);
    if (it != insts.end()) return it->second.get();
    auto& slot = insts[key];  // stays null on a missing file: cached, no retry
    const std::string path = stem + "_M" + std::to_string(M) + "_K" +
                             std::to_string(K) + "_N" + std::to_string(N) + ".bin";
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f || f.tellg() <= 0) {
      std::fprintf(stderr, "npu split: no %s, GPU-only at M %zu K %zu N %zu\n", path.c_str(), M, K, N);
      return nullptr;
    }
    std::vector<std::uint32_t> words(std::size_t(f.tellg()) / 4);
    f.seekg(0);
    f.read(reinterpret_cast<char*>(words.data()), words.size() * 4);
    xrt::bo bo(dev, words.size() * 4, xrt::bo::flags::cacheable, kern.group_id(1));
    bo.write(words.data());
    bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    slot = std::make_unique<Insts>(std::move(bo), unsigned(words.size()));
    return slot.get();
  }

  void Loop() try {
    for (int n = 0;; ++n) {
      Job job;
      {
        std::unique_lock<std::mutex> lock(mu);
        cv.wait(lock, [&] { return !jobs.empty(); });
        job = jobs.front();
        jobs.pop_front();
      }
      using Clock = std::chrono::steady_clock;
      const auto ms = [](Clock::time_point a, Clock::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
      };
      const auto tw = Clock::now();
      HIP_CHECK(hipEventSynchronize(job.ready));
      HIP_CHECK(hipEventDestroy(job.ready));
      const auto t0 = Clock::now();
      if (bo_sync) {
        bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        if (job.slot < 0) bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      }
      xrt::bo& b = job.slot >= 0 ? bcache->WaitReady(job.slot) : bo_b;  // t0..t1 includes an SSD wait
      const auto t1 = Clock::now();
      auto run = kern(3, job.insts->first, job.insts->second, bo_a, b, bo_c);
      if (run.wait() != ERT_CMD_STATE_COMPLETED) {
        // A missing flag would hang the GPU stream; wrong rows are worse than a stop.
        std::fprintf(stderr, "npu split: NPU run %u failed, aborting\n", job.seq);
        std::abort();
      }
      const auto t2 = Clock::now();
      if (job.slot >= 0) bcache->Release(job.slot);
      if (bo_sync) bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
      __atomic_store_n(flag, job.seq, __ATOMIC_RELEASE);
      const auto t3 = Clock::now();
      // event wait = GPU convert+pack (and anything queued before it)
      if (std::size_t(n) < log_jobs)
        std::fprintf(stderr,
                     "npu split: job %u M %zu K %zu N %zu wait %.3f sync_in %.3f run %.3f sync_out %.3f ms\n",
                     job.seq, job.M, job.K, job.N, ms(tw, t0), ms(t0, t1), ms(t1, t2), ms(t2, t3));
    }
  } catch (const std::exception& e) {  // detached thread: say why, then stop
    std::fprintf(stderr, "npu split: worker failed: %s\n", e.what());
    std::abort();
  }
};

Runtime* Get() {
  // ponytail: leaked singleton + detached worker, no shutdown path; a DEV
  // flag process exits with the server.
  static Runtime* rt = []() -> Runtime* {
    const char* xclbin = std::getenv("GUFO_NPU_SPLIT");
    if (xclbin == nullptr || xclbin[0] == '\0') return nullptr;
    try {  // a throwing init would be retried on every call: fail once, stay off
    auto* r = new Runtime;
    r->stem = xclbin;
    if (r->stem.size() > 7 && r->stem.ends_with(".xclbin"))
      r->stem.resize(r->stem.size() - 7);
    r->down_rows = EnvSize("GUFO_NPU_ROWS", 1536);
    r->gu_rows = EnvSize("GUFO_NPU_GU_ROWS", 0);
    r->out_rows = EnvSize("GUFO_NPU_OUT_ROWS", 0);
    r->store_pct = EnvSize("GUFO_NPU_STORE_PCT", 0);
    r->max_m = EnvSize("GUFO_NPU_MAX_M", 8192) / 256 * 256;
    // Below min M the NPU's fixed cost per job (~5 ms at M 512, measured) exceeds the GPU time it saves.
    r->min_m = EnvSize("GUFO_NPU_MIN_M", 2048);
    r->log_jobs = EnvSize("GUFO_NPU_LOG_JOBS", 4);
    // dma-buf BOs: XRT sync may be a no-op or a CPU cache flush; GUFO_NPU_BO_SYNC=0 skips it (for A/B tests).
    r->bo_sync = EnvSize("GUFO_NPU_BO_SYNC", 1) != 0;
    // down rows: whole 256-row GPU tiles; gate/up rows: whole 128-row dual tiles
    // (64-row tiles under GUFO_DEV_HALF_BM128 divide 128 too).
    if (r->down_rows % 256 || r->down_rows >= kHidden || r->gu_rows % 128 ||
        r->gu_rows >= kFfn || r->out_rows % 256 || r->out_rows >= kHidden ||
        r->store_pct >= 100 || r->down_rows + r->gu_rows + r->out_rows + r->store_pct == 0 ||
        r->max_m == 0) {
      std::fprintf(stderr, "npu split: bad GUFO_NPU_ROWS/GU_ROWS/OUT_ROWS/STORE_PCT/MAX_M, off\n");
      return nullptr;
    }
    r->use_side = EnvSize("GUFO_NPU_SIDE", 0) != 0;
    if (r->use_side) {
      int least = 0, greatest = 0;
      HIP_CHECK(hipDeviceGetStreamPriorityRange(&least, &greatest));
      HIP_CHECK(hipStreamCreateWithPriority(&r->side, hipStreamNonBlocking, greatest));
    }
    r->dev = xrt::device(0);
    auto xb = xrt::xclbin(std::string(xclbin));
    r->dev.register_xclbin(xb);
    r->ctx = xrt::hw_context(r->dev, xb.get_uuid());
    r->kern = xrt::kernel(r->ctx, xb.get_kernels().at(0).get_name());
    const std::size_t max_k = r->down_rows ? kFfn : r->out_rows ? kOut : kHidden;
    const std::size_t store_rows = r->StoreRows(kFfn);  // largest store op
    const std::size_t a_bytes = PageUp(r->max_m * max_k * 2),
                      b_bytes = PageUp(std::max({kFfn * r->down_rows, kHidden * 2 * r->gu_rows,
                                                 kOut * r->out_rows, kHidden * store_rows}) / 8 * 9),
                      c_bytes = PageUp(r->max_m * std::max({r->down_rows, 2 * r->gu_rows, r->out_rows,
                                                            store_rows}) * 2);
    if (const char* bc = std::getenv("GUFO_NPU_BCACHE"); bc != nullptr && bc[0] != '\0') {
      auto c = std::make_unique<BCache>();
      c->path = bc;
      std::memcpy(c->want.magic, "GNPB0003", 8);
      c->want.down_rows = r->down_rows;
      c->want.gu_rows = r->gu_rows;
      c->want.out_rows = r->out_rows;
      c->want.store_pct = r->store_pct;
      c->slot_bytes = Page4k(b_bytes);
      r->bcache_slots = std::max<std::size_t>(EnvSize("GUFO_NPU_BCACHE_SLOTS", 4), 2);
      HIP_CHECK(hipStreamCreateWithFlags(&c->fp_stream, hipStreamNonBlocking));
      HIP_CHECK(hipHostMalloc(&c->fp_host, sizeof(std::uint64_t)));
      c->check = EnvSize("GUFO_NPU_BCACHE_CHECK", 0);
      if (!c->OpenReplay(r->dev, r->kern, r->bcache_slots)) {
        const std::string tmp = c->path + ".tmp";
        c->fd_w = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT, 0644);
        if (c->fd_w >= 0) c->staging = static_cast<std::uint8_t*>(std::aligned_alloc(4096, c->slot_bytes));
        if (c->fd_w >= 0 && c->staging == nullptr) {
          ::close(c->fd_w);
          c->fd_w = -1;
        }
        if (c->fd_w >= 0) {
          std::fprintf(stderr, "npu bcache: recording %s (first pass packs on the GPU and waits for the SSD)\n", tmp.c_str());
        } else {
          std::fprintf(stderr, "npu bcache: cannot open %s, off\n", tmp.c_str());
        }
      }
      if (c->fd_r >= 0 || c->fd_w >= 0) r->bcache = std::move(c);
    }
    HIP_CHECK(hipMalloc(&r->a, a_bytes));
    HIP_CHECK(hipMalloc(&r->b, b_bytes));
    r->bo_a = ImportGpuBuffer(r->dev, r->a, a_bytes);
    r->bo_b = ImportGpuBuffer(r->dev, r->b, b_bytes);
    // C is written by the NPU and read by the GPU: fine-grained (coherent)
    // memory, so GPU L2 cannot serve a stale line from the previous layer.
    // Falls back to coarse hipMalloc if that memory cannot be exported.
    const char* c_kind = "fine-grained";
    HIP_CHECK(hipExtMallocWithFlags(&r->c, c_bytes, hipDeviceMallocFinegrained));
    try {
      r->bo_c = ImportGpuBuffer(r->dev, r->c, c_bytes);
    } catch (const std::exception&) {
      HIP_CHECK(hipFree(r->c));
      HIP_CHECK(hipMalloc(&r->c, c_bytes));
      r->bo_c = ImportGpuBuffer(r->dev, r->c, c_bytes);
      c_kind = "coarse (fine-grained export failed)";
    }
    // Measured: on a fresh C, some 64-B lines of the first NPU result read back as the allocation's zeros
    // (no fix: 4/4 processes, text changed 3/4). Not a dma-buf fence (poll idle). The GPU writing C once and waiting,
    // before any NPU job, fixed it 4/4 (CPU clflush and a 300 ms sleep also did). ~1 ms, once per process.
    HIP_CHECK(hipMemset(r->c, 0, c_bytes));
    HIP_CHECK(hipDeviceSynchronize());
    const char* flag_env = std::getenv("GUFO_NPU_FLAG");
    const bool signal_flag = flag_env != nullptr && std::string(flag_env) == "signal";
    if (signal_flag)
      HIP_CHECK(hipExtMallocWithFlags(reinterpret_cast<void**>(&r->flag), 64,
                                      hipMallocSignalMemory));
    else
      HIP_CHECK(hipHostMalloc(&r->flag, 64, hipHostMallocCoherent));
    *r->flag = 0;
    std::thread([r] { r->Loop(); }).detach();
    std::fprintf(stderr, "npu split: on, %s, rows %zu, gu rows %zu, out rows %zu, store pct %zu, max M %zu, buffers %.0f MB, C %s, flag %s\n",
                 xclbin, r->down_rows, r->gu_rows, r->out_rows, r->store_pct, r->max_m, (a_bytes + b_bytes + c_bytes) / 1e6,
                 c_kind, signal_flag ? "signal" : "host");
    return r;
    } catch (const std::exception& e) {  // ponytail: partial allocations leak, once
      std::fprintf(stderr, "npu split: init failed, off: %s\n", e.what());
      return nullptr;
    }
  }();
  return rt;
}

// A call the split may take: not captured (host waits cannot be captured),
// runtime on, and the one prefill stream. Null = GPU-only.
Runtime* Usable(hipStream_t stream) {
  hipStreamCaptureStatus capture = hipStreamCaptureStatusNone;
  HIP_CHECK(hipStreamIsCapturing(stream, &capture));
  if (capture != hipStreamCaptureStatusNone) return nullptr;
  Runtime* r = Get();
  if (r == nullptr) return nullptr;
  if (!r->stream_set) {
    r->stream = stream;
    r->stream_set = true;
  }
  return stream == r->stream ? r : nullptr;
}

struct Pending {
  hipStream_t ps;  // where convert+pack run
  const Insts* ins;
  std::size_t M, K, N;
  bool timed;
  int slot = -1;
};

// Front half shared by both ops: M range and instruction file (else false =
// GPU-only, nothing queued), then x -> A on the pack stream.
bool Prepare(Runtime* r, const void* x, std::size_t batch, std::size_t k, std::size_t n,
             hipStream_t stream, Pending* p) {
  if (batch == 0) return false;
  const std::size_t M = (batch + 255) / 256 * 256;
  if (M > r->max_m || M < r->min_m) return false;
  const Insts* ins = r->FindInsts(M, k, n);
  if (ins == nullptr) return false;
  r->DrainTimings(r->seq >= 2 ? r->seq - 2 : 0);
  // Side stream: wait for x on the prefill stream (and, through it, the previous
  // merge that reads C), and for the previous NPU job (it still reads the single
  // A/B buffers) before overwriting them.
  hipStream_t ps = stream;
  if (r->use_side) {
    ps = r->side;
    hipEvent_t x_ready;
    HIP_CHECK(hipEventCreateWithFlags(&x_ready, hipEventDisableTiming));
    HIP_CHECK(hipEventRecord(x_ready, stream));
    HIP_CHECK(hipStreamWaitEvent(ps, x_ready, 0));
    HIP_CHECK(hipEventDestroy(x_ready));
    if (r->seq > 0)
      HIP_CHECK(hipStreamWaitValue32(ps, r->flag, r->seq, hipStreamWaitValueGte, 0xffffffffU));
  }
  const bool timed = r->seq < r->log_jobs;
  if (timed) {
    Timing t{r->seq + 1, M, n, {}};
    for (auto& e : t.e) HIP_CHECK(hipEventCreate(&e));
    HIP_CHECK(hipEventRecord(t.e[0], ps));
    r->timings.push_back(t);
  }
  HalfToBf16Pad<<<1024, 256, 0, ps>>>(static_cast<const __half*>(x),
                                          static_cast<__hip_bfloat16*>(r->a),
                                          batch * k, M * k);
  HIP_CHECK(hipGetLastError());
  *p = {ps, ins, M, k, n, timed, -1};
  return true;
}

// Weight rows [row0, row0 + n_count) -> B columns [n_off, n_off + n_count).
void Pack(Runtime* r, const Pending& p, core::GgmlType type, const void* w,
          std::size_t row0, std::size_t n_off, std::size_t n_count) {
  const Bfp16PackShape shape = Shape(p.K, p.N);
  const auto* wb = static_cast<const std::uint8_t*>(w);
  auto* b = static_cast<std::uint8_t*>(r->b);
  static const bool pack_old = std::getenv("GUFO_NPU_PACK_OLD") != nullptr;  // A/B only
  if (pack_old)
    PackRows<<<1024, 256, 0, p.ps>>>(type, wb, row0, n_off, n_count, b, shape);
  else
    PackRows8<<<1024, 256, 0, p.ps>>>(type, wb, row0, n_off, n_count, b, shape);
  HIP_CHECK(hipGetLastError());
  // First 3 packs in process order (any op): old vs new pack, same weights.
  static int checks = EnvSize("GUFO_NPU_PACK_CHECK", 0) != 0 ? 3 : 0;
  if (checks == 0) return;
  --checks;
  // columns [n_off, n_off + n_count), n_off % 64 == 0: one contiguous byte range (cb is outermost)
  const std::size_t off = bfp16_group_index(shape, std::uint32_t(n_off), 0) * 9,
                    bytes = p.K * n_count / 8 * 9;
  std::uint8_t* ref = nullptr;
  HIP_CHECK(hipMalloc(&ref, PageUp(p.K * p.N / 8 * 9)));
  PackRows<<<1024, 256, 0, p.ps>>>(type, wb, row0, n_off, n_count, ref, shape);
  HIP_CHECK(hipGetLastError());
  std::vector<std::uint8_t> h0(bytes), h1(bytes);
  HIP_CHECK(hipMemcpyAsync(h0.data(), ref + off, bytes, hipMemcpyDeviceToHost, p.ps));
  HIP_CHECK(hipMemcpyAsync(h1.data(), b + off, bytes, hipMemcpyDeviceToHost, p.ps));
  HIP_CHECK(hipStreamSynchronize(p.ps));
  HIP_CHECK(hipFree(ref));
  std::size_t bad = 0;
  for (std::size_t i = 0; i < bytes; ++i) bad += h0[i] != h1[i];
  std::fprintf(stderr, "npu split: pack check type %d K %zu n_off %zu bytes %zu mismatches %zu\n",
               int(type), p.K, n_off, bytes, bad);
}

// All packs of one job: from the B cache when it has the job, else on the GPU
// (and appended to the cache file during its recording pass).
void PackJob(Runtime* r, Pending* p, const PackSpec* specs, const void* const* ws, int n) {
  if (r->bcache != nullptr) {
    p->slot = r->bcache->Acquire(specs, ws, n);
    if (p->slot >= 0 && r->bcache->check > 0) {  // debug: GPU-pack too, compare with the SSD bytes
      --r->bcache->check;
      for (int i = 0; i < n; ++i)
        Pack(r, *p, core::GgmlType(specs[i].type), ws[i], specs[i].row0, specs[i].n_off, specs[i].n_count);
      const std::size_t bytes = p->K * p->N / 8 * 9;
      std::vector<std::uint8_t> gpu(bytes);
      HIP_CHECK(hipMemcpyAsync(gpu.data(), r->b, bytes, hipMemcpyDeviceToHost, p->ps));
      HIP_CHECK(hipStreamSynchronize(p->ps));
      const std::uint8_t* ssd = r->bcache->slots[p->slot].host;
      r->bcache->WaitReady(p->slot);
      std::size_t bad = 0;
      for (std::size_t i = 0; i < bytes; ++i) bad += gpu[i] != ssd[i];
      std::fprintf(stderr, "npu bcache: check K %zu N %zu bytes %zu mismatches %zu\n", p->K, p->N, bytes, bad);
    }
    if (p->slot >= 0) return;
  }
  for (int i = 0; i < n; ++i)
    Pack(r, *p, core::GgmlType(specs[i].type), ws[i], specs[i].row0, specs[i].n_off, specs[i].n_count);
  if (r->bcache != nullptr)
    r->bcache->Record(r->dev, r->kern, r->bcache_slots, r->b, p->K * p->N / 8 * 9, p->ps, specs, ws, n);
}

std::uint32_t Submit(Runtime* r, const Pending& p) {
  hipEvent_t ready;
  // ReleaseToSystem: A and B leave GPU L2 before the NPU reads them.
  HIP_CHECK(hipEventCreateWithFlags(&ready, hipEventDisableTiming | hipEventReleaseToSystem));
  HIP_CHECK(hipEventRecord(ready, p.ps));
  // with the side stream e0..e1 = convert+pack on ps, e1..e2 spans both streams (not the GPU rows)
  if (p.timed) HIP_CHECK(hipEventRecord(r->timings.back().e[1], p.ps));
  std::uint32_t seq;
  {
    std::lock_guard<std::mutex> lock(r->mu);
    seq = ++r->seq;
    r->jobs.push_back({ready, p.ins, seq, p.M, p.K, p.N, p.slot});
  }
  r->cv.notify_one();
  return seq;
}

// The stream waits for NPU job seq (no host sync).
Runtime* WaitNpu(hipStream_t stream, std::uint32_t seq) {
  Runtime* r = Get();
  Timing* t = !r->timings.empty() && r->timings.back().seq == seq ? &r->timings.back() : nullptr;
  if (t != nullptr) HIP_CHECK(hipEventRecord(t->e[2], stream));
  HIP_CHECK(hipStreamWaitValue32(stream, r->flag, seq, hipStreamWaitValueGte,
                                 0xffffffffU));
  if (t != nullptr) HIP_CHECK(hipEventRecord(t->e[3], stream));
  return r;
}
}  // namespace

void SetScratchAliases(const void* gate_scratch, const LogicalWeight& gate,
                       const void* up_scratch, const LogicalWeight& up) {
  g_alias = {{{gate_scratch, gate}, {up_scratch, up}}};
}
void ClearScratchAliases() { g_alias = {}; }

std::size_t BeginSplit(core::GgmlType type, const void* w, const void* x,
                       std::size_t batch, std::size_t m, std::size_t k,
                       hipStream_t stream, std::uint32_t* seq) {
  if (m != kHidden || (k != kFfn && k != kOut)) return m;
  Runtime* r = Usable(stream);
  if (r == nullptr) return m;
  const std::size_t rows = k == kFfn ? r->down_rows : r->out_rows;
  if (rows == 0) return m;
  Pending p;
  if (!Prepare(r, x, batch, k, rows, stream, &p)) return m;
  const std::size_t m_gpu = m - rows;
  const PackSpec spec{std::int32_t(type), std::uint32_t(p.K), std::uint32_t(p.N), std::uint32_t(m_gpu), 0, std::uint32_t(rows)};
  PackJob(r, &p, &spec, &w, 1);
  *seq = Submit(r, p);
  return m_gpu;
}

void FinishResidual(float* y, std::size_t batch, std::size_t m,
                    std::size_t m_gpu, hipStream_t stream, std::uint32_t seq) {
  Runtime* r = WaitNpu(stream, seq);
  AddNpuRows<<<1024, 256, 0, stream>>>(y, static_cast<const __hip_bfloat16*>(r->c),
                                       batch, m, m_gpu, m - m_gpu);
  HIP_CHECK(hipGetLastError());
}

std::size_t BeginStore(core::GgmlType type, const void* w, const void* x,
                       std::size_t batch, std::size_t m, std::size_t k,
                       hipStream_t stream, std::uint32_t* seq) {
  // m % 256: GPU tiles end exactly at m_grid
  // m == kFfn = unpaired ffn_gate fallback: left on the GPU (gate/up has its own split)
  if (k != kHidden || m < kStoreMinM || m >= kFfn || m % 256) return m;
  Runtime* r = Usable(stream);
  if (r == nullptr) return m;
  const std::size_t rows = r->StoreRows(m);
  if (rows == 0) return m;
  Pending p;
  if (!Prepare(r, x, batch, k, rows, stream, &p)) return m;
  const std::size_t m_gpu = m - rows;
  const PackSpec spec{std::int32_t(type), std::uint32_t(p.K), std::uint32_t(p.N), std::uint32_t(m_gpu), 0, std::uint32_t(rows)};
  PackJob(r, &p, &spec, &w, 1);
  *seq = Submit(r, p);
  return m_gpu;
}

void FinishStore(float* y, std::size_t batch, std::size_t m, std::size_t m_gpu,
                 hipStream_t stream, std::uint32_t seq) {
  Runtime* r = WaitNpu(stream, seq);
  StoreNpuRows<<<1024, 256, 0, stream>>>(y, static_cast<const __hip_bfloat16*>(r->c),
                                         batch, m, m_gpu, m - m_gpu);
  HIP_CHECK(hipGetLastError());
}

std::size_t BeginGateUp(core::GgmlType gate_type, core::GgmlType up_type,
                        const void* gate_w, const void* up_w, const void* x,
                        std::size_t batch, std::size_t m, std::size_t k,
                        hipStream_t stream, std::uint32_t* seq) {
  if (k != kHidden || m != kFfn) return m;
  Runtime* r = Usable(stream);
  if (r == nullptr || r->gu_rows == 0) return m;
  const std::size_t n = r->gu_rows;
  Pending p;
  if (!Prepare(r, x, batch, k, 2 * n, stream, &p)) return m;
  const std::size_t m_gpu = m - n;
  const PackSpec specs[2] = {
      {std::int32_t(gate_type), std::uint32_t(p.K), std::uint32_t(p.N), std::uint32_t(m_gpu), 0, std::uint32_t(n)},
      {std::int32_t(up_type), std::uint32_t(p.K), std::uint32_t(p.N), std::uint32_t(m_gpu), std::uint32_t(n), std::uint32_t(n)}};
  const void* ws[2] = {gate_w, up_w};
  PackJob(r, &p, specs, ws, 2);
  *seq = Submit(r, p);
  return m_gpu;
}

void FinishGateUp(void* y, std::size_t batch, std::size_t m, std::size_t m_gpu,
                  hipStream_t stream, std::uint32_t seq) {
  Runtime* r = WaitNpu(stream, seq);
  GateUpNpuRows<<<1024, 256, 0, stream>>>(static_cast<__half*>(y),
                                          static_cast<const __hip_bfloat16*>(r->c),
                                          batch, m, m_gpu, m - m_gpu);
  HIP_CHECK(hipGetLastError());
}
}  // namespace gufo::hip::npu
#endif
