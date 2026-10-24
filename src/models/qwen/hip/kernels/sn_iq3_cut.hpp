#ifndef GUFO_MODELS_QWEN_HIP_KERNELS_SN_IQ3_CUT_HPP_
#define GUFO_MODELS_QWEN_HIP_KERNELS_SN_IQ3_CUT_HPP_

// The decode half of OpusIq3sB8CutKernel, as plain integer
// functions so a CPU test (tests/models/qwen/sn_iq3_cut_test.cpp) can check
// them against the definition. The kernel and the test call the same code.
//
// WHY (k4pad ISA, B8, one K block = 256 weights per row): of 879 VALU, 512 are
// v_dot4; the other 367 are the issue-bound cost (measured: ~0.16-0.31 us per VALU
// op removed from the loop). 64 of them are `<< 2` on the LDS addresses: the
// codebook is indexed by qs | (qhbit | sign << 1) << 8 and ds_load_b32 wants a
// byte address, so every index is perm'd and then shifted.
//
// The table is ours, so its index function is free. Three index modes
// (Idx), all returning the same 4 signed weight bytes for the same
// (qs byte, qh bit, sign nibble):
//   Idx 0  k4's index: qs8 | qh << 8 | sign << 9. Address = index << 2
//          (one v_lshlrev per word). 32 KiB table.
//   Idx 1  address bytes b0 = qs8 & 0xFC (low 2 bits zero: dword aligned),
//          b1 = (qs8 & 3) | tag << 2, tag = qh | sign << 1. The perm of
//          (A = qs & 0xFCFCFCFC, C = qs & 0x03030303 | tag << 2) IS the byte
//          address: 2 VALU per 4 words instead of 4 shifts. 32 KiB table, so
//          LDS occupancy is k4's.
//   Idx 2  address bytes b0 = tag << 2, b1 = qs8: address = qs8 * 256 +
//          tag * 4, the perm of (tag << 2, qs) alone. 0 extra VALU per word,
//          but the table is 64 KiB (half of every 256 B row unused), so only
//          2 workgroups fit per WGP instead of 4.
// Tags are built in the "<< 2" form for Idx 1, 2 by the same op count as k4's
// (shift/mask constants change, nothing is added).

#include <cstdint>

#if defined(__HIPCC__) || defined(__HIP_DEVICE_COMPILE__)
#define GUFO_CUT_HD __host__ __device__
#else
#define GUFO_CUT_HD
#endif

namespace gufo::hip::cut {

// v_perm_b32 D = perm(s0, s1, sel): selector byte 0..3 -> s1 byte, 4..7 -> s0
// byte, 0x0C -> 0x00 (the only other value used here).
GUFO_CUT_HD inline std::uint32_t Perm(std::uint32_t s0, std::uint32_t s1,
                                      std::uint32_t sel) noexcept {
#if defined(__HIP_DEVICE_COMPILE__)
  return __builtin_amdgcn_perm(s0, s1, sel);
#else
  std::uint32_t out = 0;
  for (std::uint32_t i = 0; i < 4U; ++i) {
    const std::uint32_t c = (sel >> (8U * i)) & 0xFFU;
    std::uint32_t byte = 0;
    if (c < 4U)
      byte = (s1 >> (8U * c)) & 0xFFU;
    else if (c < 8U)
      byte = (s0 >> (8U * (c - 4U))) & 0xFFU;
    out |= byte << (8U * i);
  }
  return out;
#endif
}

// Keeps the compiler from fusing `x` into the instruction that consumes it
// (it turns and_or(Q, m, and_or(qh, m2, p)) into 3 ops, and, and, or3).
GUFO_CUT_HD inline void Opaque(std::uint32_t& x) noexcept {
#if defined(__HIP_DEVICE_COMPILE__)
  asm("" : "+v"(x));
#else
  (void)x;
#endif
}

// Words of the codebook table for index mode Idx.
template<std::uint32_t Idx>
inline constexpr std::uint32_t kTableWords = Idx == 2U ? 16384U : 8192U;

// Table word that holds the signed grid word for (qs byte, qh bit, sign
// nibble). Must stay in step with WordAddrs below (the CPU test checks it).
template<std::uint32_t Idx>
GUFO_CUT_HD constexpr std::uint32_t TableIndex(std::uint32_t qs8,
                                               std::uint32_t qh,
                                               std::uint32_t sign) noexcept {
  if constexpr (Idx == 0U)
    return qs8 | (qh << 8U) | (sign << 9U);
  else if constexpr (Idx == 1U)  // qs8 rotated right by 2 inside its byte
    return (qs8 >> 2U) | ((qs8 & 3U) << 6U) | (qh << 8U) | (sign << 9U);
  else
    return (qs8 << 6U) | qh | (sign << 1U);
}

// Two tag words for one sub-block (8 grid words): byte i of tag[i >> 2] holds
// qh bit i and sign nibble i. `signs` is the sub-block's 8 sign nibbles,
// `qh_byte_shift` selects its qh byte inside qh_word (0, 8, 16, 24).
// Idx 0: tag = qh | sign << 1 in bits 0..4 (k4). Idx 1, 2: that << 2.
template<std::uint32_t Idx>
GUFO_CUT_HD inline void Tags(std::uint32_t signs, std::uint32_t qh_word,
                             std::uint32_t qh_byte_shift,
                             std::uint32_t tag[2]) noexcept {
  // Nibble n of `signs` is sign nibble n; the low nibble of byte j is word 2j,
  // the high nibble is word 2j + 1.
  constexpr std::uint32_t kSh = Idx == 0U ? 0U : 2U;
  constexpr std::uint32_t kMask = 0x1E1E1E1EU << kSh;
  constexpr std::uint32_t kQhMul = 0x00204081U << kSh;  // bit j -> 8j + kSh
  constexpr std::uint32_t kQhMask = 0x01010101U << kSh;
  const std::uint32_t even = (signs << (1U + kSh)) & kMask;  // words 0 2 4 6
  const std::uint32_t odd = (signs >> (3U - kSh)) & kMask;   // words 1 3 5 7
  // Spread 4 qh bits to bit kSh of 4 bytes: the four shifted copies of the
  // nibble never overlap, so nothing carries.
  const std::uint32_t qh_lo = ((qh_word >> qh_byte_shift) & 0x0FU) * kQhMul;
  const std::uint32_t qh_hi =
      ((qh_word >> (qh_byte_shift + 4U)) & 0x0FU) * kQhMul;
  tag[0] = (qh_lo & kQhMask) | Perm(odd, even, 0x05010400U);
  tag[1] = (qh_hi & kQhMask) | Perm(odd, even, 0x07030602U);
}

// LDS BYTE addresses of the 8 grid words of a sub-block. qs[0], qs[1] = its 8
// qs bytes (word 4w + j = byte j of qs[w]).
template<std::uint32_t Idx>
GUFO_CUT_HD inline void WordAddrs(const std::uint32_t tag[2],
                                  const std::uint32_t qs[2],
                                  std::uint32_t addr[8]) noexcept {
  if constexpr (Idx == 0U) {
    addr[0] = Perm(tag[0], qs[0], 0x0C0C0400U) << 2U;
    addr[1] = Perm(tag[0], qs[0], 0x0C0C0501U) << 2U;
    addr[2] = Perm(tag[0], qs[0], 0x0C0C0602U) << 2U;
    addr[3] = Perm(tag[0], qs[0], 0x0C0C0703U) << 2U;
    addr[4] = Perm(tag[1], qs[1], 0x0C0C0400U) << 2U;
    addr[5] = Perm(tag[1], qs[1], 0x0C0C0501U) << 2U;
    addr[6] = Perm(tag[1], qs[1], 0x0C0C0602U) << 2U;
    addr[7] = Perm(tag[1], qs[1], 0x0C0C0703U) << 2U;
  } else if constexpr (Idx == 1U) {
#pragma unroll
    for (std::uint32_t w = 0; w < 2U; ++w) {
      std::uint32_t t = tag[w];
      Opaque(t);
      const std::uint32_t a = qs[w] & 0xFCFCFCFCU;
      const std::uint32_t c = (qs[w] & 0x03030303U) | t;
      addr[(4U * w) + 0U] = Perm(c, a, 0x0C0C0400U);
      addr[(4U * w) + 1U] = Perm(c, a, 0x0C0C0501U);
      addr[(4U * w) + 2U] = Perm(c, a, 0x0C0C0602U);
      addr[(4U * w) + 3U] = Perm(c, a, 0x0C0C0703U);
    }
  } else {
#pragma unroll
    for (std::uint32_t w = 0; w < 2U; ++w) {
      addr[(4U * w) + 0U] = Perm(tag[w], qs[w], 0x0C0C0004U);
      addr[(4U * w) + 1U] = Perm(tag[w], qs[w], 0x0C0C0105U);
      addr[(4U * w) + 2U] = Perm(tag[w], qs[w], 0x0C0C0206U);
      addr[(4U * w) + 3U] = Perm(tag[w], qs[w], 0x0C0C0307U);
    }
  }
}

}  // namespace gufo::hip::cut

#endif  // GUFO_MODELS_QWEN_HIP_KERNELS_SN_IQ3_CUT_HPP_
