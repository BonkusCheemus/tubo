// Modified for tubo: IRON's Python pack_b ported to C++/HIP (host + device); the GPU pack is byte-exact vs IRON pack_b.
// bfp16ebs8 B packing for the IRON flm GEMM (AIE2P), from a gufo-layout weight W[n][k] (row = output column n).
// Port of IRON iron/operators/flm/packing.py pack_b(bfp16=True, round_conv_even=True), Apache-2.0, AMD.
// Original: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 (full text: licenses/iron.txt).
// Byte-exact target: same bytes as pack_b(W.T, ...). One call = one 9-byte group (8 consecutive k of one n).
#pragma once
#include <cstdint>
#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#define BFP_FN __host__ __device__ __forceinline__
#else
#define BFP_FN inline
#endif

struct Bfp16PackShape {
  uint32_t K, N, k_tile, n_tile, s, t, ct_k;  // s must be 8 (one bfp16 block = one s run)
};

// Output group index of (n, k0), k0 % 8 == 0. Order (cb, kb, kslice, tb, i, t_in) [s_in inside the group].
BFP_FN uint64_t bfp16_group_index(const Bfp16PackShape& p, uint32_t n, uint32_t k0) {
  const uint32_t col_a = p.ct_k / p.s;
  const uint32_t kb = k0 / p.k_tile, kr = k0 % p.k_tile;
  const uint32_t kslice = kr / p.ct_k, i = (kr % p.ct_k) / p.s;
  const uint32_t cb = n / p.n_tile, nr = n % p.n_tile;
  const uint32_t tb = nr / p.t, t_in = nr % p.t;
  uint64_t g = cb;
  g = g * (p.K / p.k_tile) + kb;
  g = g * (p.k_tile / p.ct_k) + kslice;
  g = g * (p.n_tile / p.t) + tb;
  g = g * col_a + i;
  g = g * p.t + t_in;
  return g;
}

// f32 x8 -> shared exponent byte + 8 signed mantissa bytes (RNE), as f32_to_bfp16ebs8.
BFP_FN void bfp16_pack_group(const float* v, uint8_t* out) {
  uint32_t u[8];
  int32_t e[8], max_exp = 0;
  for (int j = 0; j < 8; ++j) {
    union { float f; uint32_t u; } c{v[j]};
    u[j] = c.u;
    e[j] = int32_t((u[j] >> 23) & 0xFF);
    max_exp = e[j] > max_exp ? e[j] : max_exp;
  }
  out[0] = uint8_t(max_exp);
  for (int j = 0; j < 8; ++j) {
    uint32_t man = u[j] & 0x007FFFFFu;
    if (e[j] != 0) man |= 0x00800000u;
    const bool neg = (u[j] & 0x80000000u) != 0;
    const int64_t mag = neg ? -int64_t(man) : int64_t(man);
    const int32_t shift = max_exp - e[j];
    int32_t total = 17 + shift;
    total = total < 0 ? 0 : (total > 62 ? 62 : total);
    int64_t q;
    if (shift >= 32) {
      // exponent 32+ below the group max (in practice: exact zeros): RNE gives 0. The old floor (-1) turned -0.0
      // (Q6_K/Q8_0 q = 0 times a negative scale) into a full negative LSB.
      q = 0;
    } else if (total == 0) {
      q = mag;
    } else {
      q = mag >> total;  // floor
      const int64_t r = mag - q * (int64_t(1) << total), half = int64_t(1) << (total - 1);
      if (r > half || (r == half && (q & 1))) ++q;
    }
    q = q < -128 ? -128 : (q > 127 ? 127 : q);
    out[1 + j] = uint8_t(int8_t(q));
  }
}
#undef BFP_FN
