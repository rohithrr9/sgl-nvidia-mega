// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Per-step ring-window expert placement (development form). World 4, 256 routed experts in 4 blocks of 64; block b's local
// experts 48..63 are also held by rank b + 1 (mod 4): a fixed replica set given by id arithmetic, independent of traffic.
// From the step's routed rows per expert, one warp computes for every block b a threshold t_b in [48, 64]: the routed
// experts of block b with local index >= t_b run on rank b + 1, the others on rank b. Integer arithmetic only, so every CTA
// and every rank that runs it on the same routed ids gets the same thresholds, with no communication.
//
// Cost of an expert with h > 0 routed rows: A + B h + P (ceil(h / pass_rows) - 1) (0 when h = 0). Block cost L_b = the sum
// over its experts; S_b(t) = the cost of block b's experts with local index >= t (t = 48..64, S_b(64) = 0). Rows only move
// forward around the ring, so the excess of any arc of blocks ending at r must leave through r's own replicated experts:
// T_LB = max(ceil(sum L / 4), max over r and arcs of n = 1..3 blocks ending at r of ceil((arc - S_r(48)) / n)) is the smallest
// per-rank maximum for which the pointwise smallest flow fits the replicated supply. Candidate j = 0..kNumT-1 takes
// T_j = T_LB + j step, the pointwise smallest flow c_b = max(0, max over n = 1..4 of (arc_n(b) - n T_j)), and per block the
// threshold t whose S_b(t) is nearest c_b (ties: the larger t, i.e. fewer moved experts); the candidate with the smallest
// resulting maximum rank cost L_b - S_b(t_b) + S_{b-1}(t_{b-1}) wins (ties: the smaller j).
// The host reference is muse_vr_plugin.ring_place.place(); tests/test_ring_place.py checks the two agree.

#pragma once

#include <cstdint>

// Placement cost-model weights and candidate spacings (relative units). Tunable: override with -D at build time.
#ifndef MUSE_VR_RING_COST_A
#define MUSE_VR_RING_COST_A 18430u  // per routed expert (relative units: only the ratios to A matter)
#endif
#ifndef MUSE_VR_RING_COST_B
#define MUSE_VR_RING_COST_B 77u     // per routed row
#endif
#ifndef MUSE_VR_RING_COST_P
#define MUSE_VR_RING_COST_P 2060u   // per pass beyond the first
#endif
#ifndef MUSE_VR_RING_STEP
#define MUSE_VR_RING_STEP 2000u     // spacing of the T candidates (place_warp: 8 candidates)
#endif
#ifndef MUSE_VR_RING_STEP2
#define MUSE_VR_RING_STEP2 4000u    // spacing of the T candidates (place_warp2: 4 candidates)
#endif

namespace muse_vr_ring {

constexpr int kWorld = 4, kBlock = 64, kMovLo = 48, kNumT = 8, kNumT2 = 4;
constexpr int kSStride = 20;  // S_b(t) row: entries t - 48 = 0..15 (16-byte aligned rows); S_b(64) = 0 is not stored
static_assert(kNumT * kWorld == 32, "one lane per (candidate, block)");

template <int kPassRows>
__device__ __forceinline__ uint32_t expert_cost(uint32_t h) {
  const uint32_t passes = (h + uint32_t(kPassRows) - 1u) / uint32_t(kPassRows);
  return h == 0u ? 0u : MUSE_VR_RING_COST_A + MUSE_VR_RING_COST_B * h + MUSE_VR_RING_COST_P * (passes - 1u);
}

// Phase 1 of the placement (one full warp): expert costs from hist, block costs L_b (every lane), S rows (written to S, then
// __syncwarp), the replicated supply S_b(48) and the Hall bound T_LB (every lane). sum = L of this lane's block.
template <int kPassRows>
__device__ __forceinline__ void stats_warp(const uint32_t* hist, uint32_t* S, int lane, uint32_t& sum, uint32_t (&L)[kWorld],
                                           uint32_t& T) {
  constexpr unsigned kFull = 0xffffffffu;
  // costs of experts 8 lane .. 8 lane + 7 (block lane / 8, local index 8 (lane % 8) ..)
  const uint4 h0 = *reinterpret_cast<const uint4*>(hist + 8 * lane);
  const uint4 h1 = *reinterpret_cast<const uint4*>(hist + 8 * lane + 4);
  uint32_t c[8] = {expert_cost<kPassRows>(h0.x), expert_cost<kPassRows>(h0.y), expert_cost<kPassRows>(h0.z),
                   expert_cost<kPassRows>(h0.w), expert_cost<kPassRows>(h1.x), expert_cost<kPassRows>(h1.y),
                   expert_cost<kPassRows>(h1.z), expert_cost<kPassRows>(h1.w)};
  uint32_t suf[8];
  uint32_t run = 0u;
#pragma unroll
  for (int k = 7; k >= 0; --k) {
    run += c[k];
    suf[k] = run;
  }
  sum = run;
  sum += __shfl_xor_sync(kFull, sum, 1);
  sum += __shfl_xor_sync(kFull, sum, 2);
  sum += __shfl_xor_sync(kFull, sum, 4);  // L_b in every lane of block b's group
  const int sub = lane & 7, blk = lane >> 3;
  const uint32_t hi = __shfl_down_sync(kFull, run, 1);  // the lane of local 56..63 -> the lane of local 48..55
  uint32_t mv_l = 0u;
  if (sub == 7) {
#pragma unroll
    for (int k = 0; k < 8; ++k) S[blk * kSStride + 8 + k] = suf[k];
  } else if (sub == 6) {
#pragma unroll
    for (int k = 0; k < 8; ++k) S[blk * kSStride + k] = suf[k] + hi;
    mv_l = suf[0] + hi;  // S_b(48)
  }
  __syncwarp();
  uint32_t mv[kWorld];
#pragma unroll
  for (int q = 0; q < kWorld; ++q) {
    L[q] = __shfl_sync(kFull, sum, 8 * q);
    mv[q] = __shfl_sync(kFull, mv_l, 8 * q + 6);
  }
  // Hall bound over the ring arcs (every lane; constant indices after unrolling)
  T = (L[0] + L[1] + L[2] + L[3] + 3u) / 4u;
#pragma unroll
  for (int r = 0; r < kWorld; ++r) {
    uint32_t arc = 0u;
#pragma unroll
    for (int n = 1; n < kWorld; ++n) {
      arc += L[(r - n + 1 + kWorld) % kWorld];
      if (arc > mv[r]) {
        const uint32_t need = (arc - mv[r] + uint32_t(n) - 1u) / uint32_t(n);
        T = need > T ? need : T;
      }
    }
  }
}

// One full warp. hist: [256] routed rows per expert (shared, 16-byte aligned); S: [kWorld * kSStride] scratch (shared,
// 16-byte aligned); thr: [kWorld] the thresholds (shared). Returns the winning key (max rank cost << 3 | candidate) in every
// lane. thr and S are complete for every lane of the warp on return. Lanes = (candidate j of kNumT, block q).
template <int kPassRows>
__device__ __forceinline__ uint32_t place_warp(const uint32_t* hist, uint32_t* S, int32_t* thr, int lane) {
  constexpr unsigned kFull = 0xffffffffu;
  uint32_t sum, L[kWorld], T;
  stats_warp<kPassRows>(hist, S, lane, sum, L, T);
  // lane = (candidate j, block q)
  const int j = lane >> 2, q = lane & 3;
  const int32_t Tj = int32_t(T + uint32_t(j) * MUSE_VR_RING_STEP);
  uint32_t Lr[kWorld];  // Lr[i] = L_{q - i}
#pragma unroll
  for (int i = 0; i < kWorld; ++i) Lr[i] = __shfl_sync(kFull, sum, 8 * ((q - i + kWorld) % kWorld));
  int32_t cq = 0, arc = 0;
#pragma unroll
  for (int n = 1; n <= kWorld; ++n) {
    arc += int32_t(Lr[n - 1]);
    const int32_t v = arc - n * Tj;
    cq = v > cq ? v : cq;
  }
  const uint4* Sq = reinterpret_cast<const uint4*>(S + q * kSStride);
  uint32_t sv[16];
#pragma unroll
  for (int u = 0; u < 4; ++u) {
    const uint4 w = Sq[u];
    sv[4 * u] = w.x;
    sv[4 * u + 1] = w.y;
    sv[4 * u + 2] = w.z;
    sv[4 * u + 3] = w.w;
  }
  int bt = 16;
  int32_t bd = cq;  // S_q(64) = 0
  uint32_t out = 0u;
#pragma unroll
  for (int k = 15; k >= 0; --k) {
    const int32_t df = int32_t(sv[k]) - cq;
    const int32_t d = df < 0 ? -df : df;
    if (d < bd) {
      bd = d;
      bt = k;
      out = sv[k];
    }
  }
  const uint32_t in = __shfl_sync(kFull, out, (lane & ~3) | ((lane + 3) & 3));
  uint32_t mx = Lr[0] - out + in;
  mx = max(mx, __shfl_xor_sync(kFull, mx, 1));
  mx = max(mx, __shfl_xor_sync(kFull, mx, 2));
  const uint32_t win = __reduce_min_sync(kFull, (mx << 3) | uint32_t(j));
  if (int(win & 7u) == j) thr[q] = kMovLo + bt;
  __syncwarp();
  return win;
}

// The same placement with 4 T candidates (spacing MUSE_VR_RING_STEP2): lanes = (candidate j, block q, half h); each half scans
// 8 of the 16 thresholds (h = 1 also the "move none" threshold 64), the pair merges by the same rule as the single scan (nearest
// S_q(t) to c_q; ties: the larger t). Same results as place_warp with kNumT2 candidates and STEP2; a shorter dependent chain.
template <int kPassRows>
__device__ __forceinline__ uint32_t place_warp2(const uint32_t* hist, uint32_t* S, int32_t* thr, int lane) {
  constexpr unsigned kFull = 0xffffffffu;
  uint32_t sum, L[kWorld], T;
  stats_warp<kPassRows>(hist, S, lane, sum, L, T);
  const int j = lane >> 3, q = (lane >> 1) & 3, h = lane & 1;
  const int32_t Tj = int32_t(T + uint32_t(j) * MUSE_VR_RING_STEP2);
  uint32_t Lr[kWorld];  // Lr[i] = L_{q - i}
#pragma unroll
  for (int i = 0; i < kWorld; ++i) Lr[i] = __shfl_sync(kFull, sum, 8 * ((q - i + kWorld) % kWorld));
  int32_t cq = 0, arc = 0;
#pragma unroll
  for (int n = 1; n <= kWorld; ++n) {
    arc += int32_t(Lr[n - 1]);
    const int32_t v = arc - n * Tj;
    cq = v > cq ? v : cq;
  }
  const uint4* Sq = reinterpret_cast<const uint4*>(S + q * kSStride + 8 * h);
  uint32_t sv[8];
#pragma unroll
  for (int u = 0; u < 2; ++u) {
    const uint4 w = Sq[u];
    sv[4 * u] = w.x;
    sv[4 * u + 1] = w.y;
    sv[4 * u + 2] = w.z;
    sv[4 * u + 3] = w.w;
  }
  int bt = h ? 16 : -1;
  int32_t bd = h ? cq : 0x7fffffff;  // S_q(64) = 0 (half 1 only)
  uint32_t out = 0u;
#pragma unroll
  for (int k = 7; k >= 0; --k) {
    const int32_t df = int32_t(sv[k]) - cq;
    const int32_t d = df < 0 ? -df : df;
    if (d < bd) {
      bd = d;
      bt = 8 * h + k;
      out = sv[k];
    }
  }
  const int32_t od = __shfl_xor_sync(kFull, bd, 1);
  const int ot = __shfl_xor_sync(kFull, bt, 1);
  const uint32_t oo = __shfl_xor_sync(kFull, out, 1);
  if (od < bd || (od == bd && ot > bt)) {
    bd = od;
    bt = ot;
    out = oo;
  }
  const uint32_t in = __shfl_sync(kFull, out, (lane & ~7) | (((q + 3) & 3) << 1) | h);
  uint32_t mx = Lr[0] - out + in;
  mx = max(mx, __shfl_xor_sync(kFull, mx, 2));
  mx = max(mx, __shfl_xor_sync(kFull, mx, 4));
  const uint32_t win = __reduce_min_sync(kFull, (mx << 3) | uint32_t(j));
  if (int(win & 7u) == j && h == 0) thr[q] = kMovLo + bt;
  __syncwarp();
  return win;
}

// Pool slot of expert e on rank r (pool of 80: own experts 0..63, then the predecessor's local experts 48..63), or -1.
__host__ __device__ __forceinline__ int window_slot(int e, int r) {
  const int ci = (e - (kBlock * r - (kBlock - kMovLo))) & 255;
  return ci >= 80 ? -1 : ci >= kBlock - kMovLo ? ci - (kBlock - kMovLo) : kBlock + ci;
}

// Whether pool slot li of rank r runs here under the thresholds (thr_own = t_r, thr_pred = t_{r-1}).
__host__ __device__ __forceinline__ bool slot_here(int li, int thr_own, int thr_pred) {
  return li < kBlock ? li < thr_own : (li < kBlock + (kBlock - kMovLo) && kMovLo + (li - kBlock) >= thr_pred);
}

}  // namespace muse_vr_ring
