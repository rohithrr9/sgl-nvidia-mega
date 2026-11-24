// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
/**
 * \brief Sliding-window verify attention on e4m3 queries with the committed part of the window staged in shared memory
 *        before the dependency wait ("staged" form of swa_verify_attn_kernel).
 *
 * Same operation, shapes, masking, arithmetic, work split and merge order as swa_verify_attn_kernel (swa_verify_attn.cuh)
 * for its e4m3-query form (no query/key prologue): the output is bitwise identical for the same configuration. What
 * differs is where the K / V fragments come from and when the kernel waits:
 *
 *  - Before griddepcontrol.wait, the block copies the K and V rows of its key range that are already in the cache
 *    (keys below seq_len - 8, written by earlier steps) into shared memory with cp.async (16-byte copies, one KV head:
 *    64 bytes of K and 64 bytes of V per key). These reads use only seq_lens, the page table and those pool rows;
 *    the kernels that run between the previous step and this call do not write them (the queries and the K / V rows of
 *    the 8 new tokens are the only inputs the preceding kernel writes).
 *  - After the wait: the queries, and the K / V rows of the new tokens in the block's key range (cp.async into the same
 *    buffer), then a block barrier; every lane then reads its fragments from shared memory instead of the pools.
 *  - kEarlyTrigger: griddepcontrol.launch_dependents right after the wait, so that the next kernel can start its own
 *    pre-wait work (for example loading its weights) while this one computes; otherwise at the same place as
 *    swa_verify_attn_kernel (after the merge stores).
 *  - kExitBar (cluster form; the default): the exit cluster barrier of swa_verify_attn_kernel
 *    (arrive after the receive wait, wait at the end), so that no block exits while a record it sent may still be in
 *    flight (CUDA Programming Guide, distributed shared memory; PTX ISA, barrier.cluster). The form without it
 *    (MUSE_VR_SWA_NO_EXIT_BARRIER_UNSAFE=1, default off) is not guaranteed by the PTX memory model.
 *
 * Shared memory: the staging buffer (kChunk keys x 128 bytes) occupies the start of the per-warp partial area, which is
 * written only after the main loop; a block barrier after the last fragment read separates the two uses. The receive
 * records and the barrier are unchanged, so the footprint equals that of the e4m3-query configuration. kMinBlocks is the
 * launch-bounds minimum of resident blocks per multiprocessor (MUSE_VR_SWA_VERIFY_ATTN_STAGED_MINB; tunable).
 */
#pragma once

#include "swa_verify_attn.cuh"

namespace muse_vr {

namespace swa {

__device__ __forceinline__ void cp_async_16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src) : "memory");
}

// merge_records for records in global memory written by other blocks of this grid (__ldcg loads): same arithmetic.
template <int n, int stride>
__device__ __forceinline__ void merge_records_global(const float* rec, int c4, float4& acc, float& mx, float& lsum) {
  float ms[n];
  mx = -INFINITY;
#pragma unroll
  for (int s = 0; s < n; ++s) {
    ms[s] = __ldcg(rec + s * stride + kHeadDim);
    mx = fmaxf(mx, ms[s]);
  }
  acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  lsum = 0.0f;
#pragma unroll
  for (int s = 0; s < n; ++s) {
    if (ms[s] != -INFINITY) {
      const float w = fast_exp2(ms[s] - mx);
      const float4 v = __ldcg(reinterpret_cast<const float4*>(rec + s * stride + 4 * c4));
      acc.x += w * v.x;
      acc.y += w * v.y;
      acc.z += w * v.z;
      acc.w += w * v.w;
      lsum += w * __ldcg(rec + s * stride + kHeadDim + 1);
    }
  }
}

__device__ __forceinline__ void red_max_relaxed_gpu_u64(unsigned long long* p, unsigned long long v) {
  asm volatile("red.relaxed.gpu.global.max.u64 [%0], %1;\n" ::"l"(p), "l"(v) : "memory");
}
__device__ __forceinline__ unsigned long long atom_add_acq_rel_gpu_u64(unsigned long long* p, unsigned long long v) {
  unsigned long long old;
  asm volatile("atom.acq_rel.gpu.global.add.u64 %0, [%1], %2;\n" : "=l"(old) : "l"(p), "l"(v) : "memory");
  return old;
}

// Ticket form arguments. Arrival words: one 64-bit word per (layer slot, request, KV head) = (epoch << 32) | count, never
// reset; the epoch is the device step word (advanced once per step by swa_staged_step_kernel), read after the wait.
struct TicketArgs {
  float* __restrict__ rec;                        // records [request x KV head][rows][kC][68] (reused by every layer)
  unsigned long long* __restrict__ words;         // this layer slot's arrival words [kTicketMaxPairs]
  const unsigned long long* __restrict__ epoch;   // the device step word (low 32 bits)
  uint32_t* __restrict__ err;                     // sticky error word (0: none)
  uint32_t err_tag;                               // this call's tag in the error word (layer slot)
};
inline constexpr int kTicketSlots = 64;       // layer slots
inline constexpr int kTicketMaxPairs = 128;   // (request, KV head) pairs per launch

__global__ void swa_staged_step_kernel(unsigned long long* epoch) { *epoch += 1ull; }

// kTicket: the blocks of a (request, KV head) are launched without a cluster. Each block stores its merged row records
// to the records workspace ([rows][kC][record]) and arrives on its arrival word (max to (E, 0), then an acquire-release
// increment); the arrival that returns (E, kC - 1) is the last, and that block merges every row in block order and writes
// the output. No block waits for another and no word is reset. An arrival that returns a count >= kC (the step word did
// not advance since an earlier call on this slot) sets the sticky error word (bit 31 | bit 30 | slot << 16 | request bit)
// and writes NaN into the (request, KV head)'s output rows. Otherwise the blocks form a cluster and exchange records
// through distributed shared memory.
template <int kChunk, int kSplit, int kRG, int kC, int kMinBlocks, bool kEarlyTrigger, bool kNormGate, bool kTicket,
          bool kExitBar>
__global__ void __launch_bounds__(Cfg<kChunk, kSplit, kRG, kC, 0>::kThreads, kMinBlocks)
    swa_verify_attn_staged_kernel(const Params p, const TicketArgs ta) {
  float* const grec = ta.rec;
  using C = Cfg<kChunk, kSplit, kRG, kC, 0>;
  static_assert(kChunk * 2 * kHeadDim <= C::kRecv - C::kPart, "the staging buffer must fit in the partial area");
  static_assert(!kTicket || C::kRowBlocks == 1,
                "the ticket form indexes its arrival words and records per (request, KV head): one row block per pair");
  extern __shared__ __align__(128) uint8_t smem[];
  float* const part = reinterpret_cast<float*>(smem + C::kPart);
  float* const recv = reinterpret_cast<float*>(smem + C::kRecv);
  uint8_t* const stk = smem + C::kPart;              // staged K rows [kChunk][64] (aliases part)
  uint8_t* const stv = stk + kChunk * kHeadDim;      // staged V rows [kChunk][64]
  const uint32_t bar = smem_u32(smem + C::kBar);

  int rank;
  if constexpr (kTicket) {
    rank = static_cast<int>(blockIdx.x);
  } else {
    cg::cluster_group cluster = cg::this_cluster();
    rank = static_cast<int>(cluster.block_rank());
  }
  const int nown = (C::kRows - rank + kC - 1) / kC;
  if constexpr (!kTicket) {
    if (threadIdx.x == 0) mbar_init_expect(bar, static_cast<uint32_t>(nown * kC) * C::kRecordBytes);
    asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" ::: "memory");
  }

  const int rb = blockIdx.y / p.num_kv_heads;
  const int kvh = blockIdx.y - rb * p.num_kv_heads;
  const int req = blockIdx.z;
  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const int wrl = warp % kRG;
  const int wk = warp / kRG;
  const int wr = rb * kRG + wrl;
  const int g = lane >> 2, q4 = lane & 3;
  const int64_t head_off = static_cast<int64_t>(kvh) * kHeadDim;
  const int32_t* pt = p.page_table + static_cast<int64_t>(req) * p.page_table_stride;
  const int page_mask = (1 << p.page_shift) - 1;

  // Pre-wait: committed keys of the block's range -> shared memory. Item i: key i / 8, K (i % 8 < 4) or V, 16-byte group
  // i % 4 of the key's 64 bytes.
  const int seq_len = ld_s32(p.seq_lens + req);
  const int first = max(0, seq_len - kTokens - p.window_left);
  const int k0 = first + rank * kChunk;
  const int nvalid = max(0, min(kChunk, seq_len - k0));
  const int ncommit = max(0, min(nvalid, seq_len - kTokens - k0));  // keys k0 .. k0 + ncommit - 1 are in the cache
  const uint32_t sbase = smem_u32(stk);
  const auto stage = [&](int lo, int hi) {
    for (int i = tid; i < kChunk * 8; i += C::kThreads) {
      const int rel = i >> 3, part8 = i & 7;
      if (rel >= lo && rel < hi) {
        const int key = k0 + rel;
        const int64_t slot = (static_cast<int64_t>(ld_s32(pt + (key >> p.page_shift))) << p.page_shift) + (key & page_mask);
        const uint8_t* src = ((part8 & 4) ? p.v_pool : p.k_pool) + slot * p.pool_row_stride + head_off + 16 * (part8 & 3);
        cp_async_16(sbase + ((part8 & 4) ? kChunk * kHeadDim : 0) + rel * kHeadDim + 16 * (part8 & 3), src);
      }
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  };
  stage(0, ncommit);

  asm volatile("griddepcontrol.wait;\n" ::: "memory");
  if constexpr (kEarlyTrigger) asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");

  // Q rows (16 * wrl + g) and (+8): tokens t0 = 2*wr and t1 = 2*wr+1, head kvh*8 + g; 16 bytes each.
  const int t0 = 2 * wr, t1 = t0 + 1;
  uint4 qa, qb;
  {
    const uint8_t* qrow = p.q + (static_cast<int64_t>(req) * kTokens + t0) * p.q_token_stride +
                          (kvh * kQPerKv + g) * kHeadDim + 16 * q4;
    qa = ld_v4(qrow);
    qb = ld_v4(qrow + p.q_token_stride);
  }
  stage(ncommit, nvalid);  // the new tokens' rows, written by the preceding kernel
  asm volatile("cp.async.wait_all;\n" ::: "memory");
  __syncthreads();

  // K and V fragments of this warp's keys from shared memory: n-tile j holds key kbase + 8j + g; keys past the
  // sequence end are zeros.
  const int kbase = wk * C::kKeysPerWarp;
  uint4 kr[C::kNTiles], vr[C::kNTiles];
#pragma unroll
  for (int j = 0; j < C::kNTiles; ++j) {
    const int rel = kbase + 8 * j + g;
    if (rel < nvalid) {
      kr[j] = *reinterpret_cast<const uint4*>(stk + rel * kHeadDim + 16 * q4);
      vr[j] = *reinterpret_cast<const uint4*>(stv + rel * kHeadDim + 16 * q4);
    } else {
      kr[j] = vr[j] = make_uint4(0u, 0u, 0u, 0u);
    }
  }

  uint32_t aq[4][4];
  {
    const uint32_t wa[4] = {qa.x, qa.y, qa.z, qa.w};
    const uint32_t wb[4] = {qb.x, qb.y, qb.z, qb.w};
#pragma unroll
    for (int sstep = 0; sstep < 4; ++sstep) {
      aq[sstep][0] = f16x2_from_e4m3x2(wa[sstep] & 0xffffu);
      aq[sstep][1] = f16x2_from_e4m3x2(wb[sstep] & 0xffffu);
      aq[sstep][2] = f16x2_from_e4m3x2(wa[sstep] >> 16);
      aq[sstep][3] = f16x2_from_e4m3x2(wb[sstep] >> 16);
    }
  }

  float s[C::kNTiles][4];
#pragma unroll
  for (int j = 0; j < C::kNTiles; ++j) {
    const uint32_t kw[4] = {kr[j].x, kr[j].y, kr[j].z, kr[j].w};
    s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.0f;
#pragma unroll
    for (int sstep = 0; sstep < 4; ++sstep)
      mma_f16(s[j], aq[sstep][0], aq[sstep][1], aq[sstep][2], aq[sstep][3], f16x2_from_e4m3x2(kw[sstep] & 0xffffu),
              f16x2_from_e4m3x2(kw[sstep] >> 16));
  }

  const int pos0 = seq_len - kTokens + t0;
  const int lo0 = max(0, pos0 - p.window_left) - k0, hi0 = min(nvalid, pos0 - k0 + 1);
  const int lo1 = max(0, pos0 + 1 - p.window_left) - k0, hi1 = min(nvalid, pos0 - k0 + 2);
  float m0 = -INFINITY, m1 = -INFINITY;
  if (kbase >= max(lo0, lo1) && kbase + C::kKeysPerWarp <= min(hi0, hi1)) {
#pragma unroll
    for (int j = 0; j < C::kNTiles; ++j) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float x = s[j][e] * p.scale_log2;
        s[j][e] = x;
        if (e < 2) m0 = fmaxf(m0, x);
        else m1 = fmaxf(m1, x);
      }
    }
  } else {
#pragma unroll
    for (int j = 0; j < C::kNTiles; ++j) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int key = kbase + 8 * j + 2 * q4 + (e & 1);
        const bool valid = (e < 2) ? (key >= lo0 && key < hi0) : (key >= lo1 && key < hi1);
        const float x = valid ? s[j][e] * p.scale_log2 : -INFINITY;
        s[j][e] = x;
        if (e < 2) m0 = fmaxf(m0, x);
        else m1 = fmaxf(m1, x);
      }
    }
  }
  m0 = quad_max(m0);
  m1 = quad_max(m1);
  const float mu0 = (m0 == -INFINITY) ? 0.0f : m0;
  const float mu1 = (m1 == -INFINITY) ? 0.0f : m1;
  float l0 = 0.0f, l1 = 0.0f;
#pragma unroll
  for (int j = 0; j < C::kNTiles; ++j) {
    const float p0 = fast_exp2(s[j][0] - mu0), p1 = fast_exp2(s[j][1] - mu0);
    const float p2 = fast_exp2(s[j][2] - mu1), p3 = fast_exp2(s[j][3] - mu1);
    l0 += p0 + p1;
    l1 += p2 + p3;
    s[j][0] = p0 * kPScale;
    s[j][1] = p1 * kPScale;
    s[j][2] = p2 * kPScale;
    s[j][3] = p3 * kPScale;
  }
  l0 = quad_sum(l0);
  l1 = quad_sum(l1);

  float o[kHeadDim / 8][4];
#pragma unroll
  for (int i = 0; i < kHeadDim / 8; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
#pragma unroll
  for (int kk = 0; kk < C::kKSteps; ++kk) {
    const uint32_t a0 = f16x2_of_e4m3(s[2 * kk][0], s[2 * kk][1]);
    const uint32_t a1 = f16x2_of_e4m3(s[2 * kk][2], s[2 * kk][3]);
    const uint32_t a2 = f16x2_of_e4m3(s[2 * kk + 1][0], s[2 * kk + 1][1]);
    const uint32_t a3 = f16x2_of_e4m3(s[2 * kk + 1][2], s[2 * kk + 1][3]);
    const uint32_t w0[4] = {vr[2 * kk].x, vr[2 * kk].y, vr[2 * kk].z, vr[2 * kk].w};
    const uint32_t w1[4] = {vr[2 * kk + 1].x, vr[2 * kk + 1].y, vr[2 * kk + 1].z, vr[2 * kk + 1].w};
#pragma unroll
    for (int i = 0; i < kHeadDim / 8; ++i) {
      const uint32_t b0 = movmatrix_trans(f16x2_from_e4m3x2((w0[i >> 1] >> (16 * (i & 1))) & 0xffffu));
      const uint32_t b1 = movmatrix_trans(f16x2_from_e4m3x2((w1[i >> 1] >> (16 * (i & 1))) & 0xffffu));
      mma_f16(o[i], a0, a1, a2, a3, b0, b1);
    }
  }

  // Every warp has read its fragments: the staging buffer becomes the partial area.
  __syncthreads();
  {
    float* rec0 = part + (wk * C::kRows + wrl * 16 + g) * kRec + 16 * q4;
    float* rec1 = rec0 + 8 * kRec;
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      *reinterpret_cast<float4*>(rec0 + 4 * u) = make_float4(o[2 * u][0], o[2 * u][1], o[2 * u + 1][0], o[2 * u + 1][1]);
      *reinterpret_cast<float4*>(rec1 + 4 * u) = make_float4(o[2 * u][2], o[2 * u][3], o[2 * u + 1][2], o[2 * u + 1][3]);
    }
    if (q4 == 0) {
      *reinterpret_cast<float2*>(rec0 + kHeadDim) = make_float2(m0, l0);
      *reinterpret_cast<float2*>(rec1 + kHeadDim) = make_float2(m1, l1);
    }
  }
  __syncthreads();

  const int rq = req * p.num_kv_heads + kvh;  // (request, KV head)
  float* const grq = kTicket ? grec + static_cast<int64_t>(rq) * C::kRows * kC * kRec : nullptr;  // [rows][kC][kRec]
  if constexpr (kTicket) {
    for (int item = tid; item < C::kRows * 16; item += C::kThreads) {
      const int row = item >> 4, c4 = item & 15;
      float4 acc;
      float mx, lsum;
      merge_records<kSplit, C::kRows * kRec>(part + row * kRec, c4, acc, mx, lsum);
      float* const d = grq + (row * kC + rank) * kRec;
      *reinterpret_cast<float4*>(d + 4 * c4) = acc;
      if (c4 == 0) *reinterpret_cast<float2*>(d + kHeadDim) = make_float2(mx, lsum);
    }
    if constexpr (!kEarlyTrigger) asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
    __threadfence();  // this block's records, before its count
    __syncthreads();
    int* const go = reinterpret_cast<int*>(smem + C::kBar);  // 0: not the last, 1: merge, 2: poison
    if (tid == 0) {
      unsigned long long* const word = ta.words + rq;
      const unsigned long long tag = static_cast<unsigned long long>(static_cast<uint32_t>(*ta.epoch)) << 32;
      red_max_relaxed_gpu_u64(word, tag);
      const uint32_t cnt = static_cast<uint32_t>(atom_add_acq_rel_gpu_u64(word, 1ull));
      int g = 0;
      if (cnt >= static_cast<uint32_t>(kC)) {
        atomicOr(ta.err, 0xC0000000u | (ta.err_tag & 0xffu) << 16 | (1u << (req & 15)));
        g = 2;
      } else if (cnt == static_cast<uint32_t>(kC - 1)) {
        __threadfence();  // every other block's records, after their arrivals
        g = 1;
      }
      *go = g;
    }
    __syncthreads();
    if (*go == 0) return;
    if (*go == 2) {  // NaN into every output row of this (request, KV head), then exit
      const __nv_bfloat162 nan2 = __floats2bfloat162_rn(__int_as_float(0x7fc00000), __int_as_float(0x7fc00000));
      for (int item = tid; item < C::kRows * 32; item += C::kThreads) {
        const int row = item >> 5, c2 = item & 31;
        const int t = 2 * (rb * kRG + row / 16) + ((row >> 3) & 1);
        const int h = kvh * kQPerKv + (row & 7);
        reinterpret_cast<__nv_bfloat162*>(p.out + ((static_cast<int64_t>(req) * kTokens + t) * p.num_q_heads + h) *
                                                      kHeadDim)[c2] = nan2;
      }
      return;
    }
  } else {
    asm volatile("barrier.cluster.wait.aligned;\n" ::: "memory");
    for (int item = tid; item < C::kRows * 16; item += C::kThreads) {
      const int row = item >> 4, c4 = item & 15;
      float4 acc;
      float mx, lsum;
      merge_records<kSplit, C::kRows * kRec>(part + row * kRec, c4, acc, mx, lsum);
      const int owner = row % kC;
      const uint32_t dst = map_rank(smem_u32(recv + ((row / kC) * kC + rank) * kRec), owner);
      const uint32_t rbar = map_rank(bar, owner);
      st_async_v4(dst + 16 * c4, acc, rbar);
      if (c4 == 0) st_async_v2(dst + 4 * kHeadDim, mx, lsum, rbar);
    }
    if constexpr (!kEarlyTrigger) asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
    mbar_wait_phase0(bar);
    // Exit barrier arrival: every record bound for this block has landed (the wait above).
    if constexpr (kExitBar) asm volatile("barrier.cluster.arrive.release.aligned;\n" ::: "memory");
  }

  // Finalize: the cluster form, the rows this block owns (rank, rank + kC, ...); the ticket form (last arriver), every row.
  const int nfin = kTicket ? C::kRows : nown;
  for (int item = tid; item < nfin * 16; item += C::kThreads) {
    const int sl = item >> 4, c4 = item & 15;
    float4 acc;
    float mx, lsum;
    int row;
    if constexpr (kTicket) {
      row = sl;
      merge_records_global<kC, kRec>(grq + row * kC * kRec, c4, acc, mx, lsum);
    } else {
      merge_records<kC, kRec>(recv + sl * kC * kRec, c4, acc, mx, lsum);
      row = sl * kC + rank;
    }
    const int t = 2 * (rb * kRG + row / 16) + ((row >> 3) & 1);
    const int h = kvh * kQPerKv + (row & 7);
    const float f = lsum > 0.0f ? p.out_scale / lsum : 0.0f;
    __nv_bfloat162 lo = __floats2bfloat162_rn(acc.x * f, acc.y * f);
    __nv_bfloat162 hi = __floats2bfloat162_rn(acc.z * f, acc.w * f);
    if constexpr (kNormGate) {
      const float2 x01 = __bfloat1622float2(lo), x23 = __bfloat1622float2(hi);
      float ss = fmaf(x01.x, x01.x, 0.0f);
      ss = fmaf(x01.y, x01.y, ss);
      ss = fmaf(x23.x, x23.x, ss);
      ss = fmaf(x23.y, x23.y, ss);
      const unsigned group = 0xffffu << (threadIdx.x & 16);
#pragma unroll
      for (int mask = 8; mask >= 1; mask >>= 1) ss = ss + __shfl_xor_sync(group, ss, mask, 16);
      const float r = rsqrtf(ss / static_cast<float>(kHeadDim) + p.norm_eps);
      const uint2 graw = *reinterpret_cast<const uint2*>(
          p.gate + (static_cast<int64_t>(req) * kTokens + t) * p.gate_token_stride + h * kHeadDim + 4 * c4);
      const float2 g01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&graw.x));
      const float2 g23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&graw.y));
      const auto sig = [](float gv) { return __fdividef(1.0f, 1.0f + __expf(-gv)); };
      float y0 = x01.x * r, y1 = x01.y * r, y2 = x23.x * r, y3 = x23.y * r;
      y0 *= sig(g01.x);
      y1 *= sig(g01.y);
      y2 *= sig(g23.x);
      y3 *= sig(g23.y);
      lo = __floats2bfloat162_rn(y0, y1);
      hi = __floats2bfloat162_rn(y2, y3);
    }
    uint2 packed;
    packed.x = *reinterpret_cast<uint32_t*>(&lo);
    packed.y = *reinterpret_cast<uint32_t*>(&hi);
    *reinterpret_cast<uint2*>(p.out + ((static_cast<int64_t>(req) * kTokens + t) * p.num_q_heads + h) * kHeadDim +
                              4 * c4) = packed;
  }
  // Exit barrier wait (cluster form): every block of the cluster has received all of its records, this block's included.
  if constexpr (kExitBar && !kTicket) asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory");
}

}  // namespace swa

template <int kChunk, int kSplit, int kRG, int kC, int kMinBlocks, bool kEarlyTrigger, bool kNormGate, bool kTicket,
          bool kExitBar>
struct SwaVerifyAttnStagedKernel {
  // kTicket: rec = fp32 records workspace of >= R * Hkv * rows * kC * 68 floats; ws = int64 [>= 64 * 128 + 8] (arrival
  // words per layer slot, then the step word, then the sticky error word), zero at allocation and never reset; slot = this
  // call's layer slot (0-63). Unused otherwise.
  static void run(const tvm::ffi::TensorView rec, const tvm::ffi::TensorView ws, int64_t slot,
                  const tvm::ffi::TensorView q,           // [R*8, Hq, 64] e4m3, heads/dims contiguous
                  const tvm::ffi::TensorView k_pool,      // [slots, Hkv*64] e4m3 bytes, rows contiguous
                  const tvm::ffi::TensorView v_pool,      // [slots, Hkv*64]
                  const tvm::ffi::TensorView page_table,  // [R, pages] int32, inner contiguous
                  const tvm::ffi::TensorView seq_lens,    // [R] int32
                  const tvm::ffi::TensorView out,         // [R*8, Hq, 64] bf16, contiguous
                  int64_t window_left, int64_t page_size, double bmm1_scale, double bmm2_scale,
                  const tvm::ffi::TensorView gate,  // kNormGate: [R*8, >= Hq*64] bf16, inner contiguous
                  double norm_eps) {
    using namespace sglang::host;
    using Cfg = swa::Cfg<kChunk, kSplit, kRG, kC, 0>;
    RuntimeCheck(k_pool.dtype().code == DLDataTypeCode::kDLFloat8_e4m3fn &&
                     v_pool.dtype().code == DLDataTypeCode::kDLFloat8_e4m3fn,
                 "k/v must be float8_e4m3fn");
    RuntimeCheck(out.dtype().code == DLDataTypeCode::kDLBfloat && out.dtype().bits == 16, "out must be bf16");
    RuntimeCheck(page_table.dtype().code == DLDataTypeCode::kDLInt && page_table.dtype().bits == 32 &&
                     seq_lens.dtype().code == DLDataTypeCode::kDLInt && seq_lens.dtype().bits == 32,
                 "page_table/seq_lens must be int32");
    RuntimeCheck(out.dim() == 3 && out.size(2) == swa::kHeadDim, "out must be [tokens, heads, 64]");
    const int64_t m = out.size(0), hq = out.size(1);
    RuntimeCheck(m % swa::kTokens == 0, "token count must be a multiple of 8");
    const int64_t reqs = m / swa::kTokens;
    RuntimeCheck(seq_lens.dim() == 1 && seq_lens.size(0) == reqs && seq_lens.stride(0) == 1, "seq_lens [R]");
    RuntimeCheck(page_table.dim() == 2 && page_table.size(0) >= reqs && page_table.stride(1) == 1, "page_table");
    RuntimeCheck(out.stride(2) == 1 && out.stride(1) == swa::kHeadDim && out.stride(0) == hq * swa::kHeadDim,
                 "out layout");
    RuntimeCheck(k_pool.dim() == 2 && v_pool.dim() == 2 && k_pool.stride(1) == 1 && v_pool.stride(1) == 1 &&
                     k_pool.stride(0) == v_pool.stride(0) && k_pool.size(0) == v_pool.size(0) &&
                     k_pool.size(1) == v_pool.size(1),
                 "pools must be [slots, row] with unit inner stride");
    const int64_t hkv = k_pool.size(1) / swa::kHeadDim;
    RuntimeCheck(hkv * swa::kHeadDim == k_pool.size(1) && hkv * swa::kQPerKv == hq, "expects 8 query heads per KV head");
    RuntimeCheck(window_left >= 0 && page_size >= 64 && (page_size & (page_size - 1)) == 0, "window/page size");
    RuntimeCheck((window_left + swa::kTokens + kChunk - 1) / kChunk == kC, "cluster size ", kC,
                 " does not cover window_left ", window_left);
    const auto aligned16 = [](const void* ptr) { return reinterpret_cast<uintptr_t>(ptr) % 16 == 0; };
    RuntimeCheck(aligned16(k_pool.data_ptr()) && aligned16(v_pool.data_ptr()) && aligned16(out.data_ptr()) &&
                     k_pool.stride(0) % 16 == 0,
                 "16-byte alignment");
    RuntimeCheck(q.dim() == 3 && q.size(0) == m && q.size(1) == hq && q.size(2) == swa::kHeadDim &&
                     q.dtype().code == DLDataTypeCode::kDLFloat8_e4m3fn && q.stride(2) == 1 &&
                     q.stride(1) == swa::kHeadDim && q.stride(0) % 16 == 0 && aligned16(q.data_ptr()),
                 "q must be e4m3 [tokens, heads, 64], heads and dims contiguous, 16-byte aligned rows");
    int page_shift = 0;
    while ((int64_t{1} << page_shift) < page_size) ++page_shift;

    swa::Params params{};
    params.q = static_cast<const uint8_t*>(q.data_ptr());
    params.q_token_stride = q.stride(0);
    params.k_pool = static_cast<const uint8_t*>(k_pool.data_ptr());
    params.v_pool = static_cast<const uint8_t*>(v_pool.data_ptr());
    params.page_table = static_cast<const int32_t*>(page_table.data_ptr());
    params.seq_lens = static_cast<const int32_t*>(seq_lens.data_ptr());
    params.out = static_cast<__nv_bfloat16*>(out.data_ptr());
    params.pool_row_stride = k_pool.stride(0);
    params.pool_slots = k_pool.size(0);
    params.page_table_stride = page_table.stride(0);
    params.num_q_heads = static_cast<int32_t>(hq);
    params.num_kv_heads = static_cast<int32_t>(hkv);
    params.window_left = static_cast<int32_t>(window_left);
    params.page_shift = page_shift;
    params.scale_log2 = static_cast<float>(bmm1_scale * M_LOG2E);
    params.out_scale = static_cast<float>(bmm2_scale / static_cast<double>(swa::kPScale));
    if constexpr (kNormGate) {
      RuntimeCheck(gate.dtype().code == DLDataTypeCode::kDLBfloat && gate.dtype().bits == 16 && gate.dim() == 2 &&
                       gate.size(0) == m && gate.size(1) >= hq * swa::kHeadDim && gate.stride(1) == 1 &&
                       gate.stride(0) % 4 == 0 && reinterpret_cast<uintptr_t>(gate.data_ptr()) % 8 == 0,
                   "gate must be bf16 [tokens, >= Hq*64], inner contiguous, 8-byte aligned rows");
      params.gate = static_cast<const __nv_bfloat16*>(gate.data_ptr());
      params.gate_token_stride = gate.stride(0);
      params.norm_eps = static_cast<float>(norm_eps);
    }

    swa::TicketArgs ta{};
    if constexpr (kTicket) {
      RuntimeCheck(rec.dtype().code == DLDataTypeCode::kDLFloat && rec.dtype().bits == 32 &&
                       rec.numel() >= reqs * hkv * Cfg::kRows * kC * swa::kRec &&
                       reinterpret_cast<uintptr_t>(rec.data_ptr()) % 16 == 0,
                   "rec must be fp32, 16-byte aligned, >= R * Hkv * rows * kC * 68 floats");
      RuntimeCheck(ws.dtype().code == DLDataTypeCode::kDLInt && ws.dtype().bits == 64 &&
                       ws.numel() >= swa::kTicketSlots * swa::kTicketMaxPairs + 8 && slot >= 0 &&
                       slot < swa::kTicketSlots && reqs * hkv <= swa::kTicketMaxPairs,
                   "ticket workspace int64 [>= 64 * 128 + 8], layer slot 0..63, R * Hkv <= 128");
      auto* const w = static_cast<unsigned long long*>(ws.data_ptr());
      ta.rec = static_cast<float*>(rec.data_ptr());
      ta.words = w + slot * swa::kTicketMaxPairs;
      ta.epoch = w + swa::kTicketSlots * swa::kTicketMaxPairs;
      ta.err = reinterpret_cast<uint32_t*>(w + swa::kTicketSlots * swa::kTicketMaxPairs + 1);
      ta.err_tag = static_cast<uint32_t>(slot);
    }
    constexpr auto kernel =
        swa::swa_verify_attn_staged_kernel<kChunk, kSplit, kRG, kC, kMinBlocks, kEarlyTrigger, kNormGate, kTicket,
                                           kExitBar>;
    static bool configured = false;
    if (!configured) {
      RuntimeDeviceCheck(::cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Cfg::kBytes));
      if constexpr (!kTicket)
        RuntimeDeviceCheck(::cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
      configured = true;
    }
    if (reqs == 0) return;
    const dim3 grid(kC, static_cast<unsigned>(Cfg::kRowBlocks * hkv), static_cast<unsigned>(reqs));
    LaunchKernel(grid, Cfg::kThreads, out.device(), Cfg::kBytes)
        .config({.use_pdl = true, .cluster_dim = dim3(kTicket ? 1 : kC, 1, 1)})(kernel, params, ta);
  }

  // Registers, static / dynamic shared bytes and spill bytes of the kernel (resource read for the residency check).
  static void attrs(const tvm::ffi::TensorView out) {  // int64 [8] host tensor
    using namespace sglang::host;
    RuntimeCheck(out.dtype().code == DLDataTypeCode::kDLInt && out.dtype().bits == 64 && out.numel() >= 8,
                 "attrs: int64 [8] host tensor");
    using Cfg = swa::Cfg<kChunk, kSplit, kRG, kC, 0>;
    constexpr auto kernel =
        swa::swa_verify_attn_staged_kernel<kChunk, kSplit, kRG, kC, kMinBlocks, kEarlyTrigger, kNormGate, kTicket,
                                           kExitBar>;
    cudaFuncAttributes a;
    RuntimeDeviceCheck(::cudaFuncGetAttributes(&a, kernel));
    int64_t* o = static_cast<int64_t*>(out.data_ptr());
    o[0] = a.numRegs;
    o[1] = static_cast<int64_t>(a.sharedSizeBytes);
    o[2] = static_cast<int64_t>(a.localSizeBytes);
    o[3] = a.maxThreadsPerBlock;
    o[4] = Cfg::kBytes;
    o[5] = Cfg::kThreads;
    o[6] = a.binaryVersion;
    o[7] = 0;
  }
};

// The ticket form's step word: one plain launch per step, ahead of the step's first staged sliding-window call.
struct SwaStagedStep {
  static void run(const tvm::ffi::TensorView ws) {
    using namespace sglang::host;
    RuntimeCheck(ws.dtype().code == DLDataTypeCode::kDLInt && ws.dtype().bits == 64 &&
                     ws.numel() >= swa::kTicketSlots * swa::kTicketMaxPairs + 8,
                 "ticket workspace int64 [>= 64 * 128 + 8]");
    auto* const w = static_cast<unsigned long long*>(ws.data_ptr());
    LaunchKernel(dim3(1), 1, ws.device(), 0)
        .config({.use_pdl = false})(swa::swa_staged_step_kernel, w + swa::kTicketSlots * swa::kTicketMaxPairs);
  }
};

}  // namespace muse_vr
