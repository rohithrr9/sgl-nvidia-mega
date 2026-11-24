// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
/**
 * \brief Paged FP8 (e4m3) full-attention verify for many requests per step: a persistent kernel whose thread blocks
 * split the summed key range of all requests evenly (one block per SM, one wave), read whole token rows of the K/V
 * pools (every KV head of the rank), and run the attention on tensor-core MMA (tcgen05); a second kernel merges the
 * pieces of requests that several blocks share.
 *
 * Shapes (per call): q [R * 8, Hq, 64] e4m3 (heads and dims contiguous, any token stride that is a multiple of 16 B);
 * K/V pools [slots, Hkv * 64] e4m3 (token-major, slot = page * 128 + offset); page_table [R, >= pages] int32; seq_lens
 * [R] int32 (KV length including the 8 verify tokens); out [R * 8, Hq, 64] bf16, contiguous. Hq = 8 Hkv, Hkv = 2 kPairs
 * (kPairs = 1 or 2: 2 or 4 KV heads per rank). Token t (0..7) of request r sits at position p = seq_len - 8 + t and
 * attends keys [0, p].
 *
 * Work split. A request has n_r = ceil(seq_len / kTileKeys) tiles of kTileKeys keys (kTileKeys = 128 / kPairs, so a K
 * or V tile is always 16 KB: kTileKeys token rows of Hkv * 64 bytes). With W = sum n_r and G = min(blocks, W), block b
 * takes the tiles [floor(b W / G), floor((b + 1) W / G)) of the concatenation of all requests (request-major). The part
 * of a request inside one block is a piece; a request that lies wholly inside one block is normalized and written by
 * it; otherwise each piece writes a partial record (unnormalized output, row maximum, row sum per query row) and
 * full_attn_wide_merge_kernel (launched next, PDL-chained) merges the request's records in block order.
 *
 * Arithmetic per query row: s = fp32 dot product of e4m3 q and k (MMA, fp32 accumulation); per tile of kTileKeys keys,
 * in key order: mt = (max of s over the tile) * scale_log2; m' = max(m, mt); ar = exp2(m - m'); at = exp2(mt - m');
 * p448 = exp2(fma(s, scale_log2, log2(448) - mt)) (= 448 exp2(s scale_log2 - mt), the probability scaled for e4m3);
 * pq = e4m3(p448) (round to nearest even, satfinite); o = fma(o, ar, (sum of pq * v over the tile) * at) (MMA into
 * fp32, then fp32 arithmetic). Row sums (in x448 units): each of the row's two threads (half of every tile's keys)
 * keeps l_h = l_h * ar + (sum of its p448 over the tile) * at, its keys i in four interleaved partial sums (i % 4)
 * combined (0 + 2) + (1 + 3); l = l_0 + l_1 at the end of the piece. Merge of a request's pieces j = 0..k-1 (block
 * order): M = max m_j, O = sum exp2(m_j - M) o_j, L = sum exp2(m_j - M) l_j in order (pieces with m_j = -inf skipped);
 * out = bf16(O * (bmm2_scale / L)). A request inside one block: out = bf16(o * (bmm2_scale / l)). exp2 is
 * ex2.approx.ftz.f32 (or, with kPoly > 0, a degree-5 polynomial for kPoly of every 8 keys).
 *
 * Grouped merge (kGM, MUSE_VR_FULL_ATTN_WIDE_GMERGE; default off; the main kernel is unchanged): full_attn_wide_gmerge_kernel
 * replaces the merge kernel and merges each split request with several warps per row (pieces dealt round-robin to the
 * warps, batches of 16 against a running maximum, then the warps combined in index order), optionally followed by the
 * output norm + gate (kGM == 2, SGLang's norm_gate arithmetic for 16 threads per row); for a request inside one block it
 * applies the norm + gate to the output the main kernel wrote.
 *
 * Length-balanced split (kLb; with the grouped merge only, 1 <= R <= kLbMaxReqs, R <= blocks <= 4096): the blocks are
 * apportioned to the requests by their tile counts n_r with 32-bit integer arithmetic only (faw_lb_lane: the smallest
 * tiles-per-block target t from ceil(W / G) upwards, at most 8 steps, then ceil(W / (G - R)), for which
 * sum_r ceil(n_r / t) <= G; request r gets B_r = ceil(n_r / t) blocks, blocks past them stay empty); block s of request
 * r takes the request's tiles [floor(s n_r / B_r), floor((s + 1) n_r / B_r)), so every block holds one piece of one
 * request. Block 0 writes each request's (first block, block count) to a small workspace after its wait; the grouped
 * merge reads it after its own wait. Records stay indexed by block.
 *
 * Query/key prologue (kQk = E > 0, with kLb; layers without RoPE): the kernel takes the bf16 QKV projection output instead
 * of e4m3 q and does the work of SGLang's fused QK-norm kernel with an FP8 KV-cache store, with the arithmetic of
 * full_verify_attn's prologue for E columns per thread (see that file): the softmax warps compute the block's request's
 * q rows (every query head of the rank) into the Q buffer before the first QK MMA; a block whose tiles hold keys of the
 * 8 new tokens also computes their K and V rows (every KV head of the rank), writes them into its shared-memory K/V
 * stage after the stage's load completed (before the tile's QK MMA) and stores them into the pools at cache_loc (rows
 * with cache_loc == 0 are skipped). The tiles of a request are split over its blocks without overlap, so only the
 * storing block reads (by TMA, completed before the store) the pool bytes it stores.
 *
 * In-kernel merge (kFd > 0, MUSE_VR_FULL_ATTN_WIDE_FD; default off; with kLb and the grouped merge's arithmetic, kFdGm =
 * the grouped merge's kGM): the main kernel merges the split requests itself and no merge kernel is launched. Every block
 * writes its record as above; a request inside one block is written and (kFdGm == 2) norm-gated in place by its block. For
 * a split request, every block arrives exactly once, on every path, after its record (one elected thread, after a block
 * barrier) on the request's arrival word of the call's layer slot: a 64-bit (epoch << 32) | count word that is never
 * reset. The epoch E is the device step word (advanced once per step by full_attn_wide_fd_step_kernel, read after the
 * dependency wait); an arrival lifts a word of an earlier epoch to (E, 0) (atomic max) and then adds one with release
 * semantics, so a count left by an earlier step never counts toward this one. kFd == 1 (last arriver): the block whose
 * increment returns (E, B_r - 1) has acquired every record of the request and merges all its rows; the other blocks exit.
 * kFd == 2 (distributed): one elected thread of every block of the request waits for (E, B_r) (acquire loads with backoff,
 * bounded in SM clock cycles; the block's other threads wait at a block barrier), then the block merges its share of the
 * request's rows, a fixed function of its index (block s of B_r: rows [s 256 / B_r, (s + 1) 256 / B_r) at 4 KV heads). The
 * per-row arithmetic is the grouped merge's exactly (groups as its launcher computes them, pieces g, g + G, .. in batches
 * of 16 against a running maximum, groups combined in index order, then norm_gate's 16-thread arithmetic), each (row,
 * group) on one warp; group partials go through shared memory (the K ring, free after the main loop). Records are read with
 * L2 loads after the acquire. kFd == 2 needs the grid resident as one wave at one block per SM (the launcher asserts the
 * occupancy and takes kFd == 1 for a launch whose block count exceeds the resident capacity). An expired wait sets the
 * sticky error word (tagged with the layer slot and request), writes NaN into the rows the block owns and exits; so does an
 * arrival that finds a count >= B_r (the step word did not advance since an earlier call on the slot), in both forms, with
 * its own error bit (a request's rows are a fixed share of each of its blocks: s 256 / B_r .. (s + 1) 256 / B_r). Dependents
 * are triggered after the arrival (kFd == 1) or after the block's own merge share (kFd == 2).
 * kFd == 4 (early trigger; no in-kernel merge, no arrival words): every block triggers its dependents right after its own
 * griddepcontrol.wait instead of at its end, so the grouped merge kernel (unchanged, still waiting for this grid's
 * completion) is launched during the main loop and its blocks take processors as this grid's blocks exit.
 * kFd == 5: kFd == 4's early trigger with the low-register grouped merge (full_attn_wide_gmerge_1w_kernel: the same
 * arithmetic and output with fewer registers per thread, so that more of its blocks are resident at once and wait for this
 * grid's completion).
 * kFd == 6 (MUSE_VR_FULL_ATTN_WIDE_FD_PRE with form 5; no prologue): form 5 with the dependency wait moved. Before
 * griddepcontrol.wait the block map is computed from seq_lens, the load warp reads its page numbers and issues the TMA of
 * its first committed tiles (tiles with no key of the 8 new tokens, at most kStages); seq_lens and the page table are read
 * there with explicit gpu-scope loads. These reads touch the step's metadata and earlier steps' K / V, which are written
 * before the predecessor kernel starts. q and the remaining tiles follow the wait; the MMA and softmax warps wait at the
 * head of their role branches and then trigger the dependents as form 5 does; block 0's MMA warp stores the merge's block
 * map after its wait. So every global store follows a dependency wait on its own path; a block without tiles reads and
 * stores nothing and does not wait.
 * Launcher forms 7 and 8 (MUSE_VR_FULL_ATTN_WIDE_GM_TRIGGER with form 5): forms 5 and 6 with the trigger form of the
 * low-register grouped merge (full_attn_wide_gmerge_1wt_kernel), which triggers its own dependents at entry.
 *
 * Block roles: kSmWarps softmax warps (2 threads per query row: warp w -> half = w / (kSmWarps / 2) of every tile's
 * keys and of the output dims; the two threads of a row exchange tile maxima through shared memory and a named
 * barrier), one load warp (lane 0: per piece the q rows of the request with the tensor memory accelerator (one box of
 * 8 tokens x 8 heads x 64 B per KV head, 64-byte swizzle) into a Q buffer; per tile the K and V boxes (kTileKeys rows
 * x 128 B per KV-head pair, 128-byte swizzle; the pairs of a tile together cover whole token rows) into a ring of
 * kStages stages), one QK warp and one PV warp (the whole warp runs the loop with warp-uniform operands, one elected
 * lane issues: kind::f8f6f4, M = 64 rows per KV head, the two heads of a pair at tensor-memory lane offsets 0 and 16 of
 * every 32-lane group). Hand-offs per tile g: K/V landed -> QK(g) into score buffer g % 2 (issued two tiles ahead,
 * once the softmax warps hold tile g - 2's scores in registers) -> softmax -> probabilities in shared memory -> PV(g)
 * into output buffer g % 2 -> added into the registers of the softmax threads during tile g + 1.
 *
 * Ordering: q, seq_lens and the page table may be written by the preceding kernels; they are read only after
 * griddepcontrol.wait (use_pdl), followed by fence.proxy.async.global before the first tensor-memory-accelerator read
 * (kFd == 6 reads seq_lens, the page table and committed K / V before the wait, as described above).
 * Pool rows past the end of a request inside its last page are loaded and masked out of the probabilities (their
 * values are finite: the pools are allocated zero-filled and only finite values are stored into them). Every mbarrier
 * wait is bounded (trap after a fixed number of polls). Development probe modes (template argument kMode != 0) exist
 * for profiling only.
 */

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cmath>
#include <cstdint>
#include <cstdio>

namespace muse_vr {

namespace faw {

constexpr int kHeadDim = 64;
constexpr int kQPerKv = 8;
constexpr int kTokens = 8;
constexpr int kRows = 64;  // query rows per KV head: token * 8 + head
constexpr int kRec = 68;  // floats per partial row record: 64 outputs, m, l, 2 pad
constexpr float kPScale = 448.0f;
constexpr float kLog2PScale = 8.807354922057604f;  // log2(448)
constexpr int kMaxReqs = 256;
constexpr int kMaxPages = 1024;  // page numbers of a block's range held in shared memory (more: read from global)
constexpr int kTileBytes = 16384;  // one K or V tile
constexpr uint32_t kTmemCols = 512;  // S0 [0,128) S1 [128,256) O0 [256,384) O1 [384,512)

template <int kPairs, int kStages, int kTprT = 2>
struct Layout {
  static constexpr int kHeads = 2 * kPairs;
  static constexpr int kTileKeys = 128 / kPairs;
  static constexpr int kTpr = kTprT;  // softmax threads per query row (2: each half of a tile's keys and of the dims)
  static_assert(kTpr == 2 || (kTpr == 1 && kPairs == 2), "one thread per row only with 64-key tiles");
  static constexpr int kSmWarps = 4 * kPairs * kTpr;  // softmax warps: kTpr threads x (kHeads x 64 rows)
  static constexpr int kThreads = 32 * (kSmWarps + 3);  // + load warp + QK-MMA warp + PV-MMA warp
  static constexpr int kKpt = kTileKeys / kTpr;  // keys of a tile per softmax thread (32 or 64)
  static constexpr int kPRowBytes = kTileKeys;  // e4m3 probabilities of one row of a tile
  static constexpr int kSmK = 0;
  static constexpr int kSmV = kSmK + kStages * kTileBytes;
  static constexpr int kSmP = kSmV + kStages * kTileBytes;           // [2 buffers][heads][64 rows][kPRowBytes]
  static constexpr int kPBuf = kHeads * kRows * kPRowBytes;          // 16 KB
  static constexpr int kSmQ = kSmP + 2 * kPBuf;                      // [2 buffers][heads][64 rows][64 B]
  static constexpr int kQBuf = kHeads * kRows * 64;
  static constexpr int kQBufs = kStages >= 5 ? 1 : 2;  // Q buffers (one when the ring takes the shared memory)
  static constexpr int kSmRed = kSmQ + kQBufs * kQBuf;               // [3][heads * 64 rows][2] fp32
  static constexpr int kSmPre = kSmRed + 3 * kHeads * kRows * 2 * 4;  // prefix tile counts [kMaxReqs + 1]
  static constexpr int kSmPages = kSmPre + (kMaxReqs + 1) * 4 + 12;  // page numbers of the block's range
  static constexpr int kSmBar = (kSmPages + kMaxPages * 4 + 15) / 16 * 16;
  static constexpr int kSmBytes = kSmBar + 256 + 1024;
  // kQk: K and V rows of the 8 new tokens [2][8 tokens][kHeads * 64 B], after the barrier area
  static constexpr int kNewRowBytes = kHeads * 64;
  static constexpr int kSmNew = kSmBar + 256;
  static constexpr int kSmBytesQk = kSmNew + 2 * 8 * kNewRowBytes + 1024;
  // barrier area: kStages full + kStages empty + 12 others, the tensor-memory address, the patch barrier, the block's map
  static_assert(8 * (2 * kStages + 14) + 16 <= 256, "barrier area");
};

struct Params {
  const int32_t* __restrict__ page_table;
  const int32_t* __restrict__ seq_lens;
  float* __restrict__ partial;      // [G][2 slots][heads * 64 rows][kRec]
  uint32_t* __restrict__ counters;  // [kMaxReqs], zero between calls
  __nv_bfloat16* __restrict__ out;
  int64_t page_table_stride;
  int32_t num_reqs;
  int32_t num_q_heads;
  float scale_log2;
  float out_scale;  // bmm2_scale / 448
};
// Grouped merge, kGM == 2: the gate rows of the output norm + gate (as full_verify_attn's), a separate kernel argument so
// that the main kernel's parameters stay as they are.
struct GateArgs {
  const __nv_bfloat16* __restrict__ gate;
  int64_t gate_token_stride;
  float norm_eps;
  const int32_t* __restrict__ lb_map;  // kLb: [kLbMaxReqs][2] (first block, block count), written by the main kernel
};
// kLb / kQk: a separate argument of the main kernel (the default instantiation does not read it).
constexpr int kLbMaxReqs = 16;
struct QkArgs {
  const __nv_bfloat16* __restrict__ qkv;  // kQk: [R * 8, >= (Hq + 2 Hkv) * 64] bf16 (q, k, v, ...), inner contiguous
  const int64_t* __restrict__ cache_loc;  // kQk: [R * 8] pool rows of the new tokens (0: not stored)
  uint8_t* __restrict__ k_store;          // kQk: the pools (the tensor maps' bases)
  uint8_t* __restrict__ v_store;
  int32_t* __restrict__ lb_map;           // kLb: [kLbMaxReqs][2], written by block 0
  int64_t qkv_token_stride;               // elements
  float qk_eps;
};
// kFd: a separate argument of the main kernel after the others (the served instantiation does not read it).
// Arrival words (kFd): one 64-bit word per (layer slot, request) = (epoch << 32) | count, never reset; the epoch is the
// device step word, advanced once per step by full_attn_wide_fd_step_kernel and read after griddepcontrol.wait.
struct FdArgs {
  const __nv_bfloat16* __restrict__ gate;  // kFdGm == 2: [R * 8, >= Hq * 64] bf16, inner contiguous
  int64_t gate_token_stride;
  float norm_eps;
  int32_t groups;                            // the grouped merge's warps per row (its launcher's choice)
  unsigned long long* __restrict__ words;    // this layer slot's arrival words [kLbMaxReqs]
  const unsigned long long* __restrict__ epoch;  // the device step word (low 32 bits)
  uint32_t* __restrict__ err;                // sticky error word (0: none)
  uint32_t err_tag;                          // this call's tag in the error word (layer slot)
  uint64_t max_wait_cycles;                  // kFd == 2: bound of the arrival wait (SM clock cycles)
  uint32_t skip_block;                       // tests only: this block does not arrive (0xffffffff: none)
  uint32_t swap;                             // tests only: 1 pieces in reversed order, 2 piece 0 from the next row
};
constexpr int kFdLayerSlots = 64;     // arrival-word layer slots (one per model layer index)
constexpr uint32_t kFdSleepNs = 64;   // backoff of the arrival wait

// ---------------------------------------------------------------------------------------------- PTX wrappers
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(bar), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(bar) : "memory");
}
// Upper bound on try_wait calls of one wait: every arrival comes from this block (its threads, MMAs and copies); the
// bound turns a wait that can only hang through a defect into a trap.
inline constexpr uint32_t kMaxPolls = 1u << 24;
__device__ __noinline__ void wait_bound_exceeded() { __trap(); }
// The poll loop is written in PTX (one asm statement) so the compiler sees convergent control flow around it. Bounded:
// after kMaxPolls failed polls the thread traps.
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      ".reg .u32 n;\n"
      "mov.u32 n, 0;\n"
      "FAW_WAIT_%=:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      "@p bra FAW_DONE_%=;\n"
      "add.u32 n, n, 1;\n"
      "setp.lt.u32 p, n, %2;\n"
      "@p bra FAW_WAIT_%=;\n"
      "trap;\n"
      "FAW_DONE_%=:\n"
      "}\n" ::"r"(bar),
      "r"(parity), "n"(kMaxPolls)
      : "memory");
}
// Pure polling (mbarrier.test_wait, never suspends); development variant of mbar_wait for barriers completed by thread
// arrivals. Bounded like mbar_wait.
__device__ __forceinline__ void mbar_wait_spin(uint32_t bar, uint32_t parity) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      ".reg .u32 n;\n"
      "mov.u32 n, 0;\n"
      "FAW_SPIN_%=:\n"
      "mbarrier.test_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      "@p bra FAW_SDONE_%=;\n"
      "add.u32 n, n, 1;\n"
      "setp.lt.u32 p, n, %2;\n"
      "@p bra FAW_SPIN_%=;\n"
      "trap;\n"
      "FAW_SDONE_%=:\n"
      "}\n" ::"r"(bar),
      "r"(parity), "n"(kMaxPolls * 4)
      : "memory");
}
__device__ __forceinline__ void tma_load_2d(uint32_t dst, const CUtensorMap* map, int32_t c0, int32_t c1, uint32_t bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n" ::"r"(dst),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(bar)
      : "memory");
}
__device__ __forceinline__ void tma_load_3d(uint32_t dst, const CUtensorMap* map, int32_t c0, int32_t c1, int32_t c2,
                                            uint32_t bar) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];\n" ::"r"(
          dst),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(c2), "r"(bar)
      : "memory");
}
__device__ __forceinline__ void tma_prefetch_2d(const CUtensorMap* map, int32_t c0, int32_t c1) {
  asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global [%0, {%1, %2}];\n" ::"l"(reinterpret_cast<uint64_t>(map)),
               "r"(c0), "r"(c1)
               : "memory");
}
__device__ __forceinline__ void tc_fence_before() { asm volatile("tcgen05.fence::before_thread_sync;\n" ::: "memory"); }
__device__ __forceinline__ void tc_fence_after() { asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory"); }
__device__ __forceinline__ void fence_proxy_async() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }
__device__ __forceinline__ void mma_f8(uint32_t d, uint64_t a, uint64_t b, uint32_t idesc, uint32_t accumulate) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %4, 0;\n"
      "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n"
      "}\n" ::"r"(d),
      "l"(a), "l"(b), "r"(idesc), "r"(accumulate)
      : "memory");
}
// Warp-uniform variants: the whole warp executes them with uniform operands, one elected lane issues.
__device__ __forceinline__ void mma_f8_e(uint32_t d, uint64_t a, uint64_t b, uint32_t idesc, uint32_t accumulate) {
  asm volatile(
      "{\n"
      ".reg .pred e, p;\n"
      "elect.sync _|e, 0xffffffff;\n"
      "setp.ne.b32 p, %4, 0;\n"
      "@e tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n"
      "}\n" ::"r"(d),
      "l"(a), "l"(b), "r"(idesc), "r"(accumulate)
      : "memory");
}
__device__ __forceinline__ void mma_commit_e(uint32_t bar) {
  asm volatile(
      "{\n"
      ".reg .pred e;\n"
      "elect.sync _|e, 0xffffffff;\n"
      "@e tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n"
      "}\n" ::"r"(bar)
      : "memory");
}
__device__ __forceinline__ void mma_commit(uint32_t bar) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n" ::"r"(bar) : "memory");
}
// 32 consecutive fp32 columns of this thread's tensor-memory lane
__device__ __forceinline__ void tmem_ld32(uint32_t taddr, float (&v)[32]) {
  uint32_t r[32];
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,"
      "%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];\n"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]),
        "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]), "=r"(r[16]),
        "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]),
        "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31])
      : "r"(taddr));
#pragma unroll
  for (int i = 0; i < 32; ++i) v[i] = __uint_as_float(r[i]);
}
__device__ __forceinline__ void tmem_wait_ld() { asm volatile("tcgen05.wait::ld.sync.aligned;\n" ::: "memory"); }

// Shared-memory matrix descriptor: start >> 4 [0,14), leading byte offset >> 4 [16,30), stride byte offset >> 4
// [32,46), descriptor version 1 [46,48), swizzle mode [61,64) (PTX ISA, tcgen05 shared memory descriptor).
__device__ __forceinline__ uint64_t sdesc(uint32_t saddr, uint32_t sbo, uint32_t layout) {
  return static_cast<uint64_t>((saddr >> 4) & 0x3FFF) | (static_cast<uint64_t>(1) << 16) |
         (static_cast<uint64_t>((sbo >> 4) & 0x3FFF) << 32) | (static_cast<uint64_t>(1) << 46) |
         (static_cast<uint64_t>(layout & 7) << 61);
}
constexpr uint32_t kSw128 = 2, kSw64 = 4;
// Instruction descriptor, kind::f8f6f4, e4m3 x e4m3 -> fp32 (PTX ISA, tcgen05 instruction descriptor).
constexpr uint32_t idesc_f8(int m, int n, int b_mn_major) {
  return (1u << 4) | (static_cast<uint32_t>(b_mn_major) << 16) | (static_cast<uint32_t>(n >> 3) << 17) |
         (static_cast<uint32_t>(m >> 4) << 24);
}

__device__ __forceinline__ float fast_exp2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
  return y;
}
// 2^y for a pair (y <= 0 or -inf) in fp32 FMA arithmetic: y clamped to -126, y = n + f (n = round(y), |f| <= 1/2),
// 2^f by a degree-5 polynomial (Horner, fp32; max relative error 2.3e-7 over [-1/2, 1/2], the order of
// ex2.approx.ftz.f32's), then n added to the exponent field. Results below 2^-126 are not flushed to zero (they are
// at most 2^-126 and vanish in the e4m3 probabilities); ex2.approx.ftz.f32 returns 0 there.
__device__ __forceinline__ float2 fma2(float2 a, float2 b, float2 c);
__device__ __forceinline__ float2 add2(float2 a, float2 b);
__device__ __forceinline__ float2 exp2_poly2(float2 y) {
  constexpr float kMagic = 12582912.0f;  // 1.5 * 2^23: adding it rounds to an integer held in the low mantissa bits
  y = make_float2(fmaxf(y.x, -126.0f), fmaxf(y.y, -126.0f));
  const float2 t = add2(y, make_float2(kMagic, kMagic));
  const float2 n = add2(t, make_float2(-kMagic, -kMagic));
  const float2 f = fma2(n, make_float2(-1.0f, -1.0f), y);
  float2 q = fma2(f, make_float2(1.327764592e-03f, 1.327764592e-03f), make_float2(9.675552137e-03f, 9.675552137e-03f));
  q = fma2(q, f, make_float2(5.550709739e-02f, 5.550709739e-02f));
  q = fma2(q, f, make_float2(2.402211875e-01f, 2.402211875e-01f));
  q = fma2(q, f, make_float2(6.931469440e-01f, 6.931469440e-01f));
  q = fma2(q, f, make_float2(1.000000119e+00f, 1.000000119e+00f));
  return make_float2(__int_as_float(__float_as_int(q.x) + (__float_as_int(t.x) << 23)),
                     __int_as_float(__float_as_int(q.y) + (__float_as_int(t.y) << 23)));
}
__device__ __forceinline__ float2 fma2(float2 a, float2 b, float2 c) {
  uint64_t d;
  asm("fma.rn.f32x2 %0, %1, %2, %3;\n"
      : "=l"(d)
      : "l"(*reinterpret_cast<uint64_t*>(&a)), "l"(*reinterpret_cast<uint64_t*>(&b)), "l"(*reinterpret_cast<uint64_t*>(&c)));
  return *reinterpret_cast<float2*>(&d);
}
__device__ __forceinline__ float2 add2(float2 a, float2 b) {
  uint64_t d;
  asm("add.rn.f32x2 %0, %1, %2;\n"
      : "=l"(d)
      : "l"(*reinterpret_cast<uint64_t*>(&a)), "l"(*reinterpret_cast<uint64_t*>(&b)));
  return *reinterpret_cast<float2*>(&d);
}
__device__ __forceinline__ float2 mul2(float2 a, float2 b) {
  uint64_t d;
  asm("mul.rn.f32x2 %0, %1, %2;\n"
      : "=l"(d)
      : "l"(*reinterpret_cast<uint64_t*>(&a)), "l"(*reinterpret_cast<uint64_t*>(&b)));
  return *reinterpret_cast<float2*>(&d);
}
__device__ __forceinline__ float max3(float a, float b, float c) {
  float d;
  asm("max.f32 %0, %1, %2, %3;\n" : "=f"(d) : "f"(a), "f"(b), "f"(c));
  return d;
}
__device__ __forceinline__ uint32_t e4m3x4(float a, float b, float c, float d) {
  uint16_t lo, hi;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(lo) : "f"(b), "f"(a));
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(hi) : "f"(d), "f"(c));
  return static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
}
__device__ __forceinline__ void fence_gpu() { asm volatile("fence.acq_rel.gpu;\n" ::: "memory"); }
__device__ __forceinline__ uint32_t atom_add_acq_rel_gpu(uint32_t* p, uint32_t v) {
  uint32_t old;
  asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;\n" : "=r"(old) : "l"(p), "r"(v) : "memory");
  return old;
}
__device__ __forceinline__ void st_relaxed_gpu(uint32_t* p, uint32_t v) {
  asm volatile("st.relaxed.gpu.global.u32 [%0], %1;\n" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ float ld_relaxed_gpu_f32(const float* p) {
  float v;
  asm volatile("ld.relaxed.gpu.global.f32 %0, [%1];\n" : "=f"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ float2 ld_relaxed_gpu_v2(const float* p) {
  float2 v;
  asm volatile("ld.relaxed.gpu.global.v2.f32 {%0, %1}, [%2];\n" : "=f"(v.x), "=f"(v.y) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ float4 ld_relaxed_gpu_v4(const float* p) {
  float4 v;
  asm volatile("ld.relaxed.gpu.global.v4.f32 {%0, %1, %2, %3}, [%4];\n"
               : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
               : "l"(p)
               : "memory");
  return v;
}
// pre-wait reads of the step's metadata (kFd == 6): explicit loads at gpu scope, never the read-only cache path
__device__ __forceinline__ int ld_relaxed_gpu_s32(const int32_t* p) {
  int v;
  asm volatile("ld.relaxed.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory");
  return v;
}
#ifdef MUSE_VR_FAW_PRE_CHECK
// check build only: a coherent 16-byte re-read (gpu scope, never the read-only path)
__device__ __forceinline__ uint4 ld_relaxed_gpu_v4(const void* p) {
  uint4 v;
  asm volatile("ld.relaxed.gpu.global.v4.u32 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p)
               : "memory");
  return v;
}
#endif
__device__ __forceinline__ unsigned long long ld_acquire_gpu_u64(const unsigned long long* p) {
  unsigned long long v;
  asm volatile("ld.acquire.gpu.global.u64 %0, [%1];\n" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void atom_max_relaxed_gpu_u64(unsigned long long* p, unsigned long long v) {
  asm volatile("red.relaxed.gpu.global.max.u64 [%0], %1;\n" ::"l"(p), "l"(v) : "memory");
}
__device__ __forceinline__ unsigned long long atom_add_acq_rel_gpu_u64(unsigned long long* p, unsigned long long v) {
  unsigned long long old;
  asm volatile("atom.acq_rel.gpu.global.add.u64 %0, [%1], %2;\n" : "=l"(old) : "l"(p), "l"(v) : "memory");
  return old;
}

// Block b of G owns tiles [first(b), first(b + 1)) of W.
__device__ __forceinline__ int range_first(int b, int G, int W) {
  return static_cast<int>((static_cast<int64_t>(b) * W) / G);
}
// The block whose range holds tile t (the largest b with range_first(b) <= t).
__device__ __forceinline__ int block_of(int t, int G, int W) {
  return static_cast<int>(((static_cast<int64_t>(t) + 1) * G - 1) / W);
}

// kLb: the block map (see the header comment), warp-collective (all 32 lanes call it with the same R and G; lane r < R
// passes request r's tile count n): lane r gets B = request r's block count and prefix = blocks of the requests before
// it. 32-bit integer arithmetic only (tile counts clamped below 2^20), so every caller computes the same map. Needs
// 1 <= R <= kLbMaxReqs and R <= G <= 4096.
__device__ __forceinline__ void faw_lb_lane(int n, int R, int G, int lane, int& B, int& prefix) {
  constexpr unsigned kAll = 0xffffffffu;
  const uint32_t tiles = lane < R ? min(static_cast<uint32_t>(max(n, 1)), (1u << 20) - 1u) : 0u;
  const uint32_t W = __reduce_add_sync(kAll, tiles);
  int b = lane < R ? 1 : 0;
  if (G > R) {
    uint32_t t = (W + G - 1) / G;
    for (int it = 0;; ++it) {
      b = lane < R ? static_cast<int>((tiles + t - 1) / t) : 0;
      if (__reduce_add_sync(kAll, static_cast<uint32_t>(b)) <= static_cast<uint32_t>(G)) break;
      if (it == 7) {
        t = (W + (G - R) - 1) / (G - R);
        b = lane < R ? static_cast<int>((tiles + t - 1) / t) : 0;
        break;
      }
      t += 1;
    }
  }
  int x = b;  // inclusive prefix over lanes
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int y = __shfl_up_sync(kAll, x, o);
    if (lane >= o) x += y;
  }
  B = b;
  prefix = x - b;
}

// ---------------------------------------------------------------- query/key prologue (kQk)
// The arithmetic of full_verify_attn's prologue (csrc/full_verify_attn.cuh, "query/key prologue"), unchanged.
// Sum of squares of one thread's E consecutive columns, in the order of the Triton kernel for that column count.
template <int E, bool kUnfused = false>
__device__ __forceinline__ float row_sumsq(const float (&x)[E]) {
  if constexpr (E == 2) {
    if constexpr (kUnfused) return __fadd_rn(__fmul_rn(x[0], x[0]), __fmul_rn(x[1], x[1]));
    return __fmaf_rn(x[0], x[0], __fmul_rn(x[1], x[1]));
  } else if constexpr (E == 4) {
    return __fadd_rn(__fadd_rn(__fmul_rn(x[0], x[0]), __fmul_rn(x[2], x[2])),
                     __fadd_rn(__fmul_rn(x[1], x[1]), __fmul_rn(x[3], x[3])));
  } else {
    const float even = __fadd_rn(__fadd_rn(__fmul_rn(x[0], x[0]), __fmul_rn(x[2], x[2])),
                                 __fadd_rn(__fmul_rn(x[4], x[4]), __fmul_rn(x[6], x[6])));
    const float odd = __fadd_rn(__fadd_rn(__fmul_rn(x[1], x[1]), __fmul_rn(x[3], x[3])),
                                __fadd_rn(__fmul_rn(x[5], x[5]), __fmul_rn(x[7], x[7])));
    return __fadd_rn(even, odd);
  }
}
// Loads are volatile and placed ahead of the arithmetic that uses them.
template <int E>
__device__ __forceinline__ void ld_bf16_raw(const __nv_bfloat16* ptr, uint32_t (&w)[E / 2]) {
  if constexpr (E == 2) {
    asm volatile("ld.global.b32 %0, [%1];\n" : "=r"(w[0]) : "l"(ptr));
  } else if constexpr (E == 4) {
    asm volatile("ld.global.v2.b32 {%0, %1}, [%2];\n" : "=r"(w[0]), "=r"(w[1]) : "l"(ptr));
  } else {
    asm volatile("ld.global.v4.b32 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(w[0]), "=r"(w[1]), "=r"(w[2]), "=r"(w[3])
                 : "l"(ptr));
  }
}
template <bool kUnfused>
__device__ __forceinline__ float qk_mean(float ss) {
  float var;
  if constexpr (kUnfused) {
    float inv;
    asm("div.approx.f32 %0, %1, %2;\n" : "=f"(inv) : "f"(1.0f), "f"(64.0f));
    var = __fmul_rn(inv, ss);
  } else {
    asm("div.full.f32 %0, %1, %2;\n" : "=f"(var) : "f"(ss), "f"(64.0f));
  }
  return var;
}
// e4m3 bytes of one row's E columns held by this thread (all 64/E threads of the row call it together): QK-norm of
// the row, or (raw) the values unchanged.
template <int E, bool kUnfused>
__device__ __forceinline__ void qk_finish(const uint32_t (&w)[E / 2], float eps, bool raw, uint16_t (&b)[E / 2]) {
  float x[E];
#pragma unroll
  for (int i = 0; i < E / 2; ++i) {
    const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w[i]));
    x[2 * i] = f.x;
    x[2 * i + 1] = f.y;
  }
  float ss = row_sumsq<E, kUnfused>(x);
#pragma unroll
  for (int m = 32 / E; m >= 1; m >>= 1) ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, m));
  float r;
  const float var = qk_mean<kUnfused>(ss);
  const float t = __fadd_rn(eps, var);
  asm("rsqrt.approx.ftz.f32 %0, %1;\n" : "=f"(r) : "f"(t));
  float y[E];
#pragma unroll
  for (int e = 0; e < E; ++e) y[e] = raw ? x[e] : __bfloat162float(__float2bfloat16_rn(__fmul_rn(r, x[e])));
#pragma unroll
  for (int m = 0; m < E / 2; ++m) {
    uint16_t e;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n"
        : "=h"(e)
        : "f"(fminf(fmaxf(y[2 * m + 1], -448.0f), 448.0f)), "f"(fminf(fmaxf(y[2 * m], -448.0f), 448.0f)));
    b[m] = e;
  }
}
// Stores the E bytes of one thread (E = 2, 4 or 8).
template <int E>
__device__ __forceinline__ void st_bytes(void* ptr, const uint16_t (&b)[E / 2]) {
  if constexpr (E == 2) {
    *reinterpret_cast<uint16_t*>(ptr) = b[0];
  } else if constexpr (E == 4) {
    *reinterpret_cast<uint32_t*>(ptr) = static_cast<uint32_t>(b[0]) | (static_cast<uint32_t>(b[1]) << 16);
  } else {
    *reinterpret_cast<uint2*>(ptr) = make_uint2(static_cast<uint32_t>(b[0]) | (static_cast<uint32_t>(b[1]) << 16),
                                                static_cast<uint32_t>(b[2]) | (static_cast<uint32_t>(b[3]) << 16));
  }
}

// ---------------------------------------------------------------------------------------------- in-kernel merge (kFd)
// The grouped merge's per-row arithmetic for one warp (lane j: output dims 2j, 2j + 1), used by the main kernel's merge
// forms. Row `row` of a request: 0 .. heads * 64 - 1 (KV head row / 64; row % 64 = token * 8 + head in group).
constexpr int kFdBatch = 16;  // the grouped merge's kBatch
__device__ __forceinline__ __nv_bfloat162* fd_out(const Params& p, int r, int row, int lane) {
  const int rw = row & 63, tk = rw >> 3, qh = (row >> 6) * kQPerKv + (rw & 7);
  return reinterpret_cast<__nv_bfloat162*>(p.out + ((static_cast<int64_t>(r) * kTokens + tk) * p.num_q_heads + qh) *
                                                       kHeadDim) +
         lane;
}
// norm_gate_row of the grouped merge (SGLang's norm_gate arithmetic, 16 threads per row), row `row` of request r
__device__ __forceinline__ __nv_bfloat162 fd_norm_gate(__nv_bfloat162 v, const FdArgs& fa, int r, int row, int lane) {
  const int rw = row & 63, tk = rw >> 3, qh = (row >> 6) * kQPerKv + (rw & 7);
  const float2 x = __bfloat1622float2(v);
  const float a = fmaf(x.y, x.y, fmaf(x.x, x.x, 0.0f));
  const float prev = __shfl_xor_sync(0xffffffffu, a, 1);
  float ss = fmaf(x.y, x.y, fmaf(x.x, x.x, prev));  // meaningful on odd lanes
#pragma unroll
  for (int mask = 16; mask >= 2; mask >>= 1) ss = ss + __shfl_xor_sync(0xffffffffu, ss, mask);
  ss = __shfl_sync(0xffffffffu, ss, lane | 1);
  const float rn = rsqrtf(ss / static_cast<float>(kHeadDim) + fa.norm_eps);
  const float2 g = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(
      fa.gate + (static_cast<int64_t>(r) * kTokens + tk) * fa.gate_token_stride + qh * kHeadDim + 2 * lane));
  const auto sig = [](float z) { return __fdividef(1.0f, 1.0f + __expf(-z)); };
  float y0 = x.x * rn, y1 = x.y * rn;
  y0 *= sig(g.x);
  y1 *= sig(g.y);
  return __floats2bfloat162_rn(y0, y1);
}
template <int kFdGm>
__device__ __forceinline__ void fd_finalize(const Params& p, const FdArgs& fa, int r, int row, int lane, float2 acc,
                                            float lw) {
  const float f = lw > 0.0f ? p.out_scale / lw : 0.0f;
  __nv_bfloat162 v = __floats2bfloat162_rn(acc.x * f, acc.y * f);
  if constexpr (kFdGm == 2) v = fd_norm_gate(v, fa, r, row, lane);
  *fd_out(p, r, row, lane) = v;
}
// one batch of up to 16 pieces of a (row, group) against its running maximum (the grouped merge's loop body)
__device__ __forceinline__ void fd_batch_update(const float2 (&ml)[kFdBatch], const float2 (&v)[kFdBatch], float& mw,
                                                float2& acc, float& lw) {
  float mb = mw;
#pragma unroll
  for (int i = 0; i < kFdBatch; ++i) mb = fmaxf(mb, ml[i].x);
  if (mb != -INFINITY) {
    const float rs = mw == -INFINITY ? 0.0f : fast_exp2(mw - mb);
    acc.x *= rs;
    acc.y *= rs;
    lw *= rs;
#pragma unroll
    for (int i = 0; i < kFdBatch; ++i) {
      const float ws = ml[i].x == -INFINITY ? 0.0f : fast_exp2(ml[i].x - mb);
      acc.x += ws * v[i].x;
      acc.y += ws * v[i].y;
      lw += ws * ml[i].y;
    }
    mw = mb;
  }
}
// the batch's records (pieces k = grp + s0 + G i of the request's B_r pieces; piece k = block b0 + k, b0's record in slot
// 1, the others in slot 0), read with L2 loads; pieces past B_r as the grouped merge pads them
template <int kRR>
__device__ __forceinline__ void fd_load_batch(const Params& p, const FdArgs& fa, bool valid, int grp, int s0, int Br,
                                              int b0r, int row, int lane, float2 (&ml)[kFdBatch],
                                              float2 (&v)[kFdBatch]) {
  const int G = fa.groups;
#pragma unroll
  for (int i = 0; i < kFdBatch; ++i) {
    const int k = grp + s0 + G * i;
    if (valid && k < Br) {
      // tests only: swap 1 = the request's pieces merged in reversed order (changes the association of every group's sum
      // and of the group combine); swap 2 = piece 0 read from the neighbouring row
      const int kk = fa.swap == 1u ? Br - 1 - k : k;
      const int rr = (fa.swap == 2u && k == 0) ? row ^ 1 : row;
      const float* const rec = p.partial + ((static_cast<int64_t>(b0r + kk) * 2 + (kk == 0 ? 1 : 0)) * kRR + rr) * kRec;
      ml[i] = __ldcg(reinterpret_cast<const float2*>(rec + kHeadDim));
      v[i] = __ldcg(reinterpret_cast<const float2*>(rec + 2 * lane));
    } else {
      ml[i] = make_float2(-INFINITY, 0.0f);
      v[i] = make_float2(0.0f, 0.0f);
    }
  }
}
// Rows [rlo, rhi) of request r (the whole block calls it): (row, group) tasks dealt over the warps, one at a time per warp
// (each with its batch's 16 record loads in flight together); group
// partials (64 outputs, row sum, maximum per (row, group), as the grouped merge's shared sums)
// through `scr`, then one warp per row combines the groups in index order and writes the row.
template <int kRR, int kNW, int kFdGm>
__device__ __forceinline__ void fd_merge_rows(const Params& p, const FdArgs& fa, float* scr, int r, int Br, int b0r,
                                              int rlo, int rhi, int warp, int lane) {
  constexpr int kSum = kHeadDim + 2;
  const int G = fa.groups, crows = kRR / G;
  fence_proxy_async();
  for (int c0 = rlo; c0 < rhi; c0 += crows) {
    const int c1 = min(rhi, c0 + crows), nt = (c1 - c0) * G;
    for (int ta = warp; ta < nt; ta += kNW) {
      const int row_a = c0 + ta / G, grp_a = ta % G;
      float mw_a = -INFINITY, lw_a = 0.0f;
      float2 acc_a = make_float2(0.0f, 0.0f);
      for (int s0 = 0; grp_a + s0 < Br; s0 += G * kFdBatch) {
        float2 ml_a[kFdBatch], v_a[kFdBatch];
        fd_load_batch<kRR>(p, fa, true, grp_a, s0, Br, b0r, row_a, lane, ml_a, v_a);
        fd_batch_update(ml_a, v_a, mw_a, acc_a, lw_a);
      }
      if (G == 1) {
        fd_finalize<kFdGm>(p, fa, r, row_a, lane, acc_a, lw_a);
      } else {
        float* const sa = scr + ta * kSum;
        sa[2 * lane] = acc_a.x;
        sa[2 * lane + 1] = acc_a.y;
        if (lane == 0) {
          sa[kHeadDim] = lw_a;
          sa[kHeadDim + 1] = mw_a;
        }
      }
    }
    if (G > 1) {
      __syncthreads();
      for (int rl = warp; rl < c1 - c0; rl += kNW) {
        const float* const S = scr + rl * G * kSum;
        float mx = -INFINITY;
        for (int g = 0; g < G; ++g) mx = fmaxf(mx, S[g * kSum + kHeadDim + 1]);
        float o0 = 0.0f, o1 = 0.0f, lt = 0.0f;
        for (int g = 0; g < G; ++g) {
          const float m_g = S[g * kSum + kHeadDim + 1];
          const float rs = m_g == -INFINITY ? 0.0f : fast_exp2(m_g - mx);
          o0 += rs * S[g * kSum + 2 * lane];
          o1 += rs * S[g * kSum + 2 * lane + 1];
          lt += rs * S[g * kSum + kHeadDim];
        }
        fd_finalize<kFdGm>(p, fa, r, c0 + rl, lane, make_float2(o0, o1), lt);
      }
      __syncthreads();  // the scratch is reused by the next chunk
    }
  }
}

// ---------------------------------------------------------------------------------------------- main kernel
template <int kPairs, int kStages, bool kUsePDL, int kPoly, int kPf, int kMode, int kTpr, bool kLb = false, int kQk = 0,
          bool kQkUnfused = false, int kFd = 0, int kFdGm = 0>
__global__ void __launch_bounds__(Layout<kPairs, kStages, kTpr>::kThreads, 1)
    full_attn_wide_kernel(const Params p, const __grid_constant__ CUtensorMap q_map,
                          const __grid_constant__ CUtensorMap k_map, const __grid_constant__ CUtensorMap v_map,
                          const QkArgs qa, const FdArgs fa) {
  using L = Layout<kPairs, kStages, kTpr>;
  static_assert(kQk == 0 || kLb, "the query/key prologue needs one piece per block (kLb)");
  static_assert(kFd >= 0 && kFd <= 2 || kFd == 4 || kFd == 5 || kFd == 6,
                "kFd: 0 off, 1 last arriver, 2 distributed, 4 early trigger, 5 early trigger + one-wave grouped merge, "
                "6 = 5 + pre-wait setup and first committed tiles");
  static_assert(kFd != 6 || kQk == 0, "the pre-wait form: without the query/key prologue");
  // kFd == 6: the block map, the page reads and the first committed tiles' loads before griddepcontrol.wait
  constexpr bool kPreW = kFd == 6;
  static_assert(kFd == 0 || (kLb && kMode == 0 && (kFdGm == 1 || kFdGm == 2)),
                "the in-kernel merge needs the length-balanced split and the grouped merge's arithmetic");
  static_assert(kFd == 0 || 8 * (2 * kStages + 14) + 20 <= 256, "barrier area (block map + request's block range)");
  static_assert(kFd == 0 || kStages * kTileBytes >= 2 * kPairs * kRows * (kHeadDim + 2) * 4,
                "merge scratch (group partials of every row) inside the K ring");
  static_assert(kQk == 0 || kQk == 2 || kQk == 4 || kQk == 8, "prologue columns per thread");
  constexpr int kHeads = L::kHeads, kTileKeys = L::kTileKeys, kSmWarps = L::kSmWarps, kKpt = L::kKpt;
  // development switch carried in kPf: kPf == 1 -> poll the thread-arrival barriers (no suspend), no L2 prefetch
  constexpr bool kSpinP = kPf == 1;
  extern __shared__ uint8_t smem_raw[];
  // 1024-byte aligned base by pointer arithmetic on the shared array (keeps the shared address space visible to the
  // compiler: st.shared / ld.shared instead of generic accesses)
  uint8_t* const sm = smem_raw + ((1024u - (smem_u32(smem_raw) & 1023u)) & 1023u);
  uint64_t* const bars = reinterpret_cast<uint64_t*>(sm + L::kSmBar);
  // bars: full[kStages], empty[kStages], s_full[2], p_full[2], o_full[2], q_full[2], q_empty[2], (tmem address)
  const uint32_t bar0 = smem_u32(bars);
  const auto full_bar = [&](int s) { return bar0 + 8u * s; };
  const auto empty_bar = [&](int s) { return bar0 + 8u * (kStages + s); };
  const auto s_full_bar = [&](int b) { return bar0 + 8u * (2 * kStages + b); };
  const auto p_full_bar = [&](int b) { return bar0 + 8u * (2 * kStages + 2 + b); };
  const auto o_full_bar = [&](int b) { return bar0 + 8u * (2 * kStages + 4 + b); };
  const auto q_full_bar = [&](int b) { return bar0 + 8u * (2 * kStages + 6 + b); };
  const auto q_empty_bar = [&](int b) { return bar0 + 8u * (2 * kStages + 8 + b); };
  const auto s_free_bar = [&](int b) { return bar0 + 8u * (2 * kStages + 10 + b); };  // scores of buffer b in registers
  uint32_t* const tmem_slot = reinterpret_cast<uint32_t*>(sm + L::kSmBar + 8 * (2 * kStages + 12));
  // kQk: new-token rows written into a K/V stage (one arrival per softmax warp; one phase per patched tile)
  const uint32_t patch_bar = bar0 + 8u * (2 * kStages + 13);
  // kLb: this block's request and tile range [T0, T1) of the concatenation (written by warp kSmWarps + 1)
  int32_t* const lbinfo = reinterpret_cast<int32_t*>(sm + L::kSmBar + 8 * (2 * kStages + 14));
  int32_t* const pre = reinterpret_cast<int32_t*>(sm + L::kSmPre);  // pre[r] = tiles of requests < r; pre[R] = W
  int32_t* const pages = reinterpret_cast<int32_t*>(sm + L::kSmPages);

  // warp index through a warp shuffle: the compiler then knows it is warp-uniform, so the role branches below are
  // convergent
  const int tid = threadIdx.x, warp = __shfl_sync(0xffffffffu, tid >> 5, 0), lane = tid & 31;
  const int b = blockIdx.x;
  const int R = p.num_reqs;
  // development mode 4: clock stamps of every hand-off for blocks 0, G/2 and G-1 (tiles < 256), written after the
  // partial-record region of the workspace: [3 blocks][16 events][256 tiles] u64
  const int trace_slot = kMode != 4 ? -1 : (b == 0 ? 0 : b == static_cast<int>(gridDim.x) / 2 ? 1 : b == static_cast<int>(gridDim.x) - 1 ? 2 : -1);
  uint64_t* const trace = reinterpret_cast<uint64_t*>(
      p.partial + static_cast<int64_t>(gridDim.x) * 2 * (2 * kPairs * kRows) * kRec);
  const auto stamp = [&](int ev, int gg) {
    if (kMode == 4 && trace_slot >= 0 && gg >= 0 && gg < 256) {
      uint64_t c;
      asm volatile("mov.u64 %0, %%clock64;\n" : "=l"(c));
      trace[(trace_slot * 16 + ev) * 256 + gg] = c;
    }
  };

  // Barriers and tensor memory first: they touch nothing the preceding kernels write.
  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < kStages; ++s) {
      mbar_init(full_bar(s), 1);
      // released by the PV MMAs of every KV-head pair (one commit each; load-only / MMA-only probes: one arrival)
      mbar_init(empty_bar(s), 1);
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      mbar_init(s_full_bar(i), 1);
      mbar_init(p_full_bar(i), kSmWarps);  // one elected lane per softmax warp
      mbar_init(o_full_bar(i), 1);
      // kQk: the softmax warps write the q rows (one elected lane per warp arrives); else one TMA arrival
      mbar_init(q_full_bar(i), kQk > 0 ? kSmWarps : 1);
      mbar_init(q_empty_bar(i), 1);
      mbar_init(s_free_bar(i), kSmWarps);
    }
    if constexpr (kQk > 0) mbar_init(patch_bar, kSmWarps);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  if (warp == 0) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n" ::"r"(smem_u32(tmem_slot)),
                 "r"(kTmemCols)
                 : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n" ::: "memory");
  }
  // kFd == 6: the wait moves below the setup (block map, page reads, first committed tiles; see the load warp)
  if constexpr (kUsePDL && !kPreW) asm volatile("griddepcontrol.wait;\n" ::: "memory");
  // kFd == 4 (early trigger): the dependents (the grouped merge) are launched once every block got here, so their blocks
  // land on the processors as this grid's blocks exit; they still wait for this grid's completion (griddepcontrol.wait)
  if constexpr ((kFd == 4 || kFd == 5) && kUsePDL) asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
  // kFd == 6: seq_lens and the page table are read before the wait with explicit gpu-scope loads (the other forms keep
  // their plain reads, written out separately so that their code stays as it is)
  int lb_prel = 0, lb_count = 0;  // kFd == 6: block 0 stores the merge's block map after its wait
#ifdef MUSE_VR_FAW_PRE_CHECK
  // Tests and units only (renamed namespace): every word kFd == 6 reads before its dependency wait is read again after
  // the wait of the same warp and compared; counters after the partial records: [0] / [1] metadata words mismatched /
  // compared, [2] / [3] pre-wait tile words (16 bytes) mismatched / compared (per-tile sum + xor checksums of the landed
  // stage against a coherent re-read of the same pool rows, so the shared-memory swizzle order does not matter).
  // ck_pre: the tile count of the prefix pass below (its own read of the lane's length); ck_n / ck_seq: the block map's.
  int ck_seq = -1, ck_n = -1, ck_pre = -1;
  unsigned long long* const ck = reinterpret_cast<unsigned long long*>(trace);  // 64-bit: hour-long in-server counts
  // A pre-wait tile's page number comes from the shared-memory copy (pages[]) whenever its index is below kMaxPages; the
  // tiles issued before the wait are the first min(kStages, ...) of the block's range, so they always do.
  static_assert(kStages <= kMaxPages, "pre-wait tiles must take their page numbers from the checked shared-memory copy");
#endif
  // Prefix tile counts (warp 9): pre[r] for r = 0..R.
  if (warp == kSmWarps + 1) {
    int carry = 0;
    for (int r0 = 0; r0 < R; r0 += 32) {
      const int r = r0 + lane;
      int n = 0;
      if constexpr (kPreW) {
        if (r < R) n = (max(ld_relaxed_gpu_s32(p.seq_lens + r), 1) + kTileKeys - 1) / kTileKeys;
#ifdef MUSE_VR_FAW_PRE_CHECK
        if (r0 == 0) ck_pre = r < R ? n : -1;
#endif
      } else {
        if (r < R) n = (max(p.seq_lens[r], 1) + kTileKeys - 1) / kTileKeys;
      }
      int x = n;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o) x += y;
      }
      if (r < R) pre[r + 1] = carry + x;
      carry += __shfl_sync(0xffffffffu, x, 31);
    }
    if (lane == 0) pre[0] = 0;
    if constexpr (kLb) {
      // the block map (R <= kLbMaxReqs: one pass above held every request; lane r's tile count is recomputed)
      int n = 0;
      if constexpr (kPreW) {
        n = lane < R ? (max(ld_relaxed_gpu_s32(p.seq_lens + lane), 1) + kTileKeys - 1) / kTileKeys : 0;
#ifdef MUSE_VR_FAW_PRE_CHECK
        ck_n = n;
        ck_seq = lane < R ? ld_relaxed_gpu_s32(p.seq_lens + lane) : -1;
#endif
      } else {
        n = lane < R ? (max(p.seq_lens[lane], 1) + kTileKeys - 1) / kTileKeys : 0;
      }
      const int Gm = static_cast<int>(gridDim.x);
      int Bl, prel;
      faw_lb_lane(n, R, Gm, lane, Bl, prel);
      const int x = n;  // exclusive prefix of the tile counts (pre[] of this lane's request)
      int incl = x;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, incl, o);
        if (lane >= o) incl += y;
      }
      const int tpre = incl - x;
      const int Wt = __shfl_sync(0xffffffffu, incl, 31);
      const unsigned hit = __ballot_sync(0xffffffffu, lane < R && b >= prel && b < prel + Bl);
      const int rr = hit ? __ffs(hit) - 1 : R - 1;
      const int Br = __shfl_sync(0xffffffffu, Bl, rr), pr = __shfl_sync(0xffffffffu, prel, rr);
      const int nr = __shfl_sync(0xffffffffu, n, rr), tp = __shfl_sync(0xffffffffu, tpre, rr);
      if (lane == 0) {
        if (hit) {
          const int s = b - pr;
          lbinfo[0] = rr;
          lbinfo[1] = tp + static_cast<int>(static_cast<int64_t>(s) * nr / Br);
          lbinfo[2] = tp + static_cast<int>(static_cast<int64_t>(s + 1) * nr / Br);
        } else {  // a block past the requests' blocks: no tiles
          lbinfo[0] = R - 1;
          lbinfo[1] = Wt;
          lbinfo[2] = Wt;
        }
        if constexpr (kFd == 1 || kFd == 2) {  // the request's block range for the in-kernel merge (0: no request)
          lbinfo[3] = hit ? Br : 0;
          lbinfo[4] = hit ? pr : 0;
        }
      }
      // the merge reads each request's (first block, block count) after its own wait
      if constexpr (kPreW) {
        lb_prel = prel;  // stored after this block's wait (nothing is stored before it)
        lb_count = Bl;
      } else {
        if (b == 0 && lane < R) {
          qa.lb_map[2 * lane] = prel;
          qa.lb_map[2 * lane + 1] = Bl;
        }
      }
    }
  }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = __shfl_sync(0xffffffffu, *tmem_slot, 0);  // warp-uniform
  if (tid == 0) stamp(15, 0);
  const int W = pre[R];
  int T0, T1;
  if constexpr (kLb) {
    T0 = lbinfo[1];
    T1 = lbinfo[2];
  } else {
    // blocks used: at most one per tile, so that every used block owns at least one tile
    const int G = min(static_cast<int>(gridDim.x), W);
    T0 = b < G ? range_first(b, G, W) : W;
    T1 = b < G ? range_first(b + 1, G, W) : W;
  }
  const int ntiles = __shfl_sync(0xffffffffu, T1 - T0, 0);  // warp-uniform (provably, for the MMA warps' code)
  // first request of the range: the largest r with pre[r] <= T0 (requests have >= 1 tile)
  int r_first = 0;
  {
    int lo = 0, hi = R - 1;
    while (lo < hi) {
      const int mid = (lo + hi + 1) >> 1;
      if (pre[mid] <= T0) lo = mid; else hi = mid - 1;
    }
    r_first = lo;
  }
  // kQk: the block's tiles [pn0, pn0 + npn) (local index) hold keys of the 8 new tokens (uniform over the block)
  int pn0 = 0, npn = 0;
  if constexpr (kQk > 0) {
    if (ntiles > 0) {
      const int sl = p.seq_lens[r_first], lt0 = T0 - pre[r_first];
      const int a = max(max(sl - kTokens, 0) / kTileKeys, lt0), e = min((max(sl, 1) - 1) / kTileKeys, lt0 + ntiles - 1);
      if (a <= e) {
        pn0 = a - lt0;
        npn = e - a + 1;
      }
    }
  }
  // kFd == 6: each role below waits at the head of its own branch (the load warp after its pre-wait issue; the MMA and
  // softmax warps first thing, then they trigger the dependents as form 5 does), so every global store of the kernel
  // follows a dependency wait on its own path; a block without tiles reads and stores nothing and does not wait

  if (ntiles > 0) {
    if (warp == kSmWarps) {
      // ------------------------------------------------------------------ loads
      // page numbers of the range -> shared memory (all lanes, all loads in flight together); kFd == 6: before the wait,
      // with explicit gpu-scope loads
      constexpr int kTilesPerPage = 128 / kTileKeys;
      {
        int r = r_first, base = 0;
        for (int t = T0; t < T1 && base < kMaxPages;) {
          const int lo = t - pre[r], hi = min(T1, pre[r + 1]) - pre[r];
          const int pg0 = lo / kTilesPerPage, pg1 = (hi - 1) / kTilesPerPage + 1;
          const int32_t* const pt = p.page_table + static_cast<int64_t>(r) * p.page_table_stride;
          if constexpr (kPreW) {
            for (int i = lane; i < pg1 - pg0 && base + i < kMaxPages; i += 32)
              pages[base + i] = ld_relaxed_gpu_s32(pt + pg0 + i);
          } else {
            for (int i = lane; i < pg1 - pg0 && base + i < kMaxPages; i += 32) pages[base + i] = pt[pg0 + i];
          }
          base += pg1 - pg0;
          t = pre[r] + hi;
          ++r;
        }
      }
      __syncwarp();
      if constexpr (kPreW) {
        // One piece per block (kLb). Before the dependency wait: the TMA of the block's first committed tiles (no key of
        // the 8 new tokens, whose K / V the predecessor writes), in tile order into stages 0 .. npre - 1. After it: q,
        // then the remaining tiles in the same order.
        const int r = r_first, lo = T0 - pre[r], hi = T1 - pre[r], pg0 = lo / kTilesPerPage;
        const int32_t* const pt = p.page_table + static_cast<int64_t>(r) * p.page_table_stride;
        // tiles >= first_new hold keys of the new tokens
        const int first_new = max(ld_relaxed_gpu_s32(p.seq_lens + r) - kTokens, 0) / kTileKeys;
        const int npre = min(min(kStages, hi - lo), max(first_new - lo, 0));
        const auto issue = [&](int g) {
          const int i = lo + g, s = g % kStages;
          if (g >= kStages) mbar_wait(empty_bar(s), ((g / kStages) - 1) & 1);
          const int pi = i / kTilesPerPage - pg0;
          const int page = pi < kMaxPages ? pages[pi] : ld_relaxed_gpu_s32(pt + i / kTilesPerPage);
          const int row0 = page * 128 + (i % kTilesPerPage) * kTileKeys;
          mbar_expect_tx(full_bar(s), 2 * kTileBytes);
#pragma unroll
          for (int c = 0; c < kPairs; ++c) {
            tma_load_2d(smem_u32(sm + L::kSmK + s * kTileBytes + c * kTileKeys * 128), &k_map, c * 128, row0,
                        full_bar(s));
            tma_load_2d(smem_u32(sm + L::kSmV + s * kTileBytes + c * kTileKeys * 128), &v_map, c * 128, row0,
                        full_bar(s));
          }
        };
        if (lane == 0) {
          asm volatile("fence.proxy.async.global;\n" ::: "memory");
          for (int g = 0; g < npre; ++g) issue(g);
        }
        __syncwarp();
        if constexpr (kUsePDL) asm volatile("griddepcontrol.wait;\n" ::: "memory");
#ifdef MUSE_VR_FAW_PRE_CHECK
        {  // the page numbers of the block's range and the length behind first_new, read again after the wait
          const int np = min((hi - 1) / kTilesPerPage + 1 - pg0, kMaxPages);
          unsigned int bad = 0, cnt = 0;
          for (int i = lane; i < np; i += 32) {
            ++cnt;
            bad += ld_relaxed_gpu_s32(pt + pg0 + i) != pages[i] ? 1u : 0u;
          }
          if (lane == 0) {
            ++cnt;
            bad += max(ld_relaxed_gpu_s32(p.seq_lens + r) - kTokens, 0) / kTileKeys != first_new ? 1u : 0u;
          }
#pragma unroll
          for (int o = 16; o > 0; o >>= 1) {
            bad += __shfl_xor_sync(0xffffffffu, bad, o);
            cnt += __shfl_xor_sync(0xffffffffu, cnt, o);
          }
          if (lane == 0) {
            atomicAdd(ck + 1, static_cast<unsigned long long>(cnt));
            if (bad) atomicAdd(ck, static_cast<unsigned long long>(bad));
          }
        }
        {  // the pre-wait tiles' bytes: once landed, against a coherent re-read of the same pool rows after the wait
          // (npre <= kStages: no stage has been reused yet, so stage g's barrier is in its first phase)
          constexpr int kRowBytes = kPairs * 128;
          unsigned int bad = 0, cnt = 0;
          for (int g = 0; g < npre; ++g) {
            const int s = g % kStages, i = lo + g;
            mbar_wait(full_bar(s), 0);
            const int pi = i / kTilesPerPage - pg0;
            const int page = pi < kMaxPages ? pages[pi] : ld_relaxed_gpu_s32(pt + i / kTilesPerPage);
            const int64_t row0 = static_cast<int64_t>(page) * 128 + (i % kTilesPerPage) * kTileKeys;
            for (int c = 0; c < kPairs; ++c) {
#pragma unroll
              for (int kv = 0; kv < 2; ++kv) {
                const uint8_t* const sb = sm + (kv ? L::kSmV : L::kSmK) + s * kTileBytes + c * kTileKeys * 128;
                const uint8_t* const gb = (kv ? qa.v_store : qa.k_store) + row0 * kRowBytes + c * 128;
                uint32_t ss = 0, sx = 0, gs = 0, gx = 0;
                for (int w = lane; w < kTileKeys * 8; w += 32) {  // 16-byte words: row w / 8, chunk w % 8
                  const uint4 a = *reinterpret_cast<const uint4*>(sb + 16 * w);
                  const uint4 v = ld_relaxed_gpu_v4(gb + static_cast<int64_t>(w >> 3) * kRowBytes + 16 * (w & 7));
                  ss += a.x + a.y + a.z + a.w;
                  sx ^= a.x ^ (a.y * 3u) ^ (a.z * 5u) ^ (a.w * 7u);
                  gs += v.x + v.y + v.z + v.w;
                  gx ^= v.x ^ (v.y * 3u) ^ (v.z * 5u) ^ (v.w * 7u);
                  ++cnt;
                }
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) {
                  ss += __shfl_xor_sync(0xffffffffu, ss, o);
                  gs += __shfl_xor_sync(0xffffffffu, gs, o);
                  sx ^= __shfl_xor_sync(0xffffffffu, sx, o);
                  gx ^= __shfl_xor_sync(0xffffffffu, gx, o);
                }
                bad += (lane == 0 && (ss != gs || sx != gx)) ? 1u : 0u;
              }
            }
          }
#pragma unroll
          for (int o = 16; o > 0; o >>= 1) cnt += __shfl_xor_sync(0xffffffffu, cnt, o);
          if (lane == 0) {
            atomicAdd(ck + 3, static_cast<unsigned long long>(cnt));
            if (bad) atomicAdd(ck + 2, static_cast<unsigned long long>(bad));
          }
        }
#endif
#ifdef MUSE_VR_FAW_PRE_PROBE
        // test builds only: the number of tiles this block issued before its wait, after the records region
        if (lane == 0) reinterpret_cast<int32_t*>(trace)[b] = npre;
#endif
        if (lane == 0) {
          asm volatile("fence.proxy.async.global;\n" ::: "memory");
          mbar_expect_tx(q_full_bar(0), kHeads * kRows * 64);
#pragma unroll
          for (int h = 0; h < kHeads; ++h)
            tma_load_3d(smem_u32(sm + L::kSmQ + h * kRows * 64), &q_map, 0, h * kQPerKv, r * kTokens, q_full_bar(0));
          for (int g = npre; g < hi - lo; ++g) issue(g);
        }
      } else if (lane == 0) {
        asm volatile("fence.proxy.async.global;\n" ::: "memory");
        int g = 0, k = 0, r = r_first, base = 0;
        for (int t = T0; t < T1; ++k) {
          const int lo = t - pre[r], hi = min(T1, pre[r + 1]) - pre[r];
          const int pg0 = lo / kTilesPerPage;
          // q rows of request r -> Q buffer k & 1 (free once the QK MMAs of piece k - 2 finished); kQk: the softmax
          // warps write them (one piece per block)
          if constexpr (kQk == 0) {
            constexpr int kQB = L::kQBufs;
            if ((kMode == 0 || kMode == 2 || kMode == 4 || kMode == 6) && k >= kQB)
              mbar_wait(q_empty_bar(k % kQB), ((k / kQB) - 1) & 1);
            mbar_expect_tx(q_full_bar(k % kQB), kHeads * kRows * 64);
#pragma unroll
            for (int h = 0; h < kHeads; ++h)
              tma_load_3d(smem_u32(sm + L::kSmQ + (k % kQB) * L::kQBuf + h * kRows * 64), &q_map, 0, h * kQPerKv,
                          r * kTokens, q_full_bar(k % kQB));
          }
          const int32_t* const pt = p.page_table + static_cast<int64_t>(r) * p.page_table_stride;
          for (int i = lo; i < hi; ++i, ++g) {
            if constexpr (kMode == 9) continue;  // tensor-core rate probe: no tile loads
            const int s = g % kStages;
            if (g >= kStages) mbar_wait(empty_bar(s), ((g / kStages) - 1) & 1);
            const int pi = base + i / kTilesPerPage - pg0;
            const int page = pi < kMaxPages ? pages[pi] : pt[i / kTilesPerPage];
            const int row0 = page * 128 + (i % kTilesPerPage) * kTileKeys;
            if constexpr (kPf > 1) {
              // L2 prefetch of the tile kPf ahead in this piece (deeper memory-level parallelism than the ring)
              const int ip = i + kPf;
              if (ip < hi) {
                const int pip = base + ip / kTilesPerPage - pg0;
                const int pagep = pip < kMaxPages ? pages[pip] : pt[ip / kTilesPerPage];
                const int rowp = pagep * 128 + (ip % kTilesPerPage) * kTileKeys;
#pragma unroll
                for (int c = 0; c < kPairs; ++c) {
                  tma_prefetch_2d(&k_map, c * 128, rowp);
                  tma_prefetch_2d(&v_map, c * 128, rowp);
                }
              }
            }
            mbar_expect_tx(full_bar(s), 2 * kTileBytes);
#pragma unroll
            for (int c = 0; c < kPairs; ++c) {
              tma_load_2d(smem_u32(sm + L::kSmK + s * kTileBytes + c * kTileKeys * 128), &k_map, c * 128, row0,
                          full_bar(s));
              tma_load_2d(smem_u32(sm + L::kSmV + s * kTileBytes + c * kTileKeys * 128), &v_map, c * 128, row0,
                          full_bar(s));
            }
            stamp(0, g);
          }
          base += (hi - 1) / kTilesPerPage + 1 - pg0;
          t = pre[r] + hi;
          ++r;
        }
      }
    } else if (warp == kSmWarps + 1) {
      // ------------------------------------------------------------------ MMA issue
      if constexpr (kPreW) {
        if constexpr (kUsePDL) {
          asm volatile("griddepcontrol.wait;\n" ::: "memory");
          asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
        }
#ifdef MUSE_VR_FAW_PRE_CHECK
        if (lane < R) {  // the block map's and the prefix pass's lengths (this warp read them before its wait)
          const int s = ld_relaxed_gpu_s32(p.seq_lens + lane);
          const int nn = (max(s, 1) + kTileKeys - 1) / kTileKeys;
          atomicAdd(ck + 1, 2ull);
          const unsigned long long bad = (s != ck_seq || nn != ck_n ? 1ull : 0ull) + (nn != ck_pre ? 1ull : 0ull);
          if (bad) atomicAdd(ck, bad);
        }
#endif
        // block 0 stores each request's (first block, block count) for the merge after its wait
        if (b == 0 && lane < R) {
          qa.lb_map[2 * lane] = lb_prel;
          qa.lb_map[2 * lane + 1] = lb_count;
        }
      }
      if (kMode == 1 && lane == 0) {
        for (int g = 0; g < ntiles; ++g) {  // load-only probe: release every stage as soon as it landed
          mbar_wait(full_bar(g % kStages), (g / kStages) & 1);
          if (g + kStages < ntiles) mbar_arrive(empty_bar(g % kStages));
        }
      } else if ((kMode == 3 || kMode == 9) && lane == 0) {
        // development probe: MMAs only (QK then PV of every tile, back to back, no softmax hand-offs); mode 9: no data
        // waits after the first stages (pure tensor-core issue/execute rate; kPf 1 = QK only, 2 = PV only)
        constexpr uint32_t id_qk = idesc_f8(64, kTileKeys, 0), id_pv = idesc_f8(64, 64, 1);
        const uint32_t sp = smem_u32(sm + L::kSmP), sq = smem_u32(sm + L::kSmQ);
        for (int g = 0; g < ntiles; ++g) {
          const int s = g % kStages, bb = g & 1;
          if (kMode == 3) mbar_wait(full_bar(s), (g / kStages) & 1);
          tc_fence_after();
          const uint32_t sk = smem_u32(sm + L::kSmK + s * kTileBytes), sv = smem_u32(sm + L::kSmV + s * kTileBytes);
#pragma unroll
          for (int c = 0; c < kPairs; ++c)
#pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
#pragma unroll
              for (int kk = 0; kk < 2; ++kk)
                if (kMode == 3 || kPf != 2)
                mma_f8(tmem + ((16u * hh) << 16) + 128u * bb + 64u * c,
                       sdesc(sq + (2 * c + hh) * kRows * 64 + 32 * kk, 512, kSw64),
                       sdesc(sk + c * kTileKeys * 128 + 64 * hh + 32 * kk, 1024, kSw128), id_qk, kk);
#pragma unroll
              for (int kk = 0; kk < kTileKeys / 32; ++kk)
                if (kMode == 3 || kPf != 1)
                mma_f8(tmem + ((16u * hh) << 16) + 256u + 128u * bb + 64u * c,
                       sdesc(sp + (bb * kHeads + 2 * c + hh) * kRows * L::kPRowBytes + 32 * kk, kPairs == 2 ? 512 : 1024,
                             kPairs == 2 ? kSw64 : kSw128),
                       sdesc(sv + c * kTileKeys * 128 + 64 * hh + 4096 * kk, 1024, kSw128), id_pv, kk);
            }
          if (kMode == 3 && g + kStages < ntiles) mma_commit(empty_bar(s));
        }
        mma_commit(o_full_bar(0));
        mbar_wait(o_full_bar(0), 0);
      } else if (kMode != 1 && kMode != 3 && kMode != 9) {
        // QK issue (whole warp, warp-uniform operands, one elected lane issues). Descriptors: uniform bases plus
        // compile-time offsets (the start-address field is linear; shared-memory offsets stay inside it).
        constexpr uint32_t id_qk = idesc_f8(64, kTileKeys, 0);
        const uint64_t dq0 = sdesc(smem_u32(sm + L::kSmQ), 512, kSw64);
        const uint64_t dk0 = sdesc(smem_u32(sm + L::kSmK), 1024, kSw128);
        int r_q = r_first, k_q = 0, end_q = __shfl_sync(0xffffffffu, min(T1, pre[r_first + 1]), 0);
        const int T0u = __shfl_sync(0xffffffffu, T0, 0);
        const auto qk_tile = [&](int gq) {
          const int t = T0u + gq;
          if (t >= end_q) {  // first tile of the next piece: release the previous Q buffer, wait for this one
            mma_commit_e(q_empty_bar(k_q % L::kQBufs));
            ++r_q;
            ++k_q;
            end_q = __shfl_sync(0xffffffffu, min(T1, pre[r_q + 1]), 0);
            mbar_wait(q_full_bar(k_q % L::kQBufs), (k_q / L::kQBufs) & 1);
            __syncwarp();
            tc_fence_after();
          }
          const int s = gq % kStages, bq = gq & 1;
          if (lane == 0) stamp(10, gq);
          mbar_wait(full_bar(s), (gq / kStages) & 1);
          if constexpr (kQk > 0)  // the new tokens' rows are written into this stage first
            if (gq >= pn0 && gq < pn0 + npn) mbar_wait(patch_bar, (gq - pn0) & 1);
          __syncwarp();
          if (lane == 0) stamp(1, gq);
          tc_fence_after();
          const uint64_t dq = dq0 + static_cast<uint64_t>(((k_q % L::kQBufs) * L::kQBuf) >> 4);
          const uint64_t dk = dk0 + static_cast<uint64_t>((s * kTileBytes) >> 4);
#pragma unroll
          for (int c = 0; c < kPairs; ++c)
#pragma unroll
            for (int hh = 0; hh < 2; ++hh)
#pragma unroll
              for (int kk = 0; kk < 2; ++kk)
                mma_f8_e(tmem + ((16u * hh) << 16) + 128u * bq + 64u * c,
                         dq + static_cast<uint64_t>(((2 * c + hh) * kRows * 64 + 32 * kk) >> 4),
                         dk + static_cast<uint64_t>((c * kTileKeys * 128 + 64 * hh + 32 * kk) >> 4), id_qk, kk);
          mma_commit_e(s_full_bar(bq));
          if (lane == 0) stamp(2, gq);
        };
        mbar_wait(q_full_bar(0), 0);
        __syncwarp();
        tc_fence_after();
        qk_tile(0);
        if (ntiles > 1) qk_tile(1);
        for (int g = 0; g + 2 < ntiles; ++g) {
          mbar_wait(s_free_bar(g & 1), (g >> 1) & 1);
          __syncwarp();
          tc_fence_after();
          qk_tile(g + 2);
        }
      }
    } else if (warp == kSmWarps + 2) {
      // ------------------------------------------------------------------ PV issue (warp-uniform, elected lane)
      if (kMode != 1 && kMode != 3 && kMode != 9) {
        constexpr uint32_t id_pv = idesc_f8(64, 64, 1);
        constexpr uint32_t kPSbo = kPairs == 2 ? 512 : 1024, kPSw = kPairs == 2 ? kSw64 : kSw128;
        const uint64_t dp0 = sdesc(smem_u32(sm + L::kSmP), kPSbo, kPSw);
        const uint64_t dv0 = sdesc(smem_u32(sm + L::kSmV), 1024, kSw128);
        for (int g = 0; g < ntiles; ++g) {
          const int s = g % kStages, bb = g & 1;
          if (lane == 0) stamp(11, g);
          mbar_wait(p_full_bar(bb), (g >> 1) & 1);
          __syncwarp();
          if (lane == 0) stamp(6, g);
          tc_fence_after();
          const uint64_t dp = dp0 + static_cast<uint64_t>((bb * kHeads * kRows * L::kPRowBytes) >> 4);
          const uint64_t dv = dv0 + static_cast<uint64_t>((s * kTileBytes) >> 4);
#pragma unroll
          for (int c = 0; c < kPairs; ++c)
#pragma unroll
            for (int hh = 0; hh < 2; ++hh)
#pragma unroll
              for (int kk = 0; kk < kTileKeys / 32; ++kk)
                mma_f8_e(tmem + ((16u * hh) << 16) + 256u + 128u * bb + 64u * c,
                         dp + static_cast<uint64_t>(((2 * c + hh) * kRows * L::kPRowBytes + 32 * kk) >> 4),
                         dv + static_cast<uint64_t>((c * kTileKeys * 128 + 64 * hh + 4096 * kk) >> 4), id_pv, kk);
          mma_commit_e(o_full_bar(bb));
          if (lane == 0) stamp(7, g);
          if (g + kStages < ntiles) mma_commit_e(empty_bar(s));
        }
      }
    } else if (kMode == 0 || kMode == 2 || kMode == 4 || kMode == 6) {
      // ------------------------------------------------------------------ softmax (warps 0 .. kSmWarps-1)
      if constexpr (kPreW && kUsePDL) {
        asm volatile("griddepcontrol.wait;\n" ::: "memory");
        asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
      }
      // Two threads per query row: warp w -> half = w / (kSmWarps / 2) (keys [kKpt half, +kKpt) of every tile, output
      // dims [32 half, +32)); wi = w % (kSmWarps / 2) -> lane group q4 = wi % 4 (tensor-memory lanes 32 q4 ..),
      // KV-head pair c = wi / 4; lane i -> head 2c + i / 16, row 16 q4 + i % 16. The two threads of a row (warps wi and
      // wi + kSmWarps / 2) exchange tile maxima and row sums through shared memory and named barrier 1 + wi.
      constexpr int kHalfWarps = kSmWarps / kTpr;
      const int half = warp / kHalfWarps, wi = warp % kHalfWarps;  // kTpr 1: half 0, wi = warp
      const int q4 = wi & 3, c = wi >> 2;
      const int hh = lane >> 4, row = 16 * q4 + (lane & 15);
      const int head = 2 * c + hh;  // KV head within the rank
      const int tok = row >> 3;
      constexpr int kDims = 64 / kTpr;  // output dims per thread
      const uint32_t lane_base = (32u * q4) << 16;
      const uint32_t s_col = 64u * c + kKpt * half;
      const uint32_t o_col = 64u * c + kDims * half;
      float* const red = reinterpret_cast<float*>(sm + L::kSmRed);
      const int rec_row = head * kRows + row;  // row of the block's record (0 .. kHeads * 64 - 1)
      constexpr int kRecRows = kHeads * kRows;
      const uint32_t xbar = 1 + wi;
      // kQk: new-token K / V rows of local tile jl -> its stage (after the stage's load completed) and the pools; then
      // one arrival per warp on the patch barrier (the QK MMA warp waits for it before the tile's QK)
      constexpr int kSmThreads = 32 * kSmWarps;
      uint8_t* const snew = sm + L::kSmNew;
      const auto patch = [&](int jl) {
        const int sl = p.seq_lens[r_first], tnew = sl - kTokens;
        const int kp = (T0 - pre[r_first] + jl) * kTileKeys;  // first key of the tile
        const int kb = max(tnew, kp), nk = min(sl, kp + kTileKeys) - kb;
        const int st = jl % kStages;
        mbar_wait(full_bar(st), (jl / kStages) & 1);
        constexpr int kCh = L::kNewRowBytes / 16;  // 16-byte chunks of a token row (kCh / 8 KV-head pairs)
        for (int i = tid; i < 2 * nk * kCh; i += kSmThreads) {
          const int kv = i / (nk * kCh), rem = i - kv * nk * kCh, key = kb + rem / kCh, cq = rem % kCh;
          const int row = key - kp, tt = key - tnew;
          const uint4 v = *reinterpret_cast<const uint4*>(snew + (kv * kTokens + tt) * L::kNewRowBytes + 16 * cq);
          uint8_t* const stage = sm + (kv ? L::kSmV : L::kSmK) + st * kTileBytes + (cq >> 3) * kTileKeys * 128;
          *reinterpret_cast<uint4*>(stage + row * 128 + (((cq & 7) ^ (row & 7)) << 4)) = v;
          const int64_t loc = qa.cache_loc[static_cast<int64_t>(r_first) * kTokens + tt];
          uint8_t* const pool = kv ? qa.v_store : qa.k_store;
          if (loc != 0) *reinterpret_cast<uint4*>(pool + loc * L::kNewRowBytes + 16 * cq) = v;
        }
        fence_proxy_async();
        __syncwarp();
        if (lane == 0) mbar_arrive(patch_bar);
      };
      if constexpr (kQk > 0) {
        static_assert(kStages >= 3, "the new-token rows of tile g + 2 are written while tile g is processed");
        // q rows from the QKV output: row Rw (0 .. kHeads * 64 - 1: KV head Rw / 64, row Rw % 64 = token * 8 + head in
        // group), 64 / E threads per row (consecutive lanes), E columns each; all loads first, then the arithmetic
        constexpr int E = kQk, kLanes = 64 / E, kRowsPass = kSmThreads / kLanes, kPasses = kHeads * kRows / kRowsPass;
        static_assert(kHeads * kRows % kRowsPass == 0, "q rows per pass");
        const int li = lane % kLanes;
        {
          uint32_t w[kPasses][E / 2];
#pragma unroll
          for (int ps = 0; ps < kPasses; ++ps) {
            const int Rw = ps * kRowsPass + tid / kLanes, hk = Rw >> 6, rq = Rw & 63;
            ld_bf16_raw<E>(qa.qkv + (static_cast<int64_t>(r_first) * kTokens + (rq >> 3)) * qa.qkv_token_stride +
                               (hk * kQPerKv + (rq & 7)) * kHeadDim + E * li,
                           w[ps]);
          }
          uint8_t* const sq = sm + L::kSmQ;
#pragma unroll
          for (int ps = 0; ps < kPasses; ++ps) {
            const int Rw = ps * kRowsPass + tid / kLanes, hk = Rw >> 6, rq = Rw & 63;
            uint16_t bq[E / 2];
            qk_finish<E, kQkUnfused>(w[ps], qa.qk_eps, false, bq);
            const int cb = E * li;  // byte column of the 64-byte row (64-byte swizzle: chunk c of row r at c ^ (r/2 % 4))
            st_bytes<E>(sq + hk * kRows * 64 + rq * 64 + ((((cb >> 4) ^ ((rq >> 1) & 3)) << 4) | (cb & 15)), bq);
          }
          fence_proxy_async();
          __syncwarp();
          if (lane == 0) mbar_arrive(q_full_bar(0));
        }
        if (npn > 0) {  // uniform over the block
          // K and V rows of the 8 new tokens, every KV head -> [K/V][token][head * 64 + column] (row J: K/V J / (8 kHeads),
          // head (J / 8) % kHeads, token J % 8)
          constexpr int kNewRows = 2 * kTokens * kHeads, kPassesN = kNewRows / kRowsPass;
          static_assert(kNewRows % kRowsPass == 0, "new-token rows per pass");
          uint32_t w[kPassesN][E / 2];
#pragma unroll
          for (int ps = 0; ps < kPassesN; ++ps) {
            const int J = ps * kRowsPass + tid / kLanes, kv = J / (kTokens * kHeads), hk = (J / kTokens) % kHeads,
                      tt = J % kTokens;
            ld_bf16_raw<E>(qa.qkv + (static_cast<int64_t>(r_first) * kTokens + tt) * qa.qkv_token_stride +
                               (p.num_q_heads + kv * kHeads + hk) * kHeadDim + E * li,
                           w[ps]);
          }
#pragma unroll
          for (int ps = 0; ps < kPassesN; ++ps) {
            const int J = ps * kRowsPass + tid / kLanes, kv = J / (kTokens * kHeads), hk = (J / kTokens) % kHeads,
                      tt = J % kTokens;
            uint16_t bn[E / 2];
            qk_finish<E, kQkUnfused>(w[ps], qa.qk_eps, kv == 1, bn);
            st_bytes<E>(snew + (kv * kTokens + tt) * L::kNewRowBytes + hk * kHeadDim + E * li, bn);
          }
          asm volatile("bar.sync 15, %0;\n" ::"r"(kSmThreads) : "memory");  // the softmax warps only
          // the block's first two tiles: their QK MMAs are issued right after the q rows
          for (int jl = pn0; jl < pn0 + npn && jl < 2; ++jl) patch(jl);
        }
      }
      int g = 0;
      int r = r_first;
      for (int t = T0; t < T1; ++r) {
        const int lo = t - pre[r], hi = min(T1, pre[r + 1]) - pre[r];
        const int n_r = pre[r + 1] - pre[r];
        const int seq_len = p.seq_lens[r];
        const int pos = seq_len - kTokens + tok;
        float2 o2[kDims / 2];
#pragma unroll
        for (int d = 0; d < kDims / 2; ++d) o2[d] = make_float2(0.0f, 0.0f);
        float m = -INFINITY, l = 0.0f, ar_prev = 0.0f, at_prev = 0.0f;
        // rescale-and-add of an output tile into o (packed fp32 pairs: the same two roundings as fmaf(o, ar, v * at))
        const auto combine = [&](int gg) {
          const int bp = gg & 1;
          mbar_wait(o_full_bar(bp), (gg >> 1) & 1);
          if (tid == 0) stamp(8, gg);
          tc_fence_after();
          const float2 a2 = make_float2(ar_prev, ar_prev), t2 = make_float2(at_prev, at_prev);
#pragma unroll
          for (int d0 = 0; d0 < kDims; d0 += 32) {
            float v[32];
            tmem_ld32(tmem + lane_base + 256u + 128u * bp + o_col + d0, v);
            tmem_wait_ld();
#pragma unroll
            for (int d = 0; d < 16; ++d)
              o2[d0 / 2 + d] = fma2(o2[d0 / 2 + d], a2, mul2(make_float2(v[2 * d], v[2 * d + 1]), t2));
          }
          tc_fence_before();
          if (tid == 0) stamp(9, gg);
        };
        for (int i = lo; i < hi; ++i, ++g) {
          const int bb = g & 1;
          const int key0 = i * kTileKeys + kKpt * half;  // this thread's first key of the tile
          mbar_wait(s_full_bar(bb), (g >> 1) & 1);
          if (tid == 0) stamp(3, g);

          tc_fence_after();
          float x[kKpt];
#pragma unroll
          for (int j = 0; j < kKpt; j += 32)
            tmem_ld32(tmem + lane_base + 128u * bb + s_col + j, *reinterpret_cast<float(*)[32]>(x + j));
          tmem_wait_ld();
          tc_fence_before();
          if (tid == 0) stamp(4, g);
          __syncwarp();
          if (lane == 0) mbar_arrive(s_free_bar(bb));  // score buffer bb may take the QK of tile g + 2
          if constexpr (kQk > 0) {  // the new tokens' rows of tile g + 2, before its QK
            if (g + 2 >= pn0 && g + 2 < pn0 + npn) patch(g + 2);
          }
          // tile edge: keys after the row's position do not take part. The test uses the smallest position of the request
          // (token 0), so it is warp-uniform: interior tiles branch around the masking instead of predicating it per key.
          if (key0 + kKpt - 1 > seq_len - kTokens) {
#pragma unroll
            for (int j = 0; j < kKpt; ++j)
              if (key0 + j > pos) x[j] = -INFINITY;
          }
          if constexpr (kMode == 2) {  // development probe: no softmax arithmetic, P = 0
#pragma unroll
            for (int j = 0; j < kKpt; ++j) x[j] = 0.0f;
          }
          // maximum of the raw scores (the scale is positive, so max(s) * scale == max(s * scale))
          float mx[kKpt / 3 + 1];
#pragma unroll
          for (int j = 0; j < kKpt / 3; ++j) mx[j] = max3(x[3 * j], x[3 * j + 1], x[3 * j + 2]);
          float smax = kKpt % 3 == 2 ? fmaxf(x[kKpt - 2], x[kKpt - 1]) : x[kKpt - 1];
#pragma unroll
          for (int j = 0; j < kKpt / 3; j += 2) smax = j + 1 < kKpt / 3 ? max3(smax, mx[j], mx[j + 1]) : fmaxf(smax, mx[j]);
          if constexpr (kTpr == 2) {
            red[(bb * kRecRows + rec_row) * 2 + half] = smax;
            asm volatile("bar.sync %0, %1;\n" ::"r"(xbar), "r"(64) : "memory");
            smax = fmaxf(red[(bb * kRecRows + rec_row) * 2], red[(bb * kRecRows + rec_row) * 2 + 1]);
          }
          // the tile's own maximum: the probabilities of the tile are quantized against it
          const float mtile = smax == -INFINITY ? -INFINITY : smax * p.scale_log2;
          const float mu = mtile == -INFINITY ? 0.0f : mtile;
          const float mnew = fmaxf(m, mtile);
          const float ar = m == -INFINITY ? 0.0f : fast_exp2(m - mnew);
          const float at = mtile == -INFINITY ? 0.0f : fast_exp2(mtile - mnew);
          // 448 * 2^y as 2^(y + log2 448): probabilities come out pre-scaled for the e4m3 conversion, and the row sums are
          // kept in the same (x 448) units; the output scale divides by them (bmm2_scale / l448)
          const float nmu448 = kLog2PScale - mu;
          const float2 sc = make_float2(p.scale_log2, p.scale_log2), nmu = make_float2(nmu448, nmu448);
          float2 ls[2] = {make_float2(0.0f, 0.0f), make_float2(0.0f, 0.0f)};
#pragma unroll
          for (int j = 0; j < kKpt && kMode != 2; j += 2) {
            const float2 y = fma2(make_float2(x[j], x[j + 1]), sc, nmu);
            // kPoly of every 8 keys take 2^y from the polynomial (exp2_poly2), the rest from ex2.approx
            const float2 pj = (j & 7) < kPoly ? exp2_poly2(y) : make_float2(fast_exp2(y.x), fast_exp2(y.y));
            ls[(j >> 1) & 1] = add2(ls[(j >> 1) & 1], pj);
            x[j] = pj.x;
            x[j + 1] = pj.y;
          }
          // this thread's part of the tile row sum (its keys in four interleaved partial sums, (0 + 2) + (1 + 3))
          const float lpart = (ls[0].x + ls[1].x) + (ls[0].y + ls[1].y);
          m = mnew;
          // probabilities -> P buffer bb. kPairs 2: rows of 64 B in the 64-byte swizzle (chunk q of row r at
          // q ^ (r / 2 % 4)); kPairs 1: rows of 128 B in the 128-byte swizzle (chunk q of row r at q ^ (r % 8)).
          // Buffer bb was last read by the MMAs of tile g - 2, finished before the output wait of the previous tile.
          uint8_t* const prow = sm + L::kSmP + (bb * kHeads + head) * kRows * L::kPRowBytes + row * L::kPRowBytes;
#pragma unroll
          for (int cc = 0; cc < kKpt / 16; ++cc) {
            uint4 w;
            w.x = e4m3x4(x[16 * cc + 0], x[16 * cc + 1], x[16 * cc + 2], x[16 * cc + 3]);
            w.y = e4m3x4(x[16 * cc + 4], x[16 * cc + 5], x[16 * cc + 6], x[16 * cc + 7]);
            w.z = e4m3x4(x[16 * cc + 8], x[16 * cc + 9], x[16 * cc + 10], x[16 * cc + 11]);
            w.w = e4m3x4(x[16 * cc + 12], x[16 * cc + 13], x[16 * cc + 14], x[16 * cc + 15]);
            const int q = (kKpt / 16) * half + cc;
            const int chunk = kPairs == 2 ? (q ^ ((row >> 1) & 3)) : (q ^ (row & 7));
            *reinterpret_cast<uint4*>(prow + (chunk << 4)) = w;
          }
          if constexpr (kMode != 6) fence_proxy_async();  // mode 6: timing probe only (P visibility not ordered)
          tc_fence_before();
          if (tid == 0) stamp(5, g);
          __syncwarp();
          if (lane == 0) mbar_arrive(p_full_bar(bb));
          // this thread's half of the row sum (the two halves are added once, at the end of the piece)
          if (i > lo) combine(g - 1);
          l = l * ar + lpart * at;
          ar_prev = ar;
          at_prev = at;
        }
        combine(g - 1);
        if constexpr (kTpr == 2) {
          red[(2 * kRecRows + rec_row) * 2 + half] = l;
          asm volatile("bar.sync %0, %1;\n" ::"r"(xbar), "r"(64) : "memory");
          l = red[(2 * kRecRows + rec_row) * 2] + red[(2 * kRecRows + rec_row) * 2 + 1];
          asm volatile("bar.sync %0, %1;\n" ::"r"(xbar), "r"(64) : "memory");  // red reused by the next piece
        }
        const int qh = head * kQPerKv + (row & 7);  // query head within the rank
        __nv_bfloat16* const orow =
            p.out + ((static_cast<int64_t>(r) * kTokens + tok) * p.num_q_heads + qh) * kHeadDim + kDims * half;
        const auto store_out = [&](const float2 (&acc)[kDims / 2], float f) {
#pragma unroll
          for (int d = 0; d < kDims / 2; d += 4) {
            uint4 w;
            __nv_bfloat162 h0 = __floats2bfloat162_rn(acc[d].x * f, acc[d].y * f);
            __nv_bfloat162 h1 = __floats2bfloat162_rn(acc[d + 1].x * f, acc[d + 1].y * f);
            __nv_bfloat162 h2 = __floats2bfloat162_rn(acc[d + 2].x * f, acc[d + 2].y * f);
            __nv_bfloat162 h3 = __floats2bfloat162_rn(acc[d + 3].x * f, acc[d + 3].y * f);
            w.x = *reinterpret_cast<uint32_t*>(&h0);
            w.y = *reinterpret_cast<uint32_t*>(&h1);
            w.z = *reinterpret_cast<uint32_t*>(&h2);
            w.w = *reinterpret_cast<uint32_t*>(&h3);
            *reinterpret_cast<uint4*>(orow + 2 * d) = w;
          }
        };
        if (lo == 0 && hi == n_r) {
          // the whole request in this block: normalize and store
          store_out(o2, l > 0.0f ? p.out_scale / l : 0.0f);
        } else {
          // partial record -> slot (block, 0) when the request started in an earlier block, else (block, 1)
          const int slot = (pre[r] < T0) ? 0 : 1;
          float* const rec = p.partial + ((static_cast<int64_t>(b) * 2 + slot) * kRecRows + rec_row) * kRec;
#pragma unroll
          for (int d = 0; d < kDims / 2; d += 2)
            *reinterpret_cast<float4*>(rec + kDims * half + 2 * d) = make_float4(o2[d].x, o2[d].y, o2[d + 1].x, o2[d + 1].y);
          if (half == 0) *reinterpret_cast<float2*>(rec + kHeadDim) = make_float2(m, l);
          // merged by full_attn_wide_merge_kernel (launched after this kernel)
        }
        t = pre[r] + hi;
      }
    }
  }
  tc_fence_before();
  __syncthreads();
  if (tid == 0) stamp(14, 0);
  if constexpr (kFd == 0)
    if constexpr (kUsePDL) asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
  if (warp == 0) {
    tc_fence_after();
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n" ::"r"(tmem), "r"(kTmemCols) : "memory");
  }
  if constexpr (kFd == 1 || kFd == 2) {
    // ------------------------------------------------------------------ in-kernel merge (kFd)
    constexpr int kNW = L::kThreads / 32;   // warps of the block
    constexpr int kRR = kHeads * kRows;     // rows of a request (and of a record)
    int32_t* const fdw = reinterpret_cast<int32_t*>(sm + L::kSmRed);  // [0]: arrival / wait outcome
    const int rq = lbinfo[0], Br = lbinfo[3], b0r = lbinfo[4];
    unsigned long long* const word = fa.words + rq;
    float* const scr = reinterpret_cast<float*>(sm + L::kSmK);  // group partials (the K ring, free after the main loop)
    if (Br <= 1) {
      if constexpr (kUsePDL) asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
      if constexpr (kFdGm == 2) {
        if (Br == 1) {  // the request lies in this block: its output is written; norm + gate in place
          for (int row = warp; row < kRR; row += kNW) {
            __nv_bfloat162* const o = fd_out(p, rq, row, lane);
            *o = fd_norm_gate(*o, fa, rq, row, lane);
          }
        }
      }
    } else {
      // Arrival, every path (after the block barrier above: every record store of the block precedes it). The word is
      // (epoch << 32) | count: the max lifts a word of an earlier epoch to (E, 0) and leaves (E, c) as it is, then the
      // release increment counts this block; the arrival that returns (E, B_r - 1) is the request's last. No reset.
      // Outcome fdw[0]: 0 = nothing to merge (form 1, not the last), 1 = merge, 2 = poison this block's rows: the add
      // returned a count >= B_r (the step word did not advance since an earlier call on this slot; error bit 30) or the
      // distributed wait expired (bit 30 clear). Both set the sticky error word (bit 31 | layer slot << 16 | request bit).
      if (tid == 0) {
        const unsigned long long tag = static_cast<unsigned long long>(static_cast<uint32_t>(*fa.epoch)) << 32;
        uint32_t go = 1u;
        if (kFd == 1 || static_cast<uint32_t>(b) != fa.skip_block) {
          atom_max_relaxed_gpu_u64(word, tag);
          const uint32_t cnt = static_cast<uint32_t>(atom_add_acq_rel_gpu_u64(word, 1ull));
          if (cnt >= static_cast<uint32_t>(Br)) {
            atomicOr(fa.err, 0xC0000000u | (fa.err_tag & 0xffu) << 16 | (1u << rq));
            go = 2u;
          } else if constexpr (kFd == 1) {
            go = cnt == static_cast<uint32_t>(Br - 1) ? 1u : 0u;
          }
        }
        if constexpr (kFd == 2) {
          if (go == 1u) {
            // one elected thread waits for (E, B_r): acquire loads with backoff, bounded in SM clock cycles
            const unsigned long long want = tag | static_cast<unsigned long long>(Br);
            const long long c0 = clock64();
            while (ld_acquire_gpu_u64(word) != want) {
              if (static_cast<unsigned long long>(clock64() - c0) > fa.max_wait_cycles) {
                atomicOr(fa.err, 0x80000000u | (fa.err_tag & 0xffu) << 16 | (1u << rq));
                go = 2u;
                break;
              }
              __nanosleep(kFdSleepNs);
            }
          }
        }
        fdw[0] = static_cast<int32_t>(go);
      }
      __syncthreads();
      const int go = fdw[0];
      const int s = b - b0r, rlo = s * kRR / Br, rhi = (s + 1) * kRR / Br;  // this block's rows (fixed by its index)
      if (go == 2) {  // NaN into this block's rows (the sticky error word is set), then exit
        const __nv_bfloat162 nan2 = __floats2bfloat162_rn(__int_as_float(0x7fc00000), __int_as_float(0x7fc00000));
        for (int row = rlo + warp; row < rhi; row += kNW) *fd_out(p, rq, row, lane) = nan2;
        if constexpr (kUsePDL) asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
      } else if constexpr (kFd == 1) {
        // the last arriver triggers at its merge start, the others at their exit (right after the arrival)
        if constexpr (kUsePDL) asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
        if (go == 1) fd_merge_rows<kRR, kNW, kFdGm>(p, fa, scr, rq, Br, b0r, 0, kRR, warp, lane);
      } else {
        fd_merge_rows<kRR, kNW, kFdGm>(p, fa, scr, rq, Br, b0r, rlo, rhi, warp, lane);
        __syncthreads();
        // the trigger after the block's own merge share
        if constexpr (kUsePDL) asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
      }
    }
  }
#ifdef MUSE_VR_FAW_PRE_CHECK
  if constexpr (kPreW) {
    // a block without tiles decided so from lengths read before any wait: it waits here (this build only) and checks them
    if (ntiles <= 0 && warp == kSmWarps + 1) {
      if constexpr (kUsePDL) asm volatile("griddepcontrol.wait;\n" ::: "memory");
      if (lane < R) {
        const int s = ld_relaxed_gpu_s32(p.seq_lens + lane);
        const int nn = (max(s, 1) + kTileKeys - 1) / kTileKeys;
        atomicAdd(ck + 1, 2ull);
        const unsigned long long bad = (s != ck_seq || nn != ck_n ? 1ull : 0ull) + (nn != ck_pre ? 1ull : 0ull);
        if (bad) atomicAdd(ck, bad);
      }
    }
  }
#endif
}

// ---------------------------------------------------------------------------------------------- merge kernel
// Requests whose tiles span more than one block of the main kernel: grid (kHeads * 64 / kMergeRows, R), 256 threads;
// block (x, r) merges rows [16 x, 16 x + 16) of request r, 16 threads per row (thread: 4 output dims). The work split is
// recomputed from seq_lens exactly as the main kernel computes it (same G, same W). Records of the pieces j = b0..b1
// (b0: slot 1, others: slot 0), in block order: M = max m_j, O = sum exp2(m_j - M) o_j, L = sum exp2(m_j - M) l_j;
// out = bf16(O * (out_scale / L)). Latency rounds: one for the tile range (every lane issues all its seq_lens loads first),
// one for the pieces' maxima (spread over the 16 threads of a row) together with the first 8 pieces' outputs, one per
// further 8 pieces. The arithmetic (maximum, weights, fused multiply-adds in block order) is that of the earlier merge.
constexpr int kMergeRows = 16;
constexpr int kMergeBatch = 8;
template <int kPairs, bool kUsePDL>
__global__ void __launch_bounds__(256) full_attn_wide_merge_kernel(const Params p, int grid_blocks) {
  constexpr int kHeads = 2 * kPairs, kTileKeys = 128 / kPairs, kRecRows = kHeads * kRows;
  __shared__ int32_t range[3];  // tiles of requests < r, of requests <= r, of all requests
  if constexpr (kUsePDL) asm volatile("griddepcontrol.wait;\n" ::: "memory");
  const int R = p.num_reqs, r = blockIdx.y, tid = threadIdx.x, lane = tid & 31;
  if (tid < 32) {
    int n[kMaxReqs / 32];
#pragma unroll
    for (int k = 0; k < kMaxReqs / 32; ++k) {
      const int rr = lane + 32 * k;
      n[k] = rr < R ? p.seq_lens[rr] : 0;
    }
    int before = 0, upto = 0, all = 0;
#pragma unroll
    for (int k = 0; k < kMaxReqs / 32; ++k) {
      const int rr = lane + 32 * k;
      const int t = rr < R ? (max(n[k], 1) + kTileKeys - 1) / kTileKeys : 0;
      all += t;
      if (rr < r) before += t;
      if (rr <= r) upto += t;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      before += __shfl_xor_sync(0xffffffffu, before, o);
      upto += __shfl_xor_sync(0xffffffffu, upto, o);
      all += __shfl_xor_sync(0xffffffffu, all, o);
    }
    if (lane == 0) {
      range[0] = before;
      range[1] = upto;
      range[2] = all;
    }
  }
  __syncthreads();
  const int W = range[2];
  const int G = min(grid_blocks, W);
  const int b0 = block_of(range[0], G, W), b1 = block_of(range[1] - 1, G, W);
  if (b0 == b1) return;  // the request lies in one block: written by the main kernel
  const int row = kMergeRows * blockIdx.x + (tid >> 4), part = tid & 15;  // row of the request's records, 4-dim part
  const auto rec_of = [&](int j) {
    return p.partial + ((static_cast<int64_t>(j) * 2 + (j == b0 ? 1 : 0)) * kRecRows + row) * kRec;
  };
  // first batch of outputs and row sums in flight together with the maxima
  float2 ml[kMergeBatch];
  float4 v[kMergeBatch];
#pragma unroll
  for (int u = 0; u < kMergeBatch; ++u) {
    if (b0 + u > b1) continue;
    const float* const rj = rec_of(b0 + u);
    ml[u] = *reinterpret_cast<const float2*>(rj + kHeadDim);
    v[u] = *reinterpret_cast<const float4*>(rj + 4 * part);
  }
  // maximum over the pieces: thread `part` of the row reads pieces b0 + part, b0 + part + 16, ...; 16-lane max
  float mx = -INFINITY;
  for (int j = b0 + part; j <= b1; j += 16) mx = fmaxf(mx, rec_of(j)[kHeadDim]);
#pragma unroll
  for (int o = 8; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
  // weighted sums in block order
  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float lsum = 0.0f;
  for (int j0 = b0;;) {
#pragma unroll
    for (int u = 0; u < kMergeBatch; ++u) {
      if (j0 + u > b1 || ml[u].x == -INFINITY) continue;
      const float w = fast_exp2(ml[u].x - mx);
      acc[0] += w * v[u].x;
      acc[1] += w * v[u].y;
      acc[2] += w * v[u].z;
      acc[3] += w * v[u].w;
      lsum += w * ml[u].y;
    }
    j0 += kMergeBatch;
    if (j0 > b1) break;
#pragma unroll
    for (int u = 0; u < kMergeBatch; ++u) {
      if (j0 + u > b1) continue;
      const float* const rj = rec_of(j0 + u);
      ml[u] = *reinterpret_cast<const float2*>(rj + kHeadDim);
      v[u] = *reinterpret_cast<const float4*>(rj + 4 * part);
    }
  }
  const float f = lsum > 0.0f ? p.out_scale / lsum : 0.0f;
  const int head = row >> 6, rr = row & 63, tok = rr >> 3, qh = head * kQPerKv + (rr & 7);
  __nv_bfloat16* const orow =
      p.out + ((static_cast<int64_t>(r) * kTokens + tok) * p.num_q_heads + qh) * kHeadDim + 4 * part;
  __nv_bfloat162 h0 = __floats2bfloat162_rn(acc[0] * f, acc[1] * f);
  __nv_bfloat162 h1 = __floats2bfloat162_rn(acc[2] * f, acc[3] * f);
  uint2 w;
  w.x = *reinterpret_cast<uint32_t*>(&h0);
  w.y = *reinterpret_cast<uint32_t*>(&h1);
  *reinterpret_cast<uint2*>(orow) = w;
}

// ---------------------------------------------------------------------------------------------- grouped merge (kGM)
// Every request (MUSE_VR_FULL_ATTN_WIDE_GMERGE): grid (heads * 64 * G / 8, R), 256 threads, G = merge_groups warps per row.
// Warp w: row blockIdx.x * (8 / G) + w / G of request r, group g = w % G merges the request's pieces k = g, g + G, ...
// (piece k = block b0 + k; b0's record in slot 1, the others in slot 0; a request inside one block: its own record) in
// batches of 16 against a running maximum: M_g, O_g = sum exp2(m_k - M_g) o_k, L_g likewise, in piece order; then with
// M = max M_g: O = sum exp2(M_g - M) O_g, L likewise, in group order; out = bf16(O * (out_scale / L)) (lane j: dims 2j,
// 2j + 1). kGM == 2: then the output norm and gate of the row with SGLang's norm_gate arithmetic for 16 threads per row
// (as full_verify_attn's merge epilogue). The piece range is computed from seq_lens as the main kernel computes it; with
// kLbM (the main kernel's kLb) it is read from the map block 0 of the main kernel wrote (first block, block count).
template <int kPairs, bool kUsePDL, int kGM, bool kLbM = false>
__global__ void __launch_bounds__(256) full_attn_wide_gmerge_kernel(const Params p, const GateArgs ga, int grid_blocks,
                                                                     int merge_groups) {
  constexpr int kHeads = 2 * kPairs, kTileKeys = 128 / kPairs, kRecRows = kHeads * kRows;
  __shared__ int32_t range[3];
  __shared__ float sums[8][kHeadDim + 2];
  if constexpr (kUsePDL) asm volatile("griddepcontrol.wait;\n" ::: "memory");
  const int R = p.num_reqs, r = blockIdx.y, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
#ifdef MUSE_VR_FAW_GM_TEST_NEG
  const int rp = (r + 1) % R;  // tests only (negative control): the pieces of the next request
#else
  const int rp = r;
#endif
  if (!kLbM && tid < 32) {
    int n[kMaxReqs / 32];
#pragma unroll
    for (int k = 0; k < kMaxReqs / 32; ++k) {
      const int rr = lane + 32 * k;
      n[k] = rr < R ? p.seq_lens[rr] : 0;
    }
    int before = 0, upto = 0, all = 0;
#pragma unroll
    for (int k = 0; k < kMaxReqs / 32; ++k) {
      const int rr = lane + 32 * k;
      const int t = rr < R ? (max(n[k], 1) + kTileKeys - 1) / kTileKeys : 0;
      all += t;
      if (rr < rp) before += t;
      if (rr <= rp) upto += t;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      before += __shfl_xor_sync(0xffffffffu, before, o);
      upto += __shfl_xor_sync(0xffffffffu, upto, o);
      all += __shfl_xor_sync(0xffffffffu, all, o);
    }
    if (lane == 0) {
      range[0] = before;
      range[1] = upto;
      range[2] = all;
    }
  }
  int b0, b1;
  if constexpr (kLbM) {
    b0 = ga.lb_map[2 * rp];
    b1 = b0 + ga.lb_map[2 * rp + 1] - 1;
  } else {
    __syncthreads();
    const int W = range[2];
    const int Gb = min(grid_blocks, W);
    b0 = block_of(range[0], Gb, W);
    b1 = block_of(range[1] - 1, Gb, W);
  }
  const int G = merge_groups;
  const int row = blockIdx.x * (8 / G) + warp / G, grp = warp % G;
  const int np = b1 - b0 + 1;
  const int head = row >> 6, rr = row & 63, tok = rr >> 3, qh = head * kQPerKv + (rr & 7);
  __nv_bfloat162* const orow = reinterpret_cast<__nv_bfloat162*>(
      p.out + ((static_cast<int64_t>(r) * kTokens + tok) * p.num_q_heads + qh) * kHeadDim + 2 * lane);
  const auto norm_gate_row = [&](__nv_bfloat162 v) -> __nv_bfloat162 {
    // Output norm and gate of the row (x: the bf16 attention output; lane j holds dims 2j, 2j + 1): SGLang's norm_gate
    // arithmetic (no weight) for 16 threads per row, thread i = the lane pair (2i, 2i + 1) (full_verify_attn's epilogue).
    const float2 x = __bfloat1622float2(v);
    const float a = fmaf(x.y, x.y, fmaf(x.x, x.x, 0.0f));
    const float prev = __shfl_xor_sync(0xffffffffu, a, 1);
    float ss = fmaf(x.y, x.y, fmaf(x.x, x.x, prev));  // meaningful on odd lanes
#pragma unroll
    for (int mask = 16; mask >= 2; mask >>= 1) ss = ss + __shfl_xor_sync(0xffffffffu, ss, mask);
    ss = __shfl_sync(0xffffffffu, ss, lane | 1);
    const float rn = rsqrtf(ss / static_cast<float>(kHeadDim) + ga.norm_eps);
    const float2 g = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(
        ga.gate + (static_cast<int64_t>(r) * kTokens + tok) * ga.gate_token_stride + qh * kHeadDim + 2 * lane));
    const auto sig = [](float z) { return __fdividef(1.0f, 1.0f + __expf(-z)); };
    float y0 = x.x * rn, y1 = x.y * rn;
    y0 *= sig(g.x);
    y1 *= sig(g.y);
    return __floats2bfloat162_rn(y0, y1);
  };
  if (b0 == b1) {  // the request lies in one block: the main kernel wrote its output (and the norm + gate follow here)
    if constexpr (kGM == 2) {
      if (grp == 0) *orow = norm_gate_row(*orow);
    }
    return;
  }
  const auto rec_of = [&](int k) -> const float* {
    const int j = b0 + k;
    return p.partial + ((static_cast<int64_t>(j) * 2 + (k == 0 ? 1 : 0)) * kRecRows + row) * kRec;
  };
  float mw = -INFINITY;
  float2 acc = make_float2(0.0f, 0.0f);
  float lw = 0.0f;
  constexpr int kBatch = 16;  // pieces whose loads are issued together
  for (int s0 = grp; s0 < np; s0 += G * kBatch) {
    float2 ml[kBatch], v[kBatch];
#pragma unroll
    for (int i = 0; i < kBatch; ++i) {
      const int k = s0 + G * i;
      if (k < np) {
        const float* rec = rec_of(k);
        ml[i] = *reinterpret_cast<const float2*>(rec + kHeadDim);
        v[i] = *reinterpret_cast<const float2*>(rec + 2 * lane);
      } else {
        ml[i] = make_float2(-INFINITY, 0.0f);
        v[i] = make_float2(0.0f, 0.0f);
      }
    }
    float mb = mw;
#pragma unroll
    for (int i = 0; i < kBatch; ++i) mb = fmaxf(mb, ml[i].x);
    if (mb != -INFINITY) {
      const float rs = mw == -INFINITY ? 0.0f : fast_exp2(mw - mb);
      acc.x *= rs;
      acc.y *= rs;
      lw *= rs;
#pragma unroll
      for (int i = 0; i < kBatch; ++i) {
        const float ws = ml[i].x == -INFINITY ? 0.0f : fast_exp2(ml[i].x - mb);
        acc.x += ws * v[i].x;
        acc.y += ws * v[i].y;
        lw += ws * ml[i].y;
      }
      mw = mb;
    }
  }
  if (G > 1) {
    sums[warp][2 * lane] = acc.x;
    sums[warp][2 * lane + 1] = acc.y;
    if (lane == 0) {
      sums[warp][kHeadDim] = lw;
      sums[warp][kHeadDim + 1] = mw;
    }
    __syncthreads();
    if (grp != 0) return;
    float mx = -INFINITY;
    for (int g = 0; g < G; ++g) mx = fmaxf(mx, sums[warp + g][kHeadDim + 1]);
    float o0 = 0.0f, o1 = 0.0f, lt = 0.0f;
    for (int g = 0; g < G; ++g) {
      const float m_g = sums[warp + g][kHeadDim + 1];
      const float rs = m_g == -INFINITY ? 0.0f : fast_exp2(m_g - mx);
      o0 += rs * sums[warp + g][2 * lane];
      o1 += rs * sums[warp + g][2 * lane + 1];
      lt += rs * sums[warp + g][kHeadDim];
    }
    acc = make_float2(o0, o1);
    lw = lt;
  }
  const float f = lw > 0.0f ? p.out_scale / lw : 0.0f;
  __nv_bfloat162 v = __floats2bfloat162_rn(acc.x * f, acc.y * f);
  if constexpr (kGM == 2) v = norm_gate_row(v);
  *orow = v;
}

// Low-register grouped merge (kFd == 5): full_attn_wide_gmerge_kernel's grid, rows, groups, piece order and floating-point
// operations, with fewer registers per thread (launch bound: four blocks per processor), so that more of its blocks are
// resident at once and wait for the main grid's completion. In each batch of 16 pieces, lane i (i < 16) loads only
// piece i's (m, l) pair instead of every lane loading all 16: the batch maximum is a warp maximum over those lanes and the
// running maximum (exact in any order), and piece i's weight exp2(m_i - M) and l_i, computed once by lane i with the same
// expression, are broadcast by shuffle. Every lane then accumulates the same terms in the same piece order as the grouped
// merge (zero-weight terms of out-of-range pieces included), so the output is bitwise the grouped merge's.
template <int kPairs, bool kUsePDL, int kGM, bool kLbM = false>
__global__ void __launch_bounds__(256, 4) full_attn_wide_gmerge_1w_kernel(const Params p, const GateArgs ga,
                                                                          int grid_blocks, int merge_groups) {
  constexpr int kHeads = 2 * kPairs, kTileKeys = 128 / kPairs, kRecRows = kHeads * kRows;
  __shared__ int32_t range[3];
  __shared__ float sums[8][kHeadDim + 2];
  if constexpr (kUsePDL) asm volatile("griddepcontrol.wait;\n" ::: "memory");
  const int R = p.num_reqs, r = blockIdx.y, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (!kLbM && tid < 32) {
    int n[kMaxReqs / 32];
#pragma unroll
    for (int k = 0; k < kMaxReqs / 32; ++k) {
      const int rr = lane + 32 * k;
      n[k] = rr < R ? p.seq_lens[rr] : 0;
    }
    int before = 0, upto = 0, all = 0;
#pragma unroll
    for (int k = 0; k < kMaxReqs / 32; ++k) {
      const int rr = lane + 32 * k;
      const int t = rr < R ? (max(n[k], 1) + kTileKeys - 1) / kTileKeys : 0;
      all += t;
      if (rr < r) before += t;
      if (rr <= r) upto += t;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      before += __shfl_xor_sync(0xffffffffu, before, o);
      upto += __shfl_xor_sync(0xffffffffu, upto, o);
      all += __shfl_xor_sync(0xffffffffu, all, o);
    }
    if (lane == 0) {
      range[0] = before;
      range[1] = upto;
      range[2] = all;
    }
  }
  int b0, b1;
  if constexpr (kLbM) {
    b0 = ga.lb_map[2 * r];
    b1 = b0 + ga.lb_map[2 * r + 1] - 1;
  } else {
    __syncthreads();
    const int W = range[2];
    const int Gb = min(grid_blocks, W);
    b0 = block_of(range[0], Gb, W);
    b1 = block_of(range[1] - 1, Gb, W);
  }
  const int G = merge_groups;
  const int row = blockIdx.x * (8 / G) + warp / G, grp = warp % G;
  const int np = b1 - b0 + 1;
  const int head = row >> 6, rr = row & 63, tok = rr >> 3, qh = head * kQPerKv + (rr & 7);
  __nv_bfloat162* const orow = reinterpret_cast<__nv_bfloat162*>(
      p.out + ((static_cast<int64_t>(r) * kTokens + tok) * p.num_q_heads + qh) * kHeadDim + 2 * lane);
  const auto norm_gate_row = [&](__nv_bfloat162 v) -> __nv_bfloat162 {
    // the grouped merge's output norm and gate of the row (SGLang's norm_gate arithmetic for 16 threads per row)
    const float2 x = __bfloat1622float2(v);
    const float a = fmaf(x.y, x.y, fmaf(x.x, x.x, 0.0f));
    const float prev = __shfl_xor_sync(0xffffffffu, a, 1);
    float ss = fmaf(x.y, x.y, fmaf(x.x, x.x, prev));  // meaningful on odd lanes
#pragma unroll
    for (int mask = 16; mask >= 2; mask >>= 1) ss = ss + __shfl_xor_sync(0xffffffffu, ss, mask);
    ss = __shfl_sync(0xffffffffu, ss, lane | 1);
    const float rn = rsqrtf(ss / static_cast<float>(kHeadDim) + ga.norm_eps);
    const float2 g = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(
        ga.gate + (static_cast<int64_t>(r) * kTokens + tok) * ga.gate_token_stride + qh * kHeadDim + 2 * lane));
    const auto sig = [](float z) { return __fdividef(1.0f, 1.0f + __expf(-z)); };
    float y0 = x.x * rn, y1 = x.y * rn;
    y0 *= sig(g.x);
    y1 *= sig(g.y);
    return __floats2bfloat162_rn(y0, y1);
  };
  if (b0 == b1) {  // the request lies in one block: the main kernel wrote its output (and the norm + gate follow here)
    if constexpr (kGM == 2) {
      if (grp == 0) *orow = norm_gate_row(*orow);
    }
    return;
  }
  const auto rec_of = [&](int k) -> const float* {
    const int j = b0 + k;
    return p.partial + ((static_cast<int64_t>(j) * 2 + (k == 0 ? 1 : 0)) * kRecRows + row) * kRec;
  };
  float mw = -INFINITY;
  float2 acc = make_float2(0.0f, 0.0f);
  float lw = 0.0f;
  constexpr int kBatch = 16;  // pieces whose loads are issued together
  static_assert(kBatch <= 32, "one (m, l) pair per lane");
  for (int s0 = grp; s0 < np; s0 += G * kBatch) {
    float2 mine = make_float2(-INFINITY, 0.0f);  // lane i < kBatch: piece s0 + G i's (m, l)
    if (lane < kBatch && s0 + G * lane < np) mine = *reinterpret_cast<const float2*>(rec_of(s0 + G * lane) + kHeadDim);
    float2 v[kBatch];
#pragma unroll
    for (int i = 0; i < kBatch; ++i) {
      const int k = s0 + G * i;
      v[i] = k < np ? *reinterpret_cast<const float2*>(rec_of(k) + 2 * lane) : make_float2(0.0f, 0.0f);
    }
    float mb = mine.x;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) mb = fmaxf(mb, __shfl_xor_sync(0xffffffffu, mb, o));
    mb = fmaxf(mw, mb);
    if (mb != -INFINITY) {
      const float rs = mw == -INFINITY ? 0.0f : fast_exp2(mw - mb);
      acc.x *= rs;
      acc.y *= rs;
      lw *= rs;
      const float wmine = mine.x == -INFINITY ? 0.0f : fast_exp2(mine.x - mb);
#pragma unroll
      for (int i = 0; i < kBatch; ++i) {
        const float ws = __shfl_sync(0xffffffffu, wmine, i);
        const float li = __shfl_sync(0xffffffffu, mine.y, i);
        acc.x += ws * v[i].x;
        acc.y += ws * v[i].y;
        lw += ws * li;
      }
      mw = mb;
    }
  }
  if (G > 1) {
    sums[warp][2 * lane] = acc.x;
    sums[warp][2 * lane + 1] = acc.y;
    if (lane == 0) {
      sums[warp][kHeadDim] = lw;
      sums[warp][kHeadDim + 1] = mw;
    }
    __syncthreads();
    if (grp != 0) return;
    float mx = -INFINITY;
    for (int g = 0; g < G; ++g) mx = fmaxf(mx, sums[warp + g][kHeadDim + 1]);
    float o0 = 0.0f, o1 = 0.0f, lt = 0.0f;
    for (int g = 0; g < G; ++g) {
      const float m_g = sums[warp + g][kHeadDim + 1];
      const float rs = m_g == -INFINITY ? 0.0f : fast_exp2(m_g - mx);
      o0 += rs * sums[warp + g][2 * lane];
      o1 += rs * sums[warp + g][2 * lane + 1];
      lt += rs * sums[warp + g][kHeadDim];
    }
    acc = make_float2(o0, o1);
    lw = lt;
  }
  const float f = lw > 0.0f ? p.out_scale / lw : 0.0f;
  __nv_bfloat162 v = __floats2bfloat162_rn(acc.x * f, acc.y * f);
  if constexpr (kGM == 2) v = norm_gate_row(v);
  *orow = v;
}

// Trigger form of the low-register grouped merge (launcher forms 7 / 8): full_attn_wide_gmerge_1w_kernel's body
// unchanged, plus griddepcontrol.launch_dependents at entry, before its own dependency wait. The whole merge grid is
// resident by then, so the dependent (the output projection) takes only the resources the merge leaves free and can stage
// its weights while the merge runs; it still waits for this grid's completion before it reads the merged rows. (A separate
// copy, so that the code of full_attn_wide_gmerge_1w_kernel stays exactly as it is.)
template <int kPairs, bool kUsePDL, int kGM, bool kLbM = false>
__global__ void __launch_bounds__(256, 4) full_attn_wide_gmerge_1wt_kernel(const Params p, const GateArgs ga,
                                                                           int grid_blocks, int merge_groups) {
  constexpr int kHeads = 2 * kPairs, kTileKeys = 128 / kPairs, kRecRows = kHeads * kRows;
  __shared__ int32_t range[3];
  __shared__ float sums[8][kHeadDim + 2];
  if constexpr (kUsePDL) asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
  if constexpr (kUsePDL) asm volatile("griddepcontrol.wait;\n" ::: "memory");
  const int R = p.num_reqs, r = blockIdx.y, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (!kLbM && tid < 32) {
    int n[kMaxReqs / 32];
#pragma unroll
    for (int k = 0; k < kMaxReqs / 32; ++k) {
      const int rr = lane + 32 * k;
      n[k] = rr < R ? p.seq_lens[rr] : 0;
    }
    int before = 0, upto = 0, all = 0;
#pragma unroll
    for (int k = 0; k < kMaxReqs / 32; ++k) {
      const int rr = lane + 32 * k;
      const int t = rr < R ? (max(n[k], 1) + kTileKeys - 1) / kTileKeys : 0;
      all += t;
      if (rr < r) before += t;
      if (rr <= r) upto += t;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      before += __shfl_xor_sync(0xffffffffu, before, o);
      upto += __shfl_xor_sync(0xffffffffu, upto, o);
      all += __shfl_xor_sync(0xffffffffu, all, o);
    }
    if (lane == 0) {
      range[0] = before;
      range[1] = upto;
      range[2] = all;
    }
  }
  int b0, b1;
  if constexpr (kLbM) {
    b0 = ga.lb_map[2 * r];
    b1 = b0 + ga.lb_map[2 * r + 1] - 1;
  } else {
    __syncthreads();
    const int W = range[2];
    const int Gb = min(grid_blocks, W);
    b0 = block_of(range[0], Gb, W);
    b1 = block_of(range[1] - 1, Gb, W);
  }
  const int G = merge_groups;
  const int row = blockIdx.x * (8 / G) + warp / G, grp = warp % G;
  const int np = b1 - b0 + 1;
  const int head = row >> 6, rr = row & 63, tok = rr >> 3, qh = head * kQPerKv + (rr & 7);
  __nv_bfloat162* const orow = reinterpret_cast<__nv_bfloat162*>(
      p.out + ((static_cast<int64_t>(r) * kTokens + tok) * p.num_q_heads + qh) * kHeadDim + 2 * lane);
  const auto norm_gate_row = [&](__nv_bfloat162 v) -> __nv_bfloat162 {
    // the grouped merge's output norm and gate of the row (SGLang's norm_gate arithmetic for 16 threads per row)
    const float2 x = __bfloat1622float2(v);
    const float a = fmaf(x.y, x.y, fmaf(x.x, x.x, 0.0f));
    const float prev = __shfl_xor_sync(0xffffffffu, a, 1);
    float ss = fmaf(x.y, x.y, fmaf(x.x, x.x, prev));  // meaningful on odd lanes
#pragma unroll
    for (int mask = 16; mask >= 2; mask >>= 1) ss = ss + __shfl_xor_sync(0xffffffffu, ss, mask);
    ss = __shfl_sync(0xffffffffu, ss, lane | 1);
    const float rn = rsqrtf(ss / static_cast<float>(kHeadDim) + ga.norm_eps);
    const float2 g = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(
        ga.gate + (static_cast<int64_t>(r) * kTokens + tok) * ga.gate_token_stride + qh * kHeadDim + 2 * lane));
    const auto sig = [](float z) { return __fdividef(1.0f, 1.0f + __expf(-z)); };
    float y0 = x.x * rn, y1 = x.y * rn;
    y0 *= sig(g.x);
    y1 *= sig(g.y);
    return __floats2bfloat162_rn(y0, y1);
  };
  if (b0 == b1) {  // the request lies in one block: the main kernel wrote its output (and the norm + gate follow here)
    if constexpr (kGM == 2) {
      if (grp == 0) *orow = norm_gate_row(*orow);
    }
    return;
  }
  const auto rec_of = [&](int k) -> const float* {
    const int j = b0 + k;
    return p.partial + ((static_cast<int64_t>(j) * 2 + (k == 0 ? 1 : 0)) * kRecRows + row) * kRec;
  };
  float mw = -INFINITY;
  float2 acc = make_float2(0.0f, 0.0f);
  float lw = 0.0f;
  constexpr int kBatch = 16;  // pieces whose loads are issued together
  static_assert(kBatch <= 32, "one (m, l) pair per lane");
  for (int s0 = grp; s0 < np; s0 += G * kBatch) {
    float2 mine = make_float2(-INFINITY, 0.0f);  // lane i < kBatch: piece s0 + G i's (m, l)
    if (lane < kBatch && s0 + G * lane < np) mine = *reinterpret_cast<const float2*>(rec_of(s0 + G * lane) + kHeadDim);
    float2 v[kBatch];
#pragma unroll
    for (int i = 0; i < kBatch; ++i) {
      const int k = s0 + G * i;
      v[i] = k < np ? *reinterpret_cast<const float2*>(rec_of(k) + 2 * lane) : make_float2(0.0f, 0.0f);
    }
    float mb = mine.x;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) mb = fmaxf(mb, __shfl_xor_sync(0xffffffffu, mb, o));
    mb = fmaxf(mw, mb);
    if (mb != -INFINITY) {
      const float rs = mw == -INFINITY ? 0.0f : fast_exp2(mw - mb);
      acc.x *= rs;
      acc.y *= rs;
      lw *= rs;
      const float wmine = mine.x == -INFINITY ? 0.0f : fast_exp2(mine.x - mb);
#pragma unroll
      for (int i = 0; i < kBatch; ++i) {
        const float ws = __shfl_sync(0xffffffffu, wmine, i);
        const float li = __shfl_sync(0xffffffffu, mine.y, i);
        acc.x += ws * v[i].x;
        acc.y += ws * v[i].y;
        lw += ws * li;
      }
      mw = mb;
    }
  }
  if (G > 1) {
    sums[warp][2 * lane] = acc.x;
    sums[warp][2 * lane + 1] = acc.y;
    if (lane == 0) {
      sums[warp][kHeadDim] = lw;
      sums[warp][kHeadDim + 1] = mw;
    }
    __syncthreads();
    if (grp != 0) return;
    float mx = -INFINITY;
    for (int g = 0; g < G; ++g) mx = fmaxf(mx, sums[warp + g][kHeadDim + 1]);
    float o0 = 0.0f, o1 = 0.0f, lt = 0.0f;
    for (int g = 0; g < G; ++g) {
      const float m_g = sums[warp + g][kHeadDim + 1];
      const float rs = m_g == -INFINITY ? 0.0f : fast_exp2(m_g - mx);
      o0 += rs * sums[warp + g][2 * lane];
      o1 += rs * sums[warp + g][2 * lane + 1];
      lt += rs * sums[warp + g][kHeadDim];
    }
    acc = make_float2(o0, o1);
    lw = lt;
  }
  const float f = lw > 0.0f ? p.out_scale / lw : 0.0f;
  __nv_bfloat162 v = __floats2bfloat162_rn(acc.x * f, acc.y * f);
  if constexpr (kGM == 2) v = norm_gate_row(v);
  *orow = v;
}

// The in-kernel merge's step word (kFd): advanced by one, once per step, ahead of the step's first full-attention call
// (a plain launch, so every later call of the step reads it after its dependency wait).
__global__ void full_attn_wide_fd_step_kernel(unsigned long long* epoch) { *epoch += 1ull; }

}  // namespace faw

// ---------------------------------------------------------------------------------------------- host
namespace faw_host {

using EncodeFn = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                              const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
                              CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

inline EncodeFn encode_fn() {
  static EncodeFn fn = [] {
    void* ptr = nullptr;
    cudaDriverEntryPointQueryResult q;
    sglang::host::RuntimeDeviceCheck(
        ::cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &ptr, 12000, cudaEnableDefault, &q));
    sglang::host::RuntimeCheck(q == cudaDriverEntryPointSuccess && ptr != nullptr, "cuTensorMapEncodeTiled not found");
    return reinterpret_cast<EncodeFn>(ptr);
  }();
  return fn;
}

// Tensor map over a pool [slots, row_bytes] (bytes), box = 128 bytes x rows, 128-byte swizzle.
inline CUtensorMap pool_map(const void* base, int64_t slots, int64_t row_bytes, int rows) {
  CUtensorMap map;
  const cuuint64_t dims[2] = {static_cast<cuuint64_t>(row_bytes), static_cast<cuuint64_t>(slots)};
  const cuuint64_t strides[1] = {static_cast<cuuint64_t>(row_bytes)};
  const cuuint32_t box[2] = {128, static_cast<cuuint32_t>(rows)};
  const cuuint32_t elem[2] = {1, 1};
  const CUresult r = encode_fn()(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(base), dims, strides, box,
                                 elem, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                 CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  sglang::host::RuntimeCheck(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled (pool) failed: ", static_cast<int>(r));
  return map;
}

// Tensor map over q [tokens, heads, 64] (bytes; token stride in bytes), box = 64 B x 8 heads x 8 tokens, 64-byte swizzle.
inline CUtensorMap q_map(const void* base, int64_t tokens, int64_t heads, int64_t token_stride) {
  CUtensorMap map;
  const cuuint64_t dims[3] = {64, static_cast<cuuint64_t>(heads), static_cast<cuuint64_t>(tokens)};
  const cuuint64_t strides[2] = {64, static_cast<cuuint64_t>(token_stride)};
  const cuuint32_t box[3] = {64, 8, 8};
  const cuuint32_t elem[3] = {1, 1, 1};
  const CUresult r = encode_fn()(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, const_cast<void*>(base), dims, strides, box,
                                 elem, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_64B,
                                 CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  sglang::host::RuntimeCheck(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled (q) failed: ", static_cast<int>(r));
  return map;
}

}  // namespace faw_host

template <int kPairs, int kStages, bool kUsePDL, int kPoly, int kPf = 0, int kMode = 0, int kTpr = 2, int kGM = 0,
          bool kLb = false, int kQk = 0, bool kQkUnfused = false, int kFd = 0>
struct FullAttnWideKernel {
  static void run(const tvm::ffi::TensorView q,           // [R*8, Hq, 64] e4m3, heads/dims contiguous (kQk: unused)
                  const tvm::ffi::TensorView k_pool,      // [slots, Hkv*64] e4m3, rows contiguous
                  const tvm::ffi::TensorView v_pool,      // [slots, Hkv*64]
                  const tvm::ffi::TensorView page_table,  // [R, pages] int32, inner contiguous
                  const tvm::ffi::TensorView seq_lens,    // [R] int32
                  const tvm::ffi::TensorView out,         // [R*8, Hq, 64] bf16, contiguous
                  const tvm::ffi::TensorView partial,     // fp32 workspace, >= blocks * 2 * Hkv * 64 * 68
                  const tvm::ffi::TensorView counters,    // int32 [>= 256], zero
                  int64_t blocks, double bmm1_scale, double bmm2_scale,
                  const tvm::ffi::TensorView gate,  // kGM == 2: [R*8, >= Hq*64] bf16, inner contiguous (else unused)
                  double norm_eps,
                  const tvm::ffi::TensorView qkv,        // kQk: [R*8, >= (Hq + 2 Hkv) * 64] bf16, inner contiguous
                  const tvm::ffi::TensorView cache_loc,  // kQk: [R*8] int64
                  const tvm::ffi::TensorView lb_map,     // kLb: int32 [>= 2 * 16]
                  double qk_eps,
                  const tvm::ffi::TensorView fd_ws,   // kFd: int64 [>= 64 * 16 + 8]: arrival words, step word, error
                  int64_t fd_layer, int64_t fd_max_wait_ns, int64_t fd_skip_block, int64_t fd_swap) {
    using namespace sglang::host;
    using L = faw::Layout<kPairs, kStages, kTpr>;
    static_assert(!kLb || (kGM != 0 && kMode == 0), "the length-balanced split needs the grouped merge (serving form)");
    static_assert(kQk == 0 || kLb, "the query/key prologue needs the length-balanced split");
    static_assert(kFd == 0 || (kLb && kGM != 0), "the in-kernel merge needs the length-balanced split");
    // launcher forms beyond the main kernel's: 7 = 5 and 8 = 6 with the trigger form of the low-register grouped merge
    static_assert(kFd >= 0 && kFd <= 2 || kFd >= 4 && kFd <= 8, "kFd: 0, 1, 2, 4, 5, 6, 7, 8");
    static_assert(kFd < 6 || kQk == 0, "forms 6-8: without the query/key prologue");
    constexpr int kFdMain = kFd == 7 ? 5 : kFd == 8 ? 6 : kFd;  // the main kernel's form
    constexpr bool kGmTrig = kFd == 7 || kFd == 8;
    RuntimeCheck(kQk > 0 || (q.dim() == 3 && q.size(2) == faw::kHeadDim &&
                             q.dtype().code == DLDataTypeCode::kDLFloat8_e4m3fn && q.stride(2) == 1 &&
                             q.stride(1) == faw::kHeadDim && q.stride(0) % 16 == 0),
                 "q must be e4m3 [tokens, heads, 64], heads and dims contiguous");
    RuntimeCheck(k_pool.dtype().code == DLDataTypeCode::kDLFloat8_e4m3fn &&
                     v_pool.dtype().code == DLDataTypeCode::kDLFloat8_e4m3fn && k_pool.dim() == 2 && v_pool.dim() == 2 &&
                     k_pool.stride(1) == 1 && v_pool.stride(1) == 1 && k_pool.stride(0) == k_pool.size(1) &&
                     v_pool.stride(0) == v_pool.size(1) && k_pool.size(1) == v_pool.size(1),
                 "pools must be contiguous e4m3 [slots, Hkv*64]");
    RuntimeCheck(out.dtype().code == DLDataTypeCode::kDLBfloat && out.dtype().bits == 16, "out must be bf16");
    const int64_t m = out.size(0), hq = out.size(1), reqs = m / faw::kTokens;
    const int64_t hkv = k_pool.size(1) / faw::kHeadDim;
    RuntimeCheck(m % faw::kTokens == 0 && hkv == L::kHeads && hkv * faw::kQPerKv == hq &&
                     (kQk > 0 || (q.size(0) == m && q.size(1) == hq)),
                 "head layout (KV heads per rank must match the kernel)");
    RuntimeCheck(reqs <= faw::kMaxReqs, "too many requests");
    RuntimeCheck(out.dim() == 3 && out.size(2) == faw::kHeadDim && out.stride(2) == 1 && out.stride(1) == faw::kHeadDim &&
                     out.stride(0) == hq * faw::kHeadDim,
                 "out layout");
    RuntimeCheck(seq_lens.dim() == 1 && seq_lens.size(0) == reqs && seq_lens.stride(0) == 1 &&
                     seq_lens.dtype().code == DLDataTypeCode::kDLInt && seq_lens.dtype().bits == 32,
                 "seq_lens [R] int32");
    RuntimeCheck(page_table.dim() == 2 && page_table.size(0) >= reqs && page_table.stride(1) == 1 &&
                     page_table.dtype().code == DLDataTypeCode::kDLInt && page_table.dtype().bits == 32,
                 "page_table [R, pages] int32");
    static_assert(kGM >= 0 && kGM <= 2, "kGM: 0 off, 1 grouped merge, 2 grouped merge + norm-gate");
    static_assert(kGM == 0 || kMode == 0, "the grouped merge is a serving form (no development probe modes)");
    RuntimeCheck(blocks >= 1 && partial.dtype().code == DLDataTypeCode::kDLFloat && partial.dtype().bits == 32 &&
                     partial.numel() >= blocks * 2 * hkv * faw::kRows * faw::kRec,
                 "partial workspace too small");
    RuntimeCheck(counters.dtype().code == DLDataTypeCode::kDLInt && counters.dtype().bits == 32 &&
                     counters.numel() >= faw::kMaxReqs,
                 "counters int32 [>= 256]");
    const auto aligned16 = [](const void* ptr) { return reinterpret_cast<uintptr_t>(ptr) % 16 == 0; };
    RuntimeCheck((kQk > 0 || aligned16(q.data_ptr())) && aligned16(k_pool.data_ptr()) && aligned16(v_pool.data_ptr()) &&
                     aligned16(out.data_ptr()) && aligned16(partial.data_ptr()),
                 "16-byte alignment");
    faw::QkArgs qargs{};
    if constexpr (kLb) {
      RuntimeCheck(reqs >= 1 && reqs <= faw::kLbMaxReqs && blocks >= reqs && blocks <= 4096,
                   "length-balanced split: 1 <= requests <= 16, requests <= blocks <= 4096");
      RuntimeCheck(lb_map.dtype().code == DLDataTypeCode::kDLInt && lb_map.dtype().bits == 32 &&
                       lb_map.numel() >= 2 * faw::kLbMaxReqs,
                   "lb_map int32 [>= 32]");
      qargs.lb_map = static_cast<int32_t*>(lb_map.data_ptr());
    }
    if constexpr (kQk > 0) {
      RuntimeCheck(qkv.dtype().code == DLDataTypeCode::kDLBfloat && qkv.dtype().bits == 16 && qkv.dim() == 2 &&
                       qkv.size(0) == m && qkv.size(1) >= (hq + 2 * hkv) * faw::kHeadDim && qkv.stride(1) == 1 &&
                       qkv.stride(0) % 8 == 0 && aligned16(qkv.data_ptr()),
                   "qkv must be bf16 [tokens, >= (Hq + 2 Hkv) * 64], inner contiguous, 16-byte aligned rows");
      RuntimeCheck(cache_loc.dtype().code == DLDataTypeCode::kDLInt && cache_loc.dtype().bits == 64 &&
                       cache_loc.dim() == 1 && cache_loc.size(0) == m && cache_loc.stride(0) == 1,
                   "cache_loc [tokens] int64, contiguous");
      qargs.qkv = static_cast<const __nv_bfloat16*>(qkv.data_ptr());
      qargs.cache_loc = static_cast<const int64_t*>(cache_loc.data_ptr());
      qargs.k_store = static_cast<uint8_t*>(k_pool.data_ptr());
      qargs.v_store = static_cast<uint8_t*>(v_pool.data_ptr());
      qargs.qkv_token_stride = qkv.stride(0);
      qargs.qk_eps = static_cast<float>(qk_eps);
    }
#ifdef MUSE_VR_FAW_PRE_CHECK
    qargs.k_store = static_cast<uint8_t*>(k_pool.data_ptr());  // the check build's coherent re-read of the tiles
    qargs.v_store = static_cast<uint8_t*>(v_pool.data_ptr());
#endif

    faw::Params params{};
    params.page_table = static_cast<const int32_t*>(page_table.data_ptr());
    params.seq_lens = static_cast<const int32_t*>(seq_lens.data_ptr());
    params.partial = static_cast<float*>(partial.data_ptr());
    params.counters = static_cast<uint32_t*>(counters.data_ptr());
    params.out = static_cast<__nv_bfloat16*>(out.data_ptr());
    params.page_table_stride = page_table.stride(0);
    params.num_reqs = static_cast<int32_t>(reqs);
    params.num_q_heads = static_cast<int32_t>(hq);
    params.scale_log2 = static_cast<float>(bmm1_scale * M_LOG2E);
    params.out_scale = static_cast<float>(bmm2_scale);  // row sums are kept in x448 units
    faw::GateArgs gargs{};
    if constexpr (kGM == 2) {
      RuntimeCheck(gate.dtype().code == DLDataTypeCode::kDLBfloat && gate.dtype().bits == 16 && gate.dim() == 2 &&
                       gate.size(0) == m && gate.size(1) >= hq * faw::kHeadDim && gate.stride(1) == 1 &&
                       gate.stride(0) % 2 == 0 && reinterpret_cast<uintptr_t>(gate.data_ptr()) % 4 == 0,
                   "gate must be bf16 [tokens, >= Hq*64], inner contiguous, 4-byte aligned rows");
      gargs.gate = static_cast<const __nv_bfloat16*>(gate.data_ptr());
      gargs.gate_token_stride = gate.stride(0);
      gargs.norm_eps = static_cast<float>(norm_eps);
    }
    gargs.lb_map = qargs.lb_map;
    CUtensorMap qm{};  // kQk: unused (the kernel computes the q rows)
    if constexpr (kQk == 0) qm = faw_host::q_map(q.data_ptr(), m, hq, q.stride(0));
    const CUtensorMap km = faw_host::pool_map(k_pool.data_ptr(), k_pool.size(0), k_pool.size(1), L::kTileKeys);
    const CUtensorMap vm = faw_host::pool_map(v_pool.data_ptr(), v_pool.size(0), v_pool.size(1), L::kTileKeys);

    static_assert(kPoly >= 0 && kPoly <= 8 && kPoly % 2 == 0, "kPoly: even, 0..8");
    // warps per row of the grouped merge: the fewest (power of two, at most 8) that leave each warp at most one batch of
    // 16 pieces, with about blocks / R + 1 pieces per request
    int groups = 1;
    if (reqs > 0) {
      const int64_t pieces = blocks / reqs + 1;
      while (groups < 8 && (pieces + groups - 1) / groups > 16) groups *= 2;
    }
    faw::FdArgs fargs{};
    if constexpr (kFd == 1 || kFd == 2) {
      RuntimeCheck(fd_ws.dtype().code == DLDataTypeCode::kDLInt && fd_ws.dtype().bits == 64 &&
                       fd_ws.numel() >= faw::kFdLayerSlots * faw::kLbMaxReqs + 8 && fd_layer >= 0 &&
                       fd_layer < faw::kFdLayerSlots,
                   "in-kernel merge workspace int64 [>= 64 * 16 + 8], layer slot 0..63");
      auto* const w = static_cast<unsigned long long*>(fd_ws.data_ptr());
      fargs.gate = gargs.gate;
      fargs.gate_token_stride = gargs.gate_token_stride;
      fargs.norm_eps = gargs.norm_eps;
      fargs.groups = groups;
      fargs.words = w + fd_layer * faw::kLbMaxReqs;
      fargs.epoch = w + faw::kFdLayerSlots * faw::kLbMaxReqs;
      fargs.err = reinterpret_cast<uint32_t*>(w + faw::kFdLayerSlots * faw::kLbMaxReqs + 1);
      fargs.err_tag = static_cast<uint32_t>(fd_layer);
      // the wait bound in SM cycles at the device's rated clock (a lower running clock only lengthens it in time)
      static int clock_khz = 0;
      if (clock_khz == 0)
        RuntimeDeviceCheck(::cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, out.device().device_id));
      fargs.max_wait_cycles = static_cast<uint64_t>(fd_max_wait_ns) * static_cast<uint64_t>(clock_khz) / 1000000ull;
      fargs.skip_block = static_cast<uint32_t>(fd_skip_block);
      fargs.swap = static_cast<uint32_t>(fd_swap);
    }
    constexpr int kSm = kQk > 0 ? L::kSmBytesQk : L::kSmBytes;
    // one instantiation of the main kernel: configured once (dynamic shared memory attribute), then launched
    const auto configure = [&](auto kernel, bool& configured) {
      if (!configured) {
        RuntimeDeviceCheck(::cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSm));
        configured = true;
      }
    };
    const auto launch_main = [&](auto kernel, bool& configured) {
      configure(kernel, configured);
      if (reqs == 0) return;
      LaunchKernel(dim3(static_cast<unsigned>(blocks)), L::kThreads, out.device(), kSm)
          .config({.use_pdl = kUsePDL})(kernel, params, qm, km, vm, qargs, fargs);
    };
    constexpr auto kernel = faw::full_attn_wide_kernel<kPairs, kStages, kUsePDL, kPoly, kPf, kMode, kTpr, kLb, kQk,
                                                       kQkUnfused, kFdMain, (kFd != 0 ? kGM : 0)>;
    static bool configured = false;
    if constexpr (kFd == 2) {
      // the distributed wait needs the grid resident as one wave at one block per SM: the occupancy of this
      // instantiation is asserted once; a launch with more blocks than the resident capacity takes the last-arriver form
      static int capacity = -1;
      if (capacity < 0) {
        configure(kernel, configured);  // the occupancy query needs the attribute (no launch here)
        int occ = 0, sms = 0;
        RuntimeDeviceCheck(::cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kernel, L::kThreads, kSm));
        RuntimeDeviceCheck(::cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, out.device().device_id));
        RuntimeCheck(occ == 1, "in-kernel merge (distributed): one block per SM expected, occupancy ", occ);
        capacity = occ * sms;
      }
      if (blocks > capacity) {
        static int64_t fallbacks = 0;
        if (fallbacks++ == 0)
          std::fprintf(stderr, "full_attn_wide: distributed merge -> last-arriver form (%lld blocks > resident %d)\n",
                       static_cast<long long>(blocks), capacity);
        constexpr auto kernel_la = faw::full_attn_wide_kernel<kPairs, kStages, kUsePDL, kPoly, kPf, kMode, kTpr, kLb,
                                                              kQk, kQkUnfused, 1, kGM>;
        static bool configured_la = false;
        launch_main(kernel_la, configured_la);
        return;
      }
    }
    if (reqs == 0 && configured) return;
    launch_main(kernel, configured);
    if (reqs == 0) return;
    if constexpr (kFd == 1 || kFd == 2) {
      return;  // merged in the main kernel
    } else if constexpr (kGM != 0 && (kFd == 5 || kFd == 6) && !kGmTrig) {
      // the low-register grouped merge (same grid and arguments)
      constexpr auto gmerge = faw::full_attn_wide_gmerge_1w_kernel<kPairs, kUsePDL, kGM, kLb>;
      LaunchKernel(dim3(static_cast<unsigned>(L::kHeads * faw::kRows * groups / 8), static_cast<unsigned>(reqs)), 256,
                   out.device(), 0)
          .config({.use_pdl = kUsePDL})(gmerge, params, gargs, static_cast<int>(blocks), groups);
    } else if constexpr (kGM != 0 && kGmTrig) {  // its trigger form (forms 7 / 8)
      constexpr auto gmerge = faw::full_attn_wide_gmerge_1wt_kernel<kPairs, kUsePDL, kGM, kLb>;
      LaunchKernel(dim3(static_cast<unsigned>(L::kHeads * faw::kRows * groups / 8), static_cast<unsigned>(reqs)), 256,
                   out.device(), 0)
          .config({.use_pdl = kUsePDL})(gmerge, params, gargs, static_cast<int>(blocks), groups);
    } else if constexpr (kGM != 0) {
      constexpr auto gmerge = faw::full_attn_wide_gmerge_kernel<kPairs, kUsePDL, kGM, kLb>;
      LaunchKernel(dim3(static_cast<unsigned>(L::kHeads * faw::kRows * groups / 8), static_cast<unsigned>(reqs)), 256,
                   out.device(), 0)
          .config({.use_pdl = kUsePDL})(gmerge, params, gargs, static_cast<int>(blocks), groups);
    } else if constexpr (kMode == 0 || kMode == 4 || kMode == 6) {
      constexpr auto merge = faw::full_attn_wide_merge_kernel<kPairs, kUsePDL>;
      LaunchKernel(dim3(static_cast<unsigned>(L::kHeads * faw::kRows / faw::kMergeRows), static_cast<unsigned>(reqs)),
                   256, out.device(), 0)
          .config({.use_pdl = kUsePDL})(merge, params, static_cast<int>(blocks));
    }
  }
};

// The in-kernel merge's step word: one plain launch per step, ahead of the step's first full-attention call.
struct FullAttnWideFdStep {
  static void run(const tvm::ffi::TensorView fd_ws) {
    using namespace sglang::host;
    RuntimeCheck(fd_ws.dtype().code == DLDataTypeCode::kDLInt && fd_ws.dtype().bits == 64 &&
                     fd_ws.numel() >= faw::kFdLayerSlots * faw::kLbMaxReqs + 8,
                 "in-kernel merge workspace int64 [>= 64 * 16 + 8]");
    auto* const w = static_cast<unsigned long long*>(fd_ws.data_ptr());
    LaunchKernel(dim3(1), 1, fd_ws.device(), 0)
        .config({.use_pdl = false})(faw::full_attn_wide_fd_step_kernel, w + faw::kFdLayerSlots * faw::kLbMaxReqs);
  }
};

#ifdef MUSE_VR_FAW_PRE_CHECK
namespace faw {
// Tests and units only: a predecessor that triggers its dependents at entry, waits about delay_us microseconds, then copies
// new lengths, page-table words and pool bytes (e.g. one committed key row) over the live ones (the forced-stale control
// of form 6's pre-wait reads).
__global__ void full_attn_wide_late_writer_kernel(int32_t* seq, const int32_t* seq_new, int n_seq, int32_t* pt,
                                                  const int32_t* pt_new, int64_t n_pt, uint8_t* row,
                                                  const uint8_t* row_new, int64_t n_row, int delay_us, int trigger) {
  // trigger 0: no programmatic trigger (dependents start when its blocks have exited, as a plain launch's)
  if (trigger) asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
  for (int i = 0; i < delay_us; ++i) __nanosleep(1000);
  for (int i = threadIdx.x; i < n_seq; i += blockDim.x) seq[i] = seq_new[i];
  for (int64_t i = threadIdx.x; i < n_pt; i += blockDim.x) pt[i] = pt_new[i];
  for (int64_t i = threadIdx.x; i < n_row; i += blockDim.x) row[i] = row_new[i];
}
}  // namespace faw

struct FullAttnWideLateWriter {
  static void run(const tvm::ffi::TensorView seq, const tvm::ffi::TensorView seq_new, const tvm::ffi::TensorView pt,
                  const tvm::ffi::TensorView pt_new, const tvm::ffi::TensorView row, const tvm::ffi::TensorView row_new,
                  int64_t delay_us, int64_t trigger) {
    using namespace sglang::host;
    RuntimeCheck(seq.numel() == seq_new.numel() && pt.numel() == pt_new.numel() && row.numel() == row_new.numel(),
                 "late writer: equal sizes");
    LaunchKernel(dim3(1), 256, seq.device(), 0)
        .config({.use_pdl = false})(faw::full_attn_wide_late_writer_kernel, static_cast<int32_t*>(seq.data_ptr()),
                                    static_cast<const int32_t*>(seq_new.data_ptr()), static_cast<int>(seq.numel()),
                                    static_cast<int32_t*>(pt.data_ptr()), static_cast<const int32_t*>(pt_new.data_ptr()),
                                    static_cast<int64_t>(pt.numel()), static_cast<uint8_t*>(row.data_ptr()),
                                    static_cast<const uint8_t*>(row_new.data_ptr()), static_cast<int64_t>(row.numel()),
                                    static_cast<int>(delay_us), static_cast<int>(trigger != 0));
  }
};
#endif

}  // namespace muse_vr
