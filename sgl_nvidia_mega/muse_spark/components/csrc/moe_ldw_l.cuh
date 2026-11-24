// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// MoE-LD-W: routed NVFP4 experts of one expert-parallel rank for verify batches of 1..512 token rows (1..1024 / 1..2048 in the
// wide builds), in one launch of one CTA per SM that streams every active local expert's weights once per pass of up to 128 routed
// rows.
//
// Prologue (every CTA, after griddepcontrol.wait): the local routed slots are counted per expert (per 256-slot chunk), giving
// expert bases (rows in expert-major, token order, as the stock expanded-index map orders them) and passes of at most 128
// rows; each CTA builds its work list (pass, weight tile) of FC1 (128 gate/up rows, full K) and FC2 (128 output rows, full K)
// items, dealt per locality domain as in moe_ld.cuh (each domain streams the expert row halves placed on it); the weights
// of its first ring stages go out at once. Warps 4-11 copy the token rows of slots b, b + G, ... into the expert-major
// buffer with their per-token scales and token-scale images (the 32-lane tensor-memory layout of block scale factors: row
// t of a pass in lane t % 32, column t / 32 of each 64-wide K chunk's 4-column group, PTX ISA tcgen05 block scaling), write
// the expanded-index map and top-k weights of those slots, and release one count of the grid's rows-ready counter; token-row
// copies of any CTA follow the acquire of that counter (all CTAs) and a proxy fence.
// Every stage carries 32 KiB of weights + their scales and the pass's token rows (16 / 32 / 64 / 128 rows, the next bucket
// above the pass's row count) + their scale images; the MMA is M = 128 x N = that bucket; every output element is accumulated
// over the full K in K order, as the stock kernel does (tests/test_moe_ldw.py: every routed FC2 row bitwise).
// FC1 epilogue: the 16x256b tensor-memory fragment gives each thread the up and the gate row of a pair (no shuffles);
// SwiGLU (stock arithmetic) -> activation rows. FC1 -> FC2: per pass, the rows of the FC2 operand are dealt over the CTAs
// holding the pass's FC2 tiles (tile c: rows c, c + 32, c + 64, c + 96) and round-robin over their four builder warps; each
// row is built (FlashInfer's per-token NVFP4 recipe) once the pass's FC1 tiles are published, then one count is released;
// FC2 stages of the pass are copied after that count is acquired and a proxy fence. Builders wait only on FC1, which waits
// on nothing outside its CTA, so the waits are acyclic. Counters come in two sets used by alternating launches; a launch
// zeroes the other set for the next one. Code that runs once per launch (prologue, row copies) is kept small and out of line.
// Every wait is bounded (poll count, sticky error word); outstanding copies and MMAs are drained before exit, and a CTA
// whose wait expired then traps (MUSE_VR_LDW_TRAP, default 1), so an expired wait is never a silent result.
// Error-word build (-DMUSE_VR_LDW_ERRWORD=1, default 0): no trap. A CTA whose wait expired, or whose slot claim failed
// (slot_claim.cuh's error-word form: the CTA takes no work items), stores bit 31 | the call's tag into the caller's
// error word (one 32-bit word of page-locked, device-mapped host memory, system scope; Params::page_err) at the same point
// and the kernel completes. The verify commit's gate reads that word, so a step with an expired wait never commits.
// Residency: start tickets as in moe_ld.cuh. Slots: claimed exactly as in moe_ld.cuh (slot_claim.cuh: the SM's slot is
// only the preferred one; one claim per CTA, kept in shared memory for every role; the previous form only with
// -DMUSE_VR_SLOT_UNSAFE=1). The claim area is words kClaimWord.. of the control tensor, which the launches that share
// it already order by completion.

#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cstddef>
#include <cstdint>

#ifndef MUSE_VR_LDW_ERRWORD
#define MUSE_VR_LDW_ERRWORD 0
#endif
#if MUSE_VR_LDW_ERRWORD && !defined(MUSE_VR_SLOT_ERRWORD)
#define MUSE_VR_SLOT_ERRWORD 1
#endif
#include "ld_queue.cuh"
#include "slot_claim.cuh"
#include "tensorrt_llm/kernels/quantization_utils.cuh"

// Warp roles are inlined into the kernel body (measured faster in chain A/B than separate role functions);
// -DMUSE_VR_LDW_ROLE=__noinline__ keeps them separate (e.g. to read per-role code size).
// weight stages (beyond the ring) whose bytes are prefetched into L2 right after the first stages go out: the ring cannot
// take them before the token rows of the first pass are copied, so their DRAM reads would otherwise wait (0: off)
#ifndef MUSE_VR_LDW_HEAD_PF
#define MUSE_VR_LDW_HEAD_PF 0
#endif
#ifndef MUSE_VR_LDW_ROLE
#define MUSE_VR_LDW_ROLE __forceinline__
#endif
#ifndef MUSE_VR_LDW_IDLE_NS
#define MUSE_VR_LDW_IDLE_NS 256
#endif
#ifndef MUSE_VR_LDW_TRAP
#define MUSE_VR_LDW_TRAP 1
#endif
// Tests only: 1 = the start ticket and the slot claim are issued BEFORE griddepcontrol.wait (the earlier placement; the
// negative control of tests/test_moe_ldw_order.py). Default 0: both are issued right after the wait (prologue).
#ifndef MUSE_VR_LDW_TEST_PREWAIT_CLAIM
#define MUSE_VR_LDW_TEST_PREWAIT_CLAIM 0
#endif
// Early prologue (-DMUSE_VR_LDW_PRE=1; default 0): the prologue work that reads only the routed ids, the real-token count and
// static tables runs BEFORE griddepcontrol.wait: the id loads, per-expert counts, the ring placement, the expert bases and
// pass tables, and an L2 prefetch of the first ring stages' weights and weight scales for the work list of the SM's hinted
// slot. The start ticket, the slot claim (and so the work list), the token scales and every later step stay after the
// wait. Requires that the routed ids and the real-token count are complete when the kernel launches: the call site's
// producer of the ids must reach this launch through a full (non-programmatic) dependency. A claim that settles on a slot
// other than the hint only wastes the prefetch. Builds without a gather head and with the exact slot claim only.
#ifndef MUSE_VR_LDW_PRE
#define MUSE_VR_LDW_PRE 0
#endif
// Speculate and verify (-DMUSE_VR_LDW_PRE_VERIFY=1 with MUSE_VR_LDW_PRE=1; default 0): lifts the early prologue's
// requirement above. The routed ids and the real-token count read before the wait are a speculation: right after the wait
// every thread re-reads the words it used (ordered by the wait), and the CTA compares them with its copy in one barrier
// reduction. On any difference the CTA rebuilds every table from the fresh words (the same code, a second pass) before
// anything uses them; before the wait only shared-memory tables and L2 prefetch hints are built, so a stale read costs time
// and never a result. Every id is range-checked before it indexes anything. Diagnostics: given a stamps tensor, each CTA
// counts its rebuilds in stamp word kStRedo (after the wait); debug bit 2048 (tests only: the negative control) skips the
// compare. Debug bits 4096 / 8192 (tests only) change the speculative copies of the ids (one bit of every expert field) /
// of the real-token count before the first pass, so the compare must differ and the rebuild runs with no timing race (the
// deterministic form of the stale-read cases, e.g. under a sanitizer, which serializes launches).
#ifndef MUSE_VR_LDW_PRE_VERIFY
#define MUSE_VR_LDW_PRE_VERIFY 0
#endif
#if MUSE_VR_LDW_PRE_VERIFY && !MUSE_VR_LDW_PRE
#error "MUSE_VR_LDW_PRE_VERIFY needs MUSE_VR_LDW_PRE=1"
#endif
#ifndef MUSE_VR_LDW_MAX_POLLS
#define MUSE_VR_LDW_MAX_POLLS (1u << 24)
#endif
#ifndef MUSE_VR_LDW_K128
#define MUSE_VR_LDW_K128 0  // 1: K = 128 block-scaled MMAs (two 64-wide chunks per instruction)
#endif

// Every build variant (compile flags) gets its own symbol names (the ops layer sets MUSE_VR_LDW_ABI per variant), so
// variants loaded in one process never share a kernel or a launch-configuration static.
#ifndef MUSE_VR_LDW_ABI
#define MUSE_VR_LDW_ABI v0
#endif
// Ring placement build (-DMUSE_VR_LDW_RBAL=5; default 0; ring_place.cuh): expert-parallel world 4, pool 80 (this rank's
// experts 0..63, the predecessor's local experts 48..63 at 64..79; local_offset = 64 x rank). The prologue counts every routed
// slot per expert; one warp (MUSE_VR_LDW_RB_WARP, default 5: a warp idle in the prologue) waits at the prefix barrier with
// every warp, then computes the step's thresholds (place_warp2) while warp 0 scans the slots that never move: 0..31 and
// 32..47 (+ the empty 80..95); warp 0 then computes the candidate slots' (48..79) prefix sums with every candidate counted,
// meets the placement warp at named barrier MUSE_VR_LDW_RB_BAR (64 threads) and applies the thresholds' mask to those sums (a
// candidate set is a prefix of the own slots 48..63 and a suffix of the predecessor's 64..79). A candidate that runs on its
// other holder gets no rows and no position here. The routed math per (token, expert) is unchanged; bases, passes and the
// permutation equal the served build's on the virtual ids of the same placement.
#ifndef MUSE_VR_LDW_RBAL
#define MUSE_VR_LDW_RBAL 0
#endif
#if MUSE_VR_LDW_RBAL != 0 && MUSE_VR_LDW_RBAL != 5
#error "MUSE_VR_LDW_RBAL: 0 or 5"
#endif
#ifndef MUSE_VR_LDW_RB_BAR
#define MUSE_VR_LDW_RB_BAR 11
#endif
#ifndef MUSE_VR_LDW_RB_WARP
#define MUSE_VR_LDW_RB_WARP 5
#endif
#if MUSE_VR_LDW_RBAL
#include "ring_place.cuh"
static_assert(MUSE_VR_LDW_RB_WARP >= 4 && MUSE_VR_LDW_RB_WARP < 8, "MUSE_VR_LDW_RB_WARP: a warp idle in the prologue (4..7)");
static_assert(MUSE_VR_LDW_RB_BAR >= 0 && MUSE_VR_LDW_RB_BAR < 16 && MUSE_VR_LDW_RB_BAR != 3 && MUSE_VR_LDW_RB_BAR != 5 &&
                  MUSE_VR_LDW_RB_BAR != 6 && MUSE_VR_LDW_RB_BAR != 7,
              "MUSE_VR_LDW_RB_BAR: a free named barrier (0..15, not 3 / 5 / 6 / 7)");
#endif
// Tests only (default 0): 1 = the placement warp waits on a clock64 spin before the placement, 2 = warp 0 waits as long before
// its barrier-11 meeting (order-stress cases, results must not change)
#ifndef MUSE_VR_LDW_TEST_RB_SPIN
#define MUSE_VR_LDW_TEST_RB_SPIN 0
#endif
// Tests only (default 0): 1 = warp 0 skips its barrier-11 meeting (a planted race on the thresholds; results are not checked)
#ifndef MUSE_VR_LDW_TEST_RB_NOBAR
#define MUSE_VR_LDW_TEST_RB_NOBAR 0
#endif

namespace muse_vr_ldw {
inline namespace MUSE_VR_LDW_ABI {

constexpr uint32_t kMaxPolls = MUSE_VR_LDW_MAX_POLLS;
// ---------------------------------------------------------------- MoE-LD-W problem shape
constexpr int kHidden = 4096;  // FC1 K, FC2 N
constexpr int kInter = 4096;   // FC1 N / 2, FC2 K
constexpr int kTopK = 8;
// Rows per call: 512 (served default), 1024 or 2048 (-DMUSE_VR_LDW_MAX_TOKENS=1024 / 2048, the wide builds; their scratch is
// sized for them). Default builds preprocess to the same device code as before the macro existed.
#ifndef MUSE_VR_LDW_MAX_TOKENS
#define MUSE_VR_LDW_MAX_TOKENS 512
#endif
constexpr int kMaxTokens = MUSE_VR_LDW_MAX_TOKENS;
static_assert(kMaxTokens == 512 || kMaxTokens == 1024 || kMaxTokens == 2048, "rows per call: 512, 1024 or 2048");
constexpr int kMaxSlots = kMaxTokens * kTopK;  // routed slots per launch (all ranks' ids are read; local ones kept)
constexpr int kMaxPool = 128;
// rows of one expert pass = largest MMA N used (32: the RING4 build, with -DMUSE_VR_LDW_STAGES=4: a smaller token region per
// stage, so one more stage of weights is in flight; experts with more rows take more passes)
#ifndef MUSE_VR_LDW_PASS_ROWS
#define MUSE_VR_LDW_PASS_ROWS 128
#endif
constexpr int kPassRows = MUSE_VR_LDW_PASS_ROWS;
#if MUSE_VR_LDW_PASS_ROWS == 48
static_assert(kPassRows == 48, "pass rows");  // 48-row passes: token-operand heights 16 / 32 / 48
#else
static_assert(kPassRows == 32 || kPassRows == 64 || kPassRows == 128, "pass rows");
#endif
constexpr int kMaxPasses = kMaxPool + kMaxSlots / kPassRows;
static_assert(kMaxPasses <= 256, "pass index bits of a work item");
constexpr int kTile = 128;
constexpr int kStageW = 32 * 1024;
constexpr int kChunks = 8, kStagesPerTile = kHidden / (64 * kChunks);
constexpr int kStageA = kStageW + kChunks * 512;  // weights (128 rows x 512 K) + their block scales
constexpr int kStageB = kPassRows * 256;          // token rows (up to 128) x 512 K
constexpr int kStageImg = kChunks * 512;          // token-scale images of 8 K chunks
constexpr int kStageBytes = kStageA + kStageB + kStageImg;
#ifndef MUSE_VR_LDW_STAGES
#define MUSE_VR_LDW_STAGES 3
#endif
constexpr int kStages = MUSE_VR_LDW_STAGES;
// 1: the token rows and scale images of each CTA's first FC1 items (up to kHeadItems, while their pass has at most
// kHeadRows rows) are copied straight from the token tensors by row index (asynchronous copies tracked by each stage's
// barrier), so those items do not wait for the copy phase and its grid-wide rows-ready count; later items use the copy
// phase's permuted rows as before; 0: every item waits for the copy phase
#ifndef MUSE_VR_LDW_GATHER
#define MUSE_VR_LDW_GATHER 1
#endif
#if MUSE_VR_LDW_PRE && (MUSE_VR_LDW_GATHER || MUSE_VR_SLOT_UNSAFE || MUSE_VR_SLOT_EPOCH)
#error "MUSE_VR_LDW_PRE: builds without the gather head (MUSE_VR_LDW_GATHER=0) and with the exact slot claim only"
#endif
#ifndef MUSE_VR_LDW_HEAD_ITEMS
#define MUSE_VR_LDW_HEAD_ITEMS 1
#endif
#ifndef MUSE_VR_LDW_HEAD_ROWS
#define MUSE_VR_LDW_HEAD_ROWS 32
#endif
constexpr int kHeadItems = MUSE_VR_LDW_HEAD_ITEMS, kHeadRows = MUSE_VR_LDW_HEAD_ROWS;
static_assert(kHeadItems >= 1 && kHeadItems <= 4 && kHeadRows >= 1 && kHeadRows <= 128, "head items");
// 1 (the dense-token build, calls of at most kPassRows = 32 rows): every FC1 item takes the call's whole token block (rows
// 0 .. M - 1 of the token tensor, written by predecessor grids; rows past M zero-filled) by one tensor copy per stage, with
// the stage's token-scale image built from a per-CTA copy of the token scales. At <= 32 rows every active expert has one
// pass that holds all of its routed rows, so FC1 needs no expert-major permuted copy of the token rows and no grid-wide
// rows-ready count: the FC1 MMA computes every token column (N = M rounded up to the bucket) and the epilogue stores the
// routed columns only, at the same expert-major rows the permuted form uses. FC2, the builders and the outputs are
// unchanged. Every stored element is the same full-K, K-order accumulation with the same scales (columns of an MMA do not
// interact). 0: the permuted form.
#ifndef MUSE_VR_LDW_DENSE
#define MUSE_VR_LDW_DENSE 0
#endif
static_assert(!MUSE_VR_LDW_DENSE || (kPassRows == 32 && !MUSE_VR_LDW_GATHER && kMaxTokens == 512),
              "the dense-token build: 32-row passes, no head copies, the 512-row form");
// Tests only (default 0): 1 = the dense FC1 epilogue stores column x at row x of the pass for every x < M, ignoring the routed
// mask (a planted defect: the negative control of the dense-token identity rows; results must FAIL)
#ifndef MUSE_VR_LDW_TEST_DENSE_NOMASK
#define MUSE_VR_LDW_TEST_DENSE_NOMASK 0
#endif
constexpr int kDenseRows = 32;  // most token rows of a dense-token call (one pass per expert)
// row pitch of the dense-token build's per-CTA copy of the token scales: one row's scales plus 16 bytes, so the 16-byte
// reads of eight consecutive rows at the same offset fall in different shared-memory banks
constexpr int kTsfPitch = kHidden / 16 + 16;
constexpr int kFc1TilesPerExpert = 2 * kInter / kTile, kFc2TilesPerExpert = kHidden / kTile;
constexpr int kFc1TilesPerDomain = kFc1TilesPerExpert / 2, kFc2TilesPerDomain = kFc2TilesPerExpert / 2;
constexpr int kMaxItems = kMaxTokens > 1024 ? 128 : 96;  // work items per CTA (the 2048-row build has up to 2x the passes)
constexpr int kActWords = kInter / 4;  // 64-bit words per activation row (4 bf16)
constexpr int kFc1Arrivals = kFc1TilesPerExpert * 4;
constexpr int kImgBytes = (kHidden / 64) * 512;  // token-scale images of one pass (64 K chunks)
constexpr int kThreads = 384;
constexpr uint32_t kTmemCols = 512;
constexpr uint32_t kAccCol = 0;     // 2 accumulators x 128 columns
constexpr uint32_t kSfaCol = 256;   // weight scales: kStages regions of 32 columns
constexpr uint32_t kSfbCol = kSfaCol + kStages * 32;  // token scales: kStages regions of 32 columns
static_assert(kSfbCol + kStages * 32 <= kTmemCols, "tensor memory columns");
constexpr int kErrPoll = 1, kErrMbar = 2, kErrSlot = 4, kErrStart = 8, kErrDrain = 16, kErrShape = 32;
// slot claim area (slot_claim.cuh): 32-bit words kClaimWord .. kClaimWord + muse_vr_slot::kWords - 1 of the control
// tensor (word 0: start tickets)
constexpr int kClaimWord = 21;
static_assert(kClaimWord + muse_vr_slot::kWords <= 32, "claim area inside the 32-word control tensor");
constexpr unsigned kStartCycle = 1u << 12;
constexpr int kNumBuckets = 4;  // token-operand heights 16 / 32 / 64 / 128
#if MUSE_VR_LDW_PASS_ROWS == 48
__host__ __device__ constexpr int bucket_of(int rows) { return rows <= 16 ? 0 : rows <= 32 ? 1 : rows <= 48 ? 2 : 3; }
__host__ __device__ constexpr int bucket_rows(int b) { return b == 2 ? 48 : 16 << b; }
#else
__host__ __device__ constexpr int bucket_of(int rows) { return rows <= 16 ? 0 : rows <= 32 ? 1 : rows <= 64 ? 2 : 3; }
__host__ __device__ constexpr int bucket_rows(int b) { return 16 << b; }
#endif


// ---------------------------------------------------------------- PTX helpers
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
// CTA abort flag (shared memory; set on the first expired wait of any thread of the CTA). Every access to it is a
// strong CTA-scope operation on the shared state space, so the setter and the pollers are morally strong and do not
// form a data race (PTX ISA, memory consistency model: "Morally strong operations", "Conflict and Data-races").
__shared__ int g_abort;
__device__ __forceinline__ bool aborted() {
  uint32_t v;
  asm volatile("ld.relaxed.cta.shared.u32 %0, [%1];" : "=r"(v) : "r"(smem_u32(&g_abort)));
  return v != 0u;
}
__device__ __forceinline__ void abort_reset() {
  asm volatile("st.relaxed.cta.shared.u32 [%0], %1;" ::"r"(smem_u32(&g_abort)), "r"(0u) : "memory");
}
__device__ __noinline__ void fail(int32_t* err, int code) {
  atomicOr(err, code);
  uint32_t old;
  asm volatile("atom.relaxed.cta.shared.exch.b32 %0, [%1], %2;" : "=r"(old) : "r"(smem_u32(&g_abort)), "r"(1u)
               : "memory");
  (void)old;
}
#if MUSE_VR_LDW_ERRWORD
constexpr uint32_t kErrFlag = 0x80000000u;
// End of a CTA that aborted (error-word build): one system-scope store of bit 31 | tag into the caller's error word.
__device__ __noinline__ void report_error_word(uint32_t* word, uint32_t tag) {
  if (word != nullptr) asm volatile("st.relaxed.sys.global.u32 [%0], %1;" ::"l"(word), "r"(kErrFlag | tag) : "memory");
}
#endif
__device__ __forceinline__ bool elect_one() {
  uint32_t pred;
  asm volatile("{ .reg .pred P; elect.sync _|P, 0xffffffff; selp.u32 %0, 1, 0, P; }" : "=r"(pred));
  return pred != 0;
}
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ bool mbar_try(uint64_t* bar, uint32_t parity) {
  uint32_t done;
  asm volatile(
      "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
      : "=r"(done)
      : "r"(smem_u32(bar)), "r"(parity)
      : "memory");
  return done != 0;
}
// Non-blocking test of a phase (warp-uniform: lane 0's observation)
__device__ __forceinline__ bool mbar_test_warp(uint64_t* bar, uint32_t parity) {
  uint32_t done;
  asm volatile(
      "{ .reg .pred p; mbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
      : "=r"(done)
      : "r"(smem_u32(bar)), "r"(parity)
      : "memory");
  return __shfl_sync(0xffffffffu, done, 0) != 0;
}
// Bounded wait (every lane of the calling warp calls it with the same barrier and parity). Returns false when the CTA
// aborted or the bound ran out (which aborts the CTA).
__device__ __forceinline__ bool mbar_wait(uint64_t* bar, uint32_t parity, int32_t* err, uint32_t sleep_ns = 0) {
  uint32_t polls = 0;
  while (!mbar_try(bar, parity)) {
    ++polls;
    if ((polls & 63u) == 0u && aborted()) return false;
    if (polls > kMaxPolls) {
      fail(err, kErrMbar);
      return false;
    }
    if (sleep_ns > 0) __nanosleep(sleep_ns);
  }
  return true;
}
// Waits off the weight-stream critical path (epilogue, builders, control): one inline test, then an out-of-line polling
// loop shared by every call site.
__device__ __noinline__ bool mbar_wait_slow(uint64_t* bar, uint32_t parity, int32_t* err) {
  return mbar_wait(bar, parity, err, MUSE_VR_LDW_IDLE_NS);
}
__device__ __forceinline__ bool mbar_wait_idle(uint64_t* bar, uint32_t parity, int32_t* err) {
  return mbar_try(bar, parity) || mbar_wait_slow(bar, parity, err);
}
// Bounded wait for asynchronous work this CTA issued (completes on its own): not cut short by the abort flag.
__device__ __noinline__ void mbar_drain(uint64_t* bar, uint32_t parity, int32_t* err) {
  uint32_t polls = 0;
  while (!mbar_try(bar, parity)) {
    if (++polls > kMaxPolls) {
      atomicOr(err, kErrDrain);
      return;
    }
  }
}
__device__ __forceinline__ void fence_barrier_init() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}
__device__ __forceinline__ void fence_proxy_async_smem() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
__device__ __forceinline__ void fence_proxy_async_global() {
  asm volatile("fence.proxy.async.global;" ::: "memory");
}
__device__ __forceinline__ void named_bar_sync(uint32_t id, uint32_t n) {
  // non-aligned form: correct even if the calling warp is not converged at this point (PTX ISA barrier.sync)
  asm volatile("barrier.sync %0, %1;" ::"r"(id), "r"(n) : "memory");
}
__device__ __forceinline__ void named_bar_arrive(uint32_t id, uint32_t n) {
  asm volatile("barrier.arrive %0, %1;" ::"r"(id), "r"(n) : "memory");
}
// named barrier with an OR reduction of a predicate over its n threads (every thread gets the result)
__device__ __forceinline__ bool named_bar_red_or(uint32_t id, uint32_t n, bool pred) {
  uint32_t r;
  asm volatile(
      "{\n\t.reg .pred pi, po;\n\tsetp.ne.u32 pi, %1, 0;\n\tbarrier.red.or.pred po, %2, %3, pi;\n\tselp.u32 %0, 1, 0, po;\n\t}"
      : "=r"(r)
      : "r"(uint32_t(pred)), "r"(id), "r"(n)
      : "memory");
  return r != 0u;
}
__device__ __forceinline__ void grid_dep_wait() {
  asm volatile("griddepcontrol.wait;" ::: "memory");
}
__device__ __forceinline__ uint64_t policy_evict_first() {
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
  return pol;
}
// asynchronous copies global -> shared of 16 / 4 bytes (generic proxy); completion tracked by cp_async_arrive
__device__ __forceinline__ void cp_async_16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_async_4(uint32_t dst, const void* src) {
  asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 4;" ::"r"(dst), "l"(src) : "memory");
}
// cp.async groups: one per head stage; before a ring slot is written again, the issuing thread waits for the group that
// last wrote it (complete by then: its stage was consumed, which the empty barrier already showed), so the reuse is
// ordered after the copy by this thread too
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
__device__ __forceinline__ void cp_async_wait_newer(int n) {  // at most n newer groups pending (n = 0, 1, 2)
  if (n <= 0) asm volatile("cp.async.wait_group 0;" ::: "memory");
  else if (n == 1) asm volatile("cp.async.wait_group 1;" ::: "memory");
  else asm volatile("cp.async.wait_group 2;" ::: "memory");
}
// arrive-on of bar once every prior cp.async of this thread has completed; the pending count is raised by one first, so
// the phase cannot complete before those copies (PTX ISA cp.async.mbarrier.arrive)
__device__ __forceinline__ void cp_async_arrive(uint64_t* bar) {
  asm volatile("cp.async.mbarrier.arrive.shared::cta.b64 [%0];" ::"r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void tma_load_3d(uint32_t dst, const CUtensorMap* map, int32_t x, int32_t y, int32_t z,
                                            uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];" ::"r"(dst),
      "l"(map), "r"(x), "r"(y), "r"(z), "r"(smem_u32(bar))
      : "memory");
}
__device__ __forceinline__ void tma_load_3d_hint(uint32_t dst, const CUtensorMap* map, int32_t x, int32_t y, int32_t z,
                                                 uint64_t* bar, uint64_t pol) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, %3, %4}], [%5], %6;" ::"r"(dst),
      "l"(map), "r"(x), "r"(y), "r"(z), "r"(smem_u32(bar)), "l"(pol)
      : "memory");
}
__device__ __forceinline__ void l2_prefetch_3d(const CUtensorMap* map, int32_t x, int32_t y, int32_t z) {
  asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" ::"l"(map), "r"(x), "r"(y), "r"(z)
               : "memory");
}
__device__ __forceinline__ void l2_prefetch(const void* src, uint32_t bytes) {
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(src), "r"(bytes) : "memory");
}
__device__ __forceinline__ void bulk_load(uint32_t dst, const void* src, uint32_t bytes, uint64_t* bar) {
  asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst), "l"(src),
               "r"(bytes), "r"(smem_u32(bar))
               : "memory");
}
__device__ __forceinline__ void bulk_load_hint(uint32_t dst, const void* src, uint32_t bytes, uint64_t* bar, uint64_t pol) {
  asm volatile(
      "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;" ::"r"(dst),
      "l"(src), "r"(bytes), "r"(smem_u32(bar)), "l"(pol)
      : "memory");
}
__device__ __forceinline__ void prefetch_tensormap(const void* map) {
  asm volatile("prefetch.tensormap [%0];" ::"l"(map) : "memory");
}

// tcgen05 (PTX ISA, tensor core 5th generation)
__device__ __forceinline__ void tmem_alloc(uint32_t* dst_smem, uint32_t ncols) {
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(smem_u32(dst_smem)), "r"(ncols));
}
__device__ __forceinline__ void tmem_relinquish() {
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
}
__device__ __forceinline__ void tmem_dealloc(uint32_t taddr, uint32_t ncols) {
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(taddr), "r"(ncols));
}
__device__ __forceinline__ void tc_commit(uint64_t* bar) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void tc_fence_after() {
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}
__device__ __forceinline__ void tc_fence_before() {
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}
__device__ __forceinline__ void tc_wait_st() {
  asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ void tc_cp_32x128b_x4(uint32_t taddr, uint64_t sdesc) {
  asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" ::"r"(taddr), "l"(sdesc) : "memory");
}
__device__ __forceinline__ void tc_mma_nvf4(uint32_t d, uint64_t a, uint64_t b, uint32_t idesc, uint32_t sfa,
                                            uint32_t sfb, uint32_t accumulate) {
  asm volatile(
      "{ .reg .pred p; setp.ne.b32 p, %4, 0;\n"
      "  tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 [%0], %1, %2, %3, [%5], [%6], p; }" ::"r"(d),
      "l"(a), "l"(b), "r"(idesc), "r"(accumulate), "r"(sfa), "r"(sfb)
      : "memory");
}
// Shared-memory matrix descriptor, K-major, 128-byte swizzle, 8-row groups of 1024 bytes (SBO), version 1.
__device__ __forceinline__ uint64_t sdesc_sw128(uint32_t addr) {
  return uint64_t((addr & 0x3FFFFu) >> 4) | (uint64_t(1024u >> 4) << 32) | (uint64_t(1) << 46) | (uint64_t(2) << 61);
}
// Scale-factor source descriptor for tcgen05.cp: no swizzle, 8 x 16-byte rows per core matrix (SBO 128 bytes).
__device__ __forceinline__ uint64_t sdesc_sf(uint32_t addr) {
  return uint64_t((addr & 0x3FFFFu) >> 4) | (uint64_t(128u >> 4) << 32) | (uint64_t(1) << 46);
}
// Instruction descriptor, kind::mxf4nvf4: dense, E2M1 operands, K-major, UE4M3 scales, scale ids 0, M = 128, N = n.
__host__ __device__ constexpr uint32_t idesc_nvf4(uint32_t n) {
  return (1u << 7) | (1u << 10) | ((n >> 3) << 17) | ((128u >> 4) << 24);
}
__device__ __forceinline__ void sts_u64(uint32_t addr, uint64_t v) {
  asm volatile("st.shared.b64 [%0], %1;" ::"r"(addr), "l"(v) : "memory");
}
__device__ __forceinline__ void sts_u16(uint32_t addr, uint16_t v) {
  asm volatile("st.shared.u16 [%0], %1;" ::"r"(addr), "h"(v) : "memory");
}
__device__ __forceinline__ void sts_v4(uint32_t addr, uint4 v) {
  asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(addr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
__device__ __forceinline__ void sts_v2(uint32_t addr, uint2 v) {
  asm volatile("st.shared.v2.b32 [%0], {%1,%2};" ::"r"(addr), "r"(v.x), "r"(v.y) : "memory");
}
__device__ __forceinline__ uint4 lds_v4(uint32_t addr) {
  uint4 v;
  asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(addr) : "memory");
  return v;
}
__device__ __forceinline__ void st_release_cta_shared(int32_t* p, int v) {
  asm volatile("st.release.cta.shared::cta.b32 [%0], %1;" ::"r"(smem_u32(p)), "r"(v) : "memory");
}
__device__ __forceinline__ int ld_acquire_cta_shared(const int32_t* p) {
  int v;
  asm volatile("ld.acquire.cta.shared::cta.b32 %0, [%1];" : "=r"(v) : "r"(smem_u32(p)) : "memory");
  return v;
}
__device__ __forceinline__ uint32_t shfl_idx(uint32_t v, int src) {
  uint32_t r;
  asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=r"(r) : "r"(v), "r"(src));
  return r;
}
__device__ __forceinline__ void st_relaxed_u64(unsigned long long* p, unsigned long long v) {
  asm volatile("st.relaxed.gpu.global.b64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}
__device__ __forceinline__ unsigned long long ld_relaxed_u64(const unsigned long long* p) {
  unsigned long long v;
  asm volatile("ld.relaxed.gpu.global.b64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ ulonglong2 ld_relaxed_v2u64(const unsigned long long* p) {
  ulonglong2 v;
  asm volatile("ld.relaxed.gpu.global.v2.b64 {%0,%1}, [%2];" : "=l"(v.x), "=l"(v.y) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ unsigned long long ld_acquire_u64(const unsigned long long* p) {
  unsigned long long v;
  asm volatile("ld.acquire.gpu.global.b64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ int ld_relaxed_s32(const int32_t* p) {
  int v;
  asm volatile("ld.relaxed.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ ulonglong2 ld_ca_v2u64(const unsigned long long* p) {
  ulonglong2 v;
  asm volatile("ld.global.ca.v2.u64 {%0,%1}, [%2];" : "=l"(v.x), "=l"(v.y) : "l"(p));
  return v;
}
__device__ __forceinline__ uint32_t tmem_ld_x1(uint32_t taddr) {
  uint32_t r;
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];\n\ttcgen05.wait::ld.sync.aligned;" : "=r"(r) : "r"(taddr) : "memory");
  return r;
}
__device__ __forceinline__ unsigned ld_acquire_gpu_u32(const unsigned* p) {
  unsigned v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

// fp32 arithmetic with flush-to-zero (reproduces the stock results bitwise).
__device__ __forceinline__ float fmul_ftz(float a, float b) {
  float r;
  asm("mul.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float fadd_ftz(float a, float b) {
  float r;
  asm("add.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float ffma_ftz(float a, float b, float c) {
  float r;
  asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c));
  return r;
}
__device__ __forceinline__ float ex2_ftz(float x) {
  float r;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}
__device__ __forceinline__ float rcp_ftz(float x) {
  float r;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}
__device__ __forceinline__ uint32_t bf16_satfinite(float x) {
  uint32_t r;
  asm("cvt.rn.satfinite.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(0.0f), "f"(x));
  return r & 0xFFFFu;
}

// FC1 gated epilogue for one (gate, up) accumulator pair, with the stock operand clamp at its default limit.
__device__ __forceinline__ uint32_t swiglu_out(float acc_gate, float acc_up, float s_tok, float a_gate, float a_up) {
  constexpr float kLimit = 3.40282346638528859812e+38f;
  float x = fmul_ftz(acc_gate, s_tok);
  float u = fmul_ftz(acc_up, s_tok);
  const float x0 = x, u0 = u;
  x = x > kLimit ? kLimit : x;
  x = (x0 >= -kLimit || x0 != x0) ? x : -kLimit;
  u = u > kLimit ? kLimit : u;
  u = (u0 >= -kLimit || u0 != u0) ? u : -kLimit;
  const float k = fmul_ftz(a_gate, 1.4426950216293334961f);
  const float e = ex2_ftz(fmul_ftz(fmul_ftz(x, k), -1.0f));
  const float r = rcp_ftz(fadd_ftz(e, 1.0f));
  const float up = ffma_ftz(u, a_gate, 0.0f);
  return bf16_satfinite(fmul_ftz(fmul_ftz(up, fmul_ftz(x, r)), a_up));
}

// Byte offset of (row r, 16-byte chunk c) inside one 128-byte-swizzled K atom of rows.
__device__ __forceinline__ uint32_t sw128_off(int r, int c) {
  return uint32_t((r >> 3) * 1024 + (r & 7) * 128 + ((c ^ (r & 7)) << 4));
}


// routed slots whose token row one CTA copies (slots b, b + G, ...); the wide builds scale the table (m * 8 <= table x grid)
constexpr int kMaxMySlots = kMaxTokens / 16;
constexpr int kCntSet = 2 * kMaxPasses + 32;     // per launch parity: FC1 arrivals [kMaxPasses], FC2-operand rows
constexpr int kRowsReady = 2 * kMaxPasses;       // [kMaxPasses], rows-ready count of the grid (one per CTA)
constexpr int kChunkSlots = 256, kNumChunks = kMaxSlots / kChunkSlots;

struct Params {
  const int32_t* packed;     // [M, 8] (expert << 16 | bf16 weight)
  const uint8_t* hs;         // [M, 2048] FP4 token rows
  const uint8_t* hs_sf;      // [M, 256] e4m3, linear
  const float* pts;          // [M]
  const uint8_t* sf13;       // [P, 8192, 256]
  const uint8_t* sf2;        // [P, 4096, 256]
  const float* gate_alpha;
  const float* up_alpha;
  const float* out_alpha;
  const int8_t* sm_domain;
  uint8_t* hs_perm;          // [kMaxSlots, 2048] FC1 token rows, expert-major
  float* pts_perm;           // [kMaxSlots]
  uint8_t* img1;             // [kMaxPasses][64][512] FC1 token-scale images
  uint8_t* img2;             // [kMaxPasses][64][512] FC2 token-scale images
  unsigned long long* act;   // [kMaxSlots][kActWords]
  uint8_t* hs2;              // [kMaxSlots, 2048] FC2 token rows
  float* ramax;              // [kMaxSlots] FC2-input row maxima
  unsigned* cnt;             // [2][kCntSet] counters, one set per launch parity
  __nv_bfloat16* out;        // [M * 8, 4096]
  int32_t* e2p;              // [M * 8]
  uint16_t* expert_weights;  // [M * 8]
  unsigned* ticket;
  int32_t* err;
  int32_t m, pool, local_offset;
  int32_t n_dom0, n_dom1;
  float gsi;
  int32_t flags;
  int32_t dbg;               // ablation (timing studies only): 1 no MMAs, 128 no epilogue / builder roles, 256 no token
                             // copies, 512 no work items; 2048 / 4096 / 8192: tests of the verify build only (above)
  unsigned long long* stamps;  // [grid][kStampWords] per-CTA timestamps (timing builds only), or nullptr
#if MUSE_VR_LDW_RBAL
  // [1] real tokens of this call: the placement histogram counts the routed slots of tokens 0 .. n_real - 1 only (a CUDA-graph
  // replay pads the batch with rows whose token ids are left from earlier steps; they are still placed and computed)
  const int32_t* n_real;
#endif
#if MUSE_VR_LDW_ERRWORD
  uint32_t* page_err;        // the caller's error word (page-locked, device-mapped host memory), or nullptr
  uint32_t ew_tag;           // stored as bit 31 | ew_tag by a CTA whose wait expired or whose claim failed
#endif
};
constexpr int kFlagEvictFirst = 2, kFlagTailTrigger = 32;
constexpr int kStampWords = 24;
// Timing builds (-DMUSE_VR_LDW_STAMPS) include a stamp header that is not part of this package; production builds compile
// the stamps out.
#ifdef MUSE_VR_LDW_STAMPS
#include "ldw_stamps.cuh"
#else
#define LDW_STAMP(p, slot) ((void)0)
#define LDW_STAMPV(p, slot, v) ((void)0)
#endif
constexpr int kStStart = 0, kStPro = 1, kStCopy = 2, kStRows = 3, kStFirstFull = 4, kStFc1Issued = 5, kStFc2Dep = 6,
              kStAllIssued = 7, kStMma = 8, kStEpi = 9, kStBuild = 10, kStEnd = 11, kStEpiFc1 = 12, kStCopyRows = 13,
              kStCopyImg = 14, kStCopyBar = 15, kStItems = 16, kStCost = 17, kStSm = 18, kStCopyIn = 21, kStCopyLd = 22;
#if MUSE_VR_LDW_RBAL
constexpr int kStRbDone = 19;  // timing builds: the placement thresholds are written (the placement warp)
constexpr int kStRbDig = 23;   // any build given a stamps tensor: per CTA, thresholds (bits 0-31) and the winning key (32-63)
static_assert(kStRbDig < kStampWords, "digest word inside the stamps row");
#endif
#if MUSE_VR_LDW_PRE_VERIFY
constexpr int kStRedo = 20;  // any build given a stamps tensor: per CTA, the number of verify rebuilds (after the wait)
#endif

// ---------------------------------------------------------------- shared memory
struct Smem {
  alignas(1024) uint8_t st[kStages][kStageBytes];  // [weights 32K | weight scales 4K | token rows | token-scale images 4K]
  uint64_t full[kStages], empty[kStages], sf_full[kStages];
  uint64_t tmem_full[2], tmem_empty[2], drain_sf, drain_mma;
  alignas(16) uint16_t ostage[4][8][32];  // FC2 epilogue: 8 rows x 32 output columns per epilogue warp
  int16_t q_expert[kMaxPasses], q_row[kMaxPasses], q_rows[kMaxPasses];
  int16_t base[kMaxPool], qbase[kMaxPool];  // first permuted row / first pass of each local expert
  int16_t my_slot[kMaxMySlots], my_pos[kMaxMySlots], my_exp[kMaxMySlots];
  int16_t ptok[kHeadItems][kHeadRows];  // head copies: token of each row of the pass of FC1 item m < head_n
  int32_t head_n;                       // leading FC1 items whose token rows are copied by row index (0: none)
  uint32_t items[kMaxItems];
  int32_t n_items, n_fc1, n_pass, n_my, set;
  uint32_t fc1_stored[4];  // per FC1 epilogue warp: tiles whose activation words it stored (red.release / ld.acquire, cta)
  uint32_t tmem_base;
  unsigned start_base;
  unsigned slot;  // this CTA's claimed slot (domain << 7 | rank), set in the prologue
#if MUSE_VR_LDW_DENSE
  alignas(16) uint8_t tsf[kDenseRows * kTsfPitch];  // the call's token scales (rows 0 .. M - 1, pitch kTsfPitch), per CTA
  uint64_t tsf_full;                                     // their copy (complete_tx)
  uint32_t dmask[kMaxPasses];  // per pass: bit t = token t has a routed slot of the pass's expert computed here
#endif
};
static_assert(kStageBytes % 1024 == 0 && kStageA % 1024 == 0, "stage parts stay 1024-byte aligned");
// prologue scratch (the last ring stage, before its first copy): expert of each slot, per-chunk expert counts
struct Scratch {
  int16_t exp[kMaxSlots];
  uint32_t ccnt[kNumChunks][kMaxPool];
  uint32_t total[kMaxPool];
#if MUSE_VR_LDW_RBAL
  alignas(16) uint32_t hist[256];                                            // routed rows per expert (all routed slots)
  alignas(16) uint32_t rb_s[muse_vr_ring::kWorld * muse_vr_ring::kSStride];  // S_b(t) rows of the placement
  int32_t rb_thr[muse_vr_ring::kWorld];                                      // the step's thresholds
#endif
};
static_assert(sizeof(Scratch) <= kStageBytes, "prologue scratch fits one ring stage");

__device__ __forceinline__ uint32_t item_pack(int fc2, int q, int tile) {
  return uint32_t(q) | (uint32_t(tile) << 8) | (uint32_t(fc2) << 14);
}
struct Item {
  int fc2, q, tile;
};
__device__ __forceinline__ Item item_unpack(uint32_t v) {
  Item it;
  it.q = int(v & 0xFFu);
  it.tile = int((v >> 8) & 0x3Fu);
  it.fc2 = int((v >> 14) & 1u);
  return it;
}
__device__ __forceinline__ int steps_of(int units, int rank, int nr) {
  return units > rank ? (units - 1 - rank) / nr + 1 : 0;
}
// Bounded acquire poll of a counter (one lane); false when the bound ran out or the CTA aborted.
// Relaxed polls until the value is reached, then one acquire load (the acquire pattern of PTX ISA: the acquire reads
// the same location after the relaxed observations).
__device__ __forceinline__ bool acquire_at_least(const unsigned* c, unsigned want, int32_t* err) {
  uint32_t polls = 0;
  while (muse_vr_ldq::ld_relaxed_gpu(c) < want) {
    if (aborted()) return false;
    if (++polls > kMaxPolls) {
      fail(err, kErrPoll);
      return false;
    }
    __nanosleep(64);
  }
  while (ld_acquire_gpu_u32(c) < want) {  // counts only grow: the first acquire normally observes it
    if (aborted() || ++polls > kMaxPolls) {
      if (!aborted()) fail(err, kErrPoll);
      return false;
    }
  }
  return true;
}
// Two 16-lane x 16-column loads (lane bases t0 and t1), then one wait. Fragment of .16x256b (PTX ISA tcgen05 matrix
// fragments): thread t holds lane t / 4 (registers 0, 1) and lane t / 4 + 8 (registers 2, 3), columns 2 (t % 4) and
// 2 (t % 4) + 1; .x2 repeats it for the next 8 columns (registers 4..7).
__device__ __forceinline__ void tmem_ld_16x256b_x2_pair(uint32_t t0, uint32_t t1, uint32_t (&a)[8], uint32_t (&b)[8]) {
  asm volatile(
      "tcgen05.ld.sync.aligned.16x256b.x2.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%16];\n\t"
      "tcgen05.ld.sync.aligned.16x256b.x2.b32 {%8,%9,%10,%11,%12,%13,%14,%15}, [%17];\n\t"
      "tcgen05.wait::ld.sync.aligned;"
      : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]), "=r"(a[4]), "=r"(a[5]), "=r"(a[6]), "=r"(a[7]), "=r"(b[0]),
        "=r"(b[1]), "=r"(b[2]), "=r"(b[3]), "=r"(b[4]), "=r"(b[5]), "=r"(b[6]), "=r"(b[7])
      : "r"(t0), "r"(t1)
      : "memory");
}
__device__ __forceinline__ void tmem_ld_x4(uint32_t taddr, uint32_t (&r)[4]) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];\n\ttcgen05.wait::ld.sync.aligned;"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(taddr)
               : "memory");
}
__device__ __forceinline__ void tmem_ld_x16(uint32_t taddr, uint32_t (&r)[16]) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];\n\t"
      "tcgen05.wait::ld.sync.aligned;"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]),
        "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
      : "r"(taddr)
      : "memory");
}
__device__ __forceinline__ void tmem_ld_x8(uint32_t taddr, uint32_t (&r)[8]) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];\n\ttcgen05.wait::ld.sync.aligned;"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7])
               : "r"(taddr)
               : "memory");
}

// ---------------------------------------------------------------- roles
// warp 0: weight, weight-scale, token-row and token-scale copies, one stage at a time
__device__ MUSE_VR_LDW_ROLE void role_producer(Smem& s, const Params& p, const CUtensorMap* m13, const CUtensorMap* m2,
                                               const CUtensorMap* b1, const CUtensorMap* b2, const CUtensorMap* bd) {
  const bool leader = elect_one();
  const int lane = int(threadIdx.x & 31u), lead = __ffs(int(__ballot_sync(0xffffffffu, leader))) - 1;
  const int n_items = s.n_items;
  const bool ef = (p.flags & kFlagEvictFirst) != 0;
  const bool tok = (p.dbg & 256) == 0;
  const uint64_t pol = policy_evict_first();
  unsigned* const cset = p.cnt + s.set * kCntSet;
  if (n_items == 0 && (p.flags & kFlagTailTrigger)) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  int k = 0, issued = 0;
  bool rows_ready = false;
  bool tsf_ready = false;  // dense-token build: this CTA's copy of the token scales has landed
  (void)bd;
  (void)tsf_ready;
  const int head_n = MUSE_VR_LDW_GATHER ? s.head_n : 0;
  // the last ring stage held the prologue scratch (generic accesses, ordered before this point by the CTA barrier): a
  // proxy fence before any asynchronous copy into the ring (PTX ISA memory consistency model, proxies)
  fence_proxy_async_smem();
  // the weights of the first kStages stages go out at once (free slots); their token rows wait for the rows-ready count
  int pre = 0;
  if (tok && leader && !aborted()) {
    pre = min(kStages, n_items * kStagesPerTile);
#pragma unroll 1
    for (int x = 0; x < pre; ++x) {
      const Item it = item_unpack(s.items[x / kStagesPerTile]);
      const int e = s.q_expert[it.q], j = x % kStagesPerTile;
      const int tile_row = it.fc2 ? e * kFc2TilesPerExpert + it.tile : e * kFc1TilesPerExpert + it.tile;
      const uint8_t* sf = (it.fc2 ? p.sf2 : p.sf13) + (size_t(tile_row) * (kHidden / 64) + size_t(j) * kChunks) * 512;
      const uint32_t dst = smem_u32(&s.st[x][0]);
      mbar_expect_tx(&s.full[x], uint32_t(kStageA));
      if (ef) {
        tma_load_3d_hint(dst, it.fc2 ? m2 : m13, 0, tile_row * kTile, j * (kChunks / 4), &s.full[x], pol);
        bulk_load_hint(dst + kStageW, sf, uint32_t(kChunks * 512), &s.full[x], pol);
      } else {
        tma_load_3d(dst, it.fc2 ? m2 : m13, 0, tile_row * kTile, j * (kChunks / 4), &s.full[x]);
        bulk_load(dst + kStageW, sf, uint32_t(kChunks * 512), &s.full[x]);
      }
    }
    if (MUSE_VR_LDW_HEAD_PF > 0) {
      const int end = min(pre + MUSE_VR_LDW_HEAD_PF, n_items * kStagesPerTile);
#pragma unroll 1
      for (int x = pre; x < end; ++x) {
        const Item it = item_unpack(s.items[x / kStagesPerTile]);
        const int e = s.q_expert[it.q], j = x % kStagesPerTile;
        const int tile_row = it.fc2 ? e * kFc2TilesPerExpert + it.tile : e * kFc1TilesPerExpert + it.tile;
        l2_prefetch_3d(it.fc2 ? m2 : m13, 0, tile_row * kTile, j * (kChunks / 4));
        l2_prefetch((it.fc2 ? p.sf2 : p.sf13) + (size_t(tile_row) * (kHidden / 64) + size_t(j) * kChunks) * 512,
                    uint32_t(kChunks * 512));
      }
    }
  }
#pragma unroll 1
  for (int m = 0; m < n_items; ++m) {
    const Item it = item_unpack(s.items[m]);
    const int e = s.q_expert[it.q], rows = s.q_rows[it.q], qrow = s.q_row[it.q];
    const bool dn = MUSE_VR_LDW_DENSE && !it.fc2;  // dense-token FC1 item: every token row of the call
    const int bk = bucket_of(dn ? p.m : rows), nb = bucket_rows(bk);
    if (m + 1 == n_items && (p.flags & kFlagTailTrigger)) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    bool fc2_ready = !it.fc2 || (p.dbg & 128) != 0;
    const int tile_row = it.fc2 ? e * kFc2TilesPerExpert + it.tile : e * kFc1TilesPerExpert + it.tile;
    const uint8_t* sfw = it.fc2 ? p.sf2 : p.sf13;
    const CUtensorMap* wmap = it.fc2 ? m2 : m13;
    const CUtensorMap* bmap = (it.fc2 ? b2 : b1) + bk;
    const uint8_t* img = (it.fc2 ? p.img2 : p.img1) + size_t(it.q) * kImgBytes;
    const bool gat = m < head_n;  // head FC1 item: rows + scale image by row-index copies
    // dense FC1: the token rows come by tensor copy (tracked bytes), the scale image by this warp's shared-memory stores
    const uint32_t tok_tx = gat ? 0u : dn ? uint32_t(nb * 256) : uint32_t(nb * 256 + kStageImg);
#pragma unroll 1
    for (int j = 0; j < kStagesPerTile; ++j, ++k) {
      const int pos = k % kStages;
      const uint32_t use = uint32_t(k / kStages);
      if (k >= kStages) mbar_wait(&s.empty[pos], (use & 1u) ^ 1u, p.err);
      static_assert(!MUSE_VR_LDW_GATHER || kStages == 3, "the group arithmetic below assumes a 3-slot ring");
      if (MUSE_VR_LDW_GATHER && k >= kStages && k - kStages < head_n * kStagesPerTile)
        cp_async_wait_newer(min(head_n * kStagesPerTile, k) - (k - kStages) - 1);  // the group of stage k - 3
      if (gat && tok && __shfl_sync(0xffffffffu, int(issued == k && !aborted()), lead)) {
        // the stage's token rows straight from the token tensor (written by predecessor grids), in the layout the row map
        // gives them (K atom a of row i at a * nb * 128 + i * 128, 16-byte pieces in 128-byte swizzle), then their scale
        // image (the layout the copy phase writes to img1: chunk c at c * 512, row i at 16 (i % 32) + 4 (i / 32)), 4 bytes
        // per row and K chunk; every lane's copies are tracked by the stage's full barrier before the leader's arrival,
        // so the stage completes only once they landed. Rows at and above the pass's row count are not copied: their MMA
        // columns are never read.
        const uint32_t rb = smem_u32(&s.st[pos][kStageA]), im = smem_u32(&s.st[pos][kStageA + kStageB]);
        const int16_t* rt = s.ptok[m];
        const uint8_t* hsj = p.hs + size_t(j) * 256;
#pragma unroll 1
        for (int x = lane; x < rows * 16; x += 32) {
          const int i = x >> 4, a = (x >> 3) & 1, c = x & 7;
          cp_async_16(rb + uint32_t(a * nb * 128 + i * 128 + ((c ^ (i & 7)) << 4)),
                      hsj + size_t(rt[i]) * (kHidden / 2) + a * 128 + c * 16);
        }
#pragma unroll 1
        for (int x = lane; x < rows * kChunks; x += 32) {
          const int i = x >> 3, c = x & 7;  // 4 bytes of row i, K chunk c
          cp_async_4(im + uint32_t(c) * 512u + 16u * uint32_t(i & 31) + 4u * uint32_t(i >> 5),
                     p.hs_sf + size_t(rt[i]) * (kHidden / 16) + size_t(j) * 32 + 4 * c);
        }
        cp_async_arrive(&s.full[pos]);
        cp_async_commit();
        __syncwarp();
      }
#if MUSE_VR_LDW_DENSE
      // dense FC1 stage: the leader posts the stage's expected bytes and issues its copies first; the warp then builds
      // the stage's token-scale image (below) and the leader's arrival on the full barrier follows it, so the image work
      // overlaps the copies in flight instead of delaying their issue
      const bool go = __shfl_sync(0xffffffffu, int(issued == k && !aborted()), lead) != 0;  // warp-uniform
      const bool late = dn && tok && go;
      if (leader && go) {
#else
      if (leader && issued == k && !aborted()) {
#endif
        const uint32_t dst = smem_u32(&s.st[pos][0]);
        const uint8_t* sf = sfw + (size_t(tile_row) * (kHidden / 64) + size_t(j) * kChunks) * 512;
        if (k >= pre) {
#if MUSE_VR_LDW_DENSE
          if (late)
            mbar_expect_tx(&s.full[pos], uint32_t(kStageA) + tok_tx);  // the arrival follows the image below
          else
#endif
            mbar_arrive_expect_tx(&s.full[pos], uint32_t(kStageA) + (tok ? tok_tx : 0u));
          // weights first: they depend on nothing
          if (ef) {
            tma_load_3d_hint(dst, wmap, 0, tile_row * kTile, j * (kChunks / 4), &s.full[pos], pol);
            bulk_load_hint(dst + kStageW, sf, uint32_t(kChunks * 512), &s.full[pos], pol);
          } else {
            tma_load_3d(dst, wmap, 0, tile_row * kTile, j * (kChunks / 4), &s.full[pos]);
            bulk_load(dst + kStageW, sf, uint32_t(kChunks * 512), &s.full[pos]);
          }
        } else {
#if MUSE_VR_LDW_DENSE
          if (late)
            mbar_expect_tx(&s.full[pos], tok_tx);  // weights already out; the arrival follows the image below
          else
#endif
            mbar_arrive_expect_tx(&s.full[pos], tok ? tok_tx : 0u);  // weights already out (expect-tx above)
        }
#if MUSE_VR_LDW_DENSE
        if (tok && dn) {
          // dense FC1: rows 0 .. M - 1 of the token tensor (predecessor grids' output, ordered by the dependency wait in
          // the prologue, as the head-copy path reads it); rows past M are zero-filled by the copy
          tma_load_3d(dst + kStageA, bd + bk, 0, 0, j * (kChunks / 4), &s.full[pos]);
        }
#endif
        if (tok && !gat && !dn) {
          // token rows / images are stored by other CTAs with generic stores: acquire their counts, then a proxy fence
          // orders the asynchronous-proxy copies after them (FC1: the grid's rows-ready count; FC2: the pass's built rows)
          if (!MUSE_VR_LDW_DENSE && !rows_ready) {
            rows_ready = acquire_at_least(cset + kRowsReady, gridDim.x, p.err) || true;
            LDW_STAMP(p, kStRows);
          }
          if (!fc2_ready) fc2_ready = acquire_at_least(cset + kMaxPasses + it.q, unsigned(rows), p.err) || true;
          fence_proxy_async_global();
          tma_load_3d(dst + kStageA, bmap, 0, qrow, j * (kChunks / 4), &s.full[pos]);
          bulk_load(dst + kStageA + kStageB, img + size_t(j) * kStageImg, uint32_t(kStageImg), &s.full[pos]);
        }
        ++issued;
      }
#if MUSE_VR_LDW_DENSE
      if (late) {
        // the stage's token-scale image (the layout the copy phase writes to img1: chunk c at c * 512, row i at 16 i for
        // i < 32; a region no copy above writes), from this CTA's copy of the call's token scales: lane i takes row i's
        // 32 bytes of the stage's 8 K chunks by two 16-byte reads and writes their 4-byte words (rows at and above M get
        // 0; their MMA columns are never stored). The slot's previous image was read by its tensor-memory copy before
        // the MMAs that freed the slot (empty barrier above). Visible to the scale-copy warp through the leader's arrival
        // below (after __syncwarp); that warp fences the proxies before its tensor-memory copies.
        static_assert(kChunks == 8, "two 16-byte reads per row and stage");
        if (!tsf_ready) {
          mbar_wait(&s.tsf_full, 0u, p.err);
          tsf_ready = true;
        }
        uint4 a = make_uint4(0u, 0u, 0u, 0u), b = a;
        if (lane < p.m) {
          const uint4* row = reinterpret_cast<const uint4*>(&s.tsf[size_t(lane) * kTsfPitch + size_t(j) * (kChunks * 4)]);
          a = row[0];
          b = row[1];
        }
        const uint32_t w[kChunks] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
        uint8_t* im = &s.st[pos][kStageA + kStageB];
#pragma unroll
        for (int c = 0; c < kChunks; ++c) *reinterpret_cast<uint32_t*>(im + c * 512 + 16 * lane) = w[c];
        __syncwarp();
        if (leader) mbar_arrive(&s.full[pos]);  // release: the image stores (ordered by the warp sync above)
      }
#endif
      __syncwarp();
    }
  }
  if (leader) LDW_STAMP(p, kStAllIssued);
#if MUSE_VR_LDW_DENSE
  // the token-scale copy issued in the prologue has landed before the CTA can exit, whether or not an image read it
  if (tok && !tsf_ready) mbar_wait(&s.tsf_full, 0u, p.err);
#endif
  if (leader && aborted()) {
    // pre-issued weight stages without their arrival: arrive, so the phase completes once the weights land
#pragma unroll 1
    for (int kk = issued; kk < pre; ++kk) mbar_arrive(&s.full[kk]);
    const int out = issued > pre ? issued : pre;
#pragma unroll 1
    for (int kk = out > kStages ? out - kStages : 0; kk < out; ++kk)
      mbar_drain(&s.full[kk % kStages], uint32_t(kk / kStages) & 1u, p.err);
  }
  __syncwarp();
}

// warp 2: weight scales and token scales, shared memory -> tensor memory (one region of 32 columns each per ring stage)
__device__ MUSE_VR_LDW_ROLE void role_sfcopy(Smem& s, const Params& p) {
  const bool leader = elect_one();
  const int n_items = s.n_items;
  const uint32_t tbase = s.tmem_base;
  const int head_n = MUSE_VR_LDW_GATHER ? s.head_n : 0;
  int k = 0;
#pragma unroll 1
  for (int m = 0; m < n_items; ++m) {
#if MUSE_VR_LDW_DENSE
    const bool dn = !item_unpack(s.items[m]).fc2;  // dense-token FC1 item: image stored by the producer warp
#else
    constexpr bool dn = false;
#endif
#pragma unroll 1
    for (int j = 0; j < kStagesPerTile; ++j, ++k) {
      const int pos = k % kStages;
      mbar_wait(&s.full[pos], uint32_t(k / kStages) & 1u, p.err);
      if (k == 0 && leader) LDW_STAMP(p, kStFirstFull);
      if (k >= kStages) mbar_wait(&s.empty[pos], (uint32_t(k / kStages) & 1u) ^ 1u, p.err);
      // head stages: the token-scale image was written by cp.async (generic proxy), made visible to this thread by the
      // full barrier; a proxy fence orders it before the tensor-memory copies (asynchronous proxy) that read it.
      // Dense-token FC1 stages: the same, for the producer warp's shared-memory stores of the image
      if (m < head_n || dn) fence_proxy_async_smem();
      tc_fence_after();
      if (leader) {
        if (!aborted()) {
          const uint32_t ca = tbase + kSfaCol + uint32_t(pos) * 32u, cb = tbase + kSfbCol + uint32_t(pos) * 32u;
          const uint32_t sa = smem_u32(&s.st[pos][kStageW]), sb = smem_u32(&s.st[pos][kStageA + kStageB]);
#pragma unroll 1
          for (int c = 0; c < kChunks; ++c) {
            tc_cp_32x128b_x4(ca + 4u * c, sdesc_sf(sa + uint32_t(c) * 512u));
            tc_cp_32x128b_x4(cb + 4u * c, sdesc_sf(sb + uint32_t(c) * 512u));
          }
        }
        tc_commit(&s.sf_full[pos]);
      }
      __syncwarp();
    }
  }
  if (leader) tc_commit(&s.drain_sf);
  __syncwarp();
  mbar_drain(&s.drain_sf, 0u, p.err);
}

// warp 1: MMA issue (N = the pass's token-operand height)
__device__ MUSE_VR_LDW_ROLE void role_mma(Smem& s, const Params& p) {
  const bool leader = elect_one();
  const uint32_t tbase = s.tmem_base;
  const int n_items = s.n_items;
  int k = 0;
#pragma unroll 1
  for (int m = 0; m < n_items; ++m) {
    const Item it = item_unpack(s.items[m]);
    // dense-token FC1 items: N = the call's rows rounded up to the bucket (every token column)
    const int nb = bucket_rows(bucket_of((MUSE_VR_LDW_DENSE && !it.fc2) ? p.m : int(s.q_rows[it.q])));
    const uint32_t idesc = idesc_nvf4(uint32_t(nb));
    const bool gat = MUSE_VR_LDW_GATHER && m < s.head_n;
    const int acc = m & 1;
    const uint32_t aph = uint32_t(m >> 1) & 1u;
    if ((p.dbg & 128) == 0 && m >= 2) mbar_wait(&s.tmem_empty[acc], aph ^ 1u, p.err);
    tc_fence_after();
    const uint32_t d = tbase + kAccCol + uint32_t(acc) * 128u;
#pragma unroll 1
    for (int j = 0; j < kStagesPerTile; ++j, ++k) {
      const int pos = k % kStages;
      mbar_wait(&s.sf_full[pos], uint32_t(k / kStages) & 1u, p.err);
      if (gat) {
        // token rows written by cp.async (generic proxy): visible to this thread through the full barrier, then a proxy
        // fence before the MMAs (asynchronous proxy) read them
        mbar_wait(&s.full[pos], uint32_t(k / kStages) & 1u, p.err);
        fence_proxy_async_smem();
      }
      tc_fence_after();
      if (leader) {
        if ((p.dbg & 1) == 0 && !aborted()) {
          const uint64_t da0 = sdesc_sw128(smem_u32(&s.st[pos][0]));
          const uint64_t db0 = sdesc_sw128(smem_u32(&s.st[pos][kStageA]));
          const uint32_t sfa0 = tbase + kSfaCol + uint32_t(pos) * 32u, sfb0 = tbase + kSfbCol + uint32_t(pos) * 32u;
#if MUSE_VR_LDW_K128
          // K = 128 per MMA (instruction descriptor K field), two 64-wide chunks per instruction
#pragma unroll
          for (int c = 0; c < kChunks; c += 2) {
            const uint64_t da = da0 + uint64_t((uint32_t(c >> 2) * 16384u + uint32_t(c & 3) * 32u) >> 4);
            const uint64_t db = db0 + uint64_t((uint32_t(c >> 2) * uint32_t(nb * 128) + uint32_t(c & 3) * 32u) >> 4);
            tc_mma_nvf4(d, da, db, idesc | (1u << 3), sfa0 + 4u * c, sfb0 + 4u * c, (j | c) != 0 ? 1u : 0u);
          }
#else
#pragma unroll
          for (int c = 0; c < kChunks; ++c) {
            const uint64_t da = da0 + uint64_t((uint32_t(c >> 2) * 16384u + uint32_t(c & 3) * 32u) >> 4);
            const uint64_t db = db0 + uint64_t((uint32_t(c >> 2) * uint32_t(nb * 128) + uint32_t(c & 3) * 32u) >> 4);
            tc_mma_nvf4(d, da, db, idesc, sfa0 + 4u * c, sfb0 + 4u * c, (j | c) != 0 ? 1u : 0u);
          }
#endif
        }
        tc_commit(&s.empty[pos]);
      }
      __syncwarp();
    }
    if (leader) tc_commit(&s.tmem_full[acc]);
    __syncwarp();
  }
  if (leader) tc_commit(&s.drain_mma);
  __syncwarp();
  mbar_drain(&s.drain_mma, 0u, p.err);
  if (leader) LDW_STAMP(p, kStMma);
}

// Warps 4-11 (epilogue + builder warps, idle until the first accumulator / FC1 count): this CTA's share of the permuted token
// rows (slots b, b + G, ...): FP4 rows in 16-byte pieces with several loads in flight per thread, per-token scales,
// token-scale image entries, the expanded-index map and top-k weights of those slots; then one release of the grid's
// rows-ready count on behalf of the eight warps.
__device__ __noinline__ void copy_rows(Smem& s, const Params& p, int wt) {  // one shared copy of this cold code
  constexpr int kW = 256;  // worker threads
  const int n_my = s.n_my;
  // One warp per copied row (rolled loop: this code runs once per launch):
  // lane l moves row pieces l, l + 32, l + 64, l + 96 and token-scale piece l (< 16) of the row, then writes the 4
  // image words of that piece; the id / top-k weight / per-token scale of slot wt are loaded alongside.
  const int lane = wt & 31, wi = wt >> 5;
  const bool tail = wt < n_my;
  const int sl_t = tail ? s.my_slot[wt] : 0;
  const int32_t pk_t = tail ? p.packed[sl_t] : 0;
  const float pt_t = tail ? p.pts[sl_t / kTopK] : 0.0f;
  if (wt == 0) LDW_STAMP(p, kStCopyIn);
  // dense-token build: FC1 reads the token tensor itself, so no permuted rows, images or per-row scales are copied and no
  // rows-ready count is released (nothing acquires it); the expanded-index map and top-k weights below are written as before
#pragma unroll 1
  for (int j = wi; j < (MUSE_VR_LDW_DENSE ? 0 : n_my); j += kW / 32) {
    const int pos = s.my_pos[j];
    if (pos < 0) continue;  // warp-uniform
    const int t = s.my_slot[j] / kTopK;
    const uint4* src = reinterpret_cast<const uint4*>(p.hs + size_t(t) * (kHidden / 2));
    uint4 v[4];
#pragma unroll
    for (int u = 0; u < 4; ++u) v[u] = src[lane + 32 * u];
    uint4 f = make_uint4(0u, 0u, 0u, 0u);
    if (lane < 16) f = reinterpret_cast<const uint4*>(p.hs_sf + size_t(t) * (kHidden / 16))[lane];
    uint4* dst = reinterpret_cast<uint4*>(p.hs_perm + size_t(pos) * (kHidden / 2));
#pragma unroll
    for (int u = 0; u < 4; ++u) dst[lane + 32 * u] = v[u];
    if (lane < 16) {
      // image words of K chunks g = 4 lane .. 4 lane + 3 (one per 64-wide chunk)
      const int e = s.my_exp[j];
      const int r = pos - s.base[e], q = s.qbase[e] + r / kPassRows, i = r % kPassRows;
      uint8_t* img = p.img1 + size_t(q) * kImgBytes + size_t(4 * lane) * 512 + size_t(16 * (i & 31) + 4 * (i >> 5));
      *reinterpret_cast<uint32_t*>(img) = f.x;
      *reinterpret_cast<uint32_t*>(img + 512) = f.y;
      *reinterpret_cast<uint32_t*>(img + 1024) = f.z;
      *reinterpret_cast<uint32_t*>(img + 1536) = f.w;
    }
  }
  if (wt == 0) LDW_STAMP(p, kStCopyRows);
  if (tail) {
    const int pos = s.my_pos[wt];
    p.e2p[sl_t] = pos;
    p.expert_weights[sl_t] = uint16_t(pk_t & 0xFFFF);
    if (!MUSE_VR_LDW_DENSE && pos >= 0) p.pts_perm[pos] = pt_t;
  }
  if (wt == 0) LDW_STAMP(p, kStCopyImg);
  if (MUSE_VR_LDW_DENSE) return;
  named_bar_sync(7, kW);  // every worker's stores, before one release on their behalf
  if (wt == 0) LDW_STAMP(p, kStCopyBar);
  if (wt == 0) {
    asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(p.cnt + s.set * kCntSet + kRowsReady) : "memory");
    LDW_STAMP(p, kStCopy);
  }
}

// warps 4-7: epilogue (TMEM lane quadrant = warp % 4): 16 (FC1) / 8 (FC2) token columns per tensor-memory load
__device__ MUSE_VR_LDW_ROLE void role_epilogue(Smem& s, const Params& p) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int q = warp & 3;
  const bool leader = elect_one();
  const uint32_t tbase = s.tmem_base;
  const int n_items = s.n_items;
  const float gsi = p.gsi;
  unsigned* const cset = p.cnt + s.set * kCntSet;
  const int head_n = MUSE_VR_LDW_GATHER ? s.head_n : 0;
  copy_rows(s, p, threadIdx.x - 4 * 32);
  if (p.dbg & 128) return;
  // per-token scales of FC1 rows come from other CTAs' row copies (their release of the rows-ready count); head items
  // read them by token (pts, from predecessor grids), so with head copies the count is acquired before the first other
  // item
  // (dense-token build: the per-token scales come from predecessor grids, no count is acquired)
  bool need_rows = !MUSE_VR_LDW_DENSE && n_items > 0;
  if (head_n == 0 && need_rows) {
    if (lane == 0) acquire_at_least(cset + kRowsReady, gridDim.x, p.err);
    need_rows = false;
  }
  __syncwarp();
#pragma unroll 1
  for (int m = 0; m < n_items; ++m) {
    const int acc = m & 1;
    const uint32_t aph = uint32_t(m >> 1) & 1u;
    const Item it = item_unpack(s.items[m]);
#if MUSE_VR_SLOT_TEST == 3
    if (!it.fc2 && q == 3)  // tests only: warp skew on FC1 tiles
      for (int z = 0; z < 64; ++z) __nanosleep(1000);
#endif
    const int e = s.q_expert[it.q], rows = s.q_rows[it.q], qrow = s.q_row[it.q];
    if (need_rows && m >= head_n) {
      if (lane == 0) acquire_at_least(cset + kRowsReady, gridDim.x, p.err);
      need_rows = false;
    }
    if (it.fc2 && lane == 0) acquire_at_least(cset + kMaxPasses + it.q, unsigned(rows), p.err);  // the builders' maxima
    __syncwarp();
    mbar_wait_idle(&s.tmem_full[acc], aph, p.err);
    tc_fence_after();
    const uint32_t tacc = tbase + (uint32_t(q * 32) << 16) + kAccCol + uint32_t(acc) * 128u;
    if (!it.fc2) {
      // FC1: SwiGLU (stock arithmetic) -> activation words of 4 bf16, one routed row per column
      const float a_gate = p.gate_alpha[e], a_up = p.up_alpha[e];
      // TMEM lane a + 8 g + 16 b of this quadrant (a < 8): g = 0 the up row, g = 1 the gate row of intermediate column
      // 16 q + 2 a + b of this tile. The 16x256b fragment gives thread t both rows of a = t / 4 for tokens 2 (t % 4) and
      // 2 (t % 4) + 1 (+ 8 for the .x2 half); the loads at b = 0 and b = 1 give columns 2 a and 2 a + 1, stored as one
      // 32-bit word per token.
      const int ta = lane >> 2, tx = 2 * (lane & 3);
      uint32_t* dst = reinterpret_cast<uint32_t*>(p.act + size_t(qrow) * kActWords + size_t(it.tile) * 16) + (8 * q + ta);
      const uint32_t lb = tbase + (uint32_t(q * 32) << 16) + kAccCol + uint32_t(acc) * 128u;
      // dense-token build: column x = token x of the call; the routed columns of this pass's expert (computed here) go to
      // the pass's rows in token order (the permuted form's expert-major order), the others are not stored
#if MUSE_VR_LDW_DENSE
      const int ncol = p.m;
      const uint32_t dm = s.dmask[it.q];
#else
      const int ncol = rows;
#endif
#pragma unroll 1
      for (int x0 = 0; x0 < ncol; x0 += 16) {
#if MUSE_VR_LDW_DENSE && !MUSE_VR_LDW_TEST_DENSE_NOMASK
        if (((dm >> x0) & 0xFFFFu) == 0u) continue;  // no routed column of this pass in these 16 (warp-uniform)
#endif
        uint32_t v0[8], v1[8];
        tmem_ld_16x256b_x2_pair(lb + uint32_t(x0), lb + (16u << 16) + uint32_t(x0), v0, v1);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
#pragma unroll
          for (int c = 0; c < 2; ++c) {
            const int x = x0 + 8 * h + tx + c;
#if MUSE_VR_LDW_DENSE && MUSE_VR_LDW_TEST_DENSE_NOMASK
            if (x < rows) {  // tests only: the planted defect (column x stored at row x, routed or not)
              const int r = x;
              const float s_tok = p.pts[x];
              (void)dm;
#elif MUSE_VR_LDW_DENSE
            if (x < ncol && ((dm >> x) & 1u)) {
              const int r = __popc(dm & ((1u << x) - 1u));  // the token's row in the pass
              const float s_tok = p.pts[x];
#else
            if (x < rows) {
              const int r = x;
              const float s_tok = m < head_n ? p.pts[s.ptok[m][x]] : p.pts_perm[qrow + x];
#endif
              uint32_t y0 = swiglu_out(__uint_as_float(v0[4 * h + 2 + c]), __uint_as_float(v0[4 * h + c]), s_tok, a_gate, a_up);
              uint32_t y1 = swiglu_out(__uint_as_float(v1[4 * h + 2 + c]), __uint_as_float(v1[4 * h + c]), s_tok, a_gate, a_up);
              y0 = (y0 & 0x7FFFu) > 0x7F80u ? 0x7FFFu : y0;  // canonical NaN
              y1 = (y1 & 0x7FFFu) > 0x7F80u ? 0x7FFFu : y1;
              asm volatile("st.relaxed.gpu.global.b32 [%0], %1;" ::"l"(dst + size_t(r) * (kActWords * 2)), "r"(y0 | (y1 << 16))
                           : "memory");
            }
          }
        }
      }
      tc_fence_before();
      __syncwarp();
      if (leader) {
        mbar_arrive(&s.tmem_empty[acc]);
        asm volatile("red.release.cta.shared::cta.add.u32 [%0], 1;" ::"r"(smem_u32(&s.fc1_stored[MUSE_VR_SLOT_UNSAFE ? 0 : q]))
                     : "memory");
        if (q == 0 && m + 1 == s.n_fc1) LDW_STAMP(p, kStEpiFc1);
      }
    } else {
      // FC2: output rows (stock epilogue arithmetic); 8 rows staged per warp, then one 16-byte store per lane
      const float a_out = p.out_alpha[e];
      const uint32_t col = uint32_t((lane & 7) * 4 + (lane >> 3));
      const uint32_t st0 = smem_u32(&s.ostage[q][0][0]);
      uint8_t* dst = reinterpret_cast<uint8_t*>(p.out) + (size_t(it.tile) * kTile + size_t(32 * q)) * 2;
#pragma unroll 1
      for (int x0 = 0; x0 < rows; x0 += 8) {
        uint32_t v[8];
        tmem_ld_x8(tacc + uint32_t(x0), v);
        const float ra = lane < 8 && x0 + lane < rows ? p.ramax[qrow + x0 + lane] : 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const float pts2 = fmul_ftz(__shfl_sync(0xffffffffu, ra, i), gsi);
          sts_u16(st0 + uint32_t(i * 64) + col * 2u,
                  uint16_t(bf16_satfinite(fmul_ftz(fmul_ftz(__uint_as_float(v[i]), pts2), a_out))));
        }
        __syncwarp();
        const int i = lane >> 2;  // row of this lane's 16 bytes
        if (x0 + i < rows)
          *reinterpret_cast<uint4*>(dst + size_t(qrow + x0 + i) * kHidden * 2 + size_t(lane & 3) * 16) =
              lds_v4(st0 + uint32_t(i * 64) + uint32_t(lane & 3) * 16u);
        __syncwarp();
      }
      tc_fence_before();
      __syncwarp();
      if (leader) mbar_arrive(&s.tmem_empty[acc]);
    }
  }
  if (q == 0 && leader) LDW_STAMP(p, kStEpi);
}

// |max| over one 16-element block (two 16-byte halves of bf16)
__device__ __forceinline__ float block_absmax(ulonglong2 a, ulonglong2 b) {
  const uint32_t w[8] = {uint32_t(a.x), uint32_t(a.x >> 32), uint32_t(a.y), uint32_t(a.y >> 32),
                         uint32_t(b.x), uint32_t(b.x >> 32), uint32_t(b.y), uint32_t(b.y >> 32)};
  float t[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) t[i] = fmaxf(fabsf(__uint_as_float(w[i] << 16)), fabsf(__uint_as_float(w[i] & 0xFFFF0000u)));
#pragma unroll
  for (int st = 4; st > 0; st >>= 1)
#pragma unroll
    for (int i = 0; i < st; ++i) t[i] = fmaxf(t[i], t[i + st]);
  return t[0];
}
// FlashInfer's per-token NVFP4 quantization of one 16-element bf16 block: 16 e2m1 values + the block's e4m3 scale byte
__device__ __forceinline__ uint64_t quant16(ulonglong2 a, ulonglong2 b, float enc, float amax, uint8_t* sf) {
  using PV = tensorrt_llm::kernels::PackedVec<__nv_bfloat16, 16>;
  const uint32_t w8[8] = {uint32_t(a.x), uint32_t(a.x >> 32), uint32_t(a.y), uint32_t(a.y >> 32),
                          uint32_t(b.x), uint32_t(b.x >> 32), uint32_t(b.y), uint32_t(b.y >> 32)};
  PV vec;
#pragma unroll
  for (int x = 0; x < 8; ++x)
    vec.elts[x] = __halves2bfloat162(__ushort_as_bfloat16(uint16_t(w8[x] & 0xFFFFu)), __ushort_as_bfloat16(uint16_t(w8[x] >> 16)));
  return tensorrt_llm::kernels::cvt_warp_fp16_to_fp4<__nv_bfloat16, 16, 16, false, false, std::false_type>(vec, enc, sf, amax);
}

// warps 8-11: FC2 token rows. The rows this CTA builds (for every FC2 tile c of pass q it holds: rows c, c + 32, c + 64,
// c + 96 that exist) are dealt round-robin over the four warps in item order; each waits for the pass's FC1 tiles to be
// published (FC1 never waits on FC2, so the waits are acyclic), loads the row once, quantizes it with FlashInfer's
// per-token recipe, stores the FP4 row, its token-scale image entries and its row maximum, and releases one count.
__device__ MUSE_VR_LDW_ROLE void role_builder(Smem& s, const Params& p) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int w = warp & 3;
  const float gsi = p.gsi;
  unsigned* const cset = p.cnt + s.set * kCntSet;
  copy_rows(s, p, threadIdx.x - 4 * 32);
  if (p.dbg & 128) return;
  const int n_items = s.n_items, n_fc1 = s.n_fc1;
  int duty = 0;
#pragma unroll 1
  for (int m = n_fc1; m < n_items; ++m) {
    const Item it = item_unpack(s.items[m]);
    const int rows = s.q_rows[it.q], qrow = s.q_row[it.q];
#pragma unroll 1
    for (int i = it.tile % kFc2TilesPerExpert; i < rows; i += 32, ++duty) {
      if ((duty & 3) != w) continue;  // warp-uniform
      if (lane == 0) {
        const unsigned* c1 = cset + it.q;
        uint32_t polls = 0;
        bool ok = true;
        while (muse_vr_ldq::ld_relaxed_gpu(c1) < unsigned(kFc1Arrivals - 16)) {
          if (aborted() || ++polls > kMaxPolls) {
            ok = false;
            break;
          }
          __nanosleep(64);
        }
        while (ok && ld_acquire_gpu_u32(c1) < unsigned(kFc1Arrivals)) {
          if (aborted() || ++polls > kMaxPolls) {
            ok = false;
            break;
          }
          __nanosleep(32);
        }
        if (!ok && !aborted()) fail(p.err, kErrPoll);
      }
      __syncwarp();  // memory ordering among the warp's threads: the row loads follow lane 0's acquire
      const unsigned long long* row = p.act + size_t(qrow + i) * kActWords;
      // row maximum with MoE-LD's reduction structure (its builder warp w' covers blocks 64 w' .. 64 w' + 63 = this
      // lane's blocks lane + 64 w' and lane + 64 w' + 32): per quarter an integer warp max of fmaxf over the lane's two
      // blocks (|x| >= 0: the bit pattern orders like the value), then fmaxf over the quarters. All loads first.
      float bm[8];
#pragma unroll
      for (int b = 0; b < 8; ++b) {
        const unsigned long long* r = row + size_t(lane + 32 * b) * 4;
        bm[b] = block_absmax(ld_ca_v2u64(r), ld_ca_v2u64(r + 2));
      }
      float qmax[4];
#pragma unroll
      for (int qq = 0; qq < 4; ++qq)
        qmax[qq] = __uint_as_float(__reduce_max_sync(0xffffffffu, __float_as_uint(fmaxf(bm[2 * qq], bm[2 * qq + 1]))));
      const float amax = fmaxf(fmaxf(qmax[0], qmax[1]), fmaxf(qmax[2], qmax[3]));
      const float pts2 = fmul_ftz(amax, gsi);
      const float enc = pts2 != 0.0f ? tensorrt_llm::kernels::reciprocal_approximate_ftz(pts2) : 0.0f;
      uint8_t* frow = p.hs2 + size_t(qrow + i) * (kHidden / 2);
      uint8_t* img = p.img2 + size_t(it.q) * kImgBytes + size_t(16 * (i & 31) + 4 * (i >> 5));
      // quantization pass kept rolled: the row's blocks are reloaded
#pragma unroll 1
      for (int b = 0; b < 8; ++b) {
        const int blk = lane + 32 * b;
        const unsigned long long* r = row + size_t(blk) * 4;
        uint8_t sf8;
        const uint64_t q8 = quant16(ld_ca_v2u64(r), ld_ca_v2u64(r + 2), enc, amax, &sf8);
        *reinterpret_cast<uint64_t*>(frow + size_t(blk) * 8) = q8;
        // the four scale bytes of K chunk blk / 4 sit on lanes 4 k .. 4 k + 3: lane 4 k assembles the image word
        const uint32_t v = sf8;
        const uint32_t v1 = __shfl_down_sync(0xffffffffu, v, 1), v2 = __shfl_down_sync(0xffffffffu, v, 2),
                       v3 = __shfl_down_sync(0xffffffffu, v, 3);
        if ((lane & 3) == 0)
          *reinterpret_cast<uint32_t*>(img + size_t(blk >> 2) * 512) = v | (v1 << 8) | (v2 << 16) | (v3 << 24);
      }
      if (lane == 0) p.ramax[qrow + i] = amax;
      __syncwarp();  // the warp's stores before lane 0's release
      if (lane == 0) asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(cset + kMaxPasses + it.q) : "memory");
    }
  }
  if (w == 0 && lane == 0) LDW_STAMP(p, kStBuild);
}

// warp 3 after the prologue: FC1 publication per tile, residency exit wait
__device__ MUSE_VR_LDW_ROLE void role_ctrl(Smem& s, const Params& p) {
  const int lane = threadIdx.x & 31;
  unsigned* const cset = p.cnt + s.set * kCntSet;
  if ((p.dbg & 128) == 0 && lane == 0) {
#pragma unroll 1
    for (int mm = 0; mm < s.n_fc1; ++mm) {
      uint32_t polls = 0, done = 0;
      bool ok = true;
      while (true) {
        #if MUSE_VR_SLOT_UNSAFE
        // previous form: one CTA-wide count (assumes the four epilogue warps finish FC1 tiles in step)
        asm volatile("ld.relaxed.cta.shared::cta.u32 %0, [%1];" : "=r"(done) : "r"(smem_u32(&s.fc1_stored[0])) : "memory");
        if (done >= uint32_t(4 * (mm + 1))) break;
        #else
        // tile mm is stored once each of the four epilogue warps counted it (exact whatever their skew; the fence below
        // then acquires all four warps' releases)
        done = 0xFFFFFFFFu;
        #pragma unroll
        for (int w4 = 0; w4 < 4; ++w4) {
          uint32_t d;
          asm volatile("ld.relaxed.cta.shared::cta.u32 %0, [%1];" : "=r"(d) : "r"(smem_u32(&s.fc1_stored[w4])) : "memory");
          done = d < done ? d : done;
        }
        if (done >= uint32_t(mm + 1)) break;
        #endif
        if (aborted()) {
          ok = false;
          break;
        }
        if (++polls > kMaxPolls) {
          fail(p.err, kErrMbar);
          ok = false;
          break;
        }
        __nanosleep(64);
      }
      if (!ok) break;
      asm volatile("fence.acq_rel.cta;" ::: "memory");  // acquire pattern (cta)
      const Item it = item_unpack(s.items[mm]);
      asm volatile("red.release.gpu.global.add.u32 [%0], 4;" ::"l"(cset + it.q) : "memory");
    }
  }
  __syncwarp();
#if !MUSE_VR_SLOT_UNSAFE && !MUSE_VR_SLOT_EPOCH
  // this CTA's claim (prologue, before named barrier 3) is complete: the last CTA checks and clears the claim bitmap
  if (lane == 0) muse_vr_slot::finish(p.ticket + kClaimWord, p.err);
#endif
  if (lane == 0 && !aborted()) {
    const unsigned cycle = gridDim.x * kStartCycle;
    uint32_t polls = 0;
    for (;;) {
      const unsigned v = muse_vr_ldq::ld_relaxed_gpu(p.ticket);
      const unsigned d = v >= s.start_base ? v - s.start_base : v + cycle - s.start_base;
      if (d >= gridDim.x) break;
      if (aborted()) break;
      if (++polls > kMaxPolls) {
        fail(p.err, kErrStart);
        break;
      }
      __nanosleep(256);
    }
  }
  __syncwarp();
}

// Prologue, every thread of the CTA: counting the local routed slots per expert (per 256-slot chunk), expert bases and
// passes, this CTA's work list (warp 0) and the permuted positions of the slots whose rows this CTA copies (warps 8-11).
// The scratch lives in the last ring stage, which no copy targets before the work list is published (named barrier 3).
#if MUSE_VR_LDW_PRE
// (early prologue: the weight tensor maps, for the L2 prefetch of the hinted slot's first ring stages before the wait)
__device__ MUSE_VR_LDW_ROLE void prologue(Smem& s_arg, const Params& p, const CUtensorMap* m13, const CUtensorMap* m2) {
  // The caller's shared-memory block, rebuilt as an offset into the dynamic shared array (the same address), so the
  // prologue's shared tables are addressed in the shared space: none of its stores or atomics can reach global memory.
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  Smem& s = *reinterpret_cast<Smem*>(smem_raw + (smem_u32(&s_arg) - smem_u32(smem_raw)));
#else
__device__ MUSE_VR_LDW_ROLE void prologue(Smem& s, const Params& p) {
#endif
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  Scratch& x = *reinterpret_cast<Scratch*>(&s.st[kStages - 1][0]);
  const int nslot = p.m * kTopK;
  // Start ticket (residency, counter parity) and slot claim: both are atomic RMWs on control words that the previous
  // launch of this call site updates (its last CTA clears the claim bitmap in finish()). They are issued right AFTER
  // griddepcontrol.wait: under programmatic dependent launch the wait is the documented edge that orders them after the
  // completion of every prerequisite grid (PTX ISA, griddepcontrol), so the previous launch's clear is ordered before
  // this launch's claim. Issued before the routed-ids loads below, so their latency still overlaps those loads and the
  // ccnt zeroing. -DMUSE_VR_LDW_TEST_PREWAIT_CLAIM=1 (tests only: the negative control of tests/test_moe_ldw_order.py)
  // keeps the earlier pre-wait placement.
  unsigned ticket = 0;
#if !MUSE_VR_SLOT_UNSAFE && MUSE_VR_SLOT_EPOCH
  // launch-counter claim (slot_claim.cuh): issued by thread 0 after its routed-ids loads below, settled when the work
  // list needs it
  unsigned ep_smid = 0, ep_old = 0, ep_hint = 0;
#endif
#if !MUSE_VR_SLOT_UNSAFE && !MUSE_VR_SLOT_EPOCH
  unsigned claim_hint = 0, claim_old = 0;
#endif
#if MUSE_VR_LDW_TEST_PREWAIT_CLAIM
  if (tid == 0) ticket = atomicInc(p.ticket, gridDim.x * kStartCycle - 1u);
#if !MUSE_VR_SLOT_UNSAFE && !MUSE_VR_SLOT_EPOCH
  if (tid == 0) {
    claim_hint = unsigned(reinterpret_cast<const uint8_t*>(p.sm_domain)[muse_vr_slot::read_smid()]);
    claim_old = muse_vr_slot::try_hint(p.ticket + kClaimWord, claim_hint, p.n_dom0, p.n_dom1);
  }
#endif
#endif
#if !MUSE_VR_LDW_PRE
  grid_dep_wait();  // ids and token rows come from predecessors
#if MUSE_VR_LDW_DENSE
  // dense-token build: its token rows and scales are read by asynchronous-proxy copies (tensor / bulk) after this wait; a
  // proxy fence orders the predecessors' generic writes, made visible by the wait, before them (the full-attention kernel's
  // form after its dependency wait). Every thread passes it, the issuers of those copies included.
  fence_proxy_async_global();
#endif
#if !MUSE_VR_LDW_TEST_PREWAIT_CLAIM
  // start ticket: its value is first needed at the parity step below
  if (tid == 0) ticket = atomicInc(p.ticket, gridDim.x * kStartCycle - 1u);
#if !MUSE_VR_SLOT_UNSAFE && !MUSE_VR_SLOT_EPOCH
  // slot claim (slot_claim.cuh): the SM's slot is the hint; settled when the work list needs it
  if (tid == 0) {
    claim_hint = unsigned(reinterpret_cast<const uint8_t*>(p.sm_domain)[muse_vr_slot::read_smid()]);
    claim_old = muse_vr_slot::try_hint(p.ticket + kClaimWord, claim_hint, p.n_dom0, p.n_dom1);
  }
#endif
#endif
#if MUSE_VR_LDW_DENSE
  // dense-token build: this CTA's copy of the call's token scales (predecessor grids' output, ordered by the wait above),
  // first read by the producer warp's image stores; the pass masks are zeroed here and set below
  // (one bulk copy per row, into rows of pitch kTsfPitch; lane 0 of warp 3 posts every row's bytes first)
  if (tid >= 3 * 32 && tid < 4 * 32 && (p.dbg & 256) == 0) {
    const int r = tid - 3 * 32;
    if (r == 0) mbar_arrive_expect_tx(&s.tsf_full, uint32_t(p.m * (kHidden / 16)));
    __syncwarp();
    if (r < p.m)
      bulk_load(smem_u32(&s.tsf[size_t(r) * kTsfPitch]), p.hs_sf + size_t(r) * (kHidden / 16), uint32_t(kHidden / 16),
                &s.tsf_full);
  }
#endif
#endif  // !MUSE_VR_LDW_PRE (the early prologue issues the wait and everything above after its pass tables)
#if MUSE_VR_LDW_PRE_VERIFY
  // Speculate and verify: this thread's routed-id words and the real-token count, read once before the wait (32-bit
  // loads, the producer's store width); the table build below runs on these copies, and runs again on the fresh words
  // after the wait if any word of the CTA changed.
  constexpr int kPerV = (kMaxSlots + kThreads - 1) / kThreads;
  int32_t vv[kPerV];
#pragma unroll
  for (int u = 0; u < kPerV; ++u) {
    const int sl = tid + u * kThreads;
    vv[u] = sl < nslot ? ld_relaxed_s32(p.packed + sl) : -1;
  }
  if (p.dbg & 4096) {  // tests only: every speculative id word changed (its expert field), so every CTA must rebuild
#pragma unroll
    for (int u = 0; u < kPerV; ++u)
      if (tid + u * kThreads < nslot) vv[u] ^= 0x10000;
  }
#if MUSE_VR_LDW_RBAL
  int nreal_v = ld_relaxed_s32(p.n_real);
  if (p.dbg & 8192) nreal_v = nreal_v > 0 ? nreal_v - 1 : 1;  // tests only: the speculative count changed
  const int rb_rank = p.local_offset >> 6;  // local_offset = 64 x rank (host check)
#endif
  bool waited = false, synced = false;
#if MUSE_VR_LDW_RBAL
  uint32_t rb_win = 0;  // the placement's winning key (lane 0 of the placement warp): its digest store follows the loop
#endif
#pragma unroll 1
  for (;;) {  // pass 1: speculative, before the wait; pass 2 (only after a verify difference): the fresh words
#endif
#if MUSE_VR_LDW_DENSE
  for (int i = tid; i < kMaxPasses; i += kThreads) s.dmask[i] = 0u;
#endif
  for (int i = tid; i < kNumChunks * kMaxPool; i += kThreads) (&x.ccnt[0][0])[i] = 0u;
#if MUSE_VR_LDW_RBAL
  for (int i = tid; i < 256; i += kThreads) x.hist[i] = 0u;
#if !MUSE_VR_LDW_PRE_VERIFY  // (declared before the pass loop in that build: used after it)
  const int rb_rank = p.local_offset >> 6;  // local_offset = 64 x rank (host check)
#endif
  // routed slots of the call's real tokens (the placement histogram's rows; padding rows only follow the placement)
#if MUSE_VR_LDW_TEST_RB_COUNT_PAD
  const int rb_hist_slots = p.m * kTopK;  // tests only (negative control): every row counted, padding included
#elif MUSE_VR_LDW_PRE_VERIFY
  const int rb_hist_slots = min(max(nreal_v, 0), p.m) * kTopK;  // the speculative (pass 1) or fresh (pass 2) count
#else
  const int rb_hist_slots = min(max(int(p.n_real[0]), 0), p.m) * kTopK;
#endif
#endif
  named_bar_sync(6, kThreads);
  {
    // all of this thread's ids in flight at once, then the counts
    constexpr int kPer = (kMaxSlots + kThreads - 1) / kThreads;
    int32_t v[kPer];
#pragma unroll
    for (int u = 0; u < kPer; ++u) {
      const int sl = tid + u * kThreads;
#if MUSE_VR_LDW_PRE_VERIFY
      v[u] = vv[u];  // read before the loop (pass 1) or by the verify (pass 2)
#else
      v[u] = sl < nslot ? p.packed[sl] : -1;
#endif
    }
#if !MUSE_VR_SLOT_UNSAFE && MUSE_VR_SLOT_EPOCH
    if (tid == 0) {
      const uint8_t* const table = reinterpret_cast<const uint8_t*>(p.sm_domain);
      ep_smid = muse_vr_slot::epoch_test_smid(table, muse_vr_slot::read_smid());
      ep_old = muse_vr_slot::epoch_try_sm(p.ticket + 32, ep_smid, kStartCycle);  // counters: words 32.. of ctl
      ep_hint = ep_smid < gridDim.x ? unsigned(table[ep_smid]) : muse_vr_slot::kInvalid;
    }
#endif
#pragma unroll
    for (int u = 0; u < kPer; ++u) {
      const int sl = tid + u * kThreads;
      if (sl < nslot) {
#if MUSE_VR_LDW_RBAL
        const int eg = int((v[u] >> 16) & 0xFFFF);
        if (eg < 256 && sl < rb_hist_slots) atomicAdd(&x.hist[eg], 1u);
        const int e = eg < 256 ? muse_vr_ring::window_slot(eg, rb_rank) : -1;
        const bool loc = e >= 0;
#else
        const int e = int((v[u] >> 16) & 0xFFFF) - p.local_offset;
        const bool loc = e >= 0 && e < p.pool;
#endif
        x.exp[sl] = int16_t(loc ? e : -1);
        if (loc) atomicAdd(&x.ccnt[sl / kChunkSlots][e], 1u);
      }
    }
  }
  named_bar_sync(6, kThreads);
#if MUSE_VR_LDW_RBAL
#if MUSE_VR_LDW_PRE && !MUSE_VR_LDW_PRE_VERIFY
  uint32_t rb_win = 0;  // the placement's winning key (lane 0 of the placement warp): its digest store follows the wait
#endif
  if (warp == MUSE_VR_LDW_RB_WARP) {  // the step's placement beside the pass scan's slots that never move
    named_bar_sync(6, kThreads);      // the prefix barrier (with every warp), then the placement beside the pass scan
#if MUSE_VR_LDW_TEST_RB_SPIN == 1
    for (const long long t0 = clock64(); clock64() - t0 < 20000;) {
    }
#endif
    const uint32_t win = muse_vr_ring::place_warp2<kPassRows>(x.hist, x.rb_s, x.rb_thr, lane);
    if (lane == 0) {
      LDW_STAMP(p, kStRbDone);
#if MUSE_VR_LDW_PRE
      rb_win = win;  // no global store before the dependency wait
#else
      if (p.stamps != nullptr)
        p.stamps[size_t(blockIdx.x) * kStampWords + kStRbDig] =
            uint64_t(uint32_t(x.rb_thr[0]) | uint32_t(x.rb_thr[1]) << 8 | uint32_t(x.rb_thr[2]) << 16 |
                     uint32_t(x.rb_thr[3]) << 24) |
            uint64_t(win) << 32;
#endif
    }
    // with warp 0 before its candidate slots: the thresholds are performed relative to warp 0 when the barrier completes
#if MUSE_VR_LDW_TEST_RB_NOBAR
    named_bar_arrive(MUSE_VR_LDW_RB_BAR, 64);
    named_bar_arrive(MUSE_VR_LDW_RB_BAR, 64);  // completes the barrier on this warp's own arrivals (warp 0 never meets it)
#else
    named_bar_sync(MUSE_VR_LDW_RB_BAR, 64);
#endif
  }
#endif
  if (tid < kMaxPool) {  // per expert: exclusive prefix over chunks (in place) and the total
    uint32_t run = 0;
    for (int c = 0; c < kNumChunks; ++c) {
      const uint32_t n = x.ccnt[c][tid];
      x.ccnt[c][tid] = run;
      run += n;
    }
    x.total[tid] = tid < p.pool ? run : 0u;
  }
#if MUSE_VR_LDW_RBAL
  if (warp != MUSE_VR_LDW_RB_WARP) named_bar_sync(6, kThreads);  // the placement warp synced at it above
#else
  named_bar_sync(6, kThreads);
#endif
  if (warp == 0) {  // expert bases and passes (ascending expert order), pass tables
    int rows = 0, q = 0;
#pragma unroll 1
#if MUSE_VR_LDW_RBAL
    for (int c = 0; c < 64; c += 32) {  // the candidate slots follow the loop
      // slots 0..31, then 32..47 + the empty 80..95 (no candidate inside the loop)
      const int e = c == 0 ? lane : (lane < 16 ? 32 + lane : 64 + lane);
#else
    for (int c = 0; c < kMaxPool; c += 32) {
      const int e = c + lane;
#endif
      const int n = int(x.total[e]);
      const int np = (n + kPassRows - 1) / kPassRows;
      int incl = n, inclq = np;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, incl, o), yq = __shfl_up_sync(0xffffffffu, inclq, o);
        if (lane >= o) {
          incl += y;
          inclq += yq;
        }
      }
      const int b = rows + incl - n, qb = q + inclq - np;
      s.base[e] = int16_t(b);
      s.qbase[e] = int16_t(qb);
      for (int pq = 0; pq < np; ++pq) {
        if (qb + pq < kMaxPasses) {
          s.q_expert[qb + pq] = int16_t(e);
          s.q_row[qb + pq] = int16_t(b + pq * kPassRows);
          s.q_rows[qb + pq] = int16_t(min(kPassRows, n - pq * kPassRows));
        }
      }
      rows += __shfl_sync(0xffffffffu, incl, 31);
      q += __shfl_sync(0xffffffffu, inclq, 31);
    }
#if MUSE_VR_LDW_RBAL
    {  // the candidate slots 48..79 (lane l = slot 48 + l): own movable lanes 0..15, the predecessor's lanes 16..31
      constexpr unsigned kFull = 0xffffffffu;
      const int e = 48 + lane;
      const int nu = int(x.total[e]), npu = (nu + kPassRows - 1) / kPassRows;
      int su = nu, squ = npu;  // inclusive prefixes with every candidate counted
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(kFull, su, o), yq = __shfl_up_sync(kFull, squ, o);
        if (lane >= o) {
          su += y;
          squ += yq;
        }
      }
      const int s31 = __shfl_sync(kFull, su, 31), sq31 = __shfl_sync(kFull, squ, 31);
#if MUSE_VR_LDW_TEST_RB_SPIN == 2
      for (const long long t0 = clock64(); clock64() - t0 < 20000;) {
      }
#endif
#if !MUSE_VR_LDW_TEST_RB_NOBAR
      named_bar_sync(MUSE_VR_LDW_RB_BAR, 64);  // the placement's thresholds (the placement warp)
#endif
      // own lanes 0..a-1 and the predecessor's lanes bp..31 run here (thresholds in [48, 64]: a in [0, 16], bp in [16, 32])
      const int a = x.rb_thr[rb_rank] - muse_vr_ring::kMovLo;
      const int bp = 16 + x.rb_thr[(rb_rank + 3) & 3] - muse_vr_ring::kMovLo;
      const int ia = a > 0 ? a - 1 : 0;
      const int sa0 = __shfl_sync(kFull, su, ia), sqa0 = __shfl_sync(kFull, squ, ia);
      const int sb = __shfl_sync(kFull, su, bp - 1), sqb = __shfl_sync(kFull, squ, bp - 1);
      const int sa = a > 0 ? sa0 : 0, sqa = a > 0 ? sqa0 : 0;
      const bool here = lane < 16 ? lane < a : lane >= bp;
      const int n = here ? nu : 0, np = here ? npu : 0;
      // inclusive prefixes over the candidates that run here (the values a masked scan gives)
      const int incl = lane < 16 ? (lane < a ? su : sa) : sa + (lane >= bp ? su - sb : 0);
      const int inclq = lane < 16 ? (lane < a ? squ : sqa) : sqa + (lane >= bp ? squ - sqb : 0);
      const int b = rows + incl - n, qb = q + inclq - np;
      s.base[e] = int16_t(b);
      s.qbase[e] = int16_t(qb);
      for (int pq = 0; pq < np; ++pq) {
        if (qb + pq < kMaxPasses) {
          s.q_expert[qb + pq] = int16_t(e);
          s.q_row[qb + pq] = int16_t(b + pq * kPassRows);
          s.q_rows[qb + pq] = int16_t(min(kPassRows, n - pq * kPassRows));
        }
      }
      rows += sa + s31 - sb;
      q += sqa + sq31 - sqb;
    }
#endif
    if (lane == 0) {
      s.n_pass = q;
      s.head_n = 0;
#if !MUSE_VR_LDW_PRE
      s.start_base = ticket - ticket % gridDim.x;
      s.set = int((s.start_base / gridDim.x) & 1u);
#if !MUSE_VR_SLOT_UNSAFE && MUSE_VR_SLOT_EPOCH
      s.slot = muse_vr_slot::epoch_settle_sm(p.ticket + 32, p.ticket + kClaimWord + muse_vr_slot::kMissWord,
                                             reinterpret_cast<const uint8_t*>(p.sm_domain), ep_smid, ep_hint, ep_old,
                                             ticket / gridDim.x, kStartCycle, p.n_dom0, p.n_dom1, p.err);
#elif !MUSE_VR_SLOT_UNSAFE
      s.slot = muse_vr_slot::settle(p.ticket + kClaimWord, claim_hint, claim_old, p.n_dom0, p.n_dom1, p.err);
#endif
#if MUSE_VR_LDW_ERRWORD && !MUSE_VR_SLOT_UNSAFE
      if (s.slot == muse_vr_slot::kInvalid) {  // no slot left (error-word claim): this CTA takes no work items
        fail(p.err, kErrSlot);
        s.slot = 0;
      }
#endif
#endif
    }
#if MUSE_VR_LDW_PRE
    // Early prologue, still before the dependency wait: the work list the SM's hinted slot would get (the same round robin
    // as below), and an L2 prefetch of the weights and weight scales of its first kStages ring stages (static data; no
    // shared-memory or global writes). The ring copies issued after the claim then read them from L2.
    __syncwarp();  // the warp's pass-table stores before its lanes read them
    {
      unsigned hint = 0;
      if (lane == 0) hint = unsigned(reinterpret_cast<const uint8_t*>(p.sm_domain)[muse_vr_slot::read_smid()]);
      hint = shfl_idx(hint, 0);
      const int nq = s.n_pass;
      const unsigned dom = (hint >> 7) & 1u;
      const int R = dom ? p.n_dom1 : p.n_dom0, r = int(hint & 127u);
#if MUSE_VR_LDW_PRE_VERIFY
      const bool pf_pass = !waited;  // the prefetch is a pass-1 hint only
#else
      constexpr bool pf_pass = true;
#endif
      if (pf_pass && nq <= kMaxPasses && R > 0 && r < R && (p.dbg & 512) == 0) {
        const int n1 = steps_of(kFc1TilesPerDomain * nq, r, R);
        const int nt = steps_of((kFc1TilesPerDomain + kFc2TilesPerDomain) * nq, r, R);
        const int npf = nt > kMaxItems ? 0 : min(kStages, nt * kStagesPerTile);
#pragma unroll 1
        for (int xs = lane; xs < npf; xs += 32) {
          const int i = xs / kStagesPerTile, j = xs % kStagesPerTile;
          int fc2 = 0, qq, tile;
          if (i < n1) {
            const int g = r + i * R;
            qq = g / kFc1TilesPerDomain;
            tile = kFc1TilesPerDomain * int(dom) + (g % kFc1TilesPerDomain);
          } else {
            const int pc = r + i * R - kFc1TilesPerDomain * nq;
            fc2 = 1;
            qq = pc / kFc2TilesPerDomain;
            tile = kFc2TilesPerDomain * int(dom) + (pc % kFc2TilesPerDomain);
          }
          const int e = s.q_expert[qq];
          const int tile_row = fc2 ? e * kFc2TilesPerExpert + tile : e * kFc1TilesPerExpert + tile;
          l2_prefetch_3d(fc2 ? m2 : m13, 0, tile_row * kTile, j * (kChunks / 4));
          l2_prefetch((fc2 ? p.sf2 : p.sf13) + (size_t(tile_row) * (kHidden / 64) + size_t(j) * kChunks) * 512,
                      uint32_t(kChunks * 512));
        }
      }
    }
#endif
  }
#if MUSE_VR_LDW_PRE
#if MUSE_VR_LDW_PRE_VERIFY
  if (!waited) {  // pass 1 only: the wait, the claim and the token-scale copy run once
#endif
  // The early prologue's dependency wait (every thread): what follows reads the predecessors' token data (the token scales
  // below, every later role) or updates control words that the previous launch of this call site orders by its completion
  // (start ticket, slot claim, the other parity's counters), exactly as the default build does right after its wait.
  grid_dep_wait();
#if MUSE_VR_LDW_DENSE
  fence_proxy_async_global();  // as in the default build: before the asynchronous-proxy copies of token data
#endif
#if !MUSE_VR_LDW_TEST_PREWAIT_CLAIM
  if (tid == 0) ticket = atomicInc(p.ticket, gridDim.x * kStartCycle - 1u);
  if (tid == 0) {
    claim_hint = unsigned(reinterpret_cast<const uint8_t*>(p.sm_domain)[muse_vr_slot::read_smid()]);
    claim_old = muse_vr_slot::try_hint(p.ticket + kClaimWord, claim_hint, p.n_dom0, p.n_dom1);
  }
#endif
#if MUSE_VR_LDW_DENSE
  if (tid >= 3 * 32 && tid < 4 * 32 && (p.dbg & 256) == 0) {  // this CTA's copy of the token scales (as above)
    const int r = tid - 3 * 32;
    if (r == 0) mbar_arrive_expect_tx(&s.tsf_full, uint32_t(p.m * (kHidden / 16)));
    __syncwarp();
    if (r < p.m)
      bulk_load(smem_u32(&s.tsf[size_t(r) * kTsfPitch]), p.hs_sf + size_t(r) * (kHidden / 16), uint32_t(kHidden / 16),
                &s.tsf_full);
  }
#endif
#if MUSE_VR_LDW_RBAL && !MUSE_VR_LDW_PRE_VERIFY
  if (warp == MUSE_VR_LDW_RB_WARP && lane == 0 && p.stamps != nullptr)
    p.stamps[size_t(blockIdx.x) * kStampWords + kStRbDig] =
        uint64_t(uint32_t(x.rb_thr[0]) | uint32_t(x.rb_thr[1]) << 8 | uint32_t(x.rb_thr[2]) << 16 |
                 uint32_t(x.rb_thr[3]) << 24) |
        uint64_t(rb_win) << 32;
#endif
  if (tid == 0) {
    s.start_base = ticket - ticket % gridDim.x;
    s.set = int((s.start_base / gridDim.x) & 1u);
    s.slot = muse_vr_slot::settle(p.ticket + kClaimWord, claim_hint, claim_old, p.n_dom0, p.n_dom1, p.err);
#if MUSE_VR_LDW_ERRWORD
    if (s.slot == muse_vr_slot::kInvalid) {  // no slot left (error-word claim): this CTA takes no work items
      fail(p.err, kErrSlot);
      s.slot = 0;
    }
#endif
  }
#if MUSE_VR_LDW_PRE_VERIFY
  }  // !waited
  if (waited) break;  // pass 2 (the rebuild from the fresh words) is complete
  waited = true;
  // The verify: re-read every word the speculation used, ordered after the predecessors by the wait above (distinct
  // loads after the wait's memory clobber), and compare; one barrier reduction gives the CTA's answer. Pass 2 runs on the
  // fresh words held in vv / nreal_v.
  bool diff = false;
#pragma unroll
  for (int u = 0; u < kPerV; ++u) {
    const int sl = tid + u * kThreads;
    if (sl < nslot) {
      const int32_t f = ld_relaxed_s32(p.packed + sl);
      diff |= f != vv[u];
      vv[u] = f;
    }
  }
#if MUSE_VR_LDW_RBAL
  {
    const int f = ld_relaxed_s32(p.n_real);
    diff |= f != nreal_v;
    nreal_v = f;
  }
#endif
  if (p.dbg & 2048) diff = false;  // tests only: the negative control keeps the speculative tables
  if (!named_bar_red_or(6, kThreads, diff)) {
    synced = true;  // the reduction was this CTA's barrier 6 before the work list
    break;
  }
  if (tid == 0 && p.stamps != nullptr) p.stamps[size_t(blockIdx.x) * kStampWords + kStRedo] += 1u;  // diag: a rebuild
  }  // for (;;): pass 2
#if MUSE_VR_LDW_RBAL
  if (warp == MUSE_VR_LDW_RB_WARP && lane == 0 && p.stamps != nullptr)  // the final placement's digest (after the wait)
    p.stamps[size_t(blockIdx.x) * kStampWords + kStRbDig] =
        uint64_t(uint32_t(x.rb_thr[0]) | uint32_t(x.rb_thr[1]) << 8 | uint32_t(x.rb_thr[2]) << 16 |
                 uint32_t(x.rb_thr[3]) << 24) |
        uint64_t(rb_win) << 32;
#endif
#endif  // MUSE_VR_LDW_PRE_VERIFY
#endif
#if MUSE_VR_LDW_PRE_VERIFY
  if (!synced)
#endif
  named_bar_sync(6, kThreads);
  if (warp == 0) {  // this CTA's work list: one round-robin over the domain list FC1(all passes) then FC2(all passes)
    unsigned slot = 0;
#if MUSE_VR_SLOT_UNSAFE
    if (lane == 0) {
      unsigned smid;
      asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
      slot = unsigned(reinterpret_cast<const uint8_t*>(p.sm_domain)[smid]);
    }
    slot = shfl_idx(slot, 0);
#else
    slot = s.slot;  // claimed in the prologue (named barrier 6 above)
#endif
    int nq = s.n_pass;
    if (nq > kMaxPasses) {
      if (lane == 0) fail(p.err, kErrShape);
      nq = 0;
    }
    const unsigned dom = (slot >> 7) & 1u;
    const int R = dom ? p.n_dom1 : p.n_dom0, r = int(slot & 127u);
    const int n1 = R > 0 ? steps_of(kFc1TilesPerDomain * nq, r, R) : 0;
    const int n2 = R > 0 ? steps_of((kFc1TilesPerDomain + kFc2TilesPerDomain) * nq, r, R) - n1 : 0;
    int nt = (p.dbg & 512) ? 0 : n1 + n2;
#if MUSE_VR_LDW_ERRWORD
    if (aborted()) nt = 0;  // a failed claim (or an earlier expired wait): no work items
#endif
    if (nt > kMaxItems) {
      if (lane == 0) fail(p.err, kErrSlot);
      nt = 0;
    }
#pragma unroll 1
    for (int i = lane; i < nt; i += 32) {
      uint32_t v;
      if (i < n1) {
        const int g = r + i * R;
        v = item_pack(0, g / kFc1TilesPerDomain, kFc1TilesPerDomain * int(dom) + (g % kFc1TilesPerDomain));
      } else {
        const int pc = r + i * R - kFc1TilesPerDomain * nq;
        v = item_pack(1, pc / kFc2TilesPerDomain, kFc2TilesPerDomain * int(dom) + (pc % kFc2TilesPerDomain));
      }
      s.items[i] = v;
    }
    if (lane == 0) {
      s.n_fc1 = nt > 0 ? n1 : 0;
      s.n_items = nt;
    }
  } else if (MUSE_VR_LDW_GATHER && warp == 3 && (p.dbg & 256) == 0 && s.n_pass <= kMaxPasses) {
    // head copies: the token of each row of the passes of this CTA's first FC1 items (item h computes tile g = r + h R of
    // pass g / 32), while a pass has at most kHeadRows rows. Row i of pass qq of expert e is e's slot of rank
    // (qq - qbase[e]) * 128 + i in slot order - the copy phase's stable expert-major order: the per-chunk prefix counts
    // give the chunk holding the pass's first row, then an ordered scan from there (8 independent ballots per chunk).
    unsigned slot = 0;
#if MUSE_VR_SLOT_UNSAFE
    if (lane == 0) {
      unsigned smid;
      asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
      slot = unsigned(reinterpret_cast<const uint8_t*>(p.sm_domain)[smid]);
    }
    slot = shfl_idx(slot, 0);
#else
    slot = s.slot;  // the slot warp 0's work list uses (claimed in the prologue)
#endif
    const int R = ((slot >> 7) & 1u) ? p.n_dom1 : p.n_dom0, r = int(slot & 127u);
    const int nq = s.n_pass;
    int hn = 0;
#pragma unroll 1
    for (int h = 0; h < kHeadItems; ++h) {
      const int g = r + h * R;
      if (R <= 0 || g >= kFc1TilesPerDomain * nq) break;  // warp-uniform
      const int qq = g / kFc1TilesPerDomain, e = s.q_expert[qq], rows = s.q_rows[qq];
      if (rows > kHeadRows) break;
      const int r0 = (qq - int(s.qbase[e])) * kPassRows;
      const int nch = (nslot + kChunkSlots - 1) / kChunkSlots;
      int c0 = 0;  // last chunk whose prefix count of e is <= r0: it holds e's slot of rank r0
#pragma unroll
      for (int st = kNumChunks / 2; st > 0; st >>= 1)
        if (c0 + st < nch && int(x.ccnt[c0 + st][e]) <= r0) c0 += st;
      int seen = int(x.ccnt[c0][e]);
#pragma unroll 1
      for (int c = c0; c < nch && seen < r0 + rows; ++c) {
        // one chunk at a time: its 8 groups of 32 slots loaded and balloted independently, then ranked in order
        int ex[kChunkSlots / 32];
        unsigned bal[kChunkSlots / 32];
#pragma unroll
        for (int u = 0; u < kChunkSlots / 32; ++u) {
          const int sl = c * kChunkSlots + 32 * u + lane;
          ex[u] = sl < nslot ? int(x.exp[sl]) : -2;
        }
#pragma unroll
        for (int u = 0; u < kChunkSlots / 32; ++u) bal[u] = __ballot_sync(0xffffffffu, ex[u] == e);
#pragma unroll
        for (int u = 0; u < kChunkSlots / 32; ++u) {
          const int rk = seen + __popc(bal[u] & ((1u << lane) - 1u));
          if (ex[u] == e && rk >= r0 && rk < r0 + rows) s.ptok[h][rk - r0] = int16_t((c * kChunkSlots + 32 * u + lane) / kTopK);
          seen += __popc(bal[u]);
        }
      }
      hn = h + 1;
    }
#if MUSE_VR_LDW_ERRWORD
    if (aborted()) hn = 0;  // a failed claim: no head copies (no work items)
#endif
    if (lane == 0) s.head_n = hn;
#if MUSE_VR_LDW_DENSE
  } else if (warp == 1 || warp == 2) {  // dense-token build: per pass, the tokens whose slot of its expert is computed here
    // the same predicate as the positions below (a slot gets a row here iff its expert runs here); one pass per expert
    // (at most 32 rows), so the pass is qbase[e] and token t = slot / 8 < 32 is bit t
    for (int sl = tid - 32; sl < nslot; sl += 64) {
      const int e = x.exp[sl];
#if MUSE_VR_LDW_RBAL
      if (e >= 0 && muse_vr_ring::slot_here(e, x.rb_thr[rb_rank], x.rb_thr[(rb_rank + 3) & 3]))
#else
      if (e >= 0)
#endif
        atomicOr(&s.dmask[s.qbase[e]], 1u << (sl / kTopK));
    }
#endif
  } else if (warp >= 8) {  // permuted positions of this CTA's slots (one warp per slot)
    const int w = warp & 3;
    const int G = int(gridDim.x);
    int n_my = (nslot > int(blockIdx.x)) ? (nslot - 1 - int(blockIdx.x)) / G + 1 : 0;
    if (n_my > kMaxMySlots) {
      if (tid == 8 * 32) fail(p.err, kErrShape);
      n_my = 0;
    }
#pragma unroll 1
    for (int j = w; j < n_my; j += 4) {
      const int sl = int(blockIdx.x) + G * j;
      const int e = x.exp[sl];
      int pos = -1;
#if MUSE_VR_LDW_RBAL
      if (e >= 0 && muse_vr_ring::slot_here(e, x.rb_thr[rb_rank], x.rb_thr[(rb_rank + 3) & 3])) {
#else
      if (e >= 0) {
#endif
        // stable rank: earlier chunks (prefix) + earlier slots of the same expert in this chunk
        const int c0 = (sl / kChunkSlots) * kChunkSlots;
        int before = 0;
#pragma unroll 1
        for (int c = c0; c < sl; c += 32) {
          const int idx = c + lane;
          before += __popc(__ballot_sync(0xffffffffu, idx < sl && x.exp[idx] == e));
        }
        pos = int(s.base[e]) + int(x.ccnt[sl / kChunkSlots][e]) + before;
      }
      if (lane == 0) {
        s.my_slot[j] = int16_t(sl);
        s.my_pos[j] = int16_t(pos);
        s.my_exp[j] = int16_t(e);
      }
    }
    if (tid == 8 * 32) s.n_my = n_my;
  } else if (warp >= 4) {  // the other launch parity's counters: zero this CTA's share for the next launch
    unsigned* const other = p.cnt + (1 - s.set) * kCntSet;
    for (int i = int(blockIdx.x) + int(gridDim.x) * (tid - 4 * 32); i < kCntSet; i += int(gridDim.x) * 128)
      muse_vr_ldq::st_relaxed_gpu(other + i, 0u);
  }
}

struct alignas(64) Maps {
  CUtensorMap m13, m2;                           // FC1 / FC2 weight stages
  CUtensorMap b1[kNumBuckets], b2[kNumBuckets];  // FC1 / FC2 token rows, one box height per bucket
#if MUSE_VR_LDW_DENSE
  CUtensorMap bd[2];  // the dense-token build: FC1 token rows straight from the token tensor (box 16 / 32 rows)
#endif
};

__global__ void __launch_bounds__(kThreads, 1) ldw_kernel(const __grid_constant__ Maps maps, const __grid_constant__ Params p) {
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  Smem& s = *reinterpret_cast<Smem*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) {
    abort_reset();
    LDW_STAMP(p, kStStart);
  }
  if (warp == 3) {
    if (elect_one()) {
      for (int i = 0; i < kStages; ++i) {
        mbar_init(&s.full[i], 1);
        mbar_init(&s.empty[i], 1);
        mbar_init(&s.sf_full[i], 1);
      }
      for (int i = 0; i < 2; ++i) {
        mbar_init(&s.tmem_full[i], 1);
        mbar_init(&s.tmem_empty[i], 4);
      }
      mbar_init(&s.drain_sf, 1);
      mbar_init(&s.drain_mma, 1);
#if MUSE_VR_LDW_DENSE
      mbar_init(&s.tsf_full, 1);
#endif
      for (int i = 0; i < 4; ++i) s.fc1_stored[i] = 0u;
      fence_barrier_init();
      prefetch_tensormap(&maps.m13);
      prefetch_tensormap(&maps.m2);
#if MUSE_VR_LDW_DENSE
      prefetch_tensormap(&maps.bd[0]);
      prefetch_tensormap(&maps.bd[1]);
#endif
    }
    __syncwarp();
  } else if (warp == 1) {
    tmem_alloc(&s.tmem_base, kTmemCols);
    tmem_relinquish();
    tc_fence_before();
  }
  named_bar_sync(6, kThreads);  // barrier initialization, tensor memory
#if MUSE_VR_LDW_PRE
  prologue(s, p, &maps.m13, &maps.m2);
#else
  prologue(s, p);
#endif
  named_bar_sync(3, kThreads);  // work list, pass tables, positions, counter parity for every role
  if (tid == 0) {
    LDW_STAMP(p, kStPro);
    unsigned smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    int cost = 0;
    for (int i = 0; i < s.n_items; ++i) cost += bucket_rows(bucket_of(s.q_rows[item_unpack(s.items[i]).q]));
    LDW_STAMPV(p, kStItems, (uint64_t(s.n_items) << 16) | uint64_t(s.n_fc1));
    LDW_STAMPV(p, kStCost, uint64_t(cost));
    LDW_STAMPV(p, kStSm, uint64_t(smid));
    (void)cost;
  }
  tc_fence_after();
  if (warp == 0) {
#if MUSE_VR_LDW_DENSE
    role_producer(s, p, &maps.m13, &maps.m2, maps.b1, maps.b2, maps.bd);
#else
    role_producer(s, p, &maps.m13, &maps.m2, maps.b1, maps.b2, nullptr);
#endif
  } else if (warp == 1) {
    role_mma(s, p);
  } else if (warp == 2) {
    role_sfcopy(s, p);
  } else if (warp == 3) {
    role_ctrl(s, p);
  } else if (warp < 8) {
    role_epilogue(s, p);
  } else {
    role_builder(s, p);
  }
  named_bar_sync(5, kThreads);
  if (tid == 0) LDW_STAMP(p, kStEnd);
  if (warp == 1) tmem_dealloc(s.tmem_base, kTmemCols);
#if MUSE_VR_LDW_ERRWORD
  // after the drains and the tensor-memory release, as the trap below: a CTA that aborted, or that sees a failed slot
  // claim in the sticky word (the last CTA's coverage check sets it), reports and ends
  if (tid == 0 && (aborted() || (*reinterpret_cast<volatile int32_t*>(p.err) & muse_vr_slot::kErrClaim) != 0))
    report_error_word(p.page_err, p.ew_tag);
#elif MUSE_VR_LDW_TRAP
  if (tid == 0 && aborted()) __trap();  // after the drains and the tensor-memory release: the error word is already set
#endif
}

}  // inline namespace MUSE_VR_LDW_ABI
}  // namespace muse_vr_ldw

// ---------------------------------------------------------------- host
#ifndef MUSE_VR_LDW_DEVICE_ONLY
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/utils.h>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <mutex>
#include <unordered_map>

namespace muse_vr_ldw {
inline namespace MUSE_VR_LDW_ABI {
namespace {

using PFN_encodeTiled = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                                      const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
                                      CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

PFN_encodeTiled get_encode_fn() {
  static PFN_encodeTiled fn = [] {
    void* ptr = nullptr;
    cudaDriverEntryPointQueryResult q;
    sglang::host::RuntimeDeviceCheck(
        cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &ptr, 12000, cudaEnableDefault, &q));
    sglang::host::RuntimeCheck(q == cudaDriverEntryPointSuccess && ptr != nullptr, "cuTensorMapEncodeTiled unavailable");
    return reinterpret_cast<PFN_encodeTiled>(ptr);
  }();
  return fn;
}

// [rows, 2048] uint8 rows viewed as (128-byte K atom, row, atom index); a box of box_rows rows x box_atoms atoms lands as
// box_atoms consecutive 128-byte-swizzled atoms of box_rows rows.
CUtensorMap make_rows_map(const void* base, uint64_t rows, uint32_t box_rows, uint32_t box_atoms) {
  CUtensorMap map{};
  const cuuint64_t dim[3] = {128, rows, kHidden / 2 / 128};
  const cuuint64_t stride[2] = {kHidden / 2, 128};
  const cuuint32_t box[3] = {128, box_rows, box_atoms};
  const cuuint32_t es[3] = {1, 1, 1};
  const CUresult rc = get_encode_fn()(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, const_cast<void*>(base), dim, stride, box,
                                      es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                      CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  sglang::host::RuntimeCheck(rc == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed");
  return map;
}

const CUtensorMap& cached_map(const void* base, uint64_t rows, uint32_t box_rows, uint32_t box_atoms) {
  static std::mutex mu;
  static std::unordered_map<uint64_t, CUtensorMap> cache;
  uint64_t key = reinterpret_cast<uint64_t>(base);
  key = key * 1000003ull ^ rows;
  key = key * 1000003ull ^ (uint64_t(box_rows) << 8 | box_atoms);
  std::lock_guard<std::mutex> lock(mu);
  auto it = cache.find(key);
  if (it == cache.end()) it = cache.emplace(key, make_rows_map(base, rows, box_rows, box_atoms)).first;
  return it->second;
}

}  // namespace

struct MuseVrLdw {
  static constexpr std::size_t kSmem = sizeof(Smem) + 1024;

  static void preload() {
    using namespace sglang::host;
    int dev = 0, optin = 0;
    RuntimeDeviceCheck(cudaGetDevice(&dev));
    RuntimeDeviceCheck(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    RuntimeCheck(kSmem <= std::size_t(optin), "shared memory per block above the device's opt-in limit");
    RuntimeDeviceCheck(cudaFuncSetAttribute(ldw_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(kSmem)));
    cudaFuncAttributes attr;
    RuntimeDeviceCheck(cudaFuncGetAttributes(&attr, ldw_kernel));
  }

  static void run(const tvm::ffi::TensorView packed, const tvm::ffi::TensorView hs, const tvm::ffi::TensorView hs_sf,
                  const tvm::ffi::TensorView pts, const tvm::ffi::TensorView w13, const tvm::ffi::TensorView sf13,
                  const tvm::ffi::TensorView w2, const tvm::ffi::TensorView sf2, const tvm::ffi::TensorView gate_alpha,
                  const tvm::ffi::TensorView up_alpha, const tvm::ffi::TensorView out_alpha,
                  const tvm::ffi::TensorView sm_domain, const tvm::ffi::TensorView out,
                  const tvm::ffi::TensorView expert_weights, const tvm::ffi::TensorView e2p,
                  const tvm::ffi::TensorView hs_perm, const tvm::ffi::TensorView pts_perm, const tvm::ffi::TensorView img1,
                  const tvm::ffi::TensorView img2, const tvm::ffi::TensorView act, const tvm::ffi::TensorView hs2,
                  const tvm::ffi::TensorView ramax, const tvm::ffi::TensorView cnt, const tvm::ffi::TensorView ctl,
                  const tvm::ffi::TensorView err, const tvm::ffi::Optional<tvm::ffi::TensorView> stamps,
                  int64_t local_offset, int64_t n_dom0, int64_t n_dom1, double gsi, int64_t flags, int64_t dbg,
#if MUSE_VR_LDW_RBAL && MUSE_VR_LDW_ERRWORD
                  int64_t pdl, const tvm::ffi::TensorView n_real, int64_t page_err, int64_t ew_tag) {
#elif MUSE_VR_LDW_RBAL
                  int64_t pdl, const tvm::ffi::TensorView n_real) {
#elif MUSE_VR_LDW_ERRWORD
                  int64_t pdl, int64_t page_err, int64_t ew_tag) {
#else
                  int64_t pdl) {
#endif
    using namespace sglang;
    using namespace sglang::host;
    auto M = SymbolicSize{"num_tokens"};
    auto P = SymbolicSize{"pool"};
    auto NSM = SymbolicSize{"sm_count"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({M, kTopK}).with_dtype<int32_t>().with_device(device_).verify(packed);
    TensorMatcher({M, kHidden / 2}).with_dtype<uint8_t>().with_device(device_).verify(hs);
    TensorMatcher({M, kHidden / 16}).with_dtype<uint8_t>().with_device(device_).verify(hs_sf);
    TensorMatcher({M}).with_dtype<float>().with_device(device_).verify(pts);
    TensorMatcher({P, 2 * kInter, kHidden / 2}).with_dtype<uint8_t>().with_device(device_).verify(w13);
    TensorMatcher({P, 2 * kInter, kHidden / 16}).with_dtype<uint8_t>().with_device(device_).verify(sf13);
    TensorMatcher({P, kHidden, kInter / 2}).with_dtype<uint8_t>().with_device(device_).verify(w2);
    TensorMatcher({P, kHidden, kInter / 16}).with_dtype<uint8_t>().with_device(device_).verify(sf2);
    TensorMatcher({P}).with_dtype<float>().with_device(device_).verify(gate_alpha);
    TensorMatcher({P}).with_dtype<float>().with_device(device_).verify(up_alpha);
    TensorMatcher({P}).with_dtype<float>().with_device(device_).verify(out_alpha);
    TensorMatcher({NSM}).with_dtype<int8_t>().with_device(device_).verify(sm_domain);
    auto MR = SymbolicSize{"rows"};
    TensorMatcher({MR, kHidden}).with_dtype<bf16_t>().with_device(device_).verify(out);
    TensorMatcher({M, kTopK}).with_dtype<int16_t>().with_device(device_).verify(expert_weights);
    TensorMatcher({MR}).with_dtype<int32_t>().with_device(device_).verify(e2p);
    TensorMatcher({int64_t(kMaxSlots), kHidden / 2}).with_dtype<uint8_t>().with_device(device_).verify(hs_perm);
    TensorMatcher({int64_t(kMaxSlots)}).with_dtype<float>().with_device(device_).verify(pts_perm);
    TensorMatcher({int64_t(kMaxPasses) * kImgBytes}).with_dtype<uint8_t>().with_device(device_).verify(img1);
    TensorMatcher({int64_t(kMaxPasses) * kImgBytes}).with_dtype<uint8_t>().with_device(device_).verify(img2);
    TensorMatcher({int64_t(kMaxSlots) * kActWords}).with_dtype<int64_t>().with_device(device_).verify(act);
    TensorMatcher({int64_t(kMaxSlots), kHidden / 2}).with_dtype<uint8_t>().with_device(device_).verify(hs2);
    TensorMatcher({int64_t(kMaxSlots)}).with_dtype<float>().with_device(device_).verify(ramax);
    TensorMatcher({int64_t(2 * kCntSet)}).with_dtype<int32_t>().with_device(device_).verify(cnt);
#if !MUSE_VR_SLOT_UNSAFE && MUSE_VR_SLOT_EPOCH
    TensorMatcher({32 + muse_vr_slot::kEpochWords}).with_dtype<int32_t>().with_device(device_).verify(ctl);
#else
    TensorMatcher({32}).with_dtype<int32_t>().with_device(device_).verify(ctl);
#endif
    TensorMatcher({1}).with_dtype<int32_t>().with_device(device_).verify(err);
    if (stamps.has_value())
      TensorMatcher({NSM.unwrap() * int64_t(kStampWords)}).with_dtype<int64_t>().with_device(device_).verify(stamps.value());
    const int64_t m = M.unwrap(), pool = P.unwrap();
    RuntimeCheck(m >= 1 && m <= kMaxTokens, kMaxTokens > 1024 ? "token count outside 1..2048"
                                             : kMaxTokens > 512 ? "token count outside 1..1024" : "token count outside 1..512");
    RuntimeCheck(MR.unwrap() == m * kTopK, "output rows must be num_tokens * top_k");
    RuntimeCheck(pool >= 1 && pool <= kMaxPool, "pool outside 1..128");
    RuntimeCheck(local_offset >= 0 && local_offset + pool <= 32768, "bad local expert offset");
#if MUSE_VR_LDW_RBAL
    RuntimeCheck(pool == 80 && local_offset % 64 == 0 && local_offset < 256,
                 "ring placement build: pool 80 and local_offset = 64 x rank (world 4)");
#endif
    RuntimeCheck(n_dom0 >= 0 && n_dom1 >= 0 && n_dom0 + n_dom1 == NSM.unwrap(), "domain slots must cover the grid");
    RuntimeCheck(NSM.unwrap() <= 256 && n_dom0 <= 128 && n_dom1 <= 128, "slot table holds 128 ranks per domain");
    RuntimeCheck(m * kTopK <= int64_t(kMaxMySlots) * NSM.unwrap(), "routed slots per CTA above the copy table");

    Params p{};
    p.packed = static_cast<const int32_t*>(packed.data_ptr());
    p.hs = static_cast<const uint8_t*>(hs.data_ptr());
    p.hs_sf = static_cast<const uint8_t*>(hs_sf.data_ptr());
    p.pts = static_cast<const float*>(pts.data_ptr());
    p.sf13 = static_cast<const uint8_t*>(sf13.data_ptr());
    p.sf2 = static_cast<const uint8_t*>(sf2.data_ptr());
    p.gate_alpha = static_cast<const float*>(gate_alpha.data_ptr());
    p.up_alpha = static_cast<const float*>(up_alpha.data_ptr());
    p.out_alpha = static_cast<const float*>(out_alpha.data_ptr());
    p.sm_domain = static_cast<const int8_t*>(sm_domain.data_ptr());
    p.hs_perm = static_cast<uint8_t*>(hs_perm.data_ptr());
    p.pts_perm = static_cast<float*>(pts_perm.data_ptr());
    p.img1 = static_cast<uint8_t*>(img1.data_ptr());
    p.img2 = static_cast<uint8_t*>(img2.data_ptr());
    p.act = static_cast<unsigned long long*>(act.data_ptr());
    p.hs2 = static_cast<uint8_t*>(hs2.data_ptr());
    p.ramax = static_cast<float*>(ramax.data_ptr());
    p.cnt = static_cast<unsigned*>(cnt.data_ptr());
    p.out = static_cast<__nv_bfloat16*>(out.data_ptr());
    p.e2p = static_cast<int32_t*>(e2p.data_ptr());
    p.expert_weights = static_cast<uint16_t*>(expert_weights.data_ptr());
    p.ticket = static_cast<unsigned*>(ctl.data_ptr());
    p.err = static_cast<int32_t*>(err.data_ptr());
    p.m = static_cast<int32_t>(m);
    p.pool = static_cast<int32_t>(pool);
    p.local_offset = static_cast<int32_t>(local_offset);
    p.n_dom0 = static_cast<int32_t>(n_dom0);
    p.n_dom1 = static_cast<int32_t>(n_dom1);
    p.gsi = static_cast<float>(gsi);
    p.flags = static_cast<int32_t>(flags);
    p.dbg = static_cast<int32_t>(dbg);
    p.stamps = stamps.has_value() ? static_cast<unsigned long long*>(stamps.value().data_ptr()) : nullptr;
#if MUSE_VR_LDW_RBAL
    TensorMatcher({int64_t(1)}).with_dtype<int32_t>().with_device(device_).verify(n_real);
    p.n_real = static_cast<const int32_t*>(n_real.data_ptr());
#endif
#if MUSE_VR_LDW_ERRWORD
    RuntimeCheck(page_err != 0 && page_err % 4 == 0, "error-word build: null or unaligned error word address");
    RuntimeCheck(ew_tag >= 0 && ew_tag < 0x80000000LL, "error-word build: the tag must fit in 31 bits");
    p.page_err = reinterpret_cast<uint32_t*>(static_cast<uintptr_t>(page_err));
    p.ew_tag = static_cast<uint32_t>(ew_tag);
#endif

    Maps maps;
    const uint64_t r13 = uint64_t(pool) * 2 * kInter, r2 = uint64_t(pool) * kHidden;
    maps.m13 = cached_map(w13.data_ptr(), r13, kTile, kChunks / 4);
    maps.m2 = cached_map(w2.data_ptr(), r2, kTile, kChunks / 4);
    for (int b = 0; b < kNumBuckets; ++b) {
      maps.b1[b] = cached_map(hs_perm.data_ptr(), uint64_t(kMaxSlots), uint32_t(bucket_rows(b)), kChunks / 4);
      maps.b2[b] = cached_map(hs2.data_ptr(), uint64_t(kMaxSlots), uint32_t(bucket_rows(b)), kChunks / 4);
    }
#if MUSE_VR_LDW_DENSE
    // the token tensor's M rows as the FC1 operand (rows past M zero-filled), as the narrow MoE-LD's token map
    RuntimeCheck(m <= kDenseRows, "dense-token build: at most 32 token rows per call");
    for (int b = 0; b < 2; ++b) maps.bd[b] = cached_map(hs.data_ptr(), uint64_t(m), uint32_t(bucket_rows(b)), kChunks / 4);
#endif
    [[maybe_unused]] static const bool configured = [] {
      preload();
      return true;
    }();
    LaunchKernel(dim3(static_cast<uint32_t>(NSM.unwrap())), dim3(kThreads), device_.unwrap(), kSmem)
        .enable_pdl(pdl != 0)(ldw_kernel, maps, p);
  }
};

}  // inline namespace MUSE_VR_LDW_ABI
}  // namespace muse_vr_ldw
#endif  // MUSE_VR_LDW_DEVICE_ONLY
