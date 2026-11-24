# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Paged FP8 full-attention verify for many requests per step, JIT CUDA (``csrc/full_attn_wide_l.cuh``).

A second form of ``ops.full_attn_wide``, with its grouped merge, length-balanced split and in-kernel
merge forms; ``verify_attn`` runs it in place of ``ops.full_attn_wide`` when MUSE_VR_FULL_ATTN_WIDE_GMERGE=1.

A persistent kernel (one thread block per SM, grid fixed at the SM count so the launch is CUDA-graph safe) splits the
summed key range of all requests evenly over its blocks, reads whole token rows of the K/V pools (every KV head of the
rank) and runs the attention on tensor-core MMA with per-tile-maximum probability quantization (as
``ops.full_verify_attn``); a second kernel merges the pieces of requests that several blocks share. Covers 8 query tokens per request, 8 query heads per KV head,
2 or 4 KV heads per rank, head_dim 64, e4m3 Q/K/V, page size 128, bf16 output, full attention, at most 256 requests.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

import torch

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "kernels/jit/utils/__init__.py": ("load_jit", "make_cpp_args"),
}

_CSRC = Path(__file__).resolve().parent.parent / "csrc" / "full_attn_wide_l.cuh"

TOKENS_PER_REQUEST = 8
HEAD_DIM = 64
Q_PER_KV = 8
PAGE = 128
REC_FLOATS = 68
ROWS = 64
MAX_REQUESTS = 256
# fewest requests per step at which the kernel runs in place of the stock decode kernel (tunable), per KV heads
# per rank (below it the stock path runs); MUSE_VR_FULL_ATTN_WIDE_MIN_REQS overrides it
BREAK_EVEN_REQUESTS = {4: 10, 2: 9}
TRACE_FLOATS = (
    3 * 16 * 256 * 2
)  # development mode 4: clock stamps after the partial records
# K/V ring depth (5: one Q buffer; tunable)
STAGES = int(os.environ.get("MUSE_VR_FULL_ATTN_WIDE_STAGES", "5"))
# keys of every 8 whose 2^x is computed by a degree-5 polynomial instead of ex2.approx (even, 0..8)
POLY = int(os.environ.get("MUSE_VR_FULL_ATTN_WIDE_POLY", "0"))
# tiles ahead whose K/V boxes are prefetched into L2 (0: off); development probes: MODE 1 = load-only (no attention)
PF = int(os.environ.get("MUSE_VR_FULL_ATTN_WIDE_PF", "0"))
MODE = int(os.environ.get("MUSE_VR_FULL_ATTN_WIDE_MODE", "0"))
# softmax threads per query row at 4 KV heads per rank (1: the default, tunable; 2: the two-thread layout used at 2 KV
# heads per rank)
TPR = int(os.environ.get("MUSE_VR_FULL_ATTN_WIDE_TPR", "1"))


# Grouped merge (MUSE_VR_FULL_ATTN_WIDE_GMERGE=1, default off): split requests are merged by a grouped kernel (several
# warps per row, index order; csrc "grouped merge") in place of the merge kernel, and with a gate the output norm + gate
# run in its epilogue (SGLang's norm_gate arithmetic for 16 threads per row; for a request inside one block, on the
# output the main kernel wrote). The main kernel is unchanged.
def gmerge_enabled() -> bool:
    return os.environ.get("MUSE_VR_FULL_ATTN_WIDE_GMERGE", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


# Query/key prologue (MUSE_VR_FULL_ATTN_WIDE_QK=1 with the grouped merge, default off; 1-16 requests): the blocks are
# apportioned to the requests by tile count (one piece per block; csrc "length-balanced split") and the kernel takes the
# bf16 QKV output: it computes the q rows and the new tokens' K / V rows itself (SGLang's fused QK-norm arithmetic, as
# ops.full_verify_attn's prologue) and stores the K / V rows into the pools. MUSE_VR_FULL_ATTN_WIDE_LB=1 (with the
# grouped merge, default off): the length-balanced split alone, with q from the caller (the served path's QK-norm).
LB_MAX_REQUESTS = 16
_ON = ("1", "true", "yes", "on")


def qk_enabled() -> bool:
    return os.environ.get("MUSE_VR_FULL_ATTN_WIDE_QK", "0").strip().lower() in _ON


def lb_enabled() -> bool:
    return os.environ.get("MUSE_VR_FULL_ATTN_WIDE_LB", "0").strip().lower() in _ON


# In-kernel merge (MUSE_VR_FULL_ATTN_WIDE_FD, default 0 = off; with the grouped merge and the length-balanced split): the main
# kernel merges the split requests itself (the grouped merge's per-row arithmetic, so the output is the same) and no merge
# kernel runs. 1: the last block of a request to finish merges all its rows; 2: every block of the request waits (one
# elected thread, bounded wait) until the request's last block arrived, then merges its share of the rows (one wave at one
# block per SM; else form 1). csrc "in-kernel merge". A call engages it only with ``fd_layer`` (the layer slot of its
# arrival words): the caller then advances the step word once per step with ``fd_step`` ahead of the step's first call.
# 4 (early trigger): no in-kernel merge; the main kernel triggers its dependents right after its own dependency wait instead
# of at its end, so the grouped merge's blocks land as the main kernel's blocks exit (the served grouped merge kernel and
# its wait on the main grid are unchanged; no arrival words, step word or layer slot).
# 5: form 4's early trigger with the low-register grouped merge (the grouped merge's arithmetic and output bit for bit,
# with fewer registers per thread so that more of its blocks are resident at once).
# With form 5, two more default-off knobs (no query/key prologue): MUSE_VR_FULL_ATTN_WIDE_FD_PRE (form 6 = 5 with the block
# map, the page reads and the first committed tiles' loads before the main kernel's dependency wait) and
# MUSE_VR_FULL_ATTN_WIDE_GM_TRIGGER (forms 7 = 5 and 8 = 6 with the trigger form of the low-register grouped merge, which
# triggers its own dependents, the output projection, at entry). csrc "kFd == 6", "launcher forms 7 and 8".
FD_FORMS = (0, 1, 2, 4, 5, 6, 7, 8)
FD_EARLY = (
    4,
    5,
    6,
    7,
    8,
)  # the early-trigger forms (no arrival words, step word or layer slot)
FD_LAYER_SLOTS = 64
FD_REQ_SLOTS = 16
# bound of the distributed form's wait, far above the main kernel's duration (tunable)
FD_MAX_WAIT_NS = 5_000_000


def fd_mode() -> int:
    raw = os.environ.get("MUSE_VR_FULL_ATTN_WIDE_FD", "0").strip()
    return int(raw) if raw in ("0", "1", "2", "4", "5") else 0


def fd_pre_for(num_requests: int) -> bool:
    """Whether form 5 takes the pre-wait form (6) for this request count (the host-side choice, made per call from the
    query rows). MUSE_VR_FULL_ATTN_WIDE_FD_PRE: "1" / "on" (every count), "<n>+" (n requests and more, e.g. "3+"), or a
    list of counts separated by "," or "/" (e.g. "3,4"); unset or "0": none. Tunable per count."""
    raw = os.environ.get("MUSE_VR_FULL_ATTN_WIDE_FD_PRE", "0").strip().lower()
    if raw in _ON:
        return True
    if raw.endswith("+") and raw[:-1].isdigit():
        return num_requests >= int(raw[:-1])
    try:
        return num_requests in {
            int(v) for v in raw.replace("/", ",").split(",") if v.strip()
        }
    except ValueError:
        return False


def gm_trigger_enabled() -> bool:
    return (
        os.environ.get("MUSE_VR_FULL_ATTN_WIDE_GM_TRIGGER", "0").strip().lower() in _ON
    )


@functools.lru_cache(maxsize=None)
def _module(
    pairs: int,
    stages: int,
    use_pdl: bool,
    poly: int,
    pf: int = 0,
    mode: int = 0,
    tpr: int = 2,
    gm: int = 0,
    lb: bool = False,
    qk: int = 0,
    unfused: bool = False,
    fd: int = 0,
):
    from sglang.kernels.jit.utils import load_jit, make_cpp_args

    args = make_cpp_args(
        pairs, stages, use_pdl, poly, pf, mode, tpr, gm, lb, qk, unfused, fd
    )
    return load_jit(
        "muse_vr_full_attn_wide_l",
        *args,
        cuda_files=[str(_CSRC)],
        cuda_wrappers=[
            ("full_attn_wide", f"muse_vr::FullAttnWideKernel<{args}>::run"),
            ("fd_step", "muse_vr::FullAttnWideFdStep::run"),
        ],
    )


_fd_ws: dict = {}


def fd_workspace(device) -> torch.Tensor:
    """Per-device int64 [64 * 16 + 8] of the in-kernel merge: arrival words [layer slot][request], the step word, the sticky
    error word. Zero at allocation, never reset (the words carry the step word as an epoch); allocate before capture."""
    key = (
        device.type,
        device.index if device.index is not None else torch.cuda.current_device(),
    )
    ws = _fd_ws.get(key)
    if ws is None:
        ws = torch.zeros(
            FD_LAYER_SLOTS * FD_REQ_SLOTS + 8, dtype=torch.int64, device=device
        )
        _fd_ws[key] = ws
    return ws


def fd_step(device, fd: int | None = None) -> None:
    """Advance the step word by one (one plain single-thread launch on the current stream); once per step, before the
    step's first in-kernel-merge call. ``fd`` picks the module that carries the launcher (default: the knob's form)."""
    form = int(fd) if fd in FD_FORMS[1:] else (fd_mode() or 1)
    mod = _module(2, STAGES, True, POLY, 0, 0, TPR, 2, True, 0, False, form)
    mod.fd_step(fd_workspace(device))


def fd_error_word(device) -> int:
    """The sticky error word (nonzero: a distributed wait expired; bit 31 set, bits 16-23 the layer slot, bit r the
    request)."""
    return (
        int(fd_workspace(device)[FD_LAYER_SLOTS * FD_REQ_SLOTS + 1].item()) & 0xFFFFFFFF
    )


@functools.lru_cache(maxsize=None)
def _module_test_neg(
    pairs: int,
    stages: int,
    use_pdl: bool,
    tpr: int,
    gm: int,
    lb: bool = False,
    qk: int = 0,
    unfused: bool = False,
):
    """Tests only: the grouped-merge build with MUSE_VR_FAW_GM_TEST_NEG (each request's merge reads the next request's
    pieces), a negative control the accuracy checks must catch; its namespace is renamed so its instantiations are distinct
    symbols from the regular module's in one process."""
    from sglang.kernels.jit.utils import load_jit, make_cpp_args

    args = make_cpp_args(pairs, stages, use_pdl, 0, 0, 0, tpr, gm, lb, qk, unfused)
    return load_jit(
        "muse_vr_full_attn_wide_l_gmneg",
        *args,
        cuda_files=[str(_CSRC)],
        cuda_wrappers=[("full_attn_wide", f"muse_vr::FullAttnWideKernel<{args}>::run")],
        extra_cuda_cflags=["-DMUSE_VR_FAW_GM_TEST_NEG=1", "-Dmuse_vr=muse_vr_gmneg"],
    )


@functools.lru_cache(maxsize=None)
def _module_test_pre(
    pairs: int, stages: int, use_pdl: bool, tpr: int, gm: int, fd: int
):
    """Tests only: the forms 6 / 8 build with MUSE_VR_FAW_PRE_PROBE (each block with tiles records, after its wait, how
    many tiles it issued before the wait; see ``pre_probe_counts``); its namespace is renamed so its instantiations are
    distinct symbols from the regular module's in one process."""
    from sglang.kernels.jit.utils import load_jit, make_cpp_args

    args = make_cpp_args(pairs, stages, use_pdl, 0, 0, 0, tpr, gm, True, 0, False, fd)
    return load_jit(
        "muse_vr_full_attn_wide_l_preprobe",
        *args,
        cuda_files=[str(_CSRC)],
        cuda_wrappers=[("full_attn_wide", f"muse_vr::FullAttnWideKernel<{args}>::run")],
        extra_cuda_cflags=["-DMUSE_VR_FAW_PRE_PROBE=1", "-Dmuse_vr=muse_vr_preprobe"],
    )


@functools.lru_cache(maxsize=None)
def _module_ns(
    ns: str, pairs: int, stages: int, use_pdl: bool, tpr: int, gm: int, fd: int
):
    """Tests and units only: the serving form's build (poly 0, length-balanced split, no prologue) under a renamed
    namespace ``ns``, so that a unit comparing several forms in one process loads no kernel entry twice."""
    from sglang.kernels.jit.utils import load_jit, make_cpp_args

    if not ns.isidentifier() or not ns.startswith("muse_vr_"):
        raise ValueError("namespace must be an identifier starting with muse_vr_")
    args = make_cpp_args(pairs, stages, use_pdl, 0, 0, 0, tpr, gm, True, 0, False, fd)
    return load_jit(
        f"muse_vr_full_attn_wide_l_{ns}",
        *args,
        cuda_files=[str(_CSRC)],
        cuda_wrappers=[("full_attn_wide", f"muse_vr::FullAttnWideKernel<{args}>::run")],
        extra_cuda_cflags=[f"-Dmuse_vr={ns}"],
    )


@functools.lru_cache(maxsize=None)
def _module_precheck(
    pairs: int, stages: int, use_pdl: bool, tpr: int, gm: int, fd: int
):
    """Tests and units only: the serving form's build (as ``_module_ns``) with MUSE_VR_FAW_PRE_CHECK: form 6 reads every
    word it read before its dependency wait again after the wait and counts mismatches (``precheck_counters``); plus
    the late-writer predecessor of the forced-stale control. Renamed namespace, as ``_module_ns``."""
    from sglang.kernels.jit.utils import load_jit, make_cpp_args

    args = make_cpp_args(pairs, stages, use_pdl, 0, 0, 0, tpr, gm, True, 0, False, fd)
    return load_jit(
        "muse_vr_full_attn_wide_l_precheck",
        *args,
        cuda_files=[str(_CSRC)],
        cuda_wrappers=[
            ("full_attn_wide", f"muse_vr::FullAttnWideKernel<{args}>::run"),
            ("late_writer", "muse_vr::FullAttnWideLateWriter::run"),
        ],
        extra_cuda_cflags=["-DMUSE_VR_FAW_PRE_CHECK=1", "-Dmuse_vr=muse_vr_precheck"],
    )


def precheck_counters(device, num_kv_heads: int, blocks: int) -> torch.Tensor:
    """Tests and units only: the check build's counters, int64 [metadata mismatches, metadata words compared, tile
    checksum mismatches, tile 16-byte words compared] (a view of the workspace after the records; the caller zeroes it
    before the calls it counts)."""
    ws, _ = workspace(device, num_kv_heads, blocks)
    off = blocks * 2 * num_kv_heads * ROWS * REC_FLOATS
    return ws[off : off + 8].view(torch.int64)


def late_writer(
    seq_lens,
    seq_new,
    page_table,
    table_new,
    row,
    row_new,
    delay_us: int,
    num_kv_heads: int,
    trigger: bool = True,
) -> None:
    """Tests and units only: launch a metadata writer on the current stream: it waits ~delay_us, then copies seq_new /
    table_new / row_new over seq_lens / page_table / row (row is a uint8 view, e.g. one committed key row of the pool).
    trigger=True (the forced-stale control): it triggers its dependents at entry, so the next full_attn_wide call with
    use_pdl=True and _precheck=True may start before the copy lands. trigger=False: no programmatic trigger (the shape of
    an in-graph metadata kernel launched plainly)."""
    _module_precheck(
        num_kv_heads // 2, STAGES, True, TPR if num_kv_heads == 4 else 2, 2, 6
    ).late_writer(
        seq_lens,
        seq_new,
        page_table,
        table_new,
        row,
        row_new,
        int(delay_us),
        int(bool(trigger)),
    )


def pre_probe_counts(device, num_kv_heads: int, blocks: int) -> torch.Tensor:
    """Tests only: the per-block counts written by the ``_test_pre`` build (int32 view of the workspace after the
    records; the caller fills it with a sentinel before the call)."""
    ws, _ = workspace(device, num_kv_heads, blocks)
    off = blocks * 2 * num_kv_heads * ROWS * REC_FLOATS
    return ws[off : off + blocks].view(torch.int32)


@functools.lru_cache(maxsize=None)
def _sm_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def num_blocks(device) -> int:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    raw = os.environ.get("MUSE_VR_FULL_ATTN_WIDE_BLOCKS", "")
    return int(raw) if raw else _sm_count(idx)


_workspace: dict = {}


def workspace(device, num_kv_heads: int, blocks: int, gm: bool = False):
    """Per-device fp32 partial records and the per-request counters (zero; the kernel leaves them zero). Grows, never
    shrinks; allocate before graph capture. (The grouped merge uses the same records: gm does not change the size.)"""
    key = (
        device.type,
        device.index if device.index is not None else torch.cuda.current_device(),
    )
    need = blocks * 2 * num_kv_heads * ROWS * REC_FLOATS + TRACE_FLOATS
    ws = _workspace.get(key)
    if ws is None or ws[0].numel() < need:
        counters = (
            ws[1]
            if ws is not None
            else torch.zeros(MAX_REQUESTS, dtype=torch.int32, device=device)
        )
        ws = (torch.empty(need, dtype=torch.float32, device=device), counters)
        _workspace[key] = ws
    return ws


_lb_ws: dict = {}


def lb_map_buffer(device) -> torch.Tensor:
    """Per-device int32 [2 * 16]: each request's (first block, block count) of the length-balanced split, written by the
    main kernel and read by the merge (allocate before graph capture; preload does)."""
    key = (
        device.type,
        device.index if device.index is not None else torch.cuda.current_device(),
    )
    buf = _lb_ws.get(key)
    if buf is None:
        buf = torch.zeros(2 * LB_MAX_REQUESTS, dtype=torch.int32, device=device)
        _lb_ws[key] = buf
    return buf


def lb_map_reference(
    seq_lens: list[int], blocks: int, num_kv_heads: int
) -> list[tuple[int, int]]:
    """Python reference of the device map (csrc faw_lb_lane): (first block, block count) per request."""
    tile = 128 // (num_kv_heads // 2)
    n = [min(max((max(s, 1) + tile - 1) // tile, 1), (1 << 20) - 1) for s in seq_lens]
    R, G, W = len(n), blocks, sum(n)
    b = [1] * R
    if G > R:
        t = (W + G - 1) // G
        for it in range(8):
            b = [(x + t - 1) // t for x in n]
            if sum(b) <= G:
                break
            if it == 7:
                t = (W + (G - R) - 1) // (G - R)
                b = [(x + t - 1) // t for x in n]
                break
            t += 1
    out, pre = [], 0
    for x in b:
        out.append((pre, x))
        pre += x
    return out


def supported(
    num_q_heads: int, num_kv_heads: int, head_dim: int, page_size: int
) -> bool:
    return (
        head_dim == HEAD_DIM
        and num_kv_heads * Q_PER_KV == num_q_heads
        and num_kv_heads in (2, 4)
        and page_size == PAGE
    )


def full_attn_wide(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    bmm1_scale: float,
    bmm2_scale: float,
    out: torch.Tensor | None = None,
    use_pdl: bool = False,
    blocks: int | None = None,
    stages: int | None = None,
    poly: int | None = None,
    gate: torch.Tensor | None = None,
    norm_eps: float = 0.0,
    gmerge: bool | None = None,
    qkv: torch.Tensor | None = None,
    cache_loc: torch.Tensor | None = None,
    qk_eps: float = 0.0,
    num_q_heads: int | None = None,
    lb: bool | None = None,
    fd: int | None = None,
    fd_layer: int | None = None,
    fd_pre: bool | None = None,
    gm_trigger: bool | None = None,
    _test_neg: bool = False,
    _fd_test: tuple | None = None,
    _test_pre: bool = False,
    _ns: str | None = None,
    _precheck: bool = False,
) -> torch.Tensor:
    """Full attention of 8 consecutive query tokens per request over a paged FP8 KV pool.

    :param q: ``[R * 8, Hq, 64]`` e4m3, heads and dims contiguous, token stride a multiple of 16 bytes.
    :param k_pool: ``[slots, Hkv * 64]`` e4m3, contiguous (slot = page * 128 + offset); Hkv 2 or 4.
    :param v_pool: same layout as ``k_pool``.
    :param page_table: ``[>= R, pages]`` int32, inner dimension contiguous.
    :param seq_lens: ``[R]`` int32, KV length per request including the 8 query tokens.
    :param blocks: thread blocks (default: the SM count; ``MUSE_VR_FULL_ATTN_WIDE_BLOCKS`` overrides).
    :param gmerge: grouped merge (default: ``MUSE_VR_FULL_ATTN_WIDE_GMERGE``).
    :param gate: with the grouped merge only: ``[R * 8, >= Hq * 64]`` bf16 (inner dimension contiguous); the output is then
        ``rms_norm(attention, eps=norm_eps) * sigmoid(gate)`` per (token, head) row, as SGLang's norm_gate kernel computes
        it with 16 threads per row.
    :param qkv: query/key prologue (layers without RoPE; with the grouped merge, 1-16 requests, blocks >= requests):
        ``[R * 8, >= (Hq + 2 Hkv) * 64]`` bf16 QKV projection output (q, k, v, ...), inner dimension contiguous, in place
        of ``q`` (pass ``q=None``); the kernel computes q and the new tokens' K / V rows (SGLang's fused QK-norm
        arithmetic, ``qk_eps``) and stores K / V into the pools at ``cache_loc`` (``[R * 8]`` int64; 0: not stored).
        Needs ``num_q_heads`` and a row count that ``swa_verify_attn.qk_class`` covers.
    :param lb: length-balanced split without the prologue (default: ``MUSE_VR_FULL_ATTN_WIDE_LB``); applies with the
        grouped merge at 1-16 requests and blocks >= requests, else the even split runs.
    :param fd: in-kernel merge form (default: ``MUSE_VR_FULL_ATTN_WIDE_FD``; 0 off, 1 last block, 2 distributed, 4 early
        trigger of the grouped merge, 5 early trigger + the low-register grouped merge); applies with the grouped merge, the
        length-balanced split and ``fd_layer`` (form 2 only when blocks <= the SM count; forms 4 / 5 need no ``fd_layer``
        and no prologue), else off.
    :param fd_layer: the call's layer slot (0-63) of the in-kernel merge's arrival words; the caller advances the step
        word with ``fd_step`` once per step before the step's first such call.
    :param fd_pre: with form 5: the block map, page reads and first committed tiles' loads before the main kernel's
        dependency wait (form 6; default: ``MUSE_VR_FULL_ATTN_WIDE_FD_PRE`` for this call's request count, fd_pre_for).
    :param gm_trigger: with form 5: the low-register grouped merge triggers its dependents at entry (forms 7 / 8;
        default: ``MUSE_VR_FULL_ATTN_WIDE_GM_TRIGGER``).
    """
    hkv = k_pool.shape[1] // HEAD_DIM
    if gmerge is None:
        gmerge = gmerge_enabled()
    qk, unfused = 0, False
    if qkv is not None:
        from .swa_verify_attn import qk_class, qk_unfused

        m, hq, dev = qkv.shape[0], int(num_q_heads), qkv.device
        qk = qk_class(m, hq, hkv)
        if qk is None:
            raise ValueError("row count not covered by the query/key prologue")
        unfused = qk_unfused(qk)
    else:
        m, hq, dev = q.shape[0], q.shape[1], q.device
    reqs = m // TOKENS_PER_REQUEST
    if out is None:
        out = torch.empty(m, hq, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    g = blocks or num_blocks(dev)
    if gate is not None and not gmerge:
        raise ValueError("the fused norm + gate needs the grouped merge")
    lb_ok = bool(gmerge) and 1 <= reqs <= LB_MAX_REQUESTS and g >= reqs
    if qkv is not None:
        if not lb_ok:
            raise ValueError(
                "the query/key prologue needs the grouped merge, 1-16 requests and blocks >= requests"
            )
        lb = True
    else:
        lb = (lb_enabled() if lb is None else bool(lb)) and lb_ok
    gm = (2 if gate is not None else 1) if gmerge else 0
    fdm = fd_mode() if fd is None else int(fd)
    if (
        fdm not in FD_FORMS
        or not (gm and lb)
        or (fd_layer is None and fdm not in FD_EARLY)
        or (fdm in FD_EARLY and qk)
    ):
        fdm = 0  # forms 4 / 5 (early trigger) use no arrival words, so they need no layer slot
    elif fdm == 2 and g > _sm_count(
        dev.index if dev.index is not None else torch.cuda.current_device()
    ):
        fdm = 1  # one wave at one block per SM needed for the distributed wait
    elif fdm == 5:
        fdm += (1 if (fd_pre_for(reqs) if fd_pre is None else fd_pre) else 0) + (
            2 if (gm_trigger_enabled() if gm_trigger is None else gm_trigger) else 0
        )
    max_wait_ns, skip_block, swap = (
        _fd_test if _fd_test is not None else (FD_MAX_WAIT_NS, 0xFFFFFFFF, 0)
    )
    fdw = fd_workspace(dev) if fdm in (1, 2) else seq_lens
    ws, counters = workspace(dev, hkv, g, gm > 0)
    lbm = lb_map_buffer(dev)
    flat = out.view(out.shape[0], -1)
    tpr = TPR if hkv == 4 else 2
    if _test_pre and fdm not in (6, 8):
        raise ValueError("the pre-wait probe build needs form 6 or 8")
    if (_ns is not None or _precheck) and (
        qk or not lb or not gm or int(POLY if poly is None else poly) != 0
    ):
        raise ValueError("the renamed-namespace build covers the serving form only")
    if _precheck and (_ns is not None or _test_pre or fdm not in (5, 6)):
        raise ValueError("the check build: forms 5 / 6 only, alone")
    mod = (
        _module_test_neg(
            hkv // 2, stages or STAGES, bool(use_pdl), tpr, gm, lb, qk, unfused
        )
        if (gm and _test_neg)
        else _module_test_pre(hkv // 2, stages or STAGES, bool(use_pdl), tpr, gm, fdm)
        if _test_pre
        else _module_ns(_ns, hkv // 2, stages or STAGES, bool(use_pdl), tpr, gm, fdm)
        if _ns is not None
        else _module_precheck(hkv // 2, stages or STAGES, bool(use_pdl), tpr, gm, fdm)
        if _precheck
        else _module(
            hkv // 2,
            stages or STAGES,
            bool(use_pdl),
            POLY if poly is None else int(poly),
            PF if not gm else 0,
            MODE if not gm else 0,
            tpr,
            gm,
            lb,
            qk,
            unfused,
            fdm,
        )
    )
    mod.full_attn_wide(
        q if q is not None else out,
        k_pool,
        v_pool,
        page_table,
        seq_lens,
        out,
        ws,
        counters,
        int(g),
        float(bmm1_scale),
        float(bmm2_scale),
        gate if gate is not None else flat,
        float(norm_eps),
        qkv if qkv is not None else flat,
        cache_loc if cache_loc is not None else seq_lens,
        lbm,
        float(qk_eps),
        fdw,
        int(fd_layer or 0),
        int(max_wait_ns),
        int(skip_block),
        int(swap),
    )
    return out


def preload(num_q_heads: int, num_kv_heads: int, device, use_pdl: bool = False) -> None:
    """Build and load the kernel and size the workspace, so that no build, module load, function attribute call or
    allocation happens inside a CUDA-graph capture."""
    key = (
        device.type,
        device.index if device.index is not None else torch.cuda.current_device(),
        num_q_heads,
        num_kv_heads,
        use_pdl,
    )
    if key in _preloaded:
        return
    q = torch.zeros(
        TOKENS_PER_REQUEST,
        num_q_heads,
        HEAD_DIM,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    pool = torch.zeros(
        2 * PAGE, num_kv_heads * HEAD_DIM, dtype=torch.float8_e4m3fn, device=device
    )
    table = torch.zeros(1, 2, dtype=torch.int32, device=device)
    lens = torch.full((1,), TOKENS_PER_REQUEST, dtype=torch.int32, device=device)
    full_attn_wide(q, pool, pool, table, lens, 1.0, 1.0, use_pdl=use_pdl)
    if (
        gmerge_enabled()
    ):  # the fused norm + gate form too (the served path passes the gate when it may)
        gate = torch.zeros(
            TOKENS_PER_REQUEST,
            num_q_heads * HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
        )
        full_attn_wide(
            q,
            pool,
            pool,
            table,
            lens,
            1.0,
            1.0,
            use_pdl=use_pdl,
            gate=gate,
            norm_eps=1e-6,
        )
        if lb_enabled():  # above 16 requests the even split runs: its grouped-merge forms too (and the map buffer)
            lb_map_buffer(device)
            full_attn_wide(
                q, pool, pool, table, lens, 1.0, 1.0, use_pdl=use_pdl, lb=False
            )
            full_attn_wide(
                q,
                pool,
                pool,
                table,
                lens,
                1.0,
                1.0,
                use_pdl=use_pdl,
                gate=gate,
                norm_eps=1e-6,
                lb=False,
            )
        if qk_enabled():  # the prologue forms: one per column class over 2-16 requests, with and without the gate
            from .swa_verify_attn import qk_class

            lb_map_buffer(device)
            first: dict = {}
            for n in range(2, LB_MAX_REQUESTS + 1):
                first.setdefault(
                    qk_class(n * TOKENS_PER_REQUEST, num_q_heads, num_kv_heads), n
                )
            for cls, n in first.items():
                if cls is None:
                    continue
                rows = n * TOKENS_PER_REQUEST
                qkv = torch.zeros(
                    rows,
                    (2 * num_q_heads + 2 * num_kv_heads) * HEAD_DIM,
                    dtype=torch.bfloat16,
                    device=device,
                )
                loc = torch.zeros(
                    rows, dtype=torch.int64, device=device
                )  # 0: nothing stored
                tab = torch.zeros(n, 2, dtype=torch.int32, device=device)
                ln = torch.full(
                    (n,), TOKENS_PER_REQUEST, dtype=torch.int32, device=device
                )
                for g in (None, qkv[:, (num_q_heads + 2 * num_kv_heads) * HEAD_DIM :]):
                    full_attn_wide(
                        None,
                        pool,
                        pool,
                        tab,
                        ln,
                        1.0,
                        1.0,
                        use_pdl=use_pdl,
                        gate=g,
                        norm_eps=1e-6,
                        qkv=qkv,
                        cache_loc=loc,
                        qk_eps=1e-6,
                        num_q_heads=num_q_heads,
                    )
    torch.cuda.current_stream(device).synchronize()
    _preloaded.add(key)


_preloaded: set = set()
