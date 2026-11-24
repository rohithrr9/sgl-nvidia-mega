# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Component ``unpool_slice_splitk`` (env ``MUSE_VR_UNPOOL_SLICE_SPLITK=1``, default off): the MoE unpooler's per-rank
column slice of small decode / verify batches on a cluster split-K GEMM with all token rows in one tile.

On SGLang's deferred-gather path a Muse Spark MoE layer's unpooler runs as a per-rank column slice (``gemm_local``: the
rank's contiguous block of the unpooler's output columns), then the FFN boundary gathers the slices. At the row counts
where SGLang's bf16 dispatch runs that slice on FlashInfer's warp split-K GEMM (its tuned warp split-K table), this
component runs it on ``qk_fold_gemm.run_splitk_grouped_gemm`` instead: FlashInfer's cluster split-K kernel class with every
token row in one tile (each weight byte is read once), a ring deep enough to stage a CTA's whole weight share ahead of its
dependency wait where the tactic allows, and the split-K partials exchanged in groups of four values per thread (the same
sum order as FlashInfer's own exchange). Programmatic dependent launch is on, as SGLang's split-K call passes it: the weight
loads run before the grid-dependency wait, the activation loads after it, and the output is a new tensor per call, as the
stock slice's.

A call is taken only when all of these hold (every other call runs SGLang's ``gemm_local`` unchanged; the reason is
counted): a communicator whose world size divides the unpooler's output columns; the unpooler weight as a contiguous bf16
[hidden, expert hidden] matrix on the input's device; the input as one contiguous bf16 row block with a row count in
``MUSE_VR_UNPOOL_SLICE_SPLITK_ROWS`` (default ``17-32``); SGLang's dispatch would run its warp split-K table for (rows,
slice columns, expert hidden); the tactic (``MUSE_VR_UNPOOL_SLICE_SPLITK_TACTIC``, default ``64,32,4,8``: mma_m, mma_n,
split_k, stages) fits the device's per-block shared memory (runtime query); no sequence parallelism; and not a first use
of a row count inside a CUDA graph capture (the kernel compiles on an eager call; SGLang's eager warm-ups precede each
capture). Registered before ``proj_publish``, whose ``gemm_local`` rebinding hands the rows it does not publish to this one.

Numerics: the precision of the stock slice (bf16 inputs, fp32 accumulation, one bf16 rounding of each output). The K split
and its fixed-order reduction differ from the warp split-K's, so the bits can differ from the stock slice (a reduction-order
class; deterministic, call- and rank-invariant). Units: tests/test_unpool_slice_splitk.py.
"""

from __future__ import annotations

import logging
import os
import threading

import torch

from . import bed

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "srt/models/muse_spark.py": ("gemm_local",),
    # the bf16 dispatch _warp_splitk_served mirrors
    "srt/layers/quantization/unquant.py": (
        "_BF16_LOW_M_AUTOTUNE_MAX_M",
        "_active_warp_splitk_tactics",
        "_enable_bf16_splitk_gemm",
        "_mm_bf16",
        "use_bf16_splitk_gemm",
    ),
    "srt/runtime_context.py": ("get_forward",),
}

logger = logging.getLogger(__name__)

NAME = "unpool_slice_splitk"  # env MUSE_VR_UNPOOL_SLICE_SPLITK
ROWS_ENV = "MUSE_VR_UNPOOL_SLICE_SPLITK_ROWS"
TACTIC_ENV = "MUSE_VR_UNPOOL_SLICE_SPLITK_TACTIC"
DEFAULT_ROWS = "17-32"
DEFAULT_TACTIC = (
    64,
    32,
    4,
    8,
)  # (mma_m, mma_n, split_k, ab_stages): one row tile at <= 32 rows, a whole K share per ring
H, EH = 8192, 4096  # unpooler weight [hidden, expert hidden]
# the SGLang functions of the call site (gemm_local on the deferred-gather path), the stock slice it replaces and the
# dispatch _warp_splitk_served mirrors (3117f58 + #48948)
PINS = {
    "sglang/srt/models/muse_spark.py": (
        bed.Functions(
            {
                "MuseSpark1xMoE._reduce_and_unpool": "84d2840940f3c49b",
                "MuseSpark1xMoE._defers_unpool_gather": "c730eaaec130b50e",
            }
        ),
    ),
    "sglang/srt/layers/communication/muse_spark_gemm_ag.py": (
        bed.Functions({"gemm_local": "5ddfc71a56e900f7"}),
    ),
    "sglang/srt/layers/quantization/unquant.py": (
        bed.Functions(
            {
                "initialize_bf16_gemm_config": "293a36153ec50715",
                "use_bf16_splitk_gemm": "2503e44bf976740d",
                "bf16_gemm_dispatch": "a774516d6d09eec9",
                "_bf16_low_m_autotune": "a9ac9d38b2056c34",
                "_bf16_gemm_dispatch_impl": "7f678770cd557687",
                "_bf16_splitk_gemm_out": "3a2b9e7f7fe14e9d",
            }
        ),
    ),
}

_state = {
    "engaged": None,
    "rows": frozenset(),
    "tactic": None,
    "warm": set(),
    "fits": {},
}
_lock = threading.Lock()


def _rows(v: str) -> frozenset:
    out = set()
    for tok in v.split(","):
        tok = tok.strip()
        if not tok:
            continue
        a, b = (int(x) for x in tok.split("-")) if "-" in tok else (int(tok), int(tok))
        if not 1 <= a <= b <= 32:
            raise ValueError(
                f"{ROWS_ENV}={v!r}: want row counts or 'lo-hi' ranges within 1..32"
            )
        out.update(range(a, b + 1))
    if not out:
        raise ValueError(f"{ROWS_ENV}={v!r}: no row count")
    return frozenset(out)


def _tactic(v: str):
    from .qk_fold_gemm import SplitKTactic

    vals = tuple(int(x) for x in v.split(","))
    if len(vals) != 4:
        raise ValueError(f"{TACTIC_ENV}={v!r}: want mma_m,mma_n,split_k,stages")
    return SplitKTactic(*vals)


def _warp_splitk_served(m: int, n: int, k: int) -> bool:
    """Whether SGLang's bf16 dispatch runs its warp split-K table for (m, n, k) (the table it consults first, after the
    low-M autotuned kernel)."""
    import sglang.srt.layers.quantization.unquant as U

    # _bf16_low_m_autotune takes M <= _BF16_LOW_M_AUTOTUNE_MAX_M first wherever FlashInfer's autotuned mm_bf16 is bound
    # (not on SM107)
    if U._mm_bf16 is not None and m <= U._BF16_LOW_M_AUTOTUNE_MAX_M:
        return False
    if not getattr(U, "_enable_bf16_splitk_gemm", False) or not U.use_bf16_splitk_gemm(
        m, n, k
    ):
        return False
    return (
        (m, n, k) in U._active_warp_splitk_tactics
    )  # the warp split-K table in use (with the SM107 entries on SM107)


def _fits(m: int, n: int, k: int, device_index: int) -> bool:
    key = (m, n, k, device_index)
    got = _state["fits"].get(key)
    if got is None:
        from .qk_fold_gemm import grouped_tactic_fits

        got = _state["fits"][key] = grouped_tactic_fits(
            _state["tactic"], m, n, k, device_index
        )
    return got


def why_stock(comm, x: torch.Tensor, weight: torch.Tensor):
    """None when the call is taken, else the counted reason."""
    from sglang.srt.runtime_context import get_forward

    if x.dim() != 2 or x.shape[0] not in _state["rows"]:
        return "rows"
    if comm is None or getattr(comm, "world_size", 0) < 1 or H % (comm.world_size * 64):
        return "comm"
    if x.device.type != "cuda":
        return "cpu"
    if getattr(get_forward(), "sp_active", False):
        return "sp"
    if (
        x.dtype != torch.bfloat16
        or not x.is_contiguous()
        or x.shape[1] != EH
        or x.data_ptr() % 32
    ):
        return "input"
    if (
        weight.dtype != torch.bfloat16
        or not weight.is_contiguous()
        or tuple(weight.shape) != (H, EH)
        or weight.device != x.device
        or weight.data_ptr() % 32
    ):
        return "weight"
    m, n = x.shape[0], H // comm.world_size
    if not _warp_splitk_served(m, n, EH):
        return "dispatch"
    if not _fits(m, n, EH, x.device.index):
        return "tactic"
    if (m, x.device.index) not in _state[
        "warm"
    ] and torch.cuda.is_current_stream_capturing():
        return "cold_capture"
    return None


def slice_gemm(comm, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """The rank's column slice ``x @ weight[rank block].T`` as a new bf16 [rows, hidden / world size] on the kernel."""
    from .qk_fold_gemm import run_splitk_grouped_gemm

    n = weight.shape[0] // comm.world_size
    w = weight[comm.rank * n : (comm.rank + 1) * n]
    out = torch.empty(x.shape[0], n, dtype=x.dtype, device=x.device)
    return run_splitk_grouped_gemm(x, w.T, out, True, _state["tactic"], 1)


def install() -> None:
    import sglang.srt.models.muse_spark as model

    if getattr(model.gemm_local, "__wrapped_by_unpool_slice_splitk__", False):
        return
    from . import (  # noqa: F401  (the kernel module imports here, not at the first call)
        qk_fold_gemm,
    )

    orig_local = model.gemm_local

    def gemm_local(comm, x, weight, *a, **k):
        why = "args" if a or k else why_stock(comm, x, weight)
        if why is None:
            out = slice_gemm(comm, x, weight)
            if not torch.cuda.is_current_stream_capturing():
                with _lock:
                    _state["warm"].add((x.shape[0], x.device.index))
            bed.count(NAME, "take:m%d" % x.shape[0])
            return out
        bed.count(NAME, "stock:" + why)
        return orig_local(comm, x, weight, *a, **k)

    gemm_local.__wrapped__ = orig_local
    gemm_local.__wrapped_by_unpool_slice_splitk__ = True
    model.gemm_local = gemm_local


def register() -> bool:
    if _state["engaged"] is not None:
        return _state["engaged"]
    if not bed.enabled(NAME):
        _state["engaged"] = False
        return False
    for rel, sha in PINS.items():
        if not bed.pinned(rel, sha):
            bed.report(NAME, False, reason=f"pin mismatch: {rel}")
            _state["engaged"] = False
            return False
    from .qkv_qk_fold import (
        _flashinfer_pinned,  # the FlashInfer kernel source the kernel class subclasses
    )

    if not _flashinfer_pinned():
        bed.report(
            NAME,
            False,
            reason="pin mismatch: flashinfer dense_bf16_gemm_sm100_splitk.py",
        )
        _state["engaged"] = False
        return False
    try:
        _state["rows"] = _rows(os.environ.get(ROWS_ENV, "").strip() or DEFAULT_ROWS)
        raw = os.environ.get(TACTIC_ENV, "").strip()
        _state["tactic"] = (
            _tactic(raw) if raw else _tactic(",".join(str(t) for t in DEFAULT_TACTIC))
        )
        install()
    except Exception as e:  # noqa: BLE001
        bed.report(NAME, False, reason=f"install failed: {type(e).__name__}: {e}")
        _state["engaged"] = False
        return False
    t = _state["tactic"]
    bed.report(
        NAME,
        True,
        rows=",".join(str(n) for n in sorted(_state["rows"])),
        tactic=f"{t.mma_m},{t.mma_n},{t.split_k},{t.ab_stages}",
        grouped_mailbox=1,
        pdl=1,
    )
    _state["engaged"] = True
    return True
