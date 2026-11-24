# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Component ``pooler_splitk`` (env ``MUSE_VR_POOLER_SPLITK=1``, default off): the MoE pooler GEMM of small decode / verify
batches on FlashInfer's split-K bf16 GEMM.

SGLang's ``MuseSpark1xMoE._forward_experts`` computes the pooler (``pre_expert_proj``, the replicated weight) through
``bf16_gemm_dispatch``. The dispatcher has no tuned split-K entry for the pooler's shape (N 4096, K 8192), so at small row
counts the call runs ``F.linear``. With this component that one call runs FlashInfer's split-K dense bf16 GEMM
(``run_splitk_dense``: one launch, the K split reduced inside the cluster) at the tactic SGLang lists for the 4608-wide
projection of the same K at these row counts, in the call form of SGLang's own split-K path (``unquant._bf16_splitk_gemm_out``:
an output allocated per call, the transposed weight view, no bias). Programmatic dependent launch: on by default, as SGLang's
split-K call passes it; ``MUSE_VR_POOLER_SPLITK_PDL=0`` launches the GEMM without it, so it starts only after its predecessor
has completed and none of its loads run beside that predecessor (same kernel and arithmetic either way). Everything else is
SGLang's: the router GEMM and the routing op on the side stream, the join, the experts' input norm + quant, the experts
call.

The call is found by identity. The MoE forward wrapper arms that layer's pooler weight for its own call; the model module's
``bf16_gemm_dispatch`` name, rebound, takes a call only when its weight is the armed one and its row count is admitted
(``MUSE_VR_POOLER_SPLITK_ROWS``, default ``24,32``: comma-separated row counts or ``lo-hi`` ranges). Every other call (the
unpooler, other weights, other row counts, a bias or extra arguments) reaches the dispatcher unchanged. The dispatcher is
also kept (and the reason counted) for rows that are not a contiguous bf16 row block, a weight that is not a contiguous
bf16 [4096, 8192] block on the same device, a CPU tensor, sequence parallelism, and a first use of a row count inside a CUDA
graph capture (the kernel is compiled on an eager call; SGLang's eager warm-up runs precede each capture). Registered before
``proj_publish``: that component keeps the pooler rows it publishes and hands the others to this one.

Numerics: the precision of the stock pooler (bf16 inputs, fp32 accumulation, one bf16 rounding of each output). The split-K
form adds the two K halves in a fixed order, so the bits can differ from the stock GEMM's (a reduction-order class;
deterministic and call-invariant). Units: tests/test_pooler_splitk.py.
"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import os
import threading
from pathlib import Path

import torch

from . import bed

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "srt/models/muse_spark.py": ("MuseSpark1xMoE", "bf16_gemm_dispatch"),
    "srt/runtime_context.py": ("get_forward",),
}

logger = logging.getLogger(__name__)

NAME = "pooler_splitk"  # env MUSE_VR_POOLER_SPLITK
ROWS_ENV = "MUSE_VR_POOLER_SPLITK_ROWS"
DEFAULT_ROWS = "24,32"
PDL_ENV = (
    "MUSE_VR_POOLER_SPLITK_PDL"  # 1 (default): programmatic dependent launch on; 0: off
)
H, EH = 8192, 4096
TACTIC = (64, 32, 2, 9)  # FlashInfer SplitKTactic(mma_m, mma_n, split_k, ab_stages)
# the SGLang functions whose call site, MoE forward and split-K call form this component relies on (3117f58 + #48948)
PINS = {
    "sglang/srt/models/muse_spark.py": (
        bed.Functions(
            {
                "MuseSpark1xMoE._forward_experts": "04c56c86a0d682b6",
                "MuseSpark1xMoE.__init__": "38b36b568c633d3e",
            }
        ),
    ),
    "sglang/srt/layers/quantization/unquant.py": (
        bed.Functions(
            {
                "bf16_gemm_dispatch": "a774516d6d09eec9",
                "_bf16_gemm_dispatch_impl": "7f678770cd557687",
                "_bf16_splitk_gemm_out": "3a2b9e7f7fe14e9d",
            }
        ),
    ),
}
# sha256 of the FlashInfer kernel source this component calls (0.7.2), relative to the flashinfer package directory
FI_KERNEL = "gemm/kernels/dense_bf16_gemm_sm100_splitk.py"
FI_PIN = "f5edd4c11d95822c836a363dfa1dd19f7843b4e354e091c24c0acac12f52d9d2"

_state = {
    "engaged": None,
    "rows": frozenset(),
    "pdl": True,
    "warm": set(),
    "kernel": None,
}
_tls = threading.local()
_lock = threading.Lock()


def _rows(v: str) -> frozenset:
    out = set()
    for tok in v.split(","):
        tok = tok.strip()
        if not tok:
            continue
        a, b = (int(x) for x in tok.split("-")) if "-" in tok else (int(tok), int(tok))
        if not 1 <= a <= b <= 512:
            raise ValueError(
                f"{ROWS_ENV}={v!r}: want row counts or 'lo-hi' ranges within 1..512"
            )
        out.update(range(a, b + 1))
    if not out:
        raise ValueError(f"{ROWS_ENV}={v!r}: no row count")
    return frozenset(out)


def _rows_label(rows) -> str:
    return ",".join(str(n) for n in sorted(rows))


def fi_kernel_sha() -> str | None:
    """sha256 of the installed FlashInfer kernel source (found without importing the package)."""
    spec = importlib.util.find_spec("flashinfer")
    if spec is None or not spec.submodule_search_locations:
        return None
    try:
        return hashlib.sha256(
            (Path(list(spec.submodule_search_locations)[0]) / FI_KERNEL).read_bytes()
        ).hexdigest()
    except OSError:
        return None


def _kernel():
    k = _state["kernel"]
    if k is None:
        from flashinfer.gemm.kernels.dense_bf16_gemm_sm100_splitk import (
            SplitKTactic,
            run_splitk_dense,
        )

        k = _state["kernel"] = (run_splitk_dense, SplitKTactic(*TACTIC))
    return k


def _flag(env: str, default: bool) -> bool:
    v = os.environ.get(env, "").strip().lower()
    if not v:
        return default
    if v not in ("0", "1", "true", "false", "yes", "no", "on", "off"):
        raise ValueError(f"{env}={v!r}: want 0 or 1")
    return v in ("1", "true", "yes", "on")


def splitk_gemm(
    x: torch.Tensor, weight: torch.Tensor, pdl: bool | None = None
) -> torch.Tensor:
    """``x`` [m, K] bf16 times ``weight`` [N, K] bf16 transposed -> a new bf16 [m, N]: FlashInfer's split-K GEMM at TACTIC in
    the call form of SGLang's split-K path; ``pdl`` None takes the component's setting."""
    run, tactic = _kernel()
    out = torch.empty(x.shape[0], weight.shape[0], dtype=x.dtype, device=x.device)
    run(x, weight.T, None, out, _state["pdl"] if pdl is None else bool(pdl), tactic)
    return out


def _why_stock(x: torch.Tensor, weight: torch.Tensor):
    from sglang.srt.runtime_context import get_forward

    if x.dim() != 2 or x.shape[0] not in _state["rows"]:
        return "rows"
    if x.device.type != "cuda" or not torch.cuda.is_available():
        return "cpu"
    if getattr(get_forward(), "sp_active", False):
        return "sp"
    if x.dtype != torch.bfloat16 or not x.is_contiguous() or x.shape[1] != H:
        return "input"
    if (
        weight.dtype != torch.bfloat16
        or not weight.is_contiguous()
        or tuple(weight.shape) != (EH, H)
        or weight.device != x.device
    ):
        return "weight"
    if (x.shape[0], x.device.index) not in _state[
        "warm"
    ] and torch.cuda.is_current_stream_capturing():
        return "cold_capture"
    return None


def install() -> None:
    import sglang.srt.models.muse_spark as model

    moe = model.MuseSpark1xMoE
    if getattr(moe._forward_experts, "__wrapped_by_pooler_splitk__", False):
        return
    _kernel()  # the FlashInfer entry must import here, not at the first call
    orig_experts = moe._forward_experts
    orig_dispatch = model.bf16_gemm_dispatch

    def bf16_gemm_dispatch(x, weight, bias, *a, **k):
        if weight is not None and weight is getattr(_tls, "armed", None):
            _tls.armed = None  # one pooler call per armed MoE call
            why = "args" if bias is not None or a or k else _why_stock(x, weight)
            if why is None:
                out = splitk_gemm(x, weight)
                if not torch.cuda.is_current_stream_capturing():
                    with _lock:
                        _state["warm"].add((x.shape[0], x.device.index))
                bed.count(NAME, "take:m%d" % x.shape[0])
                return out
            bed.count(NAME, "stock:" + why)
        return orig_dispatch(x, weight, bias, *a, **k)

    def _forward_experts(self, hidden_states, router_input=None, *, ar_workspace=None):
        prev = getattr(_tls, "armed", None)
        proj = self.pre_expert_proj
        _tls.armed = getattr(proj, "weight", None) if proj is not None else None
        try:
            return orig_experts(
                self, hidden_states, router_input, ar_workspace=ar_workspace
            )
        finally:
            _tls.armed = prev

    bf16_gemm_dispatch.__wrapped__ = orig_dispatch
    _forward_experts.__wrapped__ = orig_experts
    _forward_experts.__wrapped_by_pooler_splitk__ = True
    model.bf16_gemm_dispatch = bf16_gemm_dispatch
    moe._forward_experts = _forward_experts


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
    got = fi_kernel_sha()
    if got != FI_PIN:
        logger.warning(
            "muse_vr_plugin: flashinfer/%s sha256 %s does not match the pin %s",
            FI_KERNEL,
            (got or "<missing>")[:16],
            FI_PIN[:16],
        )
        bed.report(NAME, False, reason=f"pin mismatch: flashinfer/{FI_KERNEL}")
        _state["engaged"] = False
        return False
    try:
        _state["rows"] = _rows(os.environ.get(ROWS_ENV, "").strip() or DEFAULT_ROWS)
        _state["pdl"] = _flag(PDL_ENV, True)
        install()
    except Exception as e:  # noqa: BLE001
        bed.report(NAME, False, reason=f"install failed: {type(e).__name__}: {e}")
        _state["engaged"] = False
        return False
    bed.report(
        NAME,
        True,
        rows=_rows_label(_state["rows"]),
        tactic=",".join(str(t) for t in TACTIC),
        pdl=int(_state["pdl"]),
    )
    _state["engaged"] = True
    return True
