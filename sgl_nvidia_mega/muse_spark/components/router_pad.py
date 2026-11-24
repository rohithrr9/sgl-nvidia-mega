# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Component ``router_pad`` (env ``MUSE_VR_ROUTER_PAD=1``, default off): the MoE router GEMM on a padded row count.

The Muse Spark router GEMM (``MuseSpark1xRouter.forward`` = ``F.linear`` of the fp32 router input under the served float32
matmul precision) goes to cuBLAS, whose kernel choice depends on the row count. With this component, for the row counts in
``MUSE_VR_ROUTER_PAD_ROWS`` (``lo-hi:P[,lo-hi:P...]``; tunable, default
none) the boundary kernel that produces the router input writes it into the first rows of a persistent ``[P, hidden]`` fp32
buffer (one per device, P and producer stream; created zero outside any graph capture and never reallocated), and the router
GEMM runs on all ``P`` rows; the router logits are rows ``[0, M)`` of that product. Rows of a GEMM are independent, so the
logits differ from the stock call only through the kernel choice. A band ``lo-hi`` is engaged only after an engage check on
its first eager call (outside graph capture, on the calling stream, with the served router weight and float32 matmul
settings): for EVERY row count of the band, the padded product's rows ``[0, M)`` must be bitwise equal to the stock call,
with the pad rows zero and with them holding non-finite and large stale values; the kernel names of the stock and the padded
call are recorded per row count. A failed check keeps the band on the stock call; a later change of the float32 matmul
settings turns the component off for every band. Two-batch / single-batch overlap refuses the component at engage.

Producers covered: SGLang's JIT boundary (``fused_norm_residual``, which takes the router output buffer as an argument) and
this plugin's ``ar_norm_residual`` / ``ar2_norm_residual`` (which ask ``router_out_for``). Any other producer leaves the
router input unpadded and the router GEMM unchanged (counted).
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Tuple

import torch

from . import bed, sgl_compat

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    # the hook targets: the router's forward and the JIT boundary that writes the router input
    "srt/models/muse_spark.py": ("MuseSpark1xRouter",),
    "kernels/ops/layernorm/muse_spark_norm_residual_jit.py": ("fused_norm_residual",),
    "srt/distributed/*.py": ("get_tp_group",),
    "srt/layers/moe/utils.py": ("is_sbo_enabled", "is_tbo_enabled"),
    "srt/plugins/hook_registry.py": ("HookRegistry", "HookType"),
}

logger = logging.getLogger(__name__)

NAME = "router_pad"
ROWS_ENV = "MUSE_VR_ROUTER_PAD_ROWS"
NAMES_ENV = "MUSE_VR_ROUTER_PAD_KERNEL_NAMES"  # DEV / units only: record kernel names at the engage check (runs the profiler)
# The SGLang sources this component hooks or relies on, at 3117f58 + #48948
PINS = {
    # the hooked router forward (the padded call must equal it bitwise) and the path that hands the boundary's router
    # output to it unchanged: decoder layer -> MoE forward -> _forward_tp -> _forward_experts' router call
    "sglang/srt/models/muse_spark.py": (
        bed.Functions(
            {
                "MuseSpark1xRouter.forward": "a662bcde1176fec9",
                "MuseSpark1xDecoderLayer.forward": "51e40503e316c464",
                "MuseSpark1xMoE.forward": "72a174aadac5ceae",
                "MuseSpark1xMoE._forward_tp": "1c1aa0fe77681cd0",
                "MuseSpark1xMoE._forward_experts": "04c56c86a0d682b6",
            }
        ),
    ),
    # the hooked JIT boundary (with want_router and no router_out it allocates the router output itself)
    "sglang/kernels/ops/muse_spark_v12/norm_residual.py": (
        bed.Functions({"fused_norm_residual": "975cd488a7dc2b2a"}),
    ),
}
# HookRegistry AROUND targets, by their original module names (sgl_compat.target gives the installed location)
T_ROUTER_FORWARD = "sglang.srt.models.muse_spark.MuseSpark1xRouter.forward"
T_FUSED_NORM_RESIDUAL = (
    "sglang.kernels.ops.muse_spark_v12.norm_residual.fused_norm_residual"
)

# rows: [(lo, hi, P)]; bufs: (device, P, hidden, stream handle) -> [P, hidden] fp32; by_ptr: data_ptr -> buffer; bands:
# (device, lo) -> "ok" | "refused" after the engage check; settings: the float32 matmul settings the checks ran under
_state: dict = {
    "engaged": False,
    "rows": [],
    "bufs": {},
    "by_ptr": {},
    "bands": {},
    "settings": None,
    "off": False,
}


def parse_rows(text: str) -> List[Tuple[int, int, int]]:
    """``lo-hi:P[,lo-hi:P...]`` -> [(lo, hi, P)]; raises ValueError on a malformed or overlapping entry."""
    out: List[Tuple[int, int, int]] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        rng, p = part.split(":")
        lo, hi = (int(v) for v in rng.split("-"))
        p = int(p)
        if not (1 <= lo <= hi < p <= 1024):
            raise ValueError(f"{ROWS_ENV}: {part!r} needs 1 <= lo <= hi < P <= 1024")
        for a, b, _ in out:
            if not (hi < a or lo > b):
                raise ValueError(f"{ROWS_ENV}: {part!r} overlaps {a}-{b}")
        out.append((lo, hi, p))
    return out


def band(m: int) -> Optional[Tuple[int, int, int]]:
    for lo, hi, p in _state["rows"]:
        if lo <= m <= hi:
            return lo, hi, p
    return None


def pad_target(m: int) -> Optional[int]:
    b = band(m)
    return None if b is None else b[2]


def _settings() -> tuple:
    return (
        torch.get_float32_matmul_precision(),
        bool(torch.backends.cuda.matmul.allow_tf32),
    )


def _active() -> bool:
    if not _state["engaged"] or _state["off"]:
        return False
    if _state["settings"] is not None and _settings() != _state["settings"]:
        _state["off"] = True
        bed.count(NAME, "off (float32 matmul settings changed after the engage check)")
        logger.warning(
            "router_pad: float32 matmul settings changed after the engage check; component off"
        )
        return False
    return True


def router_out_for(stream: torch.Tensor) -> Optional[torch.Tensor]:
    """The router-input buffer a boundary producer should write for ``stream`` ([M, hidden] fp32): rows [0, M) of the padded
    buffer of (device, P, the current stream), or None (the producer allocates its own) when off, the row count is not
    padded, its band was refused, or the buffer would have to be created inside a graph capture."""
    if not _active() or stream.dim() != 2 or stream.dtype != torch.float32:
        return None
    m, hidden = stream.shape
    b = band(m)
    if b is None:
        return None
    dev = stream.device.index or 0
    if _state["bands"].get((dev, b[0])) == "refused":
        return None
    key = (dev, b[2], hidden, torch.cuda.current_stream(stream.device).cuda_stream)
    buf = _state["bufs"].get(key)
    if buf is None:
        if torch.cuda.is_current_stream_capturing():
            bed.count(NAME, f"unpadded (buffer first seen in capture) M{m}")
            return None
        buf = torch.zeros(b[2], hidden, dtype=torch.float32, device=stream.device)
        _state["bufs"][key] = buf
        _state["by_ptr"][buf.data_ptr()] = buf
    return buf[:m]


def padded_full(x: torch.Tensor) -> Optional[torch.Tensor]:
    """The full padded buffer when ``x`` is rows [0, M) of one (and its band is not refused), else None."""
    if not _active() or x.dim() != 2 or x.dtype != torch.float32:
        return None
    b = _state["by_ptr"].get(x.data_ptr())
    p = pad_target(x.shape[0])
    if (
        b is None
        or p is None
        or b.shape[0] != p
        or b.shape[1] != x.shape[1]
        or x.stride() != b.stride()
    ):
        return None
    return b


def _kernel_names(fn) -> str:
    """Kernel names of one call of ``fn`` (profiled once, at engage)."""
    try:
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            fn()
            torch.cuda.synchronize()
        names = [
            e.name
            for e in prof.events()
            if e.device_type == torch.autograd.DeviceType.CUDA
        ]
        return " + ".join(n[:60] for n in names)
    except Exception as e:  # noqa: BLE001
        return f"unknown ({type(e).__name__})"


def engage_check(
    fwd,
    hidden: int,
    dev: torch.device,
    lo: int,
    hi: int,
    p: int,
    x_now: Optional[torch.Tensor] = None,
) -> Tuple[bool, list]:
    """``fwd`` = the served router call. For every M in [lo, hi]: rows [0, M) of fwd on a [P, hidden] buffer equal fwd on the M
    rows, bitwise, with the pad rows zero and with them non-finite / large (and, for the calling row count, on the live input).
    Returns (ok, notes)."""
    gen = torch.Generator(device=dev)
    gen.manual_seed(lo * 1009 + hi)
    notes = []
    ok = True
    for m in range(lo, hi + 1):
        rows = torch.randn(m, hidden, device=dev, generator=gen)
        if x_now is not None and x_now.shape[0] == m:
            rows = x_now.clone()
        ref = fwd(rows)
        for pad in ("zero", "garbage"):
            buf = torch.zeros(p, hidden, dtype=torch.float32, device=dev)
            buf[:m] = rows
            if pad == "garbage" and p > m:
                g = torch.randn(p - m, hidden, device=dev, generator=gen) * 1e30
                g[:, 0::7] = float("nan")
                g[:, 3::11] = float("inf")
                g[:, 5::13] = -float("inf")
                buf[m:] = g
            got = fwd(buf)[:m]
            if not torch.equal(ref.view(torch.int32), got.view(torch.int32)):
                ok = False
                notes.append(f"M{m} P{p} pad={pad}: not bitwise")
        if os.environ.get(NAMES_ENV, "0") == "1":
            names_m = _kernel_names(lambda: fwd(rows))
            names_p = _kernel_names(lambda: fwd(torch.zeros(p, hidden, device=dev)))
            bed.count(NAME, f"kernels M{m}: {names_m} | P{p}: {names_p}")
    return ok, notes


def _tp_min(ok: bool) -> bool:
    """MIN of the band verdict over the tensor-parallel group (CPU group, eager, outside capture); the local verdict when there
    is no initialized group; a refusal (counted) when the reduction itself fails."""
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return ok
    try:
        from sglang.srt.distributed import get_tp_group

        g = get_tp_group()
        if g.world_size <= 1:
            return ok
        t = torch.tensor([1 if ok else 0], dtype=torch.int32)
        dist.all_reduce(t, op=dist.ReduceOp.MIN, group=g.cpu_group)
        return bool(t.item())
    except Exception as e:  # noqa: BLE001
        bed.count(
            NAME,
            f"REFUSED (tensor-parallel verdict reduction failed: {type(e).__name__})",
        )
        logger.error(
            "router_pad: the tensor-parallel verdict reduction failed (%s); band refused on this rank",
            e,
        )
        return False


def install() -> None:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    def fused_norm_residual(
        original, branch, stream, *a, want_router: bool = False, router_out=None, **k
    ):
        if want_router and router_out is None:
            router_out = router_out_for(stream)
        return original(
            branch, stream, *a, want_router=want_router, router_out=router_out, **k
        )

    HookRegistry.register(
        sgl_compat.target(T_FUSED_NORM_RESIDUAL), fused_norm_residual, HookType.AROUND
    )

    def forward(original, self, hidden_states):
        full = padded_full(hidden_states)
        if full is None:
            if (
                _state["engaged"]
                and hidden_states.dim() == 2
                and pad_target(hidden_states.shape[0]) is not None
            ):
                bed.count(NAME, f"unpadded producer M{hidden_states.shape[0]}")
            return original(self, hidden_states)
        m = hidden_states.shape[0]
        lo, hi, p = band(m)
        dev = hidden_states.device.index or 0
        state = _state["bands"].get((dev, lo))
        if state is None:
            if torch.cuda.is_current_stream_capturing():
                bed.count(NAME, f"stock (band {lo}-{hi} unchecked in capture) M{m}")
                return original(self, hidden_states)
            if _overlap_on():
                ok, notes = False, ["two-batch / single-batch overlap on"]
            else:
                ok, notes = engage_check(
                    lambda x: original(self, x),
                    hidden_states.shape[1],
                    hidden_states.device,
                    lo,
                    hi,
                    p,
                    x_now=hidden_states,
                )
            ok_all = _tp_min(ok)
            if ok and not ok_all:
                notes.append("another tensor-parallel rank refused")
            ok = ok_all
            _state["bands"][(dev, lo)] = "ok" if ok else "refused"
            if _state["settings"] is None:
                _state["settings"] = _settings()
            bed.count(NAME, f"band {lo}-{hi}:P{p} {'checked' if ok else 'REFUSED'}")
            for n in notes[:8]:
                bed.count(NAME, f"refusal {n}")
            if not ok:
                logger.warning(
                    "router_pad: band %d-%d (P %d) refused: %s",
                    lo,
                    hi,
                    p,
                    "; ".join(notes[:4]),
                )
            return original(self, hidden_states)
        if state != "ok":
            return original(self, hidden_states)
        bed.count(NAME, f"padded M{m} P{p}")
        return original(self, full)[:m]

    HookRegistry.register(T_ROUTER_FORWARD, forward, HookType.AROUND)


def _overlap_on() -> bool:
    try:
        from sglang.srt.layers.moe.utils import is_sbo_enabled, is_tbo_enabled

        return bool(is_tbo_enabled() or is_sbo_enabled())
    except Exception:  # noqa: BLE001
        return False


def register() -> bool:
    if not bed.enabled(NAME):
        _state["engaged"] = False
        return False
    for rel, sha in PINS.items():
        if not bed.pinned(rel, sha):
            bed.report(NAME, False, reason=f"pin mismatch: {rel}")
            return False
    try:
        _state["rows"] = parse_rows(os.environ.get(ROWS_ENV, ""))
        if not _state["rows"]:
            bed.report(NAME, False, reason=f"{ROWS_ENV} empty")
            return False
        if _overlap_on():
            bed.report(
                NAME,
                False,
                reason="two-batch / single-batch overlap on (one router buffer per stream assumed)",
            )
            return False
        install()
    except Exception as e:  # noqa: BLE001
        bed.report(NAME, False, reason=f"install failed: {type(e).__name__}: {e}")
        return False
    _state["engaged"] = True
    bed.report(
        NAME, True, rows=",".join(f"{lo}-{hi}:{p}" for lo, hi, p in _state["rows"])
    )
    return True
