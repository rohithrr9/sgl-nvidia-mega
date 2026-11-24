# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Component ``router_first`` (env ``MUSE_VR_ROUTER_FIRST=1``, default off): the MoE router chain issued before the pooler.

SGLang's ``MuseSpark1xMoE._forward_experts`` forks a side stream (``side.wait_stream(main)``), then enqueues the replicated
pooler GEMM (``pre_expert_proj``) and the experts' ``prepare_input`` on the main stream, and only after them the router GEMM
and the fused routing op on the side stream, before it joins the side stream back (``main.wait_stream(side)``). In a
captured decode graph the two branches are independent.

This component enqueues the side-stream block first. After SGLang's fork, at the same point, it issues the router GEMM and
the routing op on the side stream through the module's own calls (``self.gate`` / ``self._routing_op()``, so any wrapper
installed around them still applies). It then runs SGLang's ``_forward_experts`` unchanged; that call's router GEMM and
routing op calls take the tensors already issued instead of launching again (a handover of the same objects stock would
have produced, never a recomputation). Kernels, arguments, inputs, the join and the cross-stream lifetime handling are
SGLang's; only the order in which the two independent branches are enqueued (and captured) changes. Outputs are therefore
bit-identical to stock (tests/test_router_first.py: eager, CUDA-graph capture + replay, repeats, changed inputs).

Rows: ``MUSE_VR_ROUTER_FIRST_ROWS`` (default ``17-64``; tunable). The stock
path is taken (and counted) for rows outside the range, an empty batch, a CPU tensor, sequence parallelism, and the bf16
router path without a router input (stock would cast it on the main stream first). A handover the stock call does not
take raises (fail-stop): it would mean SGLang's call sequence changed under the pin.
"""

from __future__ import annotations

import logging
import os
import threading

import torch

from . import bed

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "srt/models/muse_spark.py": (
        "MuseSpark1xMoE",
        "MuseSpark1xRouter",
        "_use_split_bf16_router",
    ),
    "srt/runtime_context.py": ("get_forward",),
}

logger = logging.getLogger(__name__)

NAME = "router_first"  # env MUSE_VR_ROUTER_FIRST
ROWS_ENV = "MUSE_VR_ROUTER_FIRST_ROWS"
DEFAULT_ROWS = (17, 64)
# the SGLang functions whose MoE forward, router and routing-op calls this component relies on (3117f58 + #48948);
# the stock _forward_experts digest keeps the handover's fail-stop meaningful (it guards that call sequence at run time)
PINS = {
    "sglang/srt/models/muse_spark.py": (
        bed.Functions(
            {
                "MuseSpark1xMoE._forward_experts": "04c56c86a0d682b6",
                "MuseSpark1xMoE._routing_op": "e7ed728a5c049f72",
                "MuseSpark1xMoE._side_stream": "e11c219451f8bc9c",
                "MuseSpark1xRouter.forward": "a662bcde1176fec9",
                "MuseSpark1xRouter.forward_split2": "6b4be78babba5eaf",
                "_use_split_bf16_router": "61af1e739a14e950",
            }
        ),
    ),
}

_state = {"engaged": None, "rows": DEFAULT_ROWS}
_tls = threading.local()


def _range(env: str, default):
    v = os.environ.get(env, "").strip()
    if not v:
        return default
    a, b = (int(x) for x in v.split("-"))
    if not 1 <= a <= b:
        raise ValueError(f"{env}={v!r}: want 'lo-hi' with 1 <= lo <= hi")
    return a, b


def _pending():
    return getattr(_tls, "pre", None)


class _Handover:
    """The routing op as SGLang's call sees it while a handover is pending: its one stock call returns the issued result."""

    __slots__ = ("op", "p")

    def __init__(self, op, p):
        self.op, self.p = op, p

    def __getattr__(self, name):
        return getattr(self.op, name)

    def __call__(self, *args, **kwargs):
        p = self.p
        if (
            not p["took_op"]
            and len(args) >= 2
            and args[0] is None
            and args[1] is p["logits"]
        ):
            if tuple(args[2:]) != (p["top_k"], True) or kwargs != {"want_packed": True}:
                raise RuntimeError(
                    f"{NAME}: SGLang's routing-op call changed: args {args[2:]} kwargs {kwargs}"
                )
            p["took_op"] = True
            return p["res"]
        return self.op(*args, **kwargs)


def install() -> None:
    import sglang.srt.models.muse_spark as model
    from sglang.srt.runtime_context import get_forward

    moe, router = model.MuseSpark1xMoE, model.MuseSpark1xRouter
    if getattr(moe._forward_experts, "__wrapped_by_router_first__", False):
        return
    orig_experts = moe._forward_experts
    orig_forward = router.forward
    orig_split2 = router.forward_split2
    orig_routing_op = moe._routing_op

    def _take(kind, gate, x):
        p = _pending()
        if (
            p is not None
            and not p["took_gate"]
            and p["gate"] is gate
            and p["kind"] == kind
            and p["x"] is x
        ):
            p["took_gate"] = True
            return p["logits"]
        return None

    def forward(self, hidden_states):
        r = _take("linear", self, hidden_states)
        return orig_forward(self, hidden_states) if r is None else r

    def forward_split2(self, hidden_states):
        r = _take("split2", self, hidden_states)
        return orig_split2(self, hidden_states) if r is None else r

    def _routing_op(self):
        op = orig_routing_op(self)
        p = _pending()
        if p is not None and p["moe"] is self and not p["took_op"]:
            return _Handover(op, p)
        return op

    def _why_stock(self, hidden_states, router_input, split2):
        n = hidden_states.shape[0]
        if n == 0:
            return "empty"
        if hidden_states.device.type != "cuda" or not torch.cuda.is_available():
            return "cpu"
        if getattr(get_forward(), "sp_active", False):
            return "sp"
        lo, hi = _state["rows"]
        if not lo <= n <= hi:
            return "rows"
        if not split2 and router_input is None:
            return "no_router_input"
        if _pending() is not None:
            return "nested"
        return None

    def _forward_experts(self, hidden_states, router_input=None, *, ar_workspace=None):
        split2 = bool(
            self.router_in_fp32
            and model._use_split_bf16_router(
                hidden_states.shape[0]
            )  # read at call time, as stock does
            and self.gate.weight.dtype == torch.float32
            and hidden_states.dtype in (torch.bfloat16, torch.float16)
        )
        why = _why_stock(self, hidden_states, router_input, split2)
        if why is not None:
            bed.count(NAME, "stock:" + why)
            return orig_experts(
                self, hidden_states, router_input, ar_workspace=ar_workspace
            )
        main = torch.cuda.current_stream()
        side = self._side_stream()
        side.wait_stream(
            main
        )  # SGLang's fork, at its position: before any side-stream work
        with torch.cuda.stream(side):
            logits = (
                self.gate.forward_split2(hidden_states)
                if split2
                else self.gate(router_input)
            )
            res = self._routing_op()(None, logits, self.top_k, True, want_packed=True)
        p = {
            "moe": self,
            "gate": self.gate,
            "kind": "split2" if split2 else "linear",
            "x": hidden_states if split2 else router_input,
            "logits": logits,
            "res": res,
            "top_k": self.top_k,
            "took_gate": False,
            "took_op": False,
        }
        _tls.pre = p
        try:
            out = orig_experts(
                self, hidden_states, router_input, ar_workspace=ar_workspace
            )
        finally:
            _tls.pre = None
        if not (p["took_gate"] and p["took_op"]):
            raise RuntimeError(
                f"{NAME}: SGLang's _forward_experts did not take the router handover "
                f"(gate {p['took_gate']}, routing op {p['took_op']}); its call sequence changed"
            )
        bed.count(NAME, "first:m%d" % hidden_states.shape[0])
        return out

    _forward_experts.__wrapped_by_router_first__ = True
    _forward_experts.__wrapped__ = orig_experts
    router.forward = forward
    router.forward_split2 = forward_split2
    moe._routing_op = _routing_op
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
    # router_pad's padded router calls never take the handover (each would fail-stop);
    # the two components were never run together
    if bed.enabled("router_pad"):
        bed.report(
            NAME,
            False,
            reason="router_pad on: its padded router calls skip the handover",
        )
        _state["engaged"] = False
        return False
    try:
        _state["rows"] = _range(ROWS_ENV, DEFAULT_ROWS)
        install()
    except Exception as e:  # noqa: BLE001
        bed.report(NAME, False, reason=f"install failed: {type(e).__name__}: {e}")
        _state["engaged"] = False
        return False
    bed.report(NAME, True, rows="%d-%d" % _state["rows"])
    _state["engaged"] = True
    return True
