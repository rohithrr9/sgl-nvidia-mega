# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Component ``aux_side`` (env ``MUSE_VR_AUX_SIDE=1``, default off).

When a draft model reads target hidden states (``set_dflash_layers_to_capture``), SGLang's Muse Spark forward converts
the fp32 residual stream to the operator dtype after each listed decoder layer
(``aux_hidden_states.append(stream.to(self.operator_dtype))``). Each conversion is an elementwise kernel on the main
stream between that layer's output boundary and the next layer's QKV projection, so it sits on the per-layer dependency
path. This component issues the same conversions on a side CUDA stream: after a listed layer returns, the output tensor
is allocated on the main stream, the side stream waits for the main stream and runs the same conversion
(``copy_``, the kernel ``Tensor.to`` uses) into it; the main stream waits for the side stream once, after the last
layer, before the forward returns. The converted tensors are bit-identical (tests/test_aux_side.py).

Ordering and memory: the decoder layer's output boundary returns a newly allocated stream tensor at every layer, so no
later layer writes a converted stream in place. Each converted stream and each output are allocated on the main
stream and kept referenced until the join, so the allocator cannot hand their memory to a later allocation while the
side stream may still use it. Inside CUDA-graph capture the fork and the join are captured as graph edges.

Mechanism (as ``final_norm``): an AROUND hook on ``MuseSpark1xModel._forward``. For decode and target-verify batches
without sequence parallelism, during the call the model's ``layers_to_capture`` is emptied and each listed layer is
replaced by a stand-in that runs the layer and then issues its conversion; the originals are restored when the call
returns. Every other call runs stock, and so does a call that captures the last layer.
"""

from __future__ import annotations

import logging

import torch

from . import bed

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "srt/layers/layernorm_sp.py": ("runs_sp",),
    "srt/plugins/hook_registry.py": ("HookRegistry", "HookType"),
}

logger = logging.getLogger(__name__)

NAME = "aux_side"  # env MUSE_VR_AUX_SIDE
TARGET = "sglang.srt.models.muse_spark.MuseSpark1xModel._forward"

# The SGLang sources this component hooks or relies on, at 3117f58 + #48948
PINS = {
    # the hooked model forward (sgl_shims' _forward) whose capture loop the stand-ins take over, the attributes it
    # reads, the decoder layer the stand-ins call, and the caller that consumes the captured list
    "sglang/srt/models/muse_spark.py": (
        bed.Functions(
            {
                "MuseSpark1xModel.forward": "5a17d5920ad718b9",
                "MuseSpark1xModel.__init__": "494fbc30068c6846",
                "MuseSpark1xDecoderLayer.forward": "51e40503e316c464",
                "MuseSparkTextForCausalLM.forward": "72ade900ba1d1069",
            }
        ),
    ),
    "sglang/srt/layers/layernorm_sp.py": (
        bed.Functions({"runs_sp": "6049807aa2400a43"}),
    ),
}

_side: dict = {}  # device index -> side stream (created once per device)


def enabled() -> bool:
    return bed.enabled(NAME)


def pins_match() -> tuple[bool, str]:
    for rel, want in PINS.items():
        if not bed.pinned(rel, want):
            return False, f"{rel} differs from the pinned revision"
    return True, "ok"


def side_stream(device: torch.device) -> torch.cuda.Stream:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    s = _side.get(idx)
    if s is None:
        s = torch.cuda.Stream(device=idx)
        _side[idx] = s
    return s


class _Fork(torch.nn.Module):
    """Stand-in for a listed decoder layer: runs the layer, then converts its output stream on the side stream."""

    def __init__(self, layer, ctx: dict):
        super().__init__()
        object.__setattr__(
            self, "_layer", layer
        )  # not registered: the layer stays owned by the model's list
        object.__setattr__(self, "_ctx", ctx)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(object.__getattribute__(self, "_layer"), name)

    def forward(self, positions, stream, attn_input, forward_batch):
        stream, attn_input = self._layer(positions, stream, attn_input, forward_batch)
        ctx = self._ctx
        main = torch.cuda.current_stream(stream.device)
        out = torch.empty_like(
            stream, dtype=ctx["dtype"]
        )  # allocated on the main stream, before the fork
        side = ctx["side"]
        side.wait_stream(main)
        with torch.cuda.stream(side):
            out.copy_(stream)
        ctx["held"].append(stream)
        ctx["aux"].append(out)
        return stream, attn_input


def _eligible(model, forward_batch) -> str | None:
    import sglang.srt.layers.layernorm_sp as layernorm_sp

    caps = list(getattr(model, "layers_to_capture", None) or ())
    if not caps:
        return "nothing captured"
    mode = forward_batch.forward_mode
    if not (mode.is_decode() or mode.is_target_verify()):
        return "not decode/verify"
    if layernorm_sp.runs_sp(mode):
        return "sequence parallel"
    n = len(model.layers)
    if len(set(caps)) != len(caps) or any(not 0 <= i < n for i in caps):
        return "capture list"
    if n - 1 in caps:
        return "last layer captured"
    if model.residual_dtype != torch.float32 or model.operator_dtype != torch.bfloat16:
        return "dtypes"
    return None


def model_forward(
    original, model, input_ids, positions, forward_batch, input_embeds=None
):
    reason = _eligible(model, forward_batch)
    if reason is not None:
        bed.count(NAME, "stock:" + reason)
        return original(model, input_ids, positions, forward_batch, input_embeds)
    caps = list(model.layers_to_capture)
    device = torch.device("cuda", torch.cuda.current_device())
    ctx = {
        "side": side_stream(device),
        "dtype": model.operator_dtype,
        "held": [],
        "aux": [],
    }
    saved = {i: model.layers[i] for i in caps}
    model.layers_to_capture = []
    try:
        for i in caps:
            model.layers[i] = _Fork(saved[i], ctx)
        out = original(model, input_ids, positions, forward_batch, input_embeds)
    finally:
        for i, layer in saved.items():
            model.layers[i] = layer
        model.layers_to_capture = caps
    torch.cuda.current_stream(device).wait_stream(
        ctx["side"]
    )  # the one join, before anything reads the outputs
    ctx["held"].clear()
    if isinstance(out, tuple) or len(ctx["aux"]) != len(caps):
        raise RuntimeError("aux_side: unexpected forward result")
    bed.count(NAME, "side")
    return out, ctx["aux"]


def register() -> bool:
    if not enabled():
        return False
    ok, why = pins_match()
    if not ok:
        bed.report(NAME, False, reason=why)
        return False
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    def hook(original, *args, **kwargs):
        names = ("input_ids", "positions", "forward_batch", "input_embeds")
        if len(args) < 4 or len(args) > 5 or set(kwargs) - set(names[len(args) - 1 :]):
            return original(*args, **kwargs)
        model, rest = args[0], list(args[1:])
        kw = dict(zip(names, rest))
        kw.update(kwargs)
        return model_forward(
            original,
            model,
            kw["input_ids"],
            kw["positions"],
            kw["forward_batch"],
            kw.get("input_embeds"),
        )

    HookRegistry.register(TARGET, hook, HookType.AROUND)
    bed.report(NAME, True)
    return True
