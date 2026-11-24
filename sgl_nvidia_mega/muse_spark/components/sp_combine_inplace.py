# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Component ``sp_combine_inplace``: write the sequence-parallel MoE combine payload straight into the all-to-all
workspace.

Under ``--enable-layernorm-sp`` the Muse Spark routed experts of a prefill pass run on the tokens the FlashInfer
all-to-all dispatched to each rank (``forward_dispatched``); the per-rank expert finalize (post-expert norm and
weighted top-k sum) writes a new ``[ep_size * max_tokens, hidden]`` tensor, and the combine then starts with a
kernel that copies that tensor, token by token, into the rank's own workspace region the peers read from. FlashInfer
supports a zero-copy form of the same combine: the payload is a view of that workspace region
(``get_combine_payload_tensor_in_workspace``) and ``combine(payload_in_workspace=True)`` skips the copy (its
preparation kernel then only advances the round counter). FlashInfer 0.7.4 (``flashinfer.moe_ep``) hands out the view
with ``get_combine_input_buffer`` and its ``combine`` recognizes the view by its address. SGLang uses that form for
another expert runner only.

This component makes the Muse Spark expert finalize write its output directly into the workspace view (the same
fused finalize kernel, the same arithmetic, only a different destination address) and lets the sequence-parallel
dispatcher run the zero-copy combine. The peers read the same bytes as before, so the combined output is bitwise
identical; the per-token copy of the payload is removed. The order of the payload writes and the combine's
readiness signal is unchanged: the payload is complete when an earlier kernel on the stream (the finalize, before:
the copy kernel) has finished. FlashInfer 0.7.1 / 0.7.2 check that a workspace payload is exactly the workspace view
and fail loudly otherwise; 0.7.4 stages any other payload with the copy, so there the loud check is this component's own
(the expert finalize must return the workspace view).

Opt in with ``MUSE_VR_SP_COMBINE_INPLACE=1`` (only meaningful with ``--enable-layernorm-sp``). Engagement is guarded
by sha256 pins on the SGLang dispatcher and expert files and the FlashInfer all-to-all files it relies on.
"""

from __future__ import annotations

import contextvars
import functools
import hashlib
import importlib.util
import logging
from pathlib import Path

from . import bed

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "srt/layers/muse_spark/muse_spark_fp4_moe.py": (
        "finalize_with_post_expert_norm",
        "MuseSpark1xFp4Experts",
    ),
    "srt/layers/muse_spark/muse_spark_bf16_moe.py": (
        "finalize_with_post_expert_norm",
        "MuseSpark1xBf16Experts",
    ),
    "srt/layers/moe/token_dispatcher/flashinfer.py": ("FlashinferDispatcher",),
}


logger = logging.getLogger("muse_vr_plugin")

NAME = "sp_combine_inplace"

SGLANG_PINS = {
    # FlashinferDispatcher: SP-prefill instance, payload_in_workspace in dispatch / combine (newer SGLang passes no
    # flag to combine: FlashInfer's combine recognizes the workspace payload by its address)
    "sglang/srt/layers/moe/token_dispatcher/flashinfer.py": (
        "61d79191720f8231f965be7f971155f139bf683b8b7625733b49e0031fbd2ad3",
        bed.Functions(
            {
                "FlashinferDispatcher.__init__": (
                    "09371915b1301975",
                    "fcc8ee39ae3319fe",
                ),
                "FlashinferDispatcher.dispatch": (
                    "e1eed8fdb4ea3433",
                    "74e35916f73b9b18",
                ),
                "FlashinferDispatcher.combine": (
                    "6f5ba7281d294200",
                    "a546e68f329486cb",
                ),
            }
        ),
    ),
    # NVFP4 experts: forward_dispatched / _finalize_deferred_output (reimplemented below with a destination)
    "sglang/srt/layers/muse_spark_fp4_moe.py": (
        "1d377424e4c1f486f72a0e8ea67ea41dca4bd3f6929aa3729f5f7480861ed352",
        "1d7676d6d086e6aa45139671e345ef32d28a81b7de1b50810d0e512ee95cc738",
        bed.Functions(
            {
                "MuseSpark1xFp4Experts.forward_dispatched": "55ee763b2d00ed84",
                "MuseSpark1xFp4Experts._finalize_deferred_output": "243e920761c83e15",
                "MuseSpark1xFp4Experts.set_post_expert_norm": "d05de6a65cd26d00",
                "finalize_with_post_expert_norm": "a3ee37c3b9b8b7cf",
            }
        ),
    ),
    # BF16 experts: forward_dispatched / _finalize_deferred_output (reimplemented below with a destination)
    "sglang/srt/layers/muse_spark_bf16_moe.py": (
        "687680077a7542cb1e2026d321c2f18a650fb56a654b50434acbf7be9c1aa6c5",
        "57ae1593b4496edfc658343d9ca039dead111484d78ec57ab3eaa14891f6afed",
        bed.Functions(
            {
                "MuseSpark1xBf16Experts.forward_dispatched": "592f952a35e9e122",
                "MuseSpark1xBf16Experts._finalize_deferred_output": "f02f932959a1e498",
                "MuseSpark1xBf16Experts._build_finalize_op": "1fb1bf113d5e7830",
            }
        ),
    ),
}
FLASHINFER_PINS = {
    # flashinfer.comm.MoeAlltoAll (FlashInfer 0.7.1 / 0.7.2; removed in 0.7.4)
    "flashinfer/comm/trtllm_moe_alltoall.py": (
        "8ae4a8e618e3956f864d0e012dced7ffbac55d6db18e61d05d4425c51375fda2",  # FlashInfer 0.7.1
        "21da7a12c1ef08e9c87fc5071e3fa1fafe9fbff04f9571ec683f339a01ea5b3c",  # FlashInfer 0.7.2
        "absent",  # FlashInfer 0.7.4
    ),
    "flashinfer/data/csrc/trtllm_moe_alltoall.cu": (
        "d672681fb81aad57b780bde8ec8e07edfea58b65ab420f91c8df6aa79652f60f",  # FlashInfer 0.7.1, 0.7.2
        "absent",  # FlashInfer 0.7.4
    ),
    "flashinfer/data/csrc/nv_internal/tensorrt_llm/kernels/communicationKernels/moeAlltoAllKernels.cu": (
        "195827d062cf18101b39491a6ecec4379d30f8e9e7e94e3992dda0420e67468f",  # FlashInfer 0.7.1, 0.7.2
        "absent",  # FlashInfer 0.7.4
    ),
    # flashinfer.moe_ep NVLinkOneSidedAlltoAll (the all-to-all of SGLang's dispatcher on FlashInfer 0.7.4):
    # get_combine_input_buffer, the combine that skips the staging copy for a payload at the combine-input address
    # (fence path; the CFT and Lamport paths read the payload as a plain source), and the Lamport combine kernels
    "flashinfer/moe_ep/backends/split/comm/nvlink_one_sided/communication.py": (
        "absent",  # FlashInfer 0.7.1
        "5d3075cf85b358dbdc92c94fe313e850dda84330a3620eba609b511b58f40803",  # FlashInfer 0.7.2 (SGLang 306ccf6: unused)
        "20e66b0dfd8175df8230f4e3020cadbf3aa30f172f6d6582498ac0f27d22c7b8",  # FlashInfer 0.7.4
    ),
    "flashinfer/data/csrc/nvlink_one_sided_alltoall.cu": (
        "absent",  # FlashInfer 0.7.1, 0.7.2
        "eecf9de49a0aaa30569d2235571975a4789ba0536e796b2cbac53fe634148ec3",  # FlashInfer 0.7.4
    ),
    "flashinfer/data/csrc/nv_internal/tensorrt_llm/kernels/moe/communication/moeAlltoAllKernels.cu": (
        "absent",  # FlashInfer 0.7.1, 0.7.2
        "92bce7109823ced9f436a7b96a5e7739d99d8e1158e406d95d23b4aea4447617",  # FlashInfer 0.7.4
    ),
    "flashinfer/data/include/flashinfer/comm/moe_alltoall_lamport.cuh": (
        "absent",  # FlashInfer 0.7.1, 0.7.2
        "c370c2874485af81693f55ffb5fa9c75e8b440a6aa35593ce55632d65ed40951",  # FlashInfer 0.7.4
    ),
}

_state: dict = {"engaged": None}
# destination of the expert finalize of the forward_dispatched call in flight (None: stock allocation)
_dest: contextvars.ContextVar = contextvars.ContextVar(
    "muse_vr_sp_combine_dest", default=None
)


def _flashinfer_pins_ok() -> tuple[bool, str]:
    spec = importlib.util.find_spec("flashinfer")
    if spec is None or not spec.submodule_search_locations:
        return False, "flashinfer"
    root = Path(list(spec.submodule_search_locations)[0]).resolve().parent
    for rel, sha in FLASHINFER_PINS.items():
        try:
            got = hashlib.sha256((root / rel).read_bytes()).hexdigest()
        except OSError:
            got = None
        if not bed.sha_ok(got, sha):
            logger.warning(
                "muse_vr_plugin: %s sha256 %s does not match the pin %s",
                rel,
                (got or "<missing>")[:16],
                bed.sha_label(sha),
            )
            return False, rel
    return True, ""


def _fp4_finalize(
    self, deferred_output, *, num_tokens, post_expert_norm, out_dtype, ar_workspace=None
):
    """``MuseSpark1xFp4Experts._finalize_deferred_output`` with the fused finalize writing into ``_dest``."""
    import sglang.srt.layers.muse_spark_fp4_moe as m
    import torch

    dest = _dest.get()
    if dest is None or ar_workspace is not None or out_dtype != torch.bfloat16:
        return _state["orig_fp4_finalize"](
            self,
            deferred_output,
            num_tokens=num_tokens,
            post_expert_norm=post_expert_norm,
            out_dtype=out_dtype,
            ar_workspace=ar_workspace,
        )
    if getattr(
        self, "expert_tp", None
    ):  # branch layout only (SGLang main has no expert-TP reduce here)
        deferred_output = m.expert_tp.reduce_deferred(deferred_output)
    use_fused = (
        self._post_expert_norm is not None
        and self._post_expert_norm is post_expert_norm
        and self._post_expert_norm_weight is not None
    )
    if not use_fused and post_expert_norm is not None:
        self.set_post_expert_norm(post_expert_norm)
        use_fused = self._post_expert_norm_weight is not None
    if not use_fused or post_expert_norm is None:
        out = m.finalize_with_post_expert_norm(
            gemm2_out=deferred_output.gemm2_out,
            expert_weights=deferred_output.expert_weights,
            expanded_idx_to_permuted_idx=deferred_output.expanded_idx_to_permuted_idx,
            num_tokens=num_tokens,
            hidden_size=self.hidden_size,
            top_k=deferred_output.top_k,
            post_expert_norm=post_expert_norm,
            out_dtype=out_dtype,
        )
        dest.copy_(out)
        bed.count(NAME, "fp4_unfused_copy")
        return dest
    self._fused_finalize_op(
        gemm2_output=deferred_output.gemm2_out,
        expert_weights=deferred_output.expert_weights,
        expanded_idx_to_permuted_idx=deferred_output.expanded_idx_to_permuted_idx,
        rms_norm_weight=self._post_expert_norm_weight,
        output=dest,
        ar_workspace=None,
    )
    bed.count(NAME, "fp4_inplace")
    return dest


def _bf16_finalize(self, deferred_output, *, num_tokens, post_expert_norm, out_dtype):
    """``MuseSpark1xBf16Experts._finalize_deferred_output`` with the finalize writing into ``_dest``."""
    import sglang.srt.layers.muse_spark_bf16_moe as m
    import torch

    dest = _dest.get()
    if dest is None or out_dtype != torch.bfloat16:
        return _state["orig_bf16_finalize"](
            self,
            deferred_output,
            num_tokens=num_tokens,
            post_expert_norm=post_expert_norm,
            out_dtype=out_dtype,
        )
    if self._finalize_op is None:
        out = m.finalize_with_post_expert_norm(
            gemm2_out=deferred_output.gemm2_out,
            expert_weights=deferred_output.expert_weights,
            expanded_idx_to_permuted_idx=deferred_output.expanded_idx_to_permuted_idx,
            num_tokens=num_tokens,
            hidden_size=self.expert_hidden_size,
            top_k=deferred_output.top_k,
            post_expert_norm=post_expert_norm,
            out_dtype=out_dtype,
        )
        dest.copy_(out)
        bed.count(NAME, "bf16_unfused_copy")
        return dest
    self._finalize_op(
        gemm2_output=deferred_output.gemm2_out,
        expert_weights=deferred_output.expert_weights,
        expanded_idx_to_permuted_idx=deferred_output.expanded_idx_to_permuted_idx,
        rms_norm_weight=self._post_expert_norm_weight,
        output=dest,
    )
    bed.count(NAME, "bf16_inplace")
    return dest


def _wrap_forward_dispatched(fn):
    @functools.wraps(fn)
    def forward_dispatched(self, dispatch_output, post_expert_norm):
        moe_output = getattr(dispatch_output, "moe_output", None)
        if moe_output is None:
            return fn(self, dispatch_output, post_expert_norm)
        hidden = moe_output.shape[-1]
        dest = moe_output.view(-1, hidden)
        if dest.shape[0] != dispatch_output.hidden_states.shape[0]:
            raise RuntimeError(
                "sp_combine_inplace: workspace payload rows do not match the dispatched rows"
            )
        token = _dest.set(dest)
        try:
            out = fn(self, dispatch_output, post_expert_norm)
        finally:
            _dest.reset(token)
        if out.hidden_states.data_ptr() != dest.data_ptr():
            raise RuntimeError(
                "sp_combine_inplace: expert finalize did not write the workspace payload"
            )
        return out

    forward_dispatched.__wrapped_by_sp_combine__ = True
    return forward_dispatched


def _wrap_dispatcher_init(fn):
    @functools.wraps(fn)
    def __init__(self, *args, **kwargs):
        fn(self, *args, **kwargs)
        if (
            kwargs.get("enable_sp_prefill_a2a", False)
            and getattr(self, "sp_prefill_moe_a2a", None) is not None
        ):
            # zero-copy combine: dispatch hands out the workspace payload view, combine skips the copy
            self.payload_in_workspace = True
            bed.count(NAME, "sp_dispatcher")

    __init__.__wrapped_by_sp_combine__ = True
    return __init__


def install() -> None:
    import sglang.srt.layers.moe.token_dispatcher.flashinfer as fdisp
    import sglang.srt.layers.muse_spark_bf16_moe as muse_spark_bf16_moe
    import sglang.srt.layers.muse_spark_fp4_moe as muse_spark_fp4_moe

    fp4 = muse_spark_fp4_moe.MuseSpark1xFp4Experts
    bf16 = muse_spark_bf16_moe.MuseSpark1xBf16Experts
    if getattr(fdisp.FlashinferDispatcher.__init__, "__wrapped_by_sp_combine__", False):
        return
    _state["orig_fp4_finalize"] = fp4._finalize_deferred_output
    _state["orig_bf16_finalize"] = bf16._finalize_deferred_output
    fp4._finalize_deferred_output = _fp4_finalize
    bf16._finalize_deferred_output = _bf16_finalize
    fp4.forward_dispatched = _wrap_forward_dispatched(fp4.forward_dispatched)
    bf16.forward_dispatched = _wrap_forward_dispatched(bf16.forward_dispatched)
    fdisp.FlashinferDispatcher.__init__ = _wrap_dispatcher_init(
        fdisp.FlashinferDispatcher.__init__
    )


def engage() -> bool:
    if _state["engaged"] is not None:
        return _state["engaged"]
    if not bed.enabled(NAME):
        _state["engaged"] = False
        return False
    for rel, sha in SGLANG_PINS.items():
        if not bed.pinned(rel, sha):
            bed.report(NAME, False, reason=f"pin mismatch: {rel}")
            _state["engaged"] = False
            return False
    ok, why = _flashinfer_pins_ok()
    if not ok:
        bed.report(NAME, False, reason=f"pin mismatch: {why}")
        _state["engaged"] = False
        return False
    try:
        install()
    except Exception as e:  # noqa: BLE001
        bed.report(NAME, False, reason=f"install failed: {type(e).__name__}: {e}")
        _state["engaged"] = False
        return False
    bed.report(NAME, True)
    _state["engaged"] = True
    return True
