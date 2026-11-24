# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Component ``qkv_qk_fold`` (env ``MUSE_VR_QKV_QK_FOLD=1``, default off).

In a target-verify step, the QKV projection of a full-attention layer without RoPE runs ``qk_fold_gemm``: the split-K
GEMM SGLang's bf16 dispatch would run for that shape (same tactic, same main loop, reduction and bf16 store) with its
QK-norm epilogue, which also writes the e4m3 queries and the e4m3 key / value cache rows exactly as the fused QK-norm
call (``FusedQKNorm._forward_triton`` with the FP8 KV store) would. The attention call then skips that QK-norm call and
uses the queries the GEMM wrote; the projection output, the queries and the cache are bitwise unchanged
(``tests/test_qk_fold_gemm.py``).

The projection is taken on the decoder-layer path of ``ar_attn_boundary`` (which runs the QKV projection itself) and
the queries are taken on ``verify_attn``'s ``full_attn_wide`` path, only when all of these hold (every other call runs
the stock projection and the stock QK-norm call; each reason is counted and logged once):

- ``verify_attn`` takes the call on its ``full_attn_wide`` path without its query/key prologue (and ``full_verify_attn``
  does not take it), so the stock QK-norm call is exactly the one at that path;
- the stock QK-norm call would run SGLang's Triton kernel (the conditions ``verify_attn`` checks for its prologues) on a
  row count the epilogue covers, with an FP8 KV store whose pools are contiguous;
- SGLang's bf16 dispatch would run FlashInfer's cluster split-K for the projection's (rows, outputs, inputs) with a
  64-row tile (one head per tile), the FlashInfer kernel source matches the pinned revision this copy was taken from, and
  the projection input is a dense bf16 matrix;
- the running Triton release has a validated arithmetic form (``ops.swa_verify_attn.qk_arith``).

With ``MUSE_VR_QKV_QK_FOLD_SWA=1`` (default off) the sliding-window layers (interleaved RoPE) take the RoPE form of the
epilogue in the same way, on ``verify_attn``'s ``swa_verify_attn`` path (when ``swa_t`` does not take the call): that path
then runs its kernel on the epilogue's queries without its query/key prologue and without the stock QK-norm call. A call
that the attention kernel's own query/key prologue would take (``verify_attn._qk_prologue_args``: its request cap, the
Triton kernel, the row class and the operand layouts) runs stock: there the QK-norm already runs inside the attention
kernel, so there is no separate QK-norm launch to remove. The exception is a request count for which the staged form
of ``swa_verify_attn`` is set (``MUSE_VR_SWA_VERIFY_ATTN_STAGED``): that form runs only on e4m3 queries, so there the
projection keeps the epilogue and the staged kernel takes the call on its queries.

With ``MUSE_VR_QKV_QK_FOLD_HALF_WARP=1`` (default off), on row counts where the Triton kernel uses 2 columns per lane the
layers without RoPE run each epilogue row on a half-warp with the 2-column sum-of-squares tree (``QkFold.half_warp_tree``):
half the rows per lane, the same bytes.

With ``MUSE_VR_QKV_QK_FOLD_GROUPED_MAILBOX=1`` (default off) the projection's split-K partials are exchanged in groups of
four values per thread (``QkFold.grouped_mailbox``); with ``MUSE_VR_QKV_QK_FOLD_DEEP_RING=1`` (default off) its operand
ring takes the most stages that fit the device's per-block shared memory (``QkFold.deep_ring``). Both give the same bytes.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import torch

from . import bed

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    # the bf16 dispatch _splitk_tactic mirrors, and the backend check
    "srt/layers/quantization/unquant.py": (
        "_BF16_LOW_M_AUTOTUNE_MAX_M",
        "_BF16_SPLITK_TUNED_TACTICS",
        "_active_warp_splitk_tactics",
        "_enable_bf16_splitk_gemm",
        "_mm_bf16",
        "_prefer_direct",
        "get_bf16_gemm_backend",
        "use_bf16_splitk_gemm",
    ),
    # the stock QK-norm call's backend choice
    "kernels/fused_op.py": ("_oot_dispatch_key", "get_fused_op_backend"),
    "kernels/spec.py": ("KernelBackend",),
}

logger = logging.getLogger(__name__)

NAME = "qkv_qk_fold"  # env MUSE_VR_QKV_QK_FOLD
TOKENS = 8
HEAD_DIM = 64
# FlashInfer's dense_bf16_gemm_sm100_splitk.py revisions qk_fold_gemm.py was taken from (FlashInfer 0.7.2)
FLASHINFER_SPLITK_SHA256 = (
    "f5edd4c11d95822c836a363dfa1dd19f7843b4e354e091c24c0acac12f52d9d2",
)
# The SGLang source whose bf16 dispatch _splitk_tactic mirrors (the stock projection's route to FlashInfer's cluster
# split-K and the tactic it runs), at 3117f58 + #48948
PINS = {
    "sglang/srt/layers/quantization/unquant.py": (
        bed.Functions(
            {
                "UnquantizedLinearMethod.apply": "bb339fbe7a9643dd",
                "initialize_bf16_gemm_config": "293a36153ec50715",
                "get_bf16_gemm_backend": "0f0111291b83034a",
                "use_bf16_splitk_gemm": "2503e44bf976740d",
                "_bf16_low_m_autotune": "a9ac9d38b2056c34",
                "_bf16_gemm_dispatch_impl": "7f678770cd557687",
                "_bf16_splitk_gemm": "904ae36bd9b00e4a",
                "_bf16_splitk_gemm_out": "3a2b9e7f7fe14e9d",
            }
        ),
    ),
}
_STASH = "_muse_vr_qk_fold"
_state = {
    "on": False,
    "unfused": False,
    "swa": False,
    "half_warp": False,
    "grouped_mailbox": False,
    "deep_ring": False,
}
_warned: set = set()


def _flashinfer_pinned() -> bool:
    try:
        from flashinfer.gemm.kernels import dense_bf16_gemm_sm100_splitk as fi

        got = hashlib.sha256(Path(fi.__file__).read_bytes()).hexdigest()
    except Exception:  # noqa: BLE001
        return False
    return got in FLASHINFER_SPLITK_SHA256


def register() -> bool:
    """Enable when requested, the FlashInfer source is pinned and the Triton arithmetic form is validated. No hook of
    its own: ``ar_attn_boundary`` calls ``project`` and ``verify_attn`` calls ``take``."""
    if not bed.enabled(NAME):
        return False
    for rel, sha in PINS.items():
        if not bed.pinned(rel, sha):
            bed.report(NAME, False, reason=f"pin mismatch: {rel}")
            return False
    from .ops.swa_verify_attn import qk_arith

    arith = qk_arith()
    pinned = _flashinfer_pinned()
    _state["on"] = arith is not None and pinned
    _state["unfused"] = arith == "unfused"
    _state["swa"] = _state["on"] and bed.enabled(NAME + "_swa")
    _state["half_warp"] = _state["on"] and bed.enabled(NAME + "_half_warp")
    _state["grouped_mailbox"] = _state["on"] and bed.enabled(NAME + "_grouped_mailbox")
    _state["deep_ring"] = _state["on"] and bed.enabled(NAME + "_deep_ring")
    bed.report(
        NAME,
        _state["on"],
        arith=arith,
        flashinfer_pinned=pinned,
        sliding_window_layers=_state["swa"],
        half_warp_rows=_state["half_warp"],
        grouped_mailbox=_state["grouped_mailbox"],
        deep_ring=_state["deep_ring"],
    )
    return _state["on"]


def active() -> bool:
    return _state["on"]


def _step_aside(reason: str) -> None:
    bed.count(NAME, "stock:" + reason)
    if reason not in _warned:
        _warned.add(reason)
        logger.warning("muse_vr_plugin: %s steps aside to stock: %s", NAME, reason)


def _splitk_tactic(m: int, n: int, k: int):
    """The cluster split-K tactic SGLang's bf16 dispatch runs for (m, n, k) (its order: the low-M autotuned kernel,
    the warp split-K table, then the direct kernel, then the split-K table), or None when it runs anything else or the
    tile is not one head."""
    import sglang.srt.layers.quantization.unquant as U

    # _bf16_low_m_autotune takes M <= _BF16_LOW_M_AUTOTUNE_MAX_M first wherever FlashInfer's autotuned mm_bf16 is bound
    # (not on SM107)
    if U._mm_bf16 is not None and m <= U._BF16_LOW_M_AUTOTUNE_MAX_M:
        return None
    if not getattr(U, "_enable_bf16_splitk_gemm", False) or not U.use_bf16_splitk_gemm(
        m, n, k
    ):
        return None
    key = (m, n, k)
    if (
        key in U._active_warp_splitk_tactics
    ):  # the warp split-K table in use (with the SM107 entries on SM107)
        return None
    if U._prefer_direct is not None and U._prefer_direct(m, n, k):
        return None
    tactic = U._BF16_SPLITK_TUNED_TACTICS.get(key)
    if tactic is None or tactic[0] != HEAD_DIM:
        return None
    return tactic


def _stock_qk_triton_eps(attn, qkv, kv_store):
    """eps when the stock fused QK-norm call of a layer without RoPE would run SGLang's Triton kernel on ``qkv`` (the
    conditions verify_attn checks for its prologues), else None."""
    import sglang.kernels.fused_op as fused_op
    from sglang.kernels.spec import KernelBackend

    qkn = attn._fused_qk_norm
    if (
        qkn is None
        or attn.rotary_emb is not None
        or qkn.is_torch_compile
        or qkn._head_dim != HEAD_DIM
    ):
        return None
    if (
        fused_op.get_fused_op_backend() is not None
        or fused_op._oot_dispatch_key() is not None
    ):
        return None
    cands = qkn._auto_backend_candidates()
    if not cands or cands[0] is not KernelBackend.TRITON:
        return None
    if not qkn.backend_eligible(KernelBackend.TRITON, qkv, None, None, kv_store):
        return None
    return float(qkn._eps)


def _stock_qk_rope_triton_eps(attn, qkv, positions, kv_store):
    """eps when the stock fused QK-norm call of a RoPE layer would run SGLang's Triton kernel with these operands (the
    conditions verify_attn checks for its RoPE prologues), else None."""
    import sglang.kernels.fused_op as fused_op
    from sglang.kernels.spec import KernelBackend

    qkn, rope = attn._fused_qk_norm, attn.rotary_emb
    if qkn is None or rope is None or qkn.is_torch_compile or qkn._head_dim != HEAD_DIM:
        return None
    if (
        fused_op.get_fused_op_backend() is not None
        or fused_op._oot_dispatch_key() is not None
    ):
        return None
    cands = qkn._auto_backend_candidates()
    cos_sin = rope.cos_sin_cache
    if not cands or cands[0] is not KernelBackend.TRITON:
        return None
    if not qkn.backend_eligible(
        KernelBackend.TRITON, qkv, positions, cos_sin, kv_store
    ):
        return None
    if not (
        positions.dtype == torch.int64
        and positions.dim() == 1
        and positions.is_contiguous()
        and positions.shape[0] >= qkv.shape[0]
        and positions.data_ptr() % 8 == 0
        and cos_sin.dtype == torch.float32
        and cos_sin.dim() == 2
        and cos_sin.shape[1] == HEAD_DIM
        and cos_sin.is_contiguous()
        and cos_sin.data_ptr() % 16 == 0
    ):
        return None
    return float(qkn._eps)


def _staged_takes(num_requests: int) -> bool:
    """True when swa_verify_attn's staged form (which runs on e4m3 queries only) is set for this request count; False
    when that form is absent from this build or not set."""
    from .ops import swa_verify_attn as op

    staged_for = getattr(op, "staged_for", None)
    return bool(staged_for is not None and staged_for(num_requests))


def _reason(attn, x, forward_batch, out, positions=None):
    """Reason the projection must run stock, or None with (tactic, kv_store, eps, rope operands or None)."""
    from . import verify_attn as va
    from .qk_fold_gemm import qk_fold_cols

    layer = attn.attn
    swa = layer.sliding_window_size is not None and layer.sliding_window_size >= 0
    rope = None
    if swa:
        # the sliding-window layers (RoPE): only with MUSE_VR_QKV_QK_FOLD_SWA=1, on verify_attn's swa_verify_attn path
        if not _state["swa"]:
            return "sliding window layer", None
        if attn.rotary_emb is None or positions is None:
            return "sliding window layer without RoPE / positions", None
        if not va._state["swa"]:
            return "swa_verify_attn path not on", None
        reason, backend = va._common_reason(attn, out, forward_batch)
        if reason is not None:
            return "attention: " + reason, None
        if (
            va._state["swa_t"]
            and va._swat_reason(attn, out, forward_batch, backend) is None
        ):
            return "swa_t takes the call", None
        r = va._swa_reason(attn, out, forward_batch, backend)
        if r is not None:
            return "swa_verify_attn: " + r, None
    else:
        if attn.rotary_emb is not None:
            return "RoPE layer", None
        if not (va._state["fullw"] and not va._state["fullw_qk"]):
            return "full_attn_wide path not on (or its prologue on)", None
        if not va._state[
            "fullw_l"
        ]:  # only the grouped-merge form's path (MUSE_VR_FULL_ATTN_WIDE_GMERGE=1) takes the queries
            return "full_attn_wide grouped-merge form not on", None
        reason, backend = va._common_reason(attn, out, forward_batch)
        if reason is not None:
            return "attention: " + reason, None
        if (
            va._state["fullv"]
            and va._fullv_reason(attn, out, forward_batch, backend) is None
        ):
            return "full_verify_attn takes the call", None
        r = va._fullw_reason(attn, out, forward_batch, backend)
        if r is not None:
            return "full_attn_wide: " + r, None
    kv_store = va._kv_store_ok(attn, forward_batch, backend, contiguous=True)
    if kv_store is None:
        return "no fused FP8 KV store / buffer mismatch", None
    loc = kv_store[2]
    if (
        loc.dtype != torch.int64
        or loc.dim() != 1
        or not loc.is_contiguous()
        or loc.shape[0] < x.shape[0]
        or loc.data_ptr() % 8
        or kv_store[0].data_ptr() % 16
        or kv_store[1].data_ptr() % 16
    ):
        return "cache location / pool layout", None
    if swa:
        # the call swa_verify_attn's query/key prologue takes (the same decision on operands of the same shape and
        # layout): the QK-norm runs inside the attention kernel there, so the projection stays stock, unless the staged
        # form (e4m3 queries only) is set for this request count
        if va._qk_prologue_args(
            attn, positions, out, kv_store
        ) is not None and not _staged_takes(x.shape[0] // TOKENS):
            return "swa_verify_attn query/key prologue takes the call", None
        eps = _stock_qk_rope_triton_eps(attn, out, positions, kv_store)
        rope = (positions, attn.rotary_emb.cos_sin_cache)
    else:
        eps = _stock_qk_triton_eps(attn, out, kv_store)
    if eps is None:
        return "stock QK-norm call not the Triton kernel", None
    m = x.shape[0]
    if qk_fold_cols(m, layer.tp_q_head_num, layer.tp_k_head_num) is None:
        return "row count", None
    proj = attn.qkv_proj
    w = proj.weight
    if (
        getattr(proj, "bias", None) is not None
        or w.dtype != torch.bfloat16
        or w.dim() != 2
        or w.requires_grad
    ):
        return "projection weight", None
    if type(proj.quant_method).__name__ != "UnquantizedLinearMethod":
        return "projection quantization", None
    import sglang.srt.layers.quantization.unquant as U

    if not U.get_bf16_gemm_backend().is_cutedsl():
        return "bf16 GEMM backend", None
    if (
        x.dtype != torch.bfloat16
        or x.dim() != 2
        or not x.is_contiguous()
        or x.data_ptr() % 32
        or w.data_ptr() % 32
    ):
        return "projection input layout", None
    tactic = _splitk_tactic(m, w.shape[0], w.shape[1])
    if tactic is None:
        return "dispatch is not the 64-row cluster split-K", None
    if w.shape[0] < (layer.tp_q_head_num + 2 * layer.tp_k_head_num) * HEAD_DIM:
        return "projection width", None
    return None, (tactic, kv_store, eps, rope)


def project(attn, x, forward_batch, positions=None):
    """``attn.qkv_proj(x)[0]``; on an eligible call the projection runs with the QK-norm epilogue and its queries are
    kept for ``take`` (the same attention module, the same projection output)."""
    if not _state["on"] or not forward_batch.forward_mode.is_target_verify():
        return attn.qkv_proj(x)[0]
    w = attn.qkv_proj.weight
    out = torch.empty(x.shape[0], w.shape[0], dtype=torch.bfloat16, device=x.device)
    reason, got = _reason(attn, x, forward_batch, out, positions)
    if reason is not None:
        if reason != "sliding window layer":
            _step_aside(reason)
        return attn.qkv_proj(x)[0]
    from .qk_fold_gemm import QkFold, SplitKTactic, qk_fold_cols, run_qk_fold_gemm

    tactic, kv_store, eps, rope = got
    layer = attn.attn
    m = x.shape[0]
    q = torch.empty(
        m, layer.tp_q_head_num * HEAD_DIM, dtype=torch.float8_e4m3fn, device=x.device
    )
    pos, cos_sin = rope if rope is not None else (None, None)
    run_qk_fold_gemm(
        x,
        w.T,
        None,
        out,
        True,
        SplitKTactic(*tactic),
        QkFold(
            q,
            kv_store[0],
            kv_store[1],
            kv_store[2],
            layer.tp_q_head_num,
            layer.tp_k_head_num,
            eps,
            _state["unfused"],
            positions=pos,
            cos_sin=cos_sin,
            half_warp_tree=_state["half_warp"],
            grouped_mailbox=_state["grouped_mailbox"],
            deep_ring=_state["deep_ring"],
        ),
    )
    if getattr(attn, _STASH, None) is not None:
        bed.count(NAME, "previous queries not taken")
    setattr(
        attn,
        _STASH,
        (
            out,
            q,
            kv_store[0].data_ptr(),
            kv_store[1].data_ptr(),
            kv_store[2].data_ptr(),
        ),
    )
    bed.count(NAME, f"{'swa_' if rope is not None else ''}fold_reqs{m // TOKENS}")
    if (
        _state["half_warp"]
        and rope is None
        and qk_fold_cols(m, layer.tp_q_head_num, layer.tp_k_head_num) == 2
    ):
        bed.count(
            NAME, f"half_warp_reqs{m // TOKENS}"
        )  # the call ran the half-warp rows (QkFold.half_warp_tree)
    return out


def take(attn, qkv, kv_store):
    """The e4m3 queries ``project`` wrote for this projection output and these cache operands, or None (stock call)."""
    s = attn.__dict__.pop(_STASH, None)
    if s is None:
        return None
    out, q, kp, vp, lp = s
    if out is not qkv or (kp, vp, lp) != (
        kv_store[0].data_ptr(),
        kv_store[1].data_ptr(),
        kv_store[2].data_ptr(),
    ):
        bed.count(NAME, "queries not taken: operands differ")
        return None
    bed.count(NAME, f"taken_reqs{qkv.shape[0] // TOKENS}")
    return q
