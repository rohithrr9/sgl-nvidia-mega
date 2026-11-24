# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Component ``smx`` (env ``MUSE_VR_SMX=1``, default off): the attention of the block rows of SGLang's DFLASH mixed
verify passes on the plugin's verify-attention ops.

SGLang's DFLASH mixed verify (``SGLANG_OPT_DFLASH_MIXED_VERIFY=1``, ``speculative/dflash_mixed_verify.py``) runs the
running batch's next DFLASH step inside an extend pass: one target forward over the extend batch's prompt tokens plus
one block of draft tokens per running request. In every attention call of such a pass the block rows leave the pass
(``layers/attention/mixed_verify_attention.py``) and take the call the served target-verify step makes for the same
running batch: a TARGET_VERIFY view of the batch and of the backend metadata (the block rows' lengths, page-table rows
and write locations), padded to the batch size the verify step's CUDA graph would replay (padding rows as the graph
runner fills them: length 1, slot 0), passed to ``forward_verify_rows``, whose own form is the hooked ``forward_extend``
chain (the verify chain). This component is an AROUND hook on that function: for sliding-window layers whose served
verify call takes the plugin's ``swa_verify_attn`` op (``verify_attn``'s request limit), the block rows take that op
without its fused prologue and epilogue (the pass has already applied the QK norm and applies the output norm and gate
after the call); every other call takes the verify chain. Counted ``block:swa_verify_attn`` / ``block:verify chain``.

SWA-T mirror (``MUSE_VR_SMX_SWAT_MIRROR=1``, default off): the block rows of a sliding-window layer take the attention op
the served verify step takes at the same (graph-padded) batch size, in ``verify_attn``'s order: SWA-T (``ops.swa_t``)
inside its request window, else ``swa_verify_attn`` within that op's limit, else the verify chain. The SWA-T call is the
served one without its QK prologue and output norm / gate (the pass applies those, as for ``swa_verify_attn``): the same
op in its ungated variant, the same cluster size (``kc_for`` of the padded batch size), PDL off (a normal launch inside
the mixed pass), pools, page rows and lengths, after the FP8 store the ``swa_verify_attn`` form uses. The checks are
``verify_attn``'s for that call (``_common_reason`` / ``_swat_reason`` / ``_kv_store_ok``), evaluated in the layer's FP8
view. On a mixed NVFP4 / FP8 pool the served verify step reaches the FP8 paths only through ``kv4_dec``'s view (or
``verify_attn``'s mixed-SWA routing); without either, every block row takes the verify chain, as the served step runs
those layers stock. Not engaged (logged and recorded at install) when SWA-T is not engaged (nothing to mirror) or
``swa_t6`` is engaged (its buckets run ahead of SWA-T; not mirrored). A call inside the window that SWA-T's checks refuse
goes on to the next form, as the served step does, counted ``block:swa_t:aside:<reason>`` and logged once per reason.
Counted ``block:swa_t`` and ``block:swa_t:reqs<n>``; every block-row call ``block:attn:reqs<n>``.

The views are copies of the pass's ForwardBatch: an AROUND hook on ``build_views`` drops a per-batch plan cached on the
pass (``nvfp4_prefill_dq``'s) from both views, so each view computes its own.

The scheduling, the pass, its logits and the split are SGLang's. Not engaged (logged and recorded at install) on an
SGLang without DFLASH mixed verify, with its switch off, or when a pinned SGLang function differs: the block rows then
take the verify chain.
"""

from __future__ import annotations

import importlib.util
import logging
import os

import torch

from . import bed

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "srt/environ.py": ("envs",),
    "srt/plugins/hook_registry.py": ("HookRegistry", "HookType"),
    "srt/layers/radix_attention.py": (
        "AttentionType",
        "attn_type",
        "layer_id",
        "sliding_window_size",
        "tp_q_head_num",
        "tp_k_head_num",
        "head_dim",
        "k_scale",
        "v_scale",
    ),
    "srt/layers/cp/utils.py": ("is_cp_active",),
    "srt/layers/attention/trtllm_mha_backend.py": (
        "forward_metadata",
        "is_ragged_verify",
        "max_seq_len_q",
        "cache_seqlens_int32",
        "is_nvfp4_kvcache",
        "is_xqa_impl",
        "_forward_extend_uses_native_fp4",
        "_fused_fp8_qkv_kv_cache",
        "_get_bmm_scales",
        "_get_layer_page_table",
    ),
    "srt/layers/attention/flashinfer_backend.py": (
        "data_type",
        "q_data_type",
        "page_size",
        "token_to_kv_pool",
    ),
}
# SGLang's DFLASH mixed verify: the two hook targets and the views they hand over. Absent on an SGLang without it, where
# register() refuses before any hook is registered. Its switch is read as getattr(envs, SWITCH): environ.py exists on
# every SGLang, so the name cannot be declared here (test/unit/test_muse_spark_smx.py checks it where the module is).
SGLANG_OPTIONAL_READS = {
    "srt/layers/attention/mixed_verify_attention.py": (
        "forward_verify_rows",
        "build_views",
        "extend_batch",
        "verify_batch",
    ),
}


logger = logging.getLogger("muse_vr_plugin")

NAME = "smx"
_M = "sglang.srt"
MODULE = _M + ".layers.attention.mixed_verify_attention"
T_ROWS = MODULE + ".forward_verify_rows"
T_VIEWS = MODULE + ".build_views"
SWITCH = "SGLANG_OPT_DFLASH_MIXED_VERIFY"

# What the hooks take over, are called from and call. The module is new with DFLASH mixed verify, so it has no earlier
# validated revision.
PINS = {
    "sglang/srt/layers/attention/mixed_verify_attention.py": (
        # the hooked functions, the call that hands the verify rows over (padded rows, verify view set) and the views
        bed.Functions(
            {
                "forward_mixed_verify": "cf79ef2001e83f21",
                "forward_verify_rows": "370f9b6dbabc8a58",
                "pad_rows": "1d271e2dbaea8688",
                "build_views": "dd1ba49d014922eb",
                "_extend_batch": "36e80af996cbb15b",
                "_verify_metadata": "05d66a24cddba8a3",
                "_verify_batch": "5b7f75ca60f5d85a",
            }
        ),
    ),
    "sglang/srt/layers/attention/base_attn_backend.py": (
        # the dispatch of a mixed verify pass's attention calls to forward_mixed_verify
        bed.Functions({"AttentionBackend.forward": "e8b4c8e2cecdcc45"}),
    ),
    "sglang/srt/layers/attention/trtllm_mha_backend.py": (
        # the verify chain, and the helpers the block-row forms call
        bed.Functions(
            {
                "TRTLLMHAAttnBackend.__init__": "70a9364ba6b6c920",
                "TRTLLMHAAttnBackend.forward_extend": "9ddde6d7d3f6d4be",
                "TRTLLMHAAttnBackend._fused_fp8_qkv_kv_cache": "208190898701e7b0",
                "TRTLLMHAAttnBackend._get_layer_cache_loc": "9253a58925623e0c",
                "TRTLLMHAAttnBackend._get_layer_page_table": "2a2d3e0c63990341",
                "TRTLLMHAAttnBackend._get_bmm_scales": "90a1387e8f561611",
                "TRTLLMHAAttnBackend._forward_extend_uses_native_fp4": "2ca6caa1239a837a",
            }
        ),
    ),
}

_state: dict = {"on": False, "swat_mirror": False, "swat_view": True}
_tally: dict = {}
_logged: set = set()


def enabled() -> bool:
    return bed.enabled(NAME)


def pins_match() -> tuple[bool, str]:
    for rel, want in PINS.items():
        if not bed.pinned(rel, want):
            return False, f"{rel} differs from the pinned revision"
    return True, "ok"


def mixed_verify_reason() -> str | None:
    """Why this SGLang runs no DFLASH mixed verify pass, or None: its switch (``SWITCH``, through SGLang's ``envs``) is
    on and the attention module the passes dispatch to is installed."""
    try:
        from sglang.srt.environ import envs

        switch = getattr(envs, SWITCH, None)
        module = importlib.util.find_spec(MODULE)
    except Exception as e:  # noqa: BLE001 - an SGLang that cannot say: no mixed verify
        return (
            f"cannot read SGLang's DFLASH mixed verify switch ({type(e).__name__}: {e})"
        )
    if switch is None or module is None:
        return f"this SGLang has no DFLASH mixed verify ({SWITCH})"
    if not switch.get():
        return f"{SWITCH} is off: SGLang runs no mixed verify pass"
    return None


def _count(key: str) -> None:
    bed.count(NAME, key)
    n = _tally[key] = _tally.get(key, 0) + 1
    if n & (n - 1) == 0:
        bed.count(NAME, f"{key}>={n}")


# ------------------------------------------------------------------------------------------------- block-row attention
def _around_views(original, *args, **kwargs):
    """AROUND hook on SGLang's ``build_views``: the views of one pass."""
    views = original(*args, **kwargs)
    for fb in (views.extend_batch, views.verify_batch):
        # a per-batch plan cached on the pass is not the view's
        fb.__dict__.pop("_muse_vr_prefill_dq_plan", None)
    return views


def _around_rows(original, backend, q, k, v, layer, forward_batch, **kwargs):
    """AROUND hook on SGLang's ``forward_verify_rows`` (``original``: the verify chain)."""
    return block_attention(original, backend, q, k, v, layer, forward_batch, kwargs)


def block_attention(original, backend, q, k, v, layer, ft, kwargs):
    """The block rows' attention: the call the served verify step makes for the same requests (see the module
    docstring). ``q`` / ``k`` / ``v``: the block rows as the pass hands them to the backend (QK norm applied once, by the
    pass), padded by SGLang to the ``ft.batch_size`` requests of the verify view ``ft``, whose metadata the backend
    holds. Returns ``8 x ft.batch_size`` output rows (padding rows last)."""
    if _state["swat_mirror"]:
        return _mirror_block_attention(original, backend, q, k, v, layer, ft, kwargs)
    sw = _swa_op_form(backend, layer, ft.batch_size)
    if sw is not None:
        _count("block:swa_verify_attn")
        return _swa_block_attention(backend, q, k, v, layer, ft, sw)
    _count("block:verify chain")
    return original(backend, q, k, v, layer, ft, **kwargs)


def _swa_op_form(backend, layer, reqs: int):
    """The sliding-window sub-pool (or None for a plain FP8 pool) when the served verify call of this layer at ``reqs``
    requests takes ``verify_attn``'s ``swa_verify_attn`` op, else None. Mirrors that component's checks for the call
    (``_common_reason`` / ``_swa_reason`` / ``_swa_path``), evaluated in the layer's FP8 view."""
    from . import nvfp4_kv_mixed as km
    from . import verify_attn as va

    if not va._state.get("swa"):
        return None
    w = layer.sliding_window_size
    if w is None or w < 0:
        return None
    sub = (
        km._fp8_sub_pool_of(getattr(backend, "token_to_kv_pool", None), layer.layer_id)
        if km.requested()
        else None
    )
    if km.requested() and sub is None:
        return None
    from contextlib import nullcontext

    import sglang.srt.layers.radix_attention as ra
    from sglang.srt.environ import envs

    from .ops import swa_verify_attn as op

    with km._fp8_view(backend, sub) if sub is not None else nullcontext():
        if layer.attn_type != ra.AttentionType.DECODER:
            return None
        if (
            backend.data_type != torch.float8_e4m3fn
            or backend.is_nvfp4_kvcache
            or backend.is_xqa_impl
        ):
            return None
        hkv = layer.tp_k_head_num
        if reqs > va._limit("max_reqs", hkv):
            return None
        if not op.supported(layer.tp_q_head_num, hkv, layer.head_dim, w):
            return None
        if backend.q_data_type != torch.bfloat16:
            return None
        if envs.SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR.get() is not None:
            return None
        bmm1, bmm2 = backend._get_bmm_scales(layer)
        if not isinstance(bmm1, float) or not isinstance(bmm2, float):
            return None
        if layer.k_scale is not None or layer.v_scale is not None:
            return None
    return sub if sub is not None else False


def _swa_block_attention(backend, q, k, v, layer, ft, sub):
    """``swa_verify_attn`` for the block rows (FP8 store of the rows, then the op without its QK prologue or norm-gate
    epilogue), in the layer's FP8 view."""
    from contextlib import nullcontext

    from . import nvfp4_kv_mixed as km
    from .ops import swa_verify_attn as op

    with km._fp8_view(backend, sub) if sub else nullcontext():
        # e4m3 store of k / v rows (and e4m3 q), unit scales
        q8 = backend._fused_fp8_qkv_kv_cache(q, k, v, layer, ft)
        if q8 is None:
            raise RuntimeError("smx: the fused FP8 store returned no query")
        k_pool, v_pool = backend.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        bmm1, bmm2 = backend._get_bmm_scales(layer)
        hq, d = layer.tp_q_head_num, layer.head_dim
        m, reqs = q8.shape[0], ft.batch_size
        key = (layer.sliding_window_size, hq, layer.tp_k_head_num, str(q8.device))
        if key not in _state.setdefault("swa_loaded", set()):
            from . import verify_attn as va

            op.preload(
                layer.sliding_window_size,
                va._limit("max_reqs", layer.tp_k_head_num),
                hq,
                layer.tp_k_head_num,
                q8.device,
                False,
            )
            _state["swa_loaded"].add(key)
        out = torch.empty(m, hq * d, dtype=torch.bfloat16, device=q8.device)
        op.swa_verify_attn(
            q8.reshape(m, hq, d),
            k_pool.view(k_pool.shape[0], -1),
            v_pool.view(v_pool.shape[0], -1),
            backend._get_layer_page_table(layer, ft),
            backend.forward_metadata.cache_seqlens_int32[:reqs],
            layer.sliding_window_size,
            backend.page_size,
            bmm1,
            bmm2,
            out=out.view(m, hq, d),
        )
    return out


# ------------------------------------------------------------------------ SWA-T mirror (MUSE_VR_SMX_SWAT_MIRROR)
# SWA-T's own reasons for not taking a call that are by design (not counted, as in verify_attn's served hook)
_SWAT_BY_DESIGN = ("full attention layer", "requests below min", "requests above max")


def _capturing() -> bool:
    return torch.cuda.is_current_stream_capturing()


def _layer_is_decoder(layer) -> bool:
    import sglang.srt.layers.radix_attention as ra

    return layer.attn_type == ra.AttentionType.DECODER


def _skip_softmax_set() -> bool:
    from sglang.srt.environ import envs

    return envs.SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR.get() is not None


def _cp_active(ft) -> bool:
    from sglang.srt.layers.cp.utils import is_cp_active

    return bool(is_cp_active(ft))


def swat_window(va_state: dict, kv_heads: int) -> tuple:
    """SWA-T's request window for this KV-head count per rank, as ``verify_attn._swat_reason`` reads it."""
    if kv_heads == 2:
        return va_state["swa_t_min"], va_state["swa_t_max"]
    return va_state["swa_t_min4"], va_state["swa_t_max4"]


def swat_mirror_install(va_state: dict, mixed_pool: bool, kv4_dec: bool) -> tuple:
    """Install decision of the SWA-T mirror: (engaged, reason it is not engaged or a caveat, served FP8 view).
    ``va_state``: ``verify_attn._state`` after that component's install; ``mixed_pool``: a mixed NVFP4 / FP8 pool is
    requested; ``kv4_dec``: that component is engaged (it gives the served verify step the FP8 view of such a pool)."""
    if not va_state.get("swa_t"):
        return (
            False,
            "SWA-T (swa_t) not engaged: the served verify step never takes it",
            True,
        )
    if va_state.get("swa_t6"):
        return (
            False,
            "swa_t6 engaged: its buckets run ahead of SWA-T in the served verify step (not mirrored)",
            True,
        )
    if mixed_pool and not (kv4_dec or va_state.get("mixed_swa")):
        return (
            True,
            (
                "mixed NVFP4 / FP8 pool without kv4_dec: the served verify step runs the sliding-window layers "
                "stock, so every block row takes the verify chain"
            ),
            False,
        )
    return True, None, True


def _swat_op_form(backend, layer, ft):
    """((FP8 sub-pool or False, cluster size), None) when the served verify call of this layer at ``ft.batch_size``
    requests takes ``verify_attn``'s SWA-T path, else (None, reason). Mirrors that hook's checks for the call
    (``_common_reason``, ``_swat_reason``, ``_kv_store_ok``), evaluated in the layer's FP8 view. The ungated modules of
    the request window are built at the first eager call; a call inside a graph capture needs them built."""
    from contextlib import nullcontext

    from . import nvfp4_kv_mixed as km
    from . import verify_attn as va
    from .ops import swa_t as op

    w = layer.sliding_window_size
    if w is None or w < 0:
        return None, "full attention layer"
    reqs, hkv = int(ft.batch_size), layer.tp_k_head_num
    lo, hi = swat_window(va._state, hkv)
    if reqs < lo:
        return None, "requests below min"
    if reqs > hi:
        return None, "requests above max"
    md = backend.forward_metadata
    if md is None or md.is_ragged_verify or md.max_seq_len_q != va.TOKENS:
        return None, "verify layout"
    sub = (
        km._fp8_sub_pool_of(getattr(backend, "token_to_kv_pool", None), layer.layer_id)
        if km.requested()
        else None
    )
    if km.requested() and sub is None:
        return None, "not in the FP8 sliding-window sub-pool"
    with km._fp8_view(backend, sub) if sub is not None else nullcontext():
        if not _layer_is_decoder(layer):
            return None, "attention type"
        if (
            backend.data_type != torch.float8_e4m3fn
            or backend.is_nvfp4_kvcache
            or backend.is_xqa_impl
        ):
            return None, "kv cache format"
        if backend._forward_extend_uses_native_fp4(ft) or _cp_active(ft):
            return None, "nvfp4 / context parallel"
        if not op.supported(
            layer.tp_q_head_num, hkv, layer.head_dim, backend.page_size
        ):
            return None, "shape"
        if backend.q_data_type != torch.bfloat16:
            return None, "output dtype"
        if _skip_softmax_set():
            return None, "skip-softmax threshold set"
        bmm1, bmm2 = backend._get_bmm_scales(layer)
        if not isinstance(bmm1, float) or not isinstance(bmm2, float):
            return None, "tensor scales"
        if (
            layer.k_scale is not None
            or layer.v_scale is not None
            or getattr(ft, "dcp_kv_mask", None) is not None
        ):
            return None, "no fused FP8 KV store"
        k_pool, v_pool = backend.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        if not (k_pool.is_contiguous() and v_pool.is_contiguous()):
            return None, "KV buffers not contiguous"
        dev = k_pool.device
        # PDL off on the block-row call: a normal launch inside the mixed pass (its PDL edges are not yet verified)
        if _capturing():
            if not op.is_loaded(reqs, dev, False, False, hkv):
                return None, "kernel not loaded before graph capture"
        else:
            key = (w, hkv, lo, hi, str(dev))
            if key not in _state.setdefault("swat_loaded", set()):
                op.preload(hi, dev, False, False, min_reqs=lo, qk=False, kv_heads=hkv)
                _state["swat_loaded"].add(key)
        kc = op.kc_for(reqs, dev, False, False, hkv)
        if not kc:
            return None, "no cluster size fits"
    return (sub if sub is not None else False, kc), None


def _served_common_reason(backend, layer, ft):
    """``verify_attn._common_reason`` for the served verify call of this layer at ``ft.batch_size`` requests (in the
    layer's FP8 view), or None. A reason sends the served call to SGLang's own method on every path."""
    from contextlib import nullcontext

    from . import nvfp4_kv_mixed as km
    from . import verify_attn as va

    md = backend.forward_metadata
    if md is None or md.is_ragged_verify or md.max_seq_len_q != va.TOKENS:
        return "verify layout"
    sub = (
        km._fp8_sub_pool_of(getattr(backend, "token_to_kv_pool", None), layer.layer_id)
        if km.requested()
        else None
    )
    if km.requested() and sub is None:
        return "kv cache format"  # an NVFP4 layer of a mixed pool: no FP8 view in the served step
    with km._fp8_view(backend, sub) if sub is not None else nullcontext():
        if not _layer_is_decoder(layer):
            return "attention type"
        if (
            backend.data_type != torch.float8_e4m3fn
            or backend.is_nvfp4_kvcache
            or backend.is_xqa_impl
        ):
            return "kv cache format"
        if backend._forward_extend_uses_native_fp4(ft) or _cp_active(ft):
            return "nvfp4 / context parallel"
    return None


def block_op_form(backend, layer, ft) -> tuple:
    """With the SWA-T mirror on: the attention form the served verify step takes for this layer's call at
    ``ft.batch_size`` requests, in ``verify_attn``'s order: ("swa_t", sub, kc), ("swa", sub) or ("chain",)."""
    if not _state["swat_view"] or _served_common_reason(backend, layer, ft) is not None:
        return ("chain",)
    st, why = _swat_op_form(backend, layer, ft)
    if st is not None:
        return ("swa_t",) + tuple(st)
    if why not in _SWAT_BY_DESIGN:
        bed.count(NAME, "block:swa_t:aside:" + why)
        if "swat:" + why not in _logged:
            _logged.add("swat:" + why)
            logger.warning(
                "muse_vr_plugin: %s SWA-T mirror steps aside (%s): the block rows take the next form",
                NAME,
                why,
            )
    sw = _swa_op_form(backend, layer, ft.batch_size)
    return ("swa", sw) if sw is not None else ("chain",)


def _mirror_block_attention(original, backend, q, k, v, layer, ft, kwargs):
    """``block_attention``'s dispatch with the SWA-T mirror on (``q`` / ``k`` / ``v`` padded, the block view set)."""
    form = block_op_form(backend, layer, ft)
    # every block-row call: == block:swa_t:reqs<n> inside the window
    bed.count(NAME, f"block:attn:reqs{int(ft.batch_size)}")
    if form[0] == "swa_t":
        _count("block:swa_t")
        bed.count(NAME, f"block:swa_t:reqs{int(ft.batch_size)}")
        return _swat_block_attention(backend, q, k, v, layer, ft, form[1], form[2])
    if form[0] == "swa":
        _count("block:swa_verify_attn")
        return _swa_block_attention(backend, q, k, v, layer, ft, form[1])
    _count("block:verify chain")
    return original(backend, q, k, v, layer, ft, **kwargs)


def _swat_block_attention(backend, q, k, v, layer, ft, sub, kc: int):
    """SWA-T for the block rows: the FP8 store of the rows, then the op without its QK prologue or norm-gate epilogue at
    the served call's cluster size with PDL off (a normal launch inside the fold pass), in the layer's FP8 view."""
    from contextlib import nullcontext

    from . import nvfp4_kv_mixed as km
    from .ops import swa_t as op

    with km._fp8_view(backend, sub) if sub else nullcontext():
        # e4m3 store of k / v rows (and e4m3 q), unit scales
        q8 = backend._fused_fp8_qkv_kv_cache(q, k, v, layer, ft)
        if q8 is None:
            raise RuntimeError("smx: the fused FP8 store returned no query")
        k_pool, v_pool = backend.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        bmm1, bmm2 = backend._get_bmm_scales(layer)
        hq, d = layer.tp_q_head_num, layer.head_dim
        m, reqs = q8.shape[0], ft.batch_size
        out = torch.empty(m, hq * d, dtype=torch.bfloat16, device=q8.device)
        res = op.swa_t_verify_attn(
            q8.reshape(m, hq, d),
            k_pool.view(k_pool.shape[0], -1),
            v_pool.view(v_pool.shape[0], -1),
            backend._get_layer_page_table(layer, ft),
            backend.forward_metadata.cache_seqlens_int32[:reqs],
            layer.sliding_window_size,
            bmm1,
            bmm2,
            out=out.view(m, hq, d),
            use_pdl=False,
            gate=None,
            kc=kc,
        )
        # cannot happen with kc given; the store above has already run
        if res is None:
            raise RuntimeError(
                "smx: SWA-T launched nothing for a call its checks admitted"
            )
    return out


def register() -> bool:
    if not enabled():
        return False
    why = mixed_verify_reason()
    if why is None:
        ok, why = pins_match()
        why = None if ok else why
    if why is not None:
        bed.report(NAME, False, reason=why)
        logger.warning(
            "muse_vr_plugin: %s requested but NOT engaged (%s): the block rows take SGLang's verify chain",
            NAME,
            why,
        )
        return False
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    swat_report = {}
    if os.environ.get("MUSE_VR_SMX_SWAT_MIRROR", "0").strip() == "1":
        from . import nvfp4_kv_mixed as km
        from . import verify_attn as va

        ok, why, view = swat_mirror_install(
            va._state,
            km.requested(),
            bool(bed.installs().get("kv4_dec", {}).get("engaged")),
        )
        _state["swat_mirror"], _state["swat_view"] = ok, view
        swat_report = {"swat_mirror": ok, "swat_mirror_view": view}
        if why:
            swat_report["swat_mirror_note"] = why
            logger.warning(
                "muse_vr_plugin: %s SWA-T mirror %s (%s)",
                NAME,
                "engaged" if ok else "requested but NOT engaged",
                why,
            )
    A = HookType.AROUND
    HookRegistry.register(T_VIEWS, _around_views, A)
    HookRegistry.register(T_ROWS, _around_rows, A)
    _state["on"] = True
    bed.report(
        NAME,
        True,
        block_rows="served verify call (graph-padded view)",
        **swat_report,
    )
    return True
