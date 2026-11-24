# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Component ``moe_ld`` (env ``MUSE_VR_MOE_LD=1``, default off): the routed NVFP4 experts of a decode / verify step in
one kernel launch per MoE layer (``moe_ld_ops.moe_ld``) for at most 16 token rows per rank.

Hooks (AROUND, by attribute replacement):

* ``MuseSparkNvFp4FusedMoEMethod.process_weights_after_loading``: after SGLang prepares a layer's expert weights, the
  shuffled ``gate_up_proj`` and ``down_proj`` tensors move into ``ld_layout.split_rows_copy`` placements (same shape,
  strides and bytes; each expert's row halves backed by the two locality domains when the device has two) and the
  layer gets its call-site workspace (``moe_ld_ops.Site``). This runs at load time, before any CUDA graph is captured.
* ``MuseSpark1xFp4Experts.forward``: marks the layer for the duration of the call (thread-local); with
  ``MUSE_VR_MOE_LDW_SP=1`` also ``MuseSpark1xFp4Experts.forward_dispatched`` (the sequence-parallel expert stage).
* ``flashinfer.fused_moe.trtllm_fp4_block_scale_routed_moe``: a call made from a marked, prepared layer with the decode
  argument set (packed top-k ids, 1..16 rows, top-k 8, SwiGLU, no biases or clamps, per-token activation scales,
  ``do_finalize=False``, at most 128 local experts) runs ``moe_ld`` and returns the same
  ``(gemm2_out, expert_weights, expanded_idx_to_permuted_idx)`` triple; every other call goes to FlashInfer unchanged.
  SGLang's finalize consumes the triple as before.

Numerics: every FC2 output row equals FlashInfer's bitwise for the same inputs (``tests/test_moe_ld.py``); the finalize
reads the rows through the returned map, so the layer output is bitwise identical.

Shadow (``MUSE_VR_MOE_LD_SHADOW=1``): calls made outside CUDA graph capture also run FlashInfer and compare every routed
row and the top-k weights bitwise (engagement record keys ``shadow:match`` / ``shadow:MISMATCH``). A mismatch turns the
component off for the rest of the process.

Steps aside: more than 16 rows or another argument set, FlashInfer autotuning, sequence-parallel MoE, two-batch or
single-batch overlap, a layer whose weights were not prepared, a disabled component.

Large verify batches (env ``MUSE_VR_MOE_LDW=1`` in addition, default off): an engaged-layer call with the same argument set
but ``MUSE_VR_MOE_LDW_MIN`` (default 64) .. 512 rows runs ``ops.moe_ldw.moe_ldw`` (one launch, one per-device scratch, the
same placed weights) instead of FlashInfer; every routed FC2 row bitwise equal (``tests/test_moe_ldw.py``). Below the
minimum and above 512 rows FlashInfer runs unchanged. Shadow comparison as for ``moe_ld``. Calls of at most
``MUSE_VR_MOE_LDW_HEAD_MAX_M`` rows (default 64) use the build that copies the first FC1 item's token rows by row index;
larger calls use the same file built without that path (``-DMUSE_VR_LDW_GATHER=0``). Both builds give the same rows.
``MUSE_VR_MOE_LDW_RING4=1`` (default off) runs MoE-LD-W calls of at most ``MUSE_VR_MOE_LDW_RING4_MAX_M`` rows (default 256)
on the RING4 build instead (32-row passes, a 4-stage ring: more weight bytes in flight per CTA; experts with more rows take
more passes, which is why larger calls keep the builds above; the same rows; its own per-device scratch).
``MUSE_VR_MOE_LDW_RING48=1`` (default off) runs MoE-LD-W calls of ``MUSE_VR_MOE_LDW_RING48_MIN_M`` (default 96) ..
``MUSE_VR_MOE_LDW_RING48_MAX_M`` (default 512) rows on the RING48 build (RING4's ring with passes of up to 48 rows: experts with
33-48 rows take one pass, hot experts fewer passes; the same rows; its own per-device scratch); it takes precedence over
RING4 and the builds above inside its range. Tally ``m<M>_p<P>_r48``.
``MUSE_VR_MOE_LDW_DENSE_MAX_M`` (default 0 = off; at most 32; needs RING4) runs RING4 calls of at most that many rows on
RING4's dense-token build (``ops.moe_ldw.RING4_DENSE``): every FC1 item takes the call's whole token block straight from the
token tensor (one tensor copy per stage) instead of an expert-major permuted copy behind a grid-wide rows-ready count, and
stores the routed token columns only, at the same rows; FC2 and the outputs as RING4's; every routed FC2 row bitwise
equal (``tests/test_moe_ldw.py`` / ``tests/test_ring_place.py`` with the build's flag); its own per-device scratch; the ring
placement of component ringbal applies as for RING4. Tally ``m<M>_p<P>_r4d``.
``MUSE_VR_MOE_LDW_PRE=1`` (default off; needs the dense-token form above) runs those calls on the dense-token build's early
prologue form (``ops.moe_ldw.RING4_DENSE_PRE``, ``-DMUSE_VR_LDW_PRE=1``): the id loads, per-expert counts, the ring placement,
the expert bases and pass tables, and an L2 prefetch of the first ring stages' weights for the SM's hinted slot run before
the kernel's dependency wait; the start ticket, the slot claim, the token data and everything after stay after it. Every
routed FC2 row bitwise the dense-token build's. It relies on the routed ids being complete when the kernel launches (the
routing kernel reaches the experts' launch through a full graph dependency, as in SGLang's side-stream order). Same tallies,
plus ``pre:m<M>`` (record ``moe_ldw_pre``). ``MUSE_VR_MOE_LDW_PRE=2`` runs the speculate-and-verify form of it
(``ops.moe_ldw.RING4_DENSE_PREV``, ``-DMUSE_VR_LDW_PRE_VERIFY=1``): the early work uses the ids as read before the wait,
the kernel re-reads them after the wait and rebuilds its tables from the fresh words on any difference, so the ids need not
be complete at launch; every routed FC2 row bitwise the dense-token build's (record ``moe_ldw_pre`` with ``verify``).
``MUSE_VR_MOE_LDW_WIDE=1`` (default off) adds a wide tail: engaged-layer calls of 513..1024 rows (the same argument set) run
MoE-LD-W's wide build (``ops.moe_ldw.WIDE``, 128-row passes, no head copies, its own per-device scratch for 1024 rows) instead
of FlashInfer; ``MUSE_VR_MOE_LDW_WIDE=2048`` takes 513..2048 rows on the ``WIDE2K`` build (replicated passes reach 1280 rows).
Every routed FC2 row bitwise equal to the stock call's, as for the builds above. Calls of at most 512 rows keep every build,
scratch and form above unchanged (RING4 / RING48 / the ring placement of component ringbal); the wide tail never runs the ring
placement. Tally ``m<M>_p<P>_wide`` (record ``moe_ldw_wide``).
``MUSE_VR_MOE_LDW_SP=1`` (default off; needs the wide tail) lets the wide tail also take sequence-parallel calls (the
all-to-all receive buffer: rows of every source rank, padding rows carrying the sentinel expert id -1, which MoE-LD-W counts as
non-local like any other rank's expert, so their token rows and scales are never read). Calls of at most 512 rows still step
aside under sequence parallelism, as before. Tally ``m<M>_p<P>_wide_sp`` (record ``moe_ldw_sp``).
``MUSE_VR_MOE_LD_MAX_M`` (default 16) bounds the rows MoE-LD takes; 0 leaves calls of up to 16 rows on FlashInfer, so only
MoE-LD-W engages.

``MUSE_VR_MOE_LDW_ERRWORD=1`` (default off): every MoE-LD-W build runs its error-word form (``ops.moe_ldw.ERRWORD``): a CTA
whose wait expired or whose slot claim failed does not trap; it sets this rank's sticky word in the group's error page
(``ar_norm_residual.consumer_error_init``; tag ``ops.moe_ldw.LDW_ROLE`` << 16 | the layer's call-site index) and the kernel
completes. The verify commit's gate (``eager_commit``, ``MUSE_VR_COMMIT_GATE``) reads the word, so such a step never
commits; this form serves only when that gate can engage (``ldw_errword_reason``), else the knob refuses (loud record) and
the trap form runs. A call made before the gate is armed or before any component has mapped the page runs the trap form
(counted ``trap:...``). Every routed row is bitwise the trap form's when no wait expires.

Early-trigger quant for MoE-LD-W (env ``MUSE_VR_MOE_LDW_EARLY=1`` in addition to ``MUSE_VR_MOE_LDW=1``, default off): the
experts' input quant of a call MoE-LD-W will take (``MuseSpark1xFp4Experts.prepare_input``, AROUND) runs a copy of SGLang's
fused norm + NVFP4 quant whose programmatic trigger is issued at kernel entry (``ops.norm_fp4_quant_early``: the same
arithmetic and launch shape), so MoE-LD-W launches while the quant runs; MoE-LD-W reads every quant output only after its
dependency wait (both builds: W and its RING4 form). Same outputs, bitwise. A call fed by the early quant that MoE-LD-W then does not take
raises (fail-stop): another consumer could read the outputs before they are complete. Tally ``early_quant:m<M>``
(component ``moe_ldw``). Pinned to SGLang's quant sources (loud record ``moe_ldw_early``).

KM2 consumer (env ``MUSE_VR_KM2=1`` in addition, default off): when a board producer component has published this layer
call's routed ids (and optionally its token operand) on the layer's call-site boards (``csrc/lb_ct1.cuh``) and set the
board context for the call (``set_board_context``), the engaged call runs ``ops.lb_km2.km2`` instead of ``moe_ld``: the
same rows, bitwise, with the ids / tokens read from the boards as soon as they are published. Pairing: the producer asks
``board_consumer_ready(m)`` before it publishes; whenever a context was set and the call steps aside after all, the
layer's consumer site still records this forward's sequence value (so a later missed per-forward advance can never make
that publication look current). ``preload()`` runs at load time for every KM2 build.

Package form: this module is the ``l`` form of ``moe_ld.py``. The package's ``moe_ld.register()`` delegates
to ``register()`` here under ``MUSE_VR_MOE_LD_FORM=l`` (one MoE form per process; the component name and its records
stay ``moe_ld``); ``ops.moe_ldw`` above is ``ops/moe_ldw_l.py`` (``csrc/moe_ldw_l.cuh``). The package has no
``ops.lb_km2`` (KM2) and none of the error-word form's helpers (``eager_commit.gate_reason`` / ``gate_armed``,
``ar_norm_residual.consumer_error_word``), so ``MUSE_VR_KM2`` and ``MUSE_VR_MOE_LDW_ERRWORD`` must stay unset (either one
makes the load-time preparation fail; record ``skipped``).
"""

from __future__ import annotations

import logging
import threading

from . import bed

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "srt/models/muse_spark.py": ("get_forward",),
    "srt/layers/muse_spark/muse_spark_fp4_moe.py": ("MuseSpark1xFp4Experts",),
    "srt/layers/muse_spark/muse_spark_nvfp4_moe.py": ("MuseSparkNvFp4FusedMoEMethod",),
    "srt/layers/moe/utils.py": ("is_sbo_enabled", "is_tbo_enabled"),
    "srt/plugins/hook_registry.py": ("HookRegistry", "HookType"),
}


logger = logging.getLogger("muse_vr_plugin")

NAME = "moe_ld"
PINS = {
    # the experts' construction, call (hooked; its routed-MoE call and the finalize of the triple), prepare_input (the
    # early quant's AROUND hook) and forward_dispatched (the SP tail's hook), at 3117f58 + #48948
    "sglang/srt/layers/muse_spark_fp4_moe.py": (
        bed.Functions(
            {
                "MuseSpark1xFp4Experts.__init__": "997db56fae18e78f",
                "MuseSpark1xFp4Experts.forward": "b0123c5c8460b210",
                "MuseSpark1xFp4Experts.prepare_input": "2c05166d10dbdde1",
                "MuseSpark1xFp4Experts._finalize_deferred_output": "243e920761c83e15",
                "MuseSpark1xFp4Experts.forward_dispatched": "55ee763b2d00ed84",
            }
        ),
    ),
    "sglang/srt/layers/quantization/muse_spark_nvfp4_moe.py": (
        "c09e0b4e32b7d5ac7af30caf74121dc0897e14e44f0e4503ea33a7acaf8f35a0",
        "6a1769ef903f1771040fc97e78c3fa46662ab0311cb45a5455984562da2c8a5d",
    ),
}
_NONE_ARGS = (
    "routing_bias",
    "gemm1_bias",
    "gemm1_alpha",
    "gemm1_beta",
    "gemm1_clamp_limit",
    "gemm2_bias",
    # FlashInfer 0.7.4: caller-owned scratch (needs do_finalize=True) and pre-built routing
    "workspace_buffer",
    "routing_metadata",
)

_tls = threading.local()
_state: dict = {
    "engaged": None,
    "orig_moe": None,
    "sites": {},
    "prepared": 0,
    "skipped": "",
    "off": "",
    "km2": False,
    "km2_sites": {},
    "ldw": False,
    "ldw_scratch": {},
    "ldw_min": 64,
    "ldw_head_max_m": 64,
    "ld_max_m": 16,
    "ldw_ring4": False,
    "ldw_ring4_max_m": 256,
    "ldw_ring48": False,
    "ldw_ring48_min_m": 96,
    "ldw_ring48_max_m": 512,
    "ldw_scratch48": {},
    "ldw_scratch4": {},
    "ldw_early": False,
    "early_armed": None,
    "ringbal": False,
    "ldw_wide": 0,
    "ldw_scratch_w": {},
    "ldw_sp": False,
    "ldw_dense_max_m": 0,
    "ldw_scratch4d": {},
    "ldw_pre": False,
}

EARLY_NAME = "moe_ldw_early"  # env MUSE_VR_MOE_LDW_EARLY (with moe_ldw)
EARLY_PINS = {  # the quant the copy in csrc/norm_fp4_quant_early.cuh was taken from + its dispatcher
    # 3117f58 + #48948: adds the static-scale mode only (kStaticScale = false is the per-token code the copy has)
    "sglang/kernels/jit/csrc/muse_spark_v12/norm_fp4_quant.cuh": "a1d08b5b20a5e1b0a5ab3c831a3f9f80297d8ba29bab267901da32c145053a94",
    # the public function, its JIT dispatch and the launch shape the copy mirrors (3117f58 + #48948)
    "sglang/kernels/ops/muse_spark_v12/norm_fp4_quant.py": (
        bed.Functions(
            {
                "_num_threads": "b5f5773e85e9f829",
                "_jit_module": "54a23b3fac82fe47",
                "_fused_norm_fp4_quant_jit": "19c6fd35415e4843",
                "_FusedNormFP4Quant.backend_eligible": "e43a0d012f4c6129",
                "_FusedNormFP4Quant.forward_jit": "42293f464c6f5f74",
                "fused_norm_fp4_quant": "4736e9fb088b713f",
            }
        ),
    ),
}
T_PREPARE = "sglang.srt.layers.muse_spark_fp4_moe.MuseSpark1xFp4Experts.prepare_input"


class BoardContext:
    """One layer call's board state, set by the producer component around the experts call it published for."""

    def __init__(
        self,
        seq,
        ids_board,
        prod_err,
        tok_hdr=None,
        tok_writers: int = 0,
        release_form: bool = False,
    ):
        self.seq, self.ids_board, self.prod_err = seq, ids_board, prod_err
        self.tok_hdr, self.tok_writers, self.release_form = (
            tok_hdr,
            int(tok_writers),
            bool(release_form),
        )


def set_board_context(ctx: BoardContext | None) -> None:
    """Producer side: the boards this layer call's producer published on (None clears it)."""
    _tls.board = ctx


def board_consumer_ready(m: int) -> bool:
    """Producer side, before publishing: would the current layer's experts call engage the KM2 consumer at m rows?"""
    from .moe_ld_ops import MAX_TOKENS

    layer = getattr(_tls, "layer", None)
    return bool(
        _state["km2"]
        and not _state["off"]
        and 1 <= m <= MAX_TOKENS
        and layer is not None
        and _state["km2_sites"].get(layer.gate_up_proj.data_ptr()) is not None
        and not _tuning()
        and not _sp_active()
        and not _overlap_enabled()
    )


def _tuning() -> bool:
    try:
        from flashinfer.autotuner import AutoTuner

        return bool(AutoTuner.get().is_tuning_mode)
    except Exception:  # noqa: BLE001
        return False


def _overlap_enabled() -> bool:
    try:
        from sglang.srt.layers.moe.utils import is_sbo_enabled, is_tbo_enabled

        return bool(is_tbo_enabled() or is_sbo_enabled())
    except Exception:  # noqa: BLE001 - unknown configuration: step aside
        return True


def _sp_active() -> bool:
    try:
        import sglang.srt.models.muse_spark as ms

        return bool(getattr(ms.get_forward(), "sp_active", False))
    except Exception:  # noqa: BLE001
        return True


def _layer_reason(layer) -> str:
    """Empty when this expert layer can use the kernel, else why not."""
    from .moe_ld_ops import HIDDEN, INTER, MAX_POOL

    if getattr(layer, "use_marlin", False) or getattr(layer, "use_megamoe", False):
        return "marlin / megamoe expert path"
    w13, w2 = getattr(layer, "gate_up_proj", None), getattr(layer, "down_proj", None)
    if w13 is None or w2 is None:
        return "no expert weights"
    if (
        w13.dim() != 3
        or tuple(w13.shape[1:]) != (2 * INTER, HIDDEN // 2)
        or not w13.is_contiguous()
    ):
        return f"gate_up_proj shape {tuple(w13.shape)}"
    if (
        w2.dim() != 3
        or tuple(w2.shape[1:]) != (HIDDEN, INTER // 2)
        or not w2.is_contiguous()
    ):
        return f"down_proj shape {tuple(w2.shape)}"
    if w13.shape[0] > MAX_POOL or w2.shape[0] != w13.shape[0]:
        return f"{w13.shape[0]} expert slots"
    if not bool(
        getattr(getattr(layer, "quant_config", None), "use_per_token_activation", False)
    ):
        return "per-token activation scaling is off"
    return ""


def _prepare_layer(layer) -> None:
    """Placement of the shuffled expert weights + the layer's call-site workspace (load time)."""
    import torch

    from . import ld_layout
    from .moe_ld_ops import Site

    why = _layer_reason(layer)
    if why:
        _state["skipped"] = why
        return
    dev = layer.gate_up_proj.device
    two = False
    try:
        two = ld_layout.domain_count(dev) == 2
    except Exception:  # noqa: BLE001 - placement only changes speed
        two = False
    if two:
        with torch.no_grad():
            for name in ("gate_up_proj", "down_proj"):
                p = getattr(layer, name)
                placed = ld_layout.split_rows_copy(p.data, 2)
                p.data = placed
                del placed
                torch.cuda.empty_cache()  # the replaced storage goes back before the next layer's placement
    if not _state["prepared"]:
        # once per process, at load time (outside any graph capture): the SM -> locality-domain table (probe kernels
        # and a synchronization) and the kernel builds for both row configurations
        from . import moe_ld_ops

        moe_ld_ops.domain_slots(
            dev.index if dev.index is not None else torch.cuda.current_device()
        )
        for n in sorted(moe_ld_ops.CONFIGS):
            moe_ld_ops._module(*moe_ld_ops.config_for(n))
    site = Site(dev)
    site.ew_index = len(
        _state["sites"]
    )  # the layer's call-site index (load order on this rank): MoE-LD-W's error-word tag
    _state["sites"][layer.gate_up_proj.data_ptr()] = site
    dix = dev.index if dev.index is not None else torch.cuda.current_device()
    if _state["ldw"] and dix not in _state["ldw_scratch"]:
        from .ops import moe_ldw_l as moe_ldw

        if _state["ldw_ring48"]:
            _ldw_preload(moe_ldw.RING48)
            _state["ldw_scratch48"][dix] = moe_ldw.Scratch(
                dev, moe_ldw.RING48_PASS_ROWS
            )
        if _state["ldw_ring4"] and _ldw_needs("r4"):
            _ldw_preload(moe_ldw.RING4)
            _state["ldw_scratch4"][dix] = moe_ldw.Scratch(dev, moe_ldw.RING4_PASS_ROWS)
        if _state["ldw_ring4"] and _ldw_needs("r4d"):
            _ldw_preload(_dense_build())
            _state["ldw_scratch4d"][dix] = moe_ldw.Scratch(dev, moe_ldw.RING4_PASS_ROWS)
        if _ldw_needs("base"):
            _ldw_preload()
            if _state["ldw_head_max_m"] < moe_ldw.MAX_TOKENS:
                _ldw_preload(_LDW_NO_HEAD)
            _state["ldw_scratch"][dix] = moe_ldw.Scratch(dev)
        if _state["ldw_early"]:
            from .ops import norm_fp4_quant_early

            norm_fp4_quant_early.preload(moe_ldw.HIDDEN)
    if _state["ldw"] and _state["ldw_wide"] and dix not in _state["ldw_scratch_w"]:
        from .ops import moe_ldw_l as moe_ldw

        mw = _state[
            "ldw_wide"
        ]  # the wide tail: 513..mw-row calls (128-row passes, no head copies)
        _ldw_preload(_LDW_NO_HEAD + moe_ldw.wide_flags(mw))
        _state["ldw_scratch_w"][dix] = moe_ldw.Scratch(dev, moe_ldw.PASS_ROWS, mw)
    if _state["km2"]:
        from .ops import lb_km2

        if not _state["km2_sites"]:
            lb_km2.preload()  # board consumers are loaded before any forward can run them next to a producer
        _state["km2_sites"][layer.gate_up_proj.data_ptr()] = lb_km2.Site(dev)
    _state["prepared"] += 1


def _call_site(kw, lo: int = 1, hi: int = 0, sp_ok: bool = False):
    """The layer's workspace when this FlashInfer call can run on the kernel (lo..hi rows; hi 0 = MoE-LD's maximum), else
    None. ``sp_ok``: the call may also run while sequence parallelism is active (the wide tail with MUSE_VR_MOE_LDW_SP)."""
    import torch

    from .moe_ld_ops import HIDDEN, INTER, MAX_TOKENS, TOP_K

    hi = hi or MAX_TOKENS

    if (
        kw.get("do_finalize", True) is not False
        or kw.get("activation_type") != 3
        or kw.get("routing_method_type") != 1
    ):
        return None
    if (
        kw.get("top_k") != TOP_K
        or kw.get("intermediate_size") != INTER
        or kw.get("n_group")
        or kw.get("topk_group")
    ):
        return None
    rsf = kw.get("routed_scaling_factor")
    if rsf is not None and float(rsf) != 1.0:
        return None
    if any(kw.get(k) is not None for k in _NONE_ARGS):
        return None
    # FlashInfer 0.7.4: an explicit recipe (None too: the environment is then ignored)
    if "nvfp4_4over6" in kw:
        return None
    packed, hs, pts = (
        kw.get("topk_ids"),
        kw.get("hidden_states"),
        kw.get("per_token_scale"),
    )
    if (
        not isinstance(packed, torch.Tensor)
        or packed.dtype != torch.int32
        or packed.dim() != 2
    ):
        return None
    m = packed.shape[0]
    if not (lo <= m <= hi) or packed.shape[1] != TOP_K or not packed.is_contiguous():
        return None
    if (
        pts is None
        or hs is None
        or hs.shape[0] != m
        or hs.shape[-1] * hs.element_size() != HIDDEN // 2
    ):
        return None
    w13 = kw.get("gemm1_weights")
    p = int(kw.get("local_num_experts") or 0)
    if w13 is None or not (1 <= p <= w13.shape[0]):
        return None
    site = _state["sites"].get(w13.data_ptr())
    if site is None or _tuning() or _overlap_enabled() or (_sp_active() and not sp_ok):
        return None
    return site


def _rows_equal(a, b) -> bool:
    """Every routed (token, k) row and the top-k weights of two (gemm2_out, expert_weights, map) triples agree bitwise."""
    import torch

    g_a, w_a, m_a = a
    g_b, w_b, m_b = b
    ia, ib = m_a.reshape(-1).long(), m_b.reshape(-1).long()
    if not torch.equal(ia >= 0, ib >= 0):
        return False
    va = g_a.index_select(0, ia.clamp(min=0)).view(torch.int16)
    vb = g_b.index_select(0, ib.clamp(min=0)).view(torch.int16)
    keep = (ia >= 0)[:, None]
    return bool(
        torch.equal(torch.where(keep, va, 0), torch.where(keep, vb, 0))
        and torch.equal(w_a.view(torch.int16), w_b.view(torch.int16))
    )


def _km2_call(kw, ctx):
    """The KM2 consumer for an engaged call with a board context, or None."""
    from .ops import lb_km2

    w13 = kw["gemm1_weights"]
    ksite = _state["km2_sites"].get(w13.data_ptr())
    if ksite is None:
        return None
    p = int(kw["local_num_experts"])
    flags = lb_km2.DEFAULT_FLAGS | lb_km2.FLAG_IDS_BOARD
    if ctx.release_form:
        flags |= lb_km2.FLAG_IDS_RELEASE
    if ctx.tok_hdr is not None and ctx.tok_writers > 0:
        flags |= lb_km2.FLAG_TOK_BOARD
    out = lb_km2.km2(
        kw["topk_ids"],
        kw["hidden_states"],
        kw["hidden_states_scale"],
        kw["per_token_scale"],
        w13[:p],
        kw["gemm1_weights_scale"][:p],
        kw["gemm2_weights"][:p],
        kw["gemm2_weights_scale"][:p],
        kw["output1_scale_gate_scalar"][:p],
        kw["output1_scale_scalar"][:p],
        kw["output2_scale_scalar"][:p],
        int(kw["local_expert_offset"]),
        ksite,
        seq=ctx.seq,
        ids_board=ctx.ids_board,
        tok_hdr=ctx.tok_hdr if flags & lb_km2.FLAG_TOK_BOARD else None,
        tok_writers=ctx.tok_writers if flags & lb_km2.FLAG_TOK_BOARD else 0,
        prod_err=ctx.prod_err,
        flags=flags,
        pdl=True,
    )
    bed.count("km2", f"m{kw['topk_ids'].shape[0]}_p{p}")
    return out


def _km2_mark_seen(kw, ctx) -> None:
    """A published call that stepped aside: the consumer site still records this forward's sequence value."""
    from .ops import lb_km2

    w13 = kw.get("gemm1_weights")
    ksite = _state["km2_sites"].get(w13.data_ptr()) if w13 is not None else None
    if ksite is not None:
        ksite.block[lb_km2.SITE_LAST_SEQ : lb_km2.SITE_LAST_SEQ + 1].copy_(
            ctx.seq.view(-1)[:1]
        )
        bed.count("km2", "stepped_aside_marked")


def _routed_moe(*args, **kw):
    fed = getattr(_tls, "early_fed", None)
    if fed is not None:
        _tls.early_fed = None
        out = (
            _ldw_call(kw)
            if (
                not args
                and not _state["off"]
                and _state["ldw"]
                and getattr(_tls, "layer", None) is fed
            )
            else None
        )
        if out is None:
            raise RuntimeError(
                "muse_vr_plugin: moe_ldw_early: an early-trigger quant fed a routed-MoE call that MoE-LD-W "
                "did not take (fail-stop: another consumer could read its outputs before they complete)"
            )
        return out
    layer = getattr(_tls, "layer", None)
    ctx = getattr(_tls, "board", None) if _state["km2"] else None
    if layer is not None and not args and not _state["off"]:
        # MoE-LD for 1..MUSE_VR_MOE_LD_MAX_M rows (default 16; 0 = MoE-LD-W only)
        site = _call_site(kw, 1, _state["ld_max_m"]) if _state["ld_max_m"] > 0 else None
        if site is not None and ctx is not None:
            out = _km2_call(kw, ctx)
            if out is not None:
                return out
        if site is not None:
            import torch

            from .moe_ld_ops import moe_ld

            if ctx is not None:
                _km2_mark_seen(kw, ctx)
            p = int(kw["local_num_experts"])
            out = moe_ld(
                kw["topk_ids"],
                kw["hidden_states"],
                kw["hidden_states_scale"],
                kw["per_token_scale"],
                kw["gemm1_weights"][:p],
                kw["gemm1_weights_scale"][:p],
                kw["gemm2_weights"][:p],
                kw["gemm2_weights_scale"][:p],
                kw["output1_scale_gate_scalar"][:p],
                kw["output1_scale_scalar"][:p],
                kw["output2_scale_scalar"][:p],
                int(kw["local_expert_offset"]),
                site,
                pdl=True,
            )
            bed.count(NAME, f"m{kw['topk_ids'].shape[0]}_p{p}")
            if (
                bed.flag("moe_ld_shadow")
                and not torch.cuda.is_current_stream_capturing()
            ):
                ref = _state["orig_moe"](**kw)[:3]
                ok = _rows_equal(out, ref)
                bed.shadow_result(NAME, ok)
                if not ok:
                    _state["off"] = "shadow mismatch"
                    logger.error(
                        "muse_vr_plugin: moe_ld shadow mismatch; the component is off for this process"
                    )
                    return ref
            return out
    if ctx is not None:
        _km2_mark_seen(kw, ctx)
    if layer is not None and not args and not _state["off"] and _state["ldw"]:
        out = _ldw_call(kw)
        if out is not None:
            return out
    return _state["orig_moe"](*args, **kw)


_LDW_NO_HEAD = (
    "-DMUSE_VR_LDW_GATHER=0",
)  # MoE-LD-W without the first-item head copies (larger calls)


LDW_ERRWORD_FLAG = "moe_ldw_errword"  # env MUSE_VR_MOE_LDW_ERRWORD


def ldw_errword_reason() -> str:
    """ "ok" when MoE-LD-W's error-word form (MUSE_VR_MOE_LDW_ERRWORD) may serve, else why not (decided once): MoE-LD-W
    is on and the verify commit's gate can engage."""
    if "ldw_ew_reason" not in _state:
        why = "ok"
        if not bed.flag(LDW_ERRWORD_FLAG):
            why = "off"
        elif not _state["ldw"]:
            why = "moe_ldw off"
        else:
            from . import eager_commit

            gate = eager_commit.gate_reason()
            if gate != "ok":
                why = f"commit gate: {gate}"
        _state["ldw_ew_reason"] = why
        if why != "off":
            bed.report("moe_ldw:errword", why == "ok", reason=why)
    return _state["ldw_ew_reason"]


def _ldw_ew(site):
    """(this rank's error-word address, tag) for an MoE-LD-W call at ``site``, or None: the trap form runs (the knob
    refuses; the gate is not armed; no component has mapped the group's error page yet; counted)."""
    if ldw_errword_reason() != "ok":
        return None
    from . import eager_commit
    from .ar_norm_residual import consumer_error_word
    from .ops.moe_ldw_l import LDW_ROLE

    if not eager_commit.gate_armed():  # the word would reach no gate: keep the trap
        bed.count("moe_ldw", "trap:gate not armed")
        return None
    err = consumer_error_word()
    if err is None:
        bed.count("moe_ldw", "trap:error page")
        return None
    return int(err), (LDW_ROLE << 16) | (
        int(getattr(site, "ew_index", 0xFFFF)) & 0xFFFF
    )


def _ldw_preload(extra: tuple = ()) -> None:
    """Load-time build of an MoE-LD-W form, and of its error-word form when that may serve."""
    from .ops import moe_ldw_l as moe_ldw

    moe_ldw.preload(extra)
    if ldw_errword_reason() == "ok":
        moe_ldw.preload(tuple(extra) + moe_ldw.ERRWORD)


def _ldw_form(m: int) -> str:
    """The MoE-LD-W build of an m-row call: "r48" (RING48 knob range), "r4d" (RING4's dense-token form, the
    MUSE_VR_MOE_LDW_DENSE_MAX_M range), "r4" (RING4 knob range) or "base"."""
    if (
        _state["ldw_ring48"]
        and _state["ldw_ring48_min_m"] <= m <= _state["ldw_ring48_max_m"]
    ):
        return "r48"
    if _state["ldw_ring4"] and m <= _state["ldw_ring4_max_m"]:
        return "r4d" if m <= _state["ldw_dense_max_m"] else "r4"
    return "base"


def _dense_build():
    """The dense-token build's flags: plain, its early prologue (MUSE_VR_MOE_LDW_PRE=1) or that form with speculate and
    verify (=2)."""
    from .ops import moe_ldw_l as moe_ldw

    return {1: moe_ldw.RING4_DENSE_PRE, 2: moe_ldw.RING4_DENSE_PREV}.get(
        _state["ldw_pre"], moe_ldw.RING4_DENSE
    )


def _ldw_needs(form: str) -> bool:
    """Whether some call size MoE-LD-W takes runs on this build (load-time preloads only what serving can use)."""
    from .ops import moe_ldw_l as moe_ldw

    return any(
        _ldw_form(m) == form
        for m in range(max(1, _state["ldw_min"]), moe_ldw.MAX_TOKENS + 1)
    )


def _unpacked_wide_sp(kw):
    """For a sequence-parallel call in FlashInfer's unpacked routing form (``topk_ids = (ids, weights)``: what SGLang's
    FP4 runner passes under deferred finalize, the SP expert stage) of 513..ceiling rows with MUSE_VR_MOE_LDW_SP: the
    expert ids packed for MoE-LD-W (``ids << 16``; its rows never read the weight half) and the caller's weights, which
    FlashInfer's unpacked mode returns as ``expert_weights`` (the caller's buffer). Else None."""
    import torch

    from .ops import moe_ldw_l as moe_ldw

    r = kw.get("topk_ids")
    if not (
        isinstance(r, tuple)
        and len(r) == 2
        and _state["ldw_sp"]
        and _state["ldw_wide"]
        and _sp_active()
    ):
        return None
    ids, w = r
    if (
        not (isinstance(ids, torch.Tensor) and isinstance(w, torch.Tensor))
        or ids.dtype != torch.int32
        or ids.dim() != 2
    ):
        return None
    if w.shape != ids.shape or w.dtype not in (torch.bfloat16, torch.float32):
        return None
    if (
        not moe_ldw.MAX_TOKENS < ids.shape[0] <= _state["ldw_wide"]
    ):  # checked before the pack: no extra launch otherwise
        return None
    return ids.contiguous() << 16, w


def _ldw_call(kw):
    """MoE-LD-W for an engaged-layer call of ``ldw_min``..512 rows (..1024 with the wide tail; the same argument set as
    MoE-LD's), else None."""
    import torch

    from .ops import moe_ldw_l as moe_ldw

    kw0 = kw
    unpacked = _unpacked_wide_sp(kw)
    if unpacked is not None:
        kw = dict(kw, topk_ids=unpacked[0])
    hi = _state["ldw_wide"] or moe_ldw.MAX_TOKENS
    m = int(kw["topk_ids"].shape[0]) if hasattr(kw.get("topk_ids"), "shape") else 0
    # sequence-parallel calls: only the wide tail (> 512 rows) with MUSE_VR_MOE_LDW_SP; every smaller call steps aside as before
    sp_ok = bool(_state["ldw_sp"] and _state["ldw_wide"] and m > moe_ldw.MAX_TOKENS)
    site = _call_site(kw, _state["ldw_min"], hi, sp_ok=sp_ok)
    if site is None:
        return None
    w13 = kw["gemm1_weights"]
    dix = (
        w13.device.index
        if w13.device.index is not None
        else torch.cuda.current_device()
    )
    form = "wide" if m > moe_ldw.MAX_TOKENS else _ldw_form(m)
    ring4 = form == "r4"
    scratch = {
        "r48": _state["ldw_scratch48"],
        "r4": _state["ldw_scratch4"],
        "r4d": _state["ldw_scratch4d"],
        "wide": _state["ldw_scratch_w"],
    }.get(form, _state["ldw_scratch"]).get(dix)
    if scratch is None:
        return None
    p = int(kw["local_num_experts"])
    if form == "r48":
        extra = moe_ldw.RING48
    elif form == "r4d":
        extra = _dense_build()
    elif ring4:
        extra = moe_ldw.RING4
    elif form == "wide":
        extra = _LDW_NO_HEAD + moe_ldw.wide_flags(_state["ldw_wide"])
    else:
        extra = () if m <= _state["ldw_head_max_m"] else _LDW_NO_HEAD
    ew = _ldw_ew(
        site
    )  # the error-word form (MUSE_VR_MOE_LDW_ERRWORD), or None: the trap form
    if ew is not None:
        bed.count("moe_ldw", f"ew:{form}")
    ring = None
    if (
        _state["ringbal"] and form != "wide"
    ):  # the ring placement build takes at most 512 rows; the wide tail keeps the static one
        from . import ringbal

        ring = ringbal.full(w13, p, int(kw["local_expert_offset"]))
        # the batch's real tokens (graph padding rows never move the placement); a capture without the count keeps the static one
        n_real = ringbal.real_rows(w13.device) if ring is not None else None
        if (
            ring is not None
            and n_real is None
            and torch.cuda.is_current_stream_capturing()
        ):
            ring = None
    if ring is not None:
        # the ring placement build on the layer's 80 slots (component ringbal): each routed row is bitwise the served row,
        # computed on one of its expert's holders
        st = ringbal.check_stamps(
            w13.device
        )  # engagement pass only (MUSE_VR_RINGBAL_CHECK=1, eager calls)
        out = moe_ldw.moe_ldw(
            kw["topk_ids"],
            kw["hidden_states"],
            kw["hidden_states_scale"],
            kw["per_token_scale"],
            ring["gate_up_proj"],
            ring["gate_up_proj_weight_scale"].view(kw["gemm1_weights_scale"].dtype),
            ring["down_proj"],
            ring["down_proj_weight_scale"].view(kw["gemm2_weights_scale"].dtype),
            ring["g1_alphas"],
            ring["g1_scale_c"],
            ring["g2_alphas"],
            int(kw["local_expert_offset"]),
            scratch,
            pdl=True,
            extra=extra + ringbal.FLAGS,
            stamps=st,
            n_real=n_real,
            ew=ew,
        )
        if st is not None:
            ringbal.check_digest(kw["topk_ids"], extra, st, n_real)
        bed.count(
            "moe_ldw",
            (f"m{m}_p{p}_{form}" if form != "base" else f"m{m}_p{p}") + "_ring",
        )
        if form == "r4d" and _state["ldw_pre"]:
            bed.count("moe_ldw", f"pre:m{m}")
        return out  # no shadow comparison: the rows a rank holds differ from the stock call's by design
    out = moe_ldw.moe_ldw(
        kw["topk_ids"],
        kw["hidden_states"],
        kw["hidden_states_scale"],
        kw["per_token_scale"],
        w13[:p],
        kw["gemm1_weights_scale"][:p],
        kw["gemm2_weights"][:p],
        kw["gemm2_weights_scale"][:p],
        kw["output1_scale_gate_scalar"][:p],
        kw["output1_scale_scalar"][:p],
        kw["output2_scale_scalar"][:p],
        int(kw["local_expert_offset"]),
        scratch,
        pdl=True,
        extra=extra,
        ew=ew,
    )
    if unpacked is not None:
        out = (out[0], unpacked[1], out[2])
    tag = f"m{m}_p{p}_{form}" if form != "base" else f"m{m}_p{p}"
    bed.count("moe_ldw", tag + "_sp" if (form == "wide" and _sp_active()) else tag)
    if form == "r4d" and _state["ldw_pre"]:
        bed.count("moe_ldw", f"pre:m{m}")
    if bed.flag("moe_ld_shadow") and not torch.cuda.is_current_stream_capturing():
        ref = _state["orig_moe"](**kw0)[:3]
        ok = _rows_equal(out, ref)
        bed.shadow_result("moe_ldw", ok)
        if not ok:
            _state["off"] = "shadow mismatch (moe_ldw)"
            logger.error(
                "muse_vr_plugin: moe_ldw shadow mismatch; the component is off for this process"
            )
            return ref
    return out


def _early_eligible(experts, x) -> bool:
    """MUSE_VR_MOE_LDW_EARLY: this prepare_input call feeds a call MoE-LD-W will take (rank-uniform: the layer's prepared
    site, the row count, the input form, the configuration)."""
    import torch

    if (
        _state["off"]
        or not _state["ldw"]
        or _tuning()
        or _sp_active()
        or _overlap_enabled()
    ):
        return False
    w13 = getattr(experts, "gate_up_proj", None)
    if (
        w13 is None
        or _state["sites"].get(w13.data_ptr()) is None
        or not (
            _state["ldw_scratch"] or _state["ldw_scratch4"] or _state["ldw_scratch4d"]
        )
    ):
        return False
    m = int(x.shape[0]) if x.dim() == 2 else 0
    lo = max(_state["ldw_min"], _state["ld_max_m"] + 1)
    return lo <= m <= 512 and x.dtype == torch.bfloat16 and x.is_contiguous()


def _early_around(original, experts, hidden_states, *a, **kw):
    """MUSE_VR_MOE_LDW_EARLY's AROUND hook on prepare_input: arms the quant dispatcher for an eligible call only."""
    _state["early_armed"] = experts if _early_eligible(experts, hidden_states) else None
    try:
        return original(experts, hidden_states, *a, **kw)
    finally:
        _state["early_armed"] = None


def _early_register(HookRegistry=None, HookType=None) -> bool:
    """MUSE_VR_MOE_LDW_EARLY: the quant dispatcher of SGLang's FP4 experts module defers to the early-trigger copy while
    an eligible prepare_input call runs (AROUND hook arms it); every other call keeps SGLang's dispatcher. (The registry
    arguments are for tests; SGLang's by default.)"""
    why = next(
        (
            f"pin mismatch: {rel}"
            for rel, sha in EARLY_PINS.items()
            if not bed.pinned(rel, sha)
        ),
        "",
    )
    if why:
        bed.report(EARLY_NAME, False, reason=why)
        return False
    import sglang.srt.layers.muse_spark_fp4_moe as fm

    from .ops import norm_fp4_quant_early as nfq

    if HookRegistry is None:
        from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    orig_quant = fm.fused_norm_fp4_quant

    def quant(x, weight, eps, global_scale_inv, *a, **kw):
        experts = _state["early_armed"]
        if experts is not None and not a and not kw:
            _state["early_armed"] = None  # one quant per prepare_input call
            _tls.early_fed = experts  # its routed-MoE call must be MoE-LD-W's
            bed.count("moe_ldw", f"early_quant:m{int(x.shape[0])}")
            return nfq.fused_norm_fp4_quant_early(x, weight, eps, global_scale_inv)
        return orig_quant(x, weight, eps, global_scale_inv, *a, **kw)

    fm.fused_norm_fp4_quant = quant
    HookRegistry.register(T_PREPARE, _early_around, HookType.AROUND)
    bed.report(EARLY_NAME, True, ldw_min=_state["ldw_min"], ld_max_m=_state["ld_max_m"])
    return True


def _layer_marking(orig):
    """``orig`` wrapped so that the routed-MoE dispatcher sees ``self`` as the marked layer for the call (thread-local; the
    previous mark is restored on every exit, exceptions included), as the ``forward`` wrapper of ``register`` does."""

    def wrapped(self, *a, **kw):
        prev = getattr(_tls, "layer", None)
        _tls.layer = self
        try:
            return orig(self, *a, **kw)
        finally:
            _tls.layer = prev

    wrapped.__wrapped__ = orig
    return wrapped


def _install_dispatched_hook(cls) -> bool:
    """MUSE_VR_MOE_LDW_SP: mark the layer in ``cls.forward_dispatched`` (the sequence-parallel expert stage); False when the
    class has no such entry point."""
    orig = getattr(cls, "forward_dispatched", None)
    if orig is None:
        return False
    cls.forward_dispatched = _layer_marking(orig)
    return True


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
    from . import fi_guard

    fi_guard.check(
        NAME
    )  # FlashInfer version / batched-GEMM artifact class; raises under MUSE_VR_REQUIRE_ENGAGED=1
    fi_guard.check("moe_ld_ops")
    import flashinfer.fused_moe as ffm
    import sglang.srt.layers.muse_spark_fp4_moe as fp4
    import sglang.srt.layers.quantization.muse_spark_nvfp4_moe as q

    orig_pwal = q.MuseSparkNvFp4FusedMoEMethod.process_weights_after_loading
    orig_forward = fp4.MuseSpark1xFp4Experts.forward

    def process_weights_after_loading(self, layer):
        out = orig_pwal(self, layer)
        try:
            _prepare_layer(layer)
        except Exception as e:  # noqa: BLE001 - an unprepared layer simply keeps FlashInfer
            _state["skipped"] = f"prepare failed: {type(e).__name__}: {e}"
        bed.report(
            NAME,
            _state["prepared"] > 0,
            layers=_state["prepared"],
            shadow=bed.flag("moe_ld_shadow"),
            skipped=_state["skipped"] or "-",
        )
        return out

    def experts_forward(self, *a, **kw):
        prev = getattr(_tls, "layer", None)
        _tls.layer = self
        try:
            return orig_forward(self, *a, **kw)
        finally:
            _tls.layer = prev

    _state["km2"] = bed.enabled("km2")
    from .moe_ld_ops import MAX_TOKENS as _LD_MAX

    _state["ld_max_m"] = max(
        0, min(_LD_MAX, bed.env_int("MUSE_VR_MOE_LD_MAX_M", _LD_MAX))
    )
    _state["ldw"] = bed.enabled("moe_ldw")
    _state["ldw_min"] = max(17, bed.env_int("MUSE_VR_MOE_LDW_MIN", 64))
    _state["ldw_head_max_m"] = bed.env_int("MUSE_VR_MOE_LDW_HEAD_MAX_M", 64)
    if _state["ldw"] and bed.env_int("MUSE_VR_MOE_LDW_EARLY", 0) == 1:
        _state["ldw_early"] = _early_register()
    _state["ldw_ring4"] = bed.env_int("MUSE_VR_MOE_LDW_RING4", 0) == 1
    _state["ldw_ring4_max_m"] = bed.env_int("MUSE_VR_MOE_LDW_RING4_MAX_M", 256)
    from .ops.moe_ldw_l import DENSE_MAX_ROWS as _DENSE_MAX

    dense = bed.env_int(
        "MUSE_VR_MOE_LDW_DENSE_MAX_M", 0
    )  # RING4's dense-token form for calls of at most this many rows
    if dense > _DENSE_MAX:
        logger.error(
            "muse_vr_plugin: MUSE_VR_MOE_LDW_DENSE_MAX_M=%s is above %s; capped",
            dense,
            _DENSE_MAX,
        )
    _state["ldw_dense_max_m"] = (
        max(0, min(_DENSE_MAX, dense)) if _state["ldw_ring4"] else 0
    )
    if dense and not _state["ldw_ring4"]:
        logger.error(
            "muse_vr_plugin: MUSE_VR_MOE_LDW_DENSE_MAX_M needs MUSE_VR_MOE_LDW_RING4=1; it stays off"
        )
    if _state["ldw_dense_max_m"]:
        bed.report("moe_ldw_dense", True, max_m=_state["ldw_dense_max_m"])
    pre = bed.env_int(
        "MUSE_VR_MOE_LDW_PRE", 0
    )  # 1: the dense-token build's early prologue; 2: its speculate-and-verify form
    _state["ldw_pre"] = pre if pre in (1, 2) and _state["ldw_dense_max_m"] > 0 else 0
    if pre in (1, 2) and not _state["ldw_pre"]:
        logger.error(
            "muse_vr_plugin: MUSE_VR_MOE_LDW_PRE=%d needs MUSE_VR_MOE_LDW_DENSE_MAX_M; it stays off",
            pre,
        )
    if _state["ldw_pre"]:
        bed.report(
            "moe_ldw_pre",
            True,
            max_m=_state["ldw_dense_max_m"],
            verify=_state["ldw_pre"] == 2,
        )
    _state["ldw_ring48"] = bed.env_int("MUSE_VR_MOE_LDW_RING48", 0) == 1
    _state["ldw_ring48_min_m"] = bed.env_int("MUSE_VR_MOE_LDW_RING48_MIN_M", 96)
    _state["ldw_ring48_max_m"] = bed.env_int("MUSE_VR_MOE_LDW_RING48_MAX_M", 512)
    _state["ringbal"] = _state["ldw"] and bed.enabled(
        "ringbal"
    )  # component ringbal serves the 80-slot tensors
    wide = bed.env_int(
        "MUSE_VR_MOE_LDW_WIDE", 0
    )  # 1 = 1024; 1024 / 2048 = the tail's ceiling; anything else off (logged)
    wide = 1024 if wide == 1 else wide
    if wide not in (0, 1024, 2048):
        logger.error(
            "muse_vr_plugin: MUSE_VR_MOE_LDW_WIDE=%s is not 0, 1, 1024 or 2048; the wide tail stays off",
            wide,
        )
        wide = 0
    _state["ldw_wide"] = wide if _state["ldw"] else 0
    if _state["ldw_wide"]:
        bed.report("moe_ldw_wide", True, max_m=_state["ldw_wide"], ring_max_m=512)
    _state["ldw_sp"] = (
        bool(_state["ldw_wide"]) and bed.env_int("MUSE_VR_MOE_LDW_SP", 0) == 1
    )
    if bed.env_int("MUSE_VR_MOE_LDW_SP", 0) == 1 and not _state["ldw_wide"]:
        logger.error(
            "muse_vr_plugin: MUSE_VR_MOE_LDW_SP=1 needs MUSE_VR_MOE_LDW_WIDE; it stays off"
        )
    dispatched = False
    if _state["ldw_sp"]:
        # the sequence-parallel expert stage enters through forward_dispatched (not forward): mark the layer there too
        dispatched = _install_dispatched_hook(fp4.MuseSpark1xFp4Experts)
        bed.report(
            "moe_ldw_sp",
            dispatched,
            min_m=513,
            max_m=_state["ldw_wide"],
            dispatched_hook=dispatched,
        )
        if not dispatched:
            _state["ldw_sp"] = (
                False  # no SP entry point to mark: the SP tail stays off (recorded)
            )
    _state["orig_moe"] = ffm.trtllm_fp4_block_scale_routed_moe
    ffm.trtllm_fp4_block_scale_routed_moe = _routed_moe
    q.MuseSparkNvFp4FusedMoEMethod.process_weights_after_loading = (
        process_weights_after_loading
    )
    fp4.MuseSpark1xFp4Experts.forward = experts_forward
    _state["engaged"] = True
    logger.warning("muse_vr_plugin: moe_ld hooks installed")
    return True
