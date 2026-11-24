# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Component ``proj_publish`` (env ``MUSE_VR_PROJ_PUBLISH=1``, default off): the MoE pooler / unpooler as published slices.

At tensor parallelism, SGLang's Muse Spark MoE block computes the pooler (``pre_expert_proj``) and the unpooler
(``post_expert_proj``) with the full replicated weight on every rank whenever its sliced paths are not admitted, which on
this device is every decode / verify call above the small-projection row cap. With this component each rank multiplies
its contiguous column block only, and the GEMM epilogue publishes the block into every rank's copy of the push-plane
gather slot (``ops/proj_publish.py``):

- site ``P`` (pooler; ``MUSE_VR_PROJ_PUBLISH_SITES`` default ``P``), rows ``MUSE_VR_PROJ_PUBLISH_ROWS`` (default ``56-336``):
  on the tensor-parallel decode / verify path of NVFP4 expert layers (per-token prepare path, no sequence parallelism),
  the pooler's ``bf16_gemm_dispatch`` call becomes ``publish_gemm`` and the experts' ``prepare_input`` of that output becomes
  ``pub_norm_fp4_quant`` (SGLang's RMS norm + per-token NVFP4 quantization of the gathered row);
- site ``U`` (unpooler; ``MUSE_VR_PROJ_PUBLISH_SITES=P,U``), rows ``MUSE_VR_PROJ_PUBLISH_U_ROWS`` (default ``192-288``):
  SGLang's deferred path is taken (``gemm_ag_supported`` admits this device for the unpooler only); its ``gemm_local`` call
  becomes ``publish_gemm`` and the FFN boundary's ``all_gather_norm_residual`` call becomes the consumer-only
  ``pub_ag_norm`` (SGLang's boundary arithmetic without its push of a local slice). Pooler rows in
  ``MUSE_VR_PROJ_PUBLISH_SK_ROWS`` (default none) publish through the split-K slice GEMM (``ops.proj_publish.sk_config``),
  rows in ``MUSE_VR_PROJ_PUBLISH_SK2_ROWS`` (default none; takes precedence) through its 2-CTA tile-pair form
  (``ops.proj_publish.sk2_config``); other rows keep the single-pass slice GEMM. Setting either also builds every 2-CTA
  publish GEMM of the process with the MMA warp's wait for the TMEM-alloc barrier's first phase (``opp_tgv_gemm``,
  ``_TMEM_ALLOC_WAIT``; the single-pass 2-CTA publishes included). Unpooler rows in
  ``MUSE_VR_PROJ_PUBLISH_UAG_ROWS`` (default none) take SGLang's own gathered unpooler path unchanged (sliced GEMM + its
  all-gather boundary), the form that measured faster than the publish at some row counts.

Each publish is consumed by its own consumer as the next use of the push plane (``ops/proj_publish.py`` refuses an
unpaired call, and a pooler publish not consumed inside its MoE call raises). Every decision depends only on rank-uniform
inputs (row count, shapes, layer kind, communicator), so every rank takes the same path. Everything else runs SGLang's
code unchanged; the reason is counted.

Numerics: the unpooler's TGV column-slice GEMM is bitwise equal to the replicated dispatch at the tested row counts (units),
and the published values reach the boundary through SGLang's all-gather marker rule (an exact +0.0 half arrives as -0.0),
as SGLang's own gathered path does at 306ccf6 (from 3117f58 SGLang's own path remaps only an all-zero 32-bit word). The
pooler's slice GEMM differs from the replicated cuBLAS GEMM in summation order below 136 rows on this image (a
reduction-order class) and is bitwise above (units). The split-K slice GEMM differs from it in summation order at every
row count (a reduction-order class; fixed split order, rank- and call-invariant).

Plane: the Muse Spark collectives' lease on the tensor-parallel symmetric workspace (``muse_workspace()``) and its
EpochCounter phase word (SGLang 3117f58 + #48948); the 306ccf6 form (the TP custom all-reduce's push plane with per-block
phase counters) is not kept.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch

from . import bed

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "srt/models/muse_spark.py": (
        "MuseSpark1xDecoderLayer",
        "MuseSpark1xMoE",
        # the model module's names install() rebinds
        "gemm_ag_supported",
        "gemm_local",
        "all_gather_norm_residual",
        "bf16_gemm_dispatch",
    ),
    "srt/layers/muse_spark/muse_spark_fp4_moe.py": ("MuseSpark1xFp4Experts",),
    "srt/layers/communication/muse_spark_gemm_ag.py": ("muse_workspace",),
    "srt/runtime_context.py": ("get_forward",),
}


logger = logging.getLogger(__name__)

NAME = "proj_publish"
ROWS_ENV = "MUSE_VR_PROJ_PUBLISH_ROWS"  # pooler (site P) rows
U_ROWS_ENV = "MUSE_VR_PROJ_PUBLISH_U_ROWS"  # unpooler (site U) rows
UAG_ROWS_ENV = "MUSE_VR_PROJ_PUBLISH_UAG_ROWS"  # unpooler rows on SGLang's own gathered path (no publish); default none
SK_ROWS_ENV = "MUSE_VR_PROJ_PUBLISH_SK_ROWS"  # pooler rows on the split-K publish GEMM; default none
SK2_ROWS_ENV = "MUSE_VR_PROJ_PUBLISH_SK2_ROWS"  # pooler rows on the 2-CTA tile-pair split-K publish GEMM; default none
SITES_ENV = "MUSE_VR_PROJ_PUBLISH_SITES"
DEFAULT_ROWS = (56, 336)
DEFAULT_U_ROWS = (192, 288)
H, EH = 8192, 4096
# The SGLang sources this component rebinds names of, calls, or copies kernels and the plane protocol from, at 3117f58 +
# #48948 (the publish and consumer kernels follow its EpochCounter plane; 306ccf6's per-block counters are not kept)
PINS = {
    # the MoE call sites (pooler dispatch and prepare, unpooler gather with the lease) and the decoder layer's FFN boundary
    "sglang/srt/models/muse_spark.py": (
        bed.Functions(
            {
                "MuseSpark1xMoE.__init__": "38b36b568c633d3e",
                "MuseSpark1xMoE._forward_tp": "1c1aa0fe77681cd0",
                "MuseSpark1xMoE._forward_experts": "04c56c86a0d682b6",
                "MuseSpark1xMoE._reduce_and_unpool": "84d2840940f3c49b",
                "MuseSpark1xMoE._defers_unpool_gather": "c730eaaec130b50e",
                "MuseSpark1xDecoderLayer.forward": "51e40503e316c464",
            }
        ),
    ),
    # the lease (the comm SGLang passes) and the unpooler's stock slice GEMM
    "sglang/srt/layers/communication/muse_spark_gemm_ag.py": (
        bed.Functions(
            {"muse_workspace": "bf91cbde371bd1ae", "gemm_local": "5ddfc71a56e900f7"}
        ),
    ),
    # the slab layout (push slots at offset 0, the local u64 phase word) and the communicator the kernels take
    "sglang/srt/distributed/device_communicators/symm_workspace.py": (
        bed.Functions(
            {
                "_Workspace.finalize": "9a3ce97166fdb429",
                "WorkspaceLease.get_communicator": "f5b7ae089508b316",
            }
        ),
    ),
    # the FFN-boundary gather pub_ag_norm stands in for, and its deferral check
    "sglang/kernels/ops/communication/muse_spark_all_gather_norm_residual.py": (
        bed.Functions(
            {
                "all_gather_norm_residual": "c0df708450513d02",
                "all_gather_norm_residual_supported": "acaf9761aae5908d",
            }
        ),
    ),
    # the phase protocol and the two kernels csrc/proj_publish.cuh includes and copies
    "sglang/kernels/jit/include/sgl_kernel/distributed/communicator.cuh": "43315a3609675cc012be57e88622d19c2fa373126d03de018f87ef371e82f27b",
    "sglang/kernels/jit/csrc/muse_spark/all_gather_norm_residual.cuh": "c0a24427f1f952ec1c5a82dacb2133f68e519666e5879c440bde8c0e4e2e1aad",
    "sglang/kernels/jit/csrc/muse_spark/norm_fp4_quant.cuh": "a1d08b5b20a5e1b0a5ab3c831a3f9f80297d8ba29bab267901da32c145053a94",
}

_state = {
    "engaged": None,
    "rows": DEFAULT_ROWS,
    "u_rows": DEFAULT_U_ROWS,
    "uag_rows": None,
    "sk_rows": None,
    "sk2_rows": None,
    "sites": ("P",),
    "markers": {},
    "allow_p": None,
}


def _range(env: str, default):
    v = os.environ.get(env, "").strip()
    if not v:
        return default
    lo, hi = (int(t) for t in v.split("-"))
    if not 1 <= lo <= hi <= 512:
        raise ValueError(f"{env}={v}: needs 1 <= lo <= hi <= 512")
    return lo, hi


def _sites():
    v = os.environ.get(SITES_ENV, "P").strip().upper()
    sites = tuple(s.strip() for s in v.split(",") if s.strip())
    if not sites or any(s not in ("U", "P") for s in sites):
        raise ValueError(f"{SITES_ENV}={v}: use P, U or P,U")
    return sites


def _in(rng, m: int) -> bool:
    return rng is not None and rng[0] <= m <= rng[1]


def _comm():
    # the plane SGLang's Muse Spark collectives use (and pass to gemm_local / all_gather_norm_residual): the lease on
    # the tensor-parallel symmetric workspace (3117f58 + #48948; 306ccf6 used the TP custom all-reduce's push plane)
    from sglang.srt.layers.communication.muse_spark_gemm_ag import muse_workspace

    try:
        return muse_workspace()
    except Exception:  # noqa: BLE001
        return None


def _marker(m: int, n: int, device, site: str) -> torch.Tensor:
    """A shape carrier for SGLang's code between a publish and its consumer (never read)."""
    t = torch.empty(m, n, dtype=torch.bfloat16, device=device)
    _state["markers"][t.data_ptr()] = site
    return t


def _take_marker(t: torch.Tensor, site: str) -> bool:
    return (
        _state["markers"].pop(t.data_ptr(), None) == site
        if isinstance(t, torch.Tensor)
        else False
    )


def install() -> None:
    import sglang.srt.models.muse_spark as model

    from .ops import proj_publish as pp

    sites = _state["sites"]
    orig_supported = model.gemm_ag_supported
    orig_local = model.gemm_local
    orig_agn = model.all_gather_norm_residual
    orig_dispatch = model.bf16_gemm_dispatch

    def publishable(comm, x, weight, shape, rows) -> Optional[str]:
        if (
            comm is None
            or x.dim() != 2
            or tuple(weight.shape) != shape
            or x.dtype != torch.bfloat16
        ):
            return "shape"
        if not _in(rows, x.shape[0]):
            return "rows"
        return pp.step_aside_reason(comm, x.shape[0], shape[0], shape[1])

    def gemm_ag_supported(comm, x, weight, *a, **k):
        if orig_supported(comm, x, weight, *a, **k):
            return True
        # the unpooler only (the pooler call site checks the device capability itself): published rows, and rows that
        # take SGLang's own gathered path unchanged
        if "U" not in sites:
            return False
        return (
            publishable(comm, x, weight, (H, EH), _state["u_rows"]) is None
            or publishable(comm, x, weight, (H, EH), _state["uag_rows"]) is None
        )

    def gemm_local(comm, x, weight):
        r = (
            publishable(comm, x, weight, (H, EH), _state["u_rows"])
            if "U" in sites
            else "site"
        )
        if r is not None:
            if (
                "U" in sites
                and publishable(comm, x, weight, (H, EH), _state["uag_rows"]) is None
            ):
                bed.count(NAME, f"U gathered M{x.shape[0]}")
            else:
                bed.count(NAME, f"U stock: {r}")
            return orig_local(comm, x, weight)
        pp.publish_gemm(comm, x.contiguous(), weight, "ag_norm")
        bed.count(NAME, f"U publish M{x.shape[0]}")
        return _marker(x.shape[0], H // comm.world_size, x.device, "U")

    def all_gather_norm_residual(comm, local, stream, *a, **k):
        if _take_marker(local, "U"):
            return pp.pub_ag_norm(comm, stream, *a, **k)
        return orig_agn(comm, local, stream, *a, **k)

    def bf16_gemm_dispatch(x, weight, bias, *a, **k):
        allow = _state["allow_p"]
        if allow is not None and bias is None and not a and not k and weight is allow:
            _state["allow_p"] = None  # one publish per admitted MoE call
            comm = _comm()
            r = publishable(comm, x, weight, (EH, H), _state["rows"])
            if r is None:
                if _in(_state["sk2_rows"], x.shape[0]):
                    cfg, tag = pp.sk2_config(x.shape[0]), "split-K pair"
                elif _in(_state["sk_rows"], x.shape[0]):
                    cfg, tag = pp.sk_config(x.shape[0]), "split-K"
                else:
                    cfg, tag = None, None
                if cfg is not None:
                    tactic, split_k, cluster, push = cfg
                    pp.publish_gemm(
                        comm,
                        x.contiguous(),
                        weight,
                        "norm_fp4_quant",
                        tactic=tactic,
                        split_k=split_k,
                        sk_cluster=cluster,
                        sk_push=push,
                    )
                    bed.count(NAME, f"P publish {tag} M{x.shape[0]}")
                else:
                    pp.publish_gemm(comm, x.contiguous(), weight, "norm_fp4_quant")
                bed.count(NAME, f"P publish M{x.shape[0]}")
                bed.count(
                    NAME, f"P layer {_state.get('layer', -1)}"
                )  # engagement record names the target layers
                return _marker(x.shape[0], EH, x.device, "P")
            bed.count(NAME, f"P stock: {r}")
        return orig_dispatch(x, weight, bias, *a, **k)

    model.gemm_ag_supported = gemm_ag_supported
    model.gemm_local = gemm_local
    model.all_gather_norm_residual = all_gather_norm_residual
    if "U" in sites:
        orig_layer_forward = model.MuseSpark1xDecoderLayer.forward

        def layer_forward(self, *a, **k):
            out = orig_layer_forward(self, *a, **k)
            # fail-stop inside the same forward: an unpooler publish must be consumed by this layer's FFN boundary
            if pp.pending(_comm()) is not None:
                raise RuntimeError(
                    f"proj_publish: a publish was not consumed inside decoder layer "
                    f"{getattr(self, 'layer_idx', '?')}: {pp.pending(_comm())}"
                )
            return out

        model.MuseSpark1xDecoderLayer.forward = layer_forward
    if "P" in sites:
        model.bf16_gemm_dispatch = bf16_gemm_dispatch
        import sglang.srt.layers.muse_spark.muse_spark_fp4_moe as fp4

        orig_prepare = fp4.MuseSpark1xFp4Experts.prepare_input
        orig_experts = model.MuseSpark1xMoE._forward_experts

        def _forward_experts(
            self, hidden_states, router_input=None, *, ar_workspace=None
        ):
            # Site P is admitted only on the tensor-parallel decode / verify path (called from _forward_tp, which passes
            # router_input), NVFP4 experts with the per-token prepare path, no sequence parallelism, no marlin path.
            from sglang.srt.runtime_context import get_forward

            ok = (
                router_input is not None
                and self.pre_expert_proj is not None
                and self.nvfp4_checkpoint
                and isinstance(self.experts, fp4.MuseSpark1xFp4Experts)
                and not getattr(self.experts, "use_marlin", False)
                and getattr(
                    self.experts.quant_config, "use_per_token_activation", False
                )
                and not getattr(get_forward(), "sp_active", False)
            )
            if not ok:
                return orig_experts(
                    self, hidden_states, router_input, ar_workspace=ar_workspace
                )
            _state["allow_p"] = self.pre_expert_proj.weight
            _state["layer"] = getattr(self, "layer_idx", -1)
            try:
                return orig_experts(
                    self, hidden_states, router_input, ar_workspace=ar_workspace
                )
            finally:
                _state["allow_p"] = None
                if pp.pending(_comm()) is not None:
                    raise RuntimeError(
                        "proj_publish: a pooler publish was not consumed inside its MoE call"
                    )

        model.MuseSpark1xMoE._forward_experts = _forward_experts

        def prepare_input(self, hidden_states, pre_expert_norm=None):
            if _take_marker(hidden_states, "P"):
                if (
                    pre_expert_norm is None
                    or not self.quant_config.use_per_token_activation
                ):
                    raise RuntimeError(
                        "proj_publish: published pooler output needs the per-token NVFP4 prepare path"
                    )
                comm = _comm()
                fp4_out, sf, pts = pp.pub_norm_fp4_quant(
                    comm,
                    hidden_states.shape[0],
                    hidden_states.shape[1],
                    self._pre_norm_weight(pre_expert_norm),
                    pre_expert_norm.eps,
                    hidden_states.device,
                )
                hs_scale = sf.view(torch.float8_e4m3fn).reshape(*sf.shape[:-1], -1)
                return fp4_out, hs_scale, pts
            return orig_prepare(self, hidden_states, pre_expert_norm)

        fp4.MuseSpark1xFp4Experts.prepare_input = prepare_input


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
    try:
        _state["rows"] = _range(ROWS_ENV, DEFAULT_ROWS)
        _state["u_rows"] = _range(U_ROWS_ENV, DEFAULT_U_ROWS)
        _state["uag_rows"] = _range(UAG_ROWS_ENV, None)
        _state["sk_rows"] = _range(SK_ROWS_ENV, None)
        _state["sk2_rows"] = _range(SK2_ROWS_ENV, None)
        from .ops import proj_publish as _pp

        if _state["sk_rows"] and _state["sk_rows"][1] > _pp.SK_TABLE[-1][0]:
            raise ValueError(
                f"{SK_ROWS_ENV}: split-K publish covers rows <= {_pp.SK_TABLE[-1][0]}"
            )
        if _state["sk2_rows"] and _state["sk2_rows"][1] > _pp.SK2_MAX_ROWS:
            raise ValueError(
                f"{SK2_ROWS_ENV}: 2-CTA split-K publish covers rows <= {_pp.SK2_MAX_ROWS}"
            )
        _state["sites"] = _sites()
        install()
    except Exception as e:  # noqa: BLE001
        bed.report(NAME, False, reason=f"install failed: {type(e).__name__}: {e}")
        _state["engaged"] = False
        return False
    uag, sk, sk2 = _state["uag_rows"], _state["sk_rows"], _state["sk2_rows"]
    bed.report(
        NAME,
        True,
        p_rows="%d-%d" % _state["rows"],
        u_rows="%d-%d" % _state["u_rows"],
        uag_rows=("%d-%d" % uag) if uag else "none",
        sk_rows=("%d-%d" % sk) if sk else "none",
        sk2_rows=("%d-%d" % sk2) if sk2 else "none",
        sites=",".join(_state["sites"]),
    )
    _state["engaged"] = True
    return True
