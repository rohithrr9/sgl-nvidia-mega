# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Projection publish: a replicated projection computed as column slices that every rank's GEMM epilogue publishes.

For a projection ``y = x @ W.T`` whose input ``x`` is identical on every tensor-parallel rank (the MoE pooler after the
attention boundary, the unpooler after the expert all-reduce), each rank multiplies only its contiguous ``N / world``
block of ``W`` rows. The GEMM (``opp_tgv_gemm`` push_mode 7, SGLang's TGV kernel) stores each output vector, through
the multicast address of the tensor-parallel push workspace, into this rank's block of a gather slot in every rank's
copy: the slot layout ``[source rank][M][N / world]`` and the phase halves are those of SGLang's own all-gather
(``gemm_ag`` / ``all_gather_norm_residual``); the empty-marker rule (+0.0 in either 16-bit half = not arrived; an exact
+0.0 half is sent as -0.0) is that all-gather's rule at 306ccf6 (from 3117f58 SGLang marks per 32-bit word, which a
payload remapped per half also satisfies). The consumers take every vector from the local copy, reset it, and run the arithmetic of
the stock kernel they replace (``csrc/proj_publish.cuh``):

- ``pub_ag_norm``: the FFN boundary of SGLang's ``all_gather_norm_residual`` without its own push of the local slice;
- ``pub_norm_fp4_quant``: SGLang's pre-expert RMS norm + per-token NVFP4 quantization (quantize-only form).

A producer and its consumer form one use of the push plane: the producer reads the phase after its dependency wait and
never advances it; the consumer, launched next on the same stream, reads the same phase and advances it at its end, as
every user of the plane does. Results equal the stock pair "replicated GEMM + stock kernel" up to (a) the GEMM's
summation order where the slice GEMM and the replicated GEMM differ, and (b) the sign of exact-zero projection outputs
(the empty-marker remap), which SGLang's all-gather path shares at 306ccf6 (from 3117f58 it remaps only an all-zero
32-bit word).

Split-K publish (``publish_gemm(..., split_k=S, sk_cluster=...)``, ``sk_config``): the slice GEMM's K tiles are divided
over S CTAs per output tile; the fp32 fragments meet in one workspace per (device, slice width), shared by every call
(calls on one stream run one after another), and are summed in split order before the publish (by the last CTA of the
tile, or with sk_cluster by every CTA of the tile's cluster for its share of the rows). The sum order depends on the row
count only (same on every rank and every call). ``sk2_config``: the same exchange on 2-CTA tile pairs (a pair covers 128
output features and one CTA loads half of the pair's rows), half the CTAs of the 1-CTA form at the same split count.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

from .. import sgl_compat

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "kernels/jit/utils/compile/paths.py": ("KERNEL_PATH",),
    "kernels/jit/utils/__init__.py": (
        "is_arch_support_pdl",
        "load_jit",
        "make_cpp_args",
    ),
}
# SGLang's symmetric-workspace lease (nightly 3117f58 and later); on an older SGLang the component's source pins
# fail first and it steps aside, so the module is never reached there
SGLANG_OPTIONAL_READS = {
    "srt/distributed/device_communicators/symm_workspace.py": ("WorkspaceLease",),
}


_CUH = Path(__file__).resolve().parents[1] / "csrc" / "proj_publish.cuh"
_WORLDS = (2, 4, 8)
NVFP4_PER_TOKEN_SCALE_INV = 1.0 / (448.0 * 6.0)


def _cache(fn):
    from functools import lru_cache

    return lru_cache(maxsize=None)(fn)


def _include_paths():
    from sglang.kernels.jit.utils.compile.paths import KERNEL_PATH

    return sgl_compat.csrc_include_paths(KERNEL_PATH / "csrc")


@_cache
def _ag_norm_module(
    hidden: int,
    world: int,
    config: Tuple[int, int, bool],
    scale: bool,
    has_post: bool,
    gate: bool,
):
    from sglang.kernels.jit.utils import is_arch_support_pdl, load_jit, make_cpp_args

    cluster, threads, prefetch = config
    args = make_cpp_args(
        hidden,
        world,
        threads,
        cluster,
        is_arch_support_pdl(),
        scale,
        prefetch,
        has_post,
        gate,
    )
    return load_jit(
        "muse_vr_pub_ag_norm",
        *args,
        cuda_files=[str(_CUH)],
        cuda_wrappers=[("run", f"MuseVrPubAgNorm<{args}>::run")],
        extra_include_paths=_include_paths(),
        extra_cuda_cflags=["-O3"],
    )


@_cache
def _quant_module(hidden: int, world: int, threads: int):
    from sglang.kernels.jit.utils import is_arch_support_pdl, load_jit, make_cpp_args

    args = make_cpp_args(hidden, world, threads, is_arch_support_pdl())
    return load_jit(
        "muse_vr_pub_norm_fp4_quant",
        *args,
        cuda_files=[str(_CUH)],
        cuda_wrappers=[("run", f"MuseVrPubNormFp4Quant<{args}>::run")],
        extra_include_paths=_include_paths(),
        extra_cuda_cflags=["-O3"],
    )


def step_aside_reason(comm, num_tokens: int, n_full: int, k: int) -> Optional[str]:
    """None when a publish of an [num_tokens, k] x [n_full, k]^T projection can run on ``comm`` (the Muse Spark
    collectives' lease on the tensor-parallel symmetric workspace, ``muse_workspace()``); else the reason."""
    from sglang.srt.distributed.device_communicators.symm_workspace import (
        WorkspaceLease,
    )

    if not isinstance(comm, WorkspaceLease):
        return "no symmetric workspace"
    if comm.world_size not in _WORLDS:
        return "world"
    if not comm.has_multicast:
        return "no multicast"
    if num_tokens <= 0:
        return "rows"
    if n_full % (comm.world_size * 128) or k % 128:
        return "shape"
    # the slot this lease asked for, as SGLang's own all-gathers bound it (the workspace's slot is at least this)
    if num_tokens * n_full * 2 > comm.request.push_slot_bytes:
        return "push slot size"
    return None


_meta: Dict[Tuple[int, int, int, int, int], Tuple[torch.Tensor, torch.Tensor]] = {}
_carriers: Dict[Tuple[int, int], torch.Tensor] = {}
_sk_bufs: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}
_MAX_ROWS = 512

# Split-K publish configs: (largest row count, opp_tgv_gemm tactic or None = single pass, split_k, cluster exchange, push
# exchange), first match wins. 1-CTA 64 x 64 tiles with 5 stages, 4 splits, cluster push exchange: 64 CTAs up to 64 rows,
# 128 CTAs for 96-128 rows; 65-95 rows keep the single pass (a second, mostly empty row tile).
SK_TABLE = (
    (64, 35, 4, True, True),
    (95, None, 1, False, False),
    (128, 35, 4, True, True),
)
# 2-CTA tile-pair split-K publish configs: (largest row count, opp_tgv_gemm tactic, split_k), first match wins. Tile pairs of
# 128 features x 64 / 128 / 192 / 256 rows, 4 splits, cluster (2, 1, 4) push exchange: 64 CTAs for a 4096-wide slice.
SK2_TABLE = (
    (64, 36, 4),
    (128, 37, 4),
    (192, 38, 4),
    (256, 39, 4),
)
SK2_MAX_ROWS = SK2_TABLE[-1][0]
SK_MAX_ROWS = max(SK_TABLE[-1][0], SK2_MAX_ROWS)  # rows any split-K publish form covers


def sk_config(m: int) -> Optional[Tuple[int, int, bool, bool]]:
    """(tactic, split_k, cluster exchange, push exchange) of the split-K publish GEMM at ``m`` rows, or None where the
    single pass is kept (m <= SK_TABLE[-1][0])."""
    for hi, tactic, s, cluster, push in SK_TABLE:
        if m <= hi:
            return None if tactic is None else (tactic, s, cluster, push)
    raise RuntimeError(f"proj_publish: no split-K config for {m} rows")


def sk2_config(m: int) -> Tuple[int, int, bool, bool]:
    """(tactic, split_k, cluster exchange, push exchange) of the 2-CTA tile-pair split-K publish GEMM at ``m`` rows."""
    for hi, tactic, s in SK2_TABLE:
        if m <= hi:
            return (tactic, s, True, True)
    raise RuntimeError(f"proj_publish: no 2-CTA split-K config for {m} rows")


def _sk_buffers(device: torch.device, n_slice: int):
    """The split-K workspace and tile counters of (device, slice width), sized once for every row count of SK_TABLE and
    SK2_TABLE (the push exchange forms never touch them; the run checks their size for every split-K call)."""
    from .. import opp_tgv_gemm as g

    key = (device.index or 0, n_slice)
    hit = _sk_bufs.get(key)
    if hit is None:
        ws_n = cnt_n = 0
        for m in range(1, SK_MAX_ROWS + 1):
            cfgs = [sk2_config(m)] if m <= SK2_MAX_ROWS else []
            if m <= SK_TABLE[-1][0]:
                cfgs.append(sk_config(m))
            for cfg in cfgs:
                if cfg is not None:
                    a, b = g.split_k_sizes(m, n_slice, cfg[0], cfg[1])
                    ws_n, cnt_n = max(ws_n, a), max(cnt_n, b)
        hit = (
            torch.empty(ws_n, dtype=torch.float32, device=device),
            torch.zeros(cnt_n, dtype=torch.int32, device=device),
        )
        _sk_bufs[key] = hit
    return hit


def _carrier(device: torch.device, n_slice: int, m: int) -> torch.Tensor:
    """A [m, n_slice] view of one shared shape-carrier buffer per (device, n_slice): the publish GEMM never writes its
    plain output, so every row count shares one allocation."""
    key = (device.index or 0, n_slice)
    buf = _carriers.get(key)
    if buf is None:
        buf = torch.empty(_MAX_ROWS, n_slice, dtype=torch.bfloat16, device=device)
        _carriers[key] = buf
    if m > _MAX_ROWS:
        raise RuntimeError(f"proj_publish: {m} rows > {_MAX_ROWS}")
    return buf[:m]


# Call-uniform pairing: a publish must be followed by its consumer as the very next use of the push plane, and a
# consumer must have a publish to take, or a half would be left holding words (or a consumer would wait forever).
# Every rank makes the same calls (the inputs of every decision are rank-uniform); this host check refuses an
# unpaired call on the rank that makes it.
_pending: Dict[int, Tuple[str, int, int]] = {}


def _begin(comm, site: str, m: int, n_full: int) -> None:
    key = id(comm)
    if key in _pending:
        raise RuntimeError(
            f"proj_publish: publish {site} while {_pending[key]} has no consumer yet"
        )
    _pending[key] = (site, m, n_full)


def _end(comm, site: str, m: int, n_full: int) -> None:
    got = _pending.get(id(comm))
    if got != (site, m, n_full):
        raise RuntimeError(
            f"proj_publish: {site} consumer for (M={m}, N={n_full}) without its publish (pending: {got})"
        )
    del _pending[id(comm)]


def pending(comm):
    """The unconsumed publish on ``comm`` (site, M, N), or None."""
    return _pending.get(id(comm))


def note_publish(comm, site: str, m: int, n_full: int) -> None:
    """For a producer other than ``publish_gemm``: record its publish of an [m, n_full] projection for the consumer ``site``
    ("ag_norm" or "norm_fp4_quant"), before its launch, so the pairing rule covers it. Such a producer must store exactly
    what ``publish_gemm`` stores: after its dependency wait, the phase = bit 0 of the high 32-bit half of the lease's
    EpochCounter word (``control.epoch``); this rank's [m, n_full / world] block at byte offset
    phase * world * capacity.push_slot_bytes + rank * m * (n_full / world) * 2 of the lease's multicast address
    (``data.push_mc``); 16-byte vectors with every all-zero 16-bit half sent as 0x8000; one multimem.st.relaxed.sys
    .global.v4.f32 per vector; every element of the block written once; no counter written."""
    _begin(comm, site, m, n_full)


def _publish_meta(
    comm, m: int, n_slice: int, device: torch.device, split_k: bool = False
):
    """(dummy output [m, n_slice], int64 argument block) of a push_mode 7 GEMM, cached per (device, m, n_slice, plane,
    split-K or not)."""
    from .. import opp_tgv_gemm as g

    push_mc = int(comm.data.push_mc)  # multicast VA of the push slots (has_multicast)
    key = (device.index or 0, m, n_slice, push_mc, int(split_k))
    hit = _meta.get(key)
    if hit is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "proj_publish: first use of a row count inside graph capture (warm-up missing)"
            )
        w = comm.world_size
        out = _carrier(device, n_slice, m)  # shape carrier only (no plain stores)
        ws, cnt = _sk_buffers(device, n_slice) if split_k else (None, None)
        meta = g.make_push_meta(
            # the plane's phase word is one u64 EpochCounter: the phase is bit 0 of its high (second) 32-bit half,
            # which the GEMM reads as a b32 word (& 1)
            counter_addr=comm.control.epoch.data_ptr() + 4,
            phase_stride=w * comm.capacity.push_slot_bytes,
            rank=comm.rank,
            rows=m,
            chunk_rows=0,
            out=out,
            peer_slot_bases=[push_mc + comm.rank * m * n_slice * 2],
            sk_workspace=ws,
            sk_counters=cnt,
        )
        hit = (out, meta)
        _meta[key] = hit
    return hit


def publish_gemm(
    comm,
    x: torch.Tensor,
    weight: torch.Tensor,
    site: str,
    tactic: Optional[int] = None,
    split_k: int = 1,
    sk_cluster: bool = False,
    sk_push: bool = False,
) -> None:
    """Publish this rank's column block of ``x @ weight.T`` (``weight`` = the full replicated [N, K]) into the gather slot
    of the current phase on every rank. ``site`` ("ag_norm" or "norm_fp4_quant") names the consumer, which must be the
    next user of the plane. ``split_k`` > 1: the split-K GEMM (``tactic`` required; ``sk_config`` / ``sk2_config`` give
    the served configs, rows <= SK_MAX_ROWS)."""
    from .. import opp_tgv_gemm as g

    assert (
        x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and x.is_contiguous()
    )
    m, (n, k) = x.shape[0], weight.shape
    r = step_aside_reason(comm, m, n, k)
    if r is not None:
        raise RuntimeError(
            f"proj_publish: cannot publish [{m}, {k}] x [{n}, {k}]^T: {r}"
        )
    w = comm.world_size
    n_slice = n // w
    w_slice = weight[comm.rank * n_slice : (comm.rank + 1) * n_slice]
    if split_k > 1 and (tactic is None or m > SK_MAX_ROWS):
        raise ValueError(
            "proj_publish: split_k > 1 needs an explicit tactic and rows <= SK_MAX_ROWS"
        )
    out, meta = _publish_meta(comm, m, n_slice, x.device, split_k > 1)
    _begin(comm, site, m, n)
    g.run_tgv(
        x,
        w_slice,
        out,
        tactic=tactic,
        push_mode=7,
        push_world=w,
        meta=meta,
        split_k=split_k,
        sk_cluster=sk_cluster,
        sk_push=sk_push,
    )


def pub_ag_norm(
    comm,
    stream: torch.Tensor,
    post_eps: float,
    pre_eps: float,
    *,
    post_norm_is_input_scale: bool = False,
    post_weight: Optional[torch.Tensor] = None,
    pre_weight: Optional[torch.Tensor] = None,
    alpha: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The FFN boundary of the published projection: ``(output bf16 [M, H], new_stream fp32 [M, H])``, as
    ``all_gather_norm_residual`` returns them, with SGLang's launch shape for ``(H, M)``."""
    norm_residual_launch_config = sgl_compat.module(
        "sglang.kernels.ops.muse_spark_v12.norm_residual"
    ).norm_residual_launch_config

    num_tokens, hidden = stream.shape
    _end(comm, "ag_norm", num_tokens, hidden)
    config = norm_residual_launch_config(hidden_size=hidden, num_tokens=num_tokens)
    output = torch.empty(num_tokens, hidden, dtype=torch.bfloat16, device=stream.device)
    new_stream = torch.empty_like(stream)
    module = _ag_norm_module(
        hidden,
        comm.world_size,
        tuple(config),
        bool(post_norm_is_input_scale),
        post_weight is not None,
        alpha is not None,
    )
    module.run(
        comm.get_communicator(),
        stream,
        output,
        new_stream,
        post_weight,
        pre_weight,
        alpha,
        beta,
        float(post_eps),
        float(pre_eps),
    )
    return output, new_stream


def pub_norm_fp4_quant(
    comm,
    num_tokens: int,
    hidden: int,
    weight: Optional[torch.Tensor],
    eps: float,
    device: torch.device,
    global_scale_inv: float = NVFP4_PER_TOKEN_SCALE_INV,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(fp4 [M, K/2] u8, sf [M, K/16] u8, per_token_scale [M] fp32)`` of the published pooler output, as
    ``fused_norm_fp4_quant(quantize=True)`` returns them."""
    _num_threads = sgl_compat.module(
        "sglang.kernels.ops.muse_spark_v12.norm_fp4_quant"
    )._num_threads

    _end(comm, "norm_fp4_quant", num_tokens, hidden)
    fp4 = torch.empty(num_tokens, hidden // 2, dtype=torch.uint8, device=device)
    sf = torch.empty(num_tokens, hidden // 16, dtype=torch.uint8, device=device)
    pts = torch.empty(num_tokens, dtype=torch.float32, device=device)
    module = _quant_module(hidden, comm.world_size, _num_threads(hidden))
    module.run(
        comm.get_communicator(),
        weight,
        fp4,
        sf,
        pts,
        float(eps),
        float(global_scale_inv),
    )
    return fp4, sf, pts
