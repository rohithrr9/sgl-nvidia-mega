# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DFlash draft dense GEMMs through FlashInfer's low-row BF16 kernels with the K reduction inside the kernel.

At a few request blocks (17-32 rows) the stock dispatch sends the draft's block-convolution coefficient projection
(``nn.Linear``), the MLP's merged gate/up projection and its down projection to cuBLAS, which runs each of them as a
split-K kernel followed by a separate reduction kernel. This module runs them through FlashInfer's dense BF16 GEMMs for
at most 32 rows instead: the cluster split-K kernel (partials reduced in distributed shared memory), the warp split-K
kernel (partials reduced across the warps of one CTA) or the direct kernel, one launch per GEMM, with programmatic
dependent launch as the stock dispatch uses them.

Rounding points are the stock ones: bf16 operands, fp32 accumulation, one bf16 rounding per output element. The fp32
summation order of the K reduction differs from cuBLAS's (tests/test_draft_splitk.py: error against an fp64 reference
within the bound of the stock call's error).

``gemm(x, w, m)`` returns ``x @ w.T`` as a new tensor. Tactics are per (rows, N, K) from ``DEFAULTS`` or the
``MUSE_VR_DRAFT_SPLITK_TACTICS`` override, entries ``NxK:m:family:a/b/c[/d[/e]]`` separated by ``;`` (family ``sk`` =
cluster split-K (mma_m / mma_n / split_k / ab_stages), ``ws`` = warp split-K (output_tile / token_tile / k_tile / stages
/ b_loader_warps), ``dr`` = direct with its default tactic).

The three FlashInfer sources are pinned at 0.7.2 (``FLASHINFER_PINS``): with another release
``available()`` is False and the projections keep the stock dispatch.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.util
import os
from pathlib import Path

import torch

# (rows, N, K) -> (family, tactic) for the DFlash2 draft at four-way tensor parallelism: coefficient projection
# 2048 x 8192, merged gate/up 14336 x 8192, down 8192 x 7168. Tunable per shape and row count; sweep with
# tests/test_draft_splitk.py --bench (cold-weight graph replay against the stock call).
DEFAULTS: dict[tuple[int, int, int], tuple[str, tuple[int, ...]]] = {
    (24, 2048, 8192): ("sk", (64, 32, 4, 8)),
    (32, 2048, 8192): ("sk", (64, 32, 4, 8)),
    (24, 14336, 8192): ("sk", (128, 32, 1, 5)),
    (32, 14336, 8192): ("sk", (128, 32, 1, 5)),
    (24, 8192, 7168): ("sk", (64, 32, 1, 9)),
    (32, 8192, 7168): ("sk", (64, 32, 1, 9)),
}

MAX_M = 32

# sha256 of the FlashInfer sources whose entry points gemm() calls (0.7.2), relative to the flashinfer package directory
FLASHINFER_PINS = {
    "gemm/kernels/dense_bf16_gemm_sm100_splitk.py": "f5edd4c11d95822c836a363dfa1dd19f7843b4e354e091c24c0acac12f52d9d2",
    "gemm/kernels/dense_bf16_gemm_warp_splitk.py": "de56bdac4f4edd385e4dff052098aa1286e15173ad56e3e49715b6ceb458dc72",
    "gemm/kernels/dense_bf16_gemm_direct.py": "605ba1d46a3a98fedd1e0d2513f0629eb1ce129142673dc7ef197163f2064c54",
}


@functools.lru_cache(maxsize=None)
def flashinfer_pinned() -> bool:
    """The installed FlashInfer sources are the pinned ones (read without importing the package)."""
    spec = importlib.util.find_spec("flashinfer")
    if spec is None or not spec.submodule_search_locations:
        return False
    root = Path(list(spec.submodule_search_locations)[0])
    try:
        return all(
            hashlib.sha256((root / rel).read_bytes()).hexdigest() == sha
            for rel, sha in FLASHINFER_PINS.items()
        )
    except OSError:
        return False


def _parse(spec: str) -> dict:
    out = {}
    for ent in (s.strip() for s in spec.split(";")):
        if not ent:
            continue
        shape, m, fam, tac = ent.split(":")
        n, k = (int(v) for v in shape.lower().split("x"))
        if fam not in ("sk", "ws", "dr"):
            raise ValueError(f"unknown family {fam!r}")
        out[(int(m), n, k)] = (
            fam,
            tuple(int(v) for v in tac.split("/")) if tac else (),
        )
    return out


@functools.lru_cache(maxsize=None)
def table() -> dict:
    t = dict(DEFAULTS)
    spec = os.environ.get("MUSE_VR_DRAFT_SPLITK_TACTICS", "")
    if spec:
        t.update(_parse(spec))
    return t


def tactic_for(m: int, n: int, k: int):
    if not available():
        return None
    return table().get((m, n, k))


@functools.lru_cache(maxsize=None)
def available() -> bool:
    """FlashInfer's three low-row BF16 GEMM modules with the entry points this module calls."""
    if not flashinfer_pinned():
        return False
    try:
        sk, ws, dr = _mods()
    except ImportError:
        return False
    return (
        all(hasattr(sk, a) for a in ("SplitKTactic", "run_splitk_dense"))
        and all(hasattr(ws, a) for a in ("WarpSplitKTactic", "run_warp_splitk_dense"))
        and all(hasattr(dr, a) for a in ("default_tactic", "run_direct_dense"))
    )


@functools.lru_cache(maxsize=None)
def _mods():
    from flashinfer.gemm.kernels import dense_bf16_gemm_direct as dr
    from flashinfer.gemm.kernels import dense_bf16_gemm_sm100_splitk as sk
    from flashinfer.gemm.kernels import dense_bf16_gemm_warp_splitk as ws

    return sk, ws, dr


def gemm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """``x [m, k] @ w [n, k]^T`` -> new bf16 [m, n] with the table's kernel for (m, n, k)."""
    x = x.contiguous()
    m, k = x.shape
    n = w.shape[0]
    fam, tac = tactic_for(m, n, k)
    out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    sk, ws, dr = _mods()
    if fam == "sk":
        sk.run_splitk_dense(x, w.T, None, out, True, sk.SplitKTactic(*tac))
    elif fam == "ws":
        ws.run_warp_splitk_dense(x, w.T, out, True, ws.WarpSplitKTactic(*tac), None)
    else:
        dr.run_direct_dense(x, w.T, out, True, dr.default_tactic(m, n, k))
    return out
