# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MoE-LD-W (``csrc/moe_ldw_l.cuh``): routed NVFP4 experts of one expert-parallel rank for 1..512 token rows (1..1024 in the wide
build ``WIDE``, 1..2048 in ``WIDE2K``) in one launch of one CTA per SM streaming each active local expert's weights once per
pass of up to 128 routed rows.

``moe_ldw(...)`` takes the arguments of ``moe_ld_ops.moe_ld`` and returns the same ``(gemm2_out, expert_weights,
expanded_idx_to_permuted_idx)`` triple (FC2 rows in a compact expert-major order, the map points at them), every routed
row bitwise equal to the stock call's.

``Scratch`` (per device; the kernel keeps it consistent across launches that are ordered by completion): permuted token
rows, token-scale images, activation rows, FC2 operand rows, two counter sets (alternating launches), start tickets,
sticky error word. Expert
weights are best placed with ``ld_layout.split_rows_copy`` (each expert's row halves on the two locality domains).

Package form: the ``l`` form of ``ops/moe_ldw.py``, the MoE-LD-W of ``moe_ld_l`` (``MUSE_VR_MOE_LD_FORM=l``).
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

import torch

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "kernels/jit/utils/__init__.py": ("load_jit",),
}


_CSRC = Path(__file__).resolve().parents[1] / "csrc"
_CUH = _CSRC / "moe_ldw_l.cuh"
HIDDEN = 4096
INTER = 4096
TOP_K = 8
MAX_TOKENS = 512
MAX_SLOTS = MAX_TOKENS * TOP_K
MAX_POOL = 128
PASS_ROWS = 128
MAX_PASSES = MAX_POOL + MAX_SLOTS // PASS_ROWS
ACT_WORDS = INTER // 4
IMG_BYTES = (HIDDEN // 64) * 512
CNT_SET = 2 * MAX_PASSES + 32
FLAG_EVICT_FIRST = 2
FLAG_TAIL_TRIGGER = 32
STAMP_WORDS = 24
DEFAULT_FLAGS = FLAG_EVICT_FIRST | FLAG_TAIL_TRIGGER
# the RING4 build: 32-row passes and a 4-stage ring (more weight bytes in flight per CTA); same rows, its own Scratch
RING4_PASS_ROWS = 32
RING4 = (
    "-DMUSE_VR_LDW_GATHER=0",
    f"-DMUSE_VR_LDW_PASS_ROWS={RING4_PASS_ROWS}",
    "-DMUSE_VR_LDW_STAGES=4",
)
# RING4 with 48-row passes (experts with 33-48 rows in one pass; fewer passes for hot experts); its own scratch
RING48_PASS_ROWS = 48
RING48 = (
    "-DMUSE_VR_LDW_GATHER=0",
    "-DMUSE_VR_LDW_PASS_ROWS=48",
    "-DMUSE_VR_LDW_STAGES=4",
)
# RING4's dense-token form for calls of at most DENSE_MAX_ROWS rows: every FC1 item takes the call's whole token block
# straight from the token tensor (no permuted copy, no grid-wide rows-ready count); the same rows; its own scratch
DENSE_MAX_ROWS = 32
DENSE_FLAG = "-DMUSE_VR_LDW_DENSE=1"
RING4_DENSE = RING4 + (DENSE_FLAG,)
# the dense-token build's early prologue form: the prologue work that reads only the routed ids (counts, ring placement, pass
# tables) and an L2 prefetch of the hinted slot's first ring stages run before the dependency wait; the same rows. Its callers'
# ids must be complete at launch (a full graph dependency from their producer)
PRE_FLAG = "-DMUSE_VR_LDW_PRE=1"
RING4_DENSE_PRE = RING4_DENSE + (PRE_FLAG,)
# the speculate-and-verify form of it: the early work runs on the ids as read before the wait, the kernel re-reads them
# after the wait and rebuilds its tables on any difference, so the callers' ids need not be complete at launch
VERIFY_FLAG = "-DMUSE_VR_LDW_PRE_VERIFY=1"
RING4_DENSE_PREV = RING4_DENSE_PRE + (VERIFY_FLAG,)
# the wide builds (added to a build's flags): up to 1024 (WIDE) or 2048 (WIDE2K) rows per call; each needs a Scratch made with
# its max_tokens. Only pass heights whose pass count fits the work item's 8-bit pass index compile with them (the kernel
# asserts: 128 / 64 for WIDE, 128 for WIDE2K).
MAX_TOKENS_WIDE = 1024
WIDE = ("-DMUSE_VR_LDW_MAX_TOKENS=1024",)
MAX_TOKENS_2K = 2048
WIDE2K = ("-DMUSE_VR_LDW_MAX_TOKENS=2048",)


# the error-word build (added to a build's flags by ``moe_ldw(..., ew=...)``): no trap; a CTA whose wait expired or whose slot
# claim failed stores bit 31 | tag into the caller's error word and the kernel completes (csrc/moe_ldw.cuh; it also sets
# slot_claim.cuh's error-word form)
ERRWORD = ("-DMUSE_VR_LDW_ERRWORD=1",)
LDW_ROLE = 7  # error-word tag: (LDW_ROLE << 16) | MoE call site index


def wide_flags(max_tokens: int) -> tuple:
    """The build flag for a rows-per-call ceiling (512: none)."""
    return {MAX_TOKENS: (), MAX_TOKENS_WIDE: WIDE, MAX_TOKENS_2K: WIDE2K}[
        int(max_tokens)
    ]


def max_tokens_of(extra) -> int:
    """Rows per call of the build ``extra`` describes (512, or 1024 / 2048 for the wide builds)."""
    e = tuple(extra)
    return (
        MAX_TOKENS_2K
        if WIDE2K[0] in e
        else MAX_TOKENS_WIDE
        if WIDE[0] in e
        else MAX_TOKENS
    )


GLOBAL_SCALE_INV = float(
    torch.tensor(1.0, dtype=torch.float32) / (torch.tensor(448.0) * torch.tensor(6.0))
)


def _flashinfer_include_paths() -> list[str]:
    import flashinfer

    data = Path(flashinfer.__file__).resolve().parent / "data"
    paths = [
        data / "csrc" / "nv_internal",
        data / "csrc" / "nv_internal" / "include",
        data / "include",
        data / "cutlass" / "include",
    ]
    return [str(p) for p in paths if p.exists()]


@lru_cache(maxsize=None)
def _module(extra: tuple = ()):
    import hashlib

    from sglang.kernels.jit.utils import load_jit

    from .. import mma_k

    extra = tuple(extra) + mma_k.build_flags(
        "moe_ldw"
    )  # MMA-K form (MUSE_VR_MOE_MMA_K; no define for the default 64)
    extra = (
        tuple(extra) + epoch_flags()
    )  # launch-counter slot claim (MUSE_VR_MOE_LDW_SLOT_EPOCH=1; default off)
    # one symbol namespace per build variant (two variants may be loaded in one process)
    abi = "v" + hashlib.sha256(" ".join(extra).encode()).hexdigest()[:12]
    from .. import slot_claim

    extra = tuple(extra) + slot_claim.build_flags(
        "moe_ldw"
    )  # exact slot claim (default) or the UNSAFE form
    name_args = [re.sub(r"[^A-Za-z0-9_]", "_", e) for e in extra]
    if len("_".join(["sgl_kernel_jit_muse_vr_moe_ldw"] + name_args)) + 3 > 255:
        # a flag set whose module file name would pass the file system's 255-byte name limit (the tagged-ids builds):
        # one hashed name component instead; every shorter name is unchanged
        name_args = ["x" + hashlib.sha256(" ".join(extra).encode()).hexdigest()[:16]]
    return load_jit(
        "muse_vr_moe_ldw",
        *name_args,
        cuda_files=[str(_CUH)],
        cuda_wrappers=[
            ("run", "muse_vr_ldw::MuseVrLdw::run"),
            ("preload", "muse_vr_ldw::MuseVrLdw::preload"),
        ],
        extra_include_paths=[str(_CSRC)] + _flashinfer_include_paths(),
        extra_cuda_cflags=[
            "-O3",
            "-use_fast_math",
            "-DENABLE_BF16",
            "-lineinfo",
            f"-DMUSE_VR_LDW_ABI={abi}",
            *extra,
        ],
    )


def preload(extra: tuple = ()) -> None:
    _module(tuple(extra)).preload()


CTL_CLAIM_WORD = 21  # first word of the slot claim area in the control tensor (the kernel's kClaimWord)
EPOCH_ENV = "MUSE_VR_MOE_LDW_SLOT_EPOCH"
EPOCH_WORDS = 256 * 8  # per-SM launch counters after the 32 control words
_epoch_recorded: list = []


def epoch() -> bool:
    """Launch-counter slot claim (slot_claim.cuh, -DMUSE_VR_SLOT_EPOCH=1): the exact claim with no bitmap and no
    last-CTA check (default off; the UNSAFE form takes precedence)."""
    from .. import slot_claim

    return os.environ.get(EPOCH_ENV, "0").strip() == "1" and not slot_claim.unsafe()


def epoch_flags() -> tuple[str, ...]:
    if not epoch():
        return ()
    if not _epoch_recorded:
        _epoch_recorded.append(True)
        from .. import bed

        bed.report("slot_epoch_moe_ldw", True, env=f"{EPOCH_ENV}=1")
    return ("-DMUSE_VR_SLOT_EPOCH=1",)


class Scratch:
    """Per-device buffers of MoE-LD-W (launches that share it must be ordered by completion). ``pass_rows`` is the build's
    pass height (``PASS_ROWS``, or ``RING4_PASS_ROWS`` for the ``RING4`` build): the per-pass buffers follow it.
    ``max_tokens`` is the build's rows per call (``MAX_TOKENS``, or ``MAX_TOKENS_WIDE`` / ``MAX_TOKENS_2K`` for a wide build)."""

    def __init__(
        self,
        device: torch.device,
        pass_rows: int = PASS_ROWS,
        max_tokens: int = MAX_TOKENS,
    ):
        z = dict(device=device)
        if max_tokens not in (MAX_TOKENS, MAX_TOKENS_WIDE, MAX_TOKENS_2K):
            raise ValueError(
                f"max_tokens {max_tokens}: {MAX_TOKENS}, {MAX_TOKENS_WIDE} or {MAX_TOKENS_2K}"
            )
        slots = max_tokens * TOP_K
        passes = MAX_POOL + slots // pass_rows
        if passes > 256:
            raise ValueError(
                f"{pass_rows}-row passes for {max_tokens} rows: {passes} passes > 256 (the kernel's pass index)"
            )
        self.pass_rows = pass_rows
        self.max_tokens = max_tokens
        self.hs_perm = torch.zeros(slots, HIDDEN // 2, dtype=torch.uint8, **z)
        self.pts_perm = torch.zeros(slots, dtype=torch.float32, **z)
        self.img1 = torch.zeros(passes * IMG_BYTES, dtype=torch.uint8, **z)
        self.img2 = torch.zeros(passes * IMG_BYTES, dtype=torch.uint8, **z)
        self.act = torch.zeros(slots * ACT_WORDS, dtype=torch.int64, **z)
        self.hs2 = torch.zeros(slots, HIDDEN // 2, dtype=torch.uint8, **z)
        self.ramax = torch.zeros(slots, dtype=torch.float32, **z)
        self.cnt = torch.zeros(2 * (2 * passes + 32), dtype=torch.int32, **z)
        self.ctl = torch.zeros(
            32 + (EPOCH_WORDS if epoch() else 0), dtype=torch.int32, **z
        )
        self.err = torch.zeros(1, dtype=torch.int32, **z)

    def error(self) -> int:
        return int(self.err.item())

    def slot_tally(self) -> dict:
        """Slot claim counters (verified launches, hinted-slot misses; exact form only)."""
        from .. import slot_claim

        return slot_claim.tally(self.ctl, CTL_CLAIM_WORD)


RBAL_PREFIX = "-DMUSE_VR_LDW_RBAL="
_N_REAL: dict = {}  # (device index, m) -> int32 [1] tensor holding m (the real-token count of an unpadded call)


def ring_build(extra: tuple = ()) -> bool:
    """The build ``extra`` selects is a ring placement build (-DMUSE_VR_LDW_RBAL=<nonzero>)."""
    return any(f.startswith(RBAL_PREFIX) and f != RBAL_PREFIX + "0" for f in extra)


def _n_real_const(dev: torch.device, m: int):
    key = (dev.index, int(m))
    t = _N_REAL.get(key)
    if t is None:
        t = torch.full((1,), int(m), dtype=torch.int32, device=dev)
        _N_REAL[key] = t
    return t


def moe_ldw(
    packed,
    hs,
    hs_sf,
    pts,
    w13,
    sf13,
    w2,
    sf2,
    gate_alpha,
    up_alpha,
    out_alpha,
    local_offset: int,
    scratch: Scratch,
    flags: int | None = None,
    dbg: int = 0,
    pdl: bool = False,
    extra: tuple = (),
    stamps=None,
    n_real=None,
    ew=None,
):
    """``n_real`` (ring builds only): int32 [1] device tensor, the call's real tokens (the placement histogram counts their routed
    slots only; a CUDA-graph replay's padding rows past it are still placed and computed). None = an unpadded call (m); a ring
    call inside a CUDA-graph capture must pass it. ``ew`` = (error word address, tag): the build's error-word form
    (``ERRWORD``) with that word instead of the trap."""
    if ew is not None:
        extra = tuple(extra) + ERRWORD
    from ..moe_ld_ops import domain_slots

    if flags is None:
        flags = DEFAULT_FLAGS
    m = packed.shape[0]
    limit = max_tokens_of(extra)
    rows_max = (
        DENSE_MAX_ROWS if DENSE_FLAG in tuple(extra) else limit
    )  # the dense-token build takes at most 32 rows
    if not 1 <= m <= rows_max:
        raise ValueError(f"num_tokens {m} outside 1..{rows_max}")
    dev = packed.device
    table, n0, n1 = domain_slots(
        dev.index if dev.index is not None else torch.cuda.current_device()
    )
    out = torch.empty(m * TOP_K, HIDDEN, dtype=torch.bfloat16, device=dev)
    ew_out = torch.empty(m, TOP_K, dtype=torch.bfloat16, device=dev)
    e2p = torch.empty(m * TOP_K, dtype=torch.int32, device=dev)
    s = scratch
    want = PASS_ROWS
    for f in extra:
        if f.startswith("-DMUSE_VR_LDW_PASS_ROWS="):
            want = int(f.split("=", 1)[1])
    if s.pass_rows != want:
        raise ValueError(
            f"scratch for {s.pass_rows}-row passes, build with {want}-row passes"
        )
    if getattr(s, "max_tokens", MAX_TOKENS) != limit:
        raise ValueError(
            f"scratch for {getattr(s, 'max_tokens', MAX_TOKENS)} rows per call, build for {limit}"
        )
    ring = ring_build(tuple(extra))
    if ring:
        if n_real is None:
            if torch.cuda.is_current_stream_capturing():
                raise ValueError(
                    "moe_ldw ring build: a CUDA-graph capture needs the real-token count tensor (n_real)"
                )
            n_real = _n_real_const(dev, m)
        if not (
            torch.is_tensor(n_real)
            and n_real.dtype == torch.int32
            and n_real.numel() == 1
            and n_real.device == dev
        ):
            raise ValueError(
                "moe_ldw ring build: n_real must be an int32 [1] tensor on the call's device"
            )
        n_real = n_real.reshape(1)
    elif n_real is not None:
        raise ValueError("moe_ldw: n_real is for ring placement builds only")
    _module(tuple(extra)).run(
        packed,
        hs.view(torch.uint8),
        hs_sf.view(torch.uint8).reshape(m, -1),
        pts.reshape(m).float(),
        w13.view(torch.uint8),
        sf13.view(torch.uint8),
        w2.view(torch.uint8),
        sf2.view(torch.uint8),
        gate_alpha,
        up_alpha,
        out_alpha,
        table,
        out,
        ew_out.view(torch.int16),
        e2p,
        s.hs_perm,
        s.pts_perm,
        s.img1,
        s.img2,
        s.act,
        s.hs2,
        s.ramax,
        s.cnt,
        s.ctl,
        s.err,
        stamps,
        int(local_offset),
        n0,
        n1,
        GLOBAL_SCALE_INV,
        int(flags),
        int(dbg),
        int(bool(pdl)),
        *((n_real,) if ring else ()),
        *((int(ew[0]), int(ew[1])) if ew is not None else ()),
    )
    return out, ew_out, e2p
