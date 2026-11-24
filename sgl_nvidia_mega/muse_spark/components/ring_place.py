# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Per-step ring-window expert placement, host reference and helpers (development form; csrc/ring_place.cuh is the device
form, built into MoE-LD-W with -DMUSE_VR_LDW_RBAL=1).

World 4, 256 routed experts in 4 blocks of 64. Block b's local experts 48..63 are also held by rank b + 1 (mod 4): a fixed
replica set given by id arithmetic, independent of traffic. Rank r's pool has 80 slots: its own experts 0..63, then the
predecessor's local experts 48..63. Each step, ``place`` turns the routed rows per expert into one threshold per block:
block b's routed experts with local index >= t_b run on rank b + 1, the others on rank b. Integer arithmetic only; the
device form gives the same thresholds (tests/test_ring_place.py).
"""

from __future__ import annotations

from typing import Sequence

WORLD = 4
BLOCK = 64
MOV_LO = 48
POOL = 80
NUM_EXPERTS = 256
# cost per routed expert, per routed row, per extra pass, and the spacing of the T candidates (relative units: only the ratios
# matter); the device defaults (MUSE_VR_RING_COST_A / _B / _P, MUSE_VR_RING_STEP) are the same numbers
COST_A, COST_B, COST_P, STEP, NUM_T = 18430, 77, 2060, 2000, 8
FLAGS = ("-DMUSE_VR_LDW_RBAL=1",)
# the overlapped build (-DMUSE_VR_LDW_RBAL=2, place_warp2): 4 T candidates spaced STEP2 (MUSE_VR_RING_STEP2)
STEP2, NUM_T2 = 4000, 4
FLAGS2 = ("-DMUSE_VR_LDW_RBAL=2",)
# the pinned-first build (-DMUSE_VR_LDW_RBAL=3): the board machinery with the placement of place_warp2 computed in the CTA
# after the pinned work list is out; candidate experts' passes follow the pinned ones (rows are bitwise the same, their order
# in the output differs); not for the compact prologue
FLAGS3 = ("-DMUSE_VR_LDW_RBAL=3",)
# form 2 with its late-start and candidates-last options (-DMUSE_VR_LDW_RBAL=4; same placement as form 2)
FLAGS4 = ("-DMUSE_VR_LDW_RBAL=4",)
STAMP_WORDS = 24  # the kernel's stamps row (kStampWords) on this revision
# the short-scan form (-DMUSE_VR_LDW_RBAL=5; same placement as form 2)
FLAGS5 = ("-DMUSE_VR_LDW_RBAL=5",)


def flags_of(form: int) -> tuple:
    return {1: FLAGS, 2: FLAGS2, 3: FLAGS3, 4: FLAGS4, 5: FLAGS5}[form]


def place_form(hist, pass_rows: int, form: int):
    """The host reference of build form 1 (place_warp) or 2 / 3 / 4 / 5 (place_warp2)."""
    if form == 1:
        return place(hist, pass_rows)
    return place(hist, pass_rows, step=STEP2, num_t=NUM_T2)


def expert_cost(
    h: int, pass_rows: int, a: int = COST_A, b: int = COST_B, p: int = COST_P
) -> int:
    if h == 0:
        return 0
    return a + b * h + p * ((h + pass_rows - 1) // pass_rows - 1)


def place(
    hist: Sequence[int],
    pass_rows: int,
    a: int = COST_A,
    b: int = COST_B,
    p: int = COST_P,
    step: int = STEP,
    num_t: int = NUM_T,
) -> tuple[list[int], int]:
    """Thresholds t_0..t_3 (each in [48, 64]) and the winning key (max rank cost << 3 | candidate) for one step."""
    assert len(hist) == NUM_EXPERTS
    cost = [expert_cost(int(h), pass_rows, a, b, p) for h in hist]
    L = [sum(cost[BLOCK * q : BLOCK * (q + 1)]) for q in range(WORLD)]
    S = [
        [sum(cost[BLOCK * q + t : BLOCK * (q + 1)]) for t in range(MOV_LO, BLOCK + 1)]
        for q in range(WORLD)
    ]
    mv = [S[q][0] for q in range(WORLD)]
    T = (sum(L) + 3) // 4
    for r in range(WORLD):
        arc = 0
        for n in range(1, WORLD):
            arc += L[(r - n + 1) % WORLD]
            if arc > mv[r]:
                T = max(T, (arc - mv[r] + n - 1) // n)
    best = None
    for j in range(num_t):
        tj = T + j * step
        ts, outs = [], []
        for q in range(WORLD):
            arc, cq = 0, 0
            for n in range(1, WORLD + 1):
                arc += L[(q - n + 1) % WORLD]
                cq = max(cq, arc - n * tj)
            bt, bd = BLOCK - MOV_LO, cq
            for k in range(BLOCK - MOV_LO - 1, -1, -1):
                d = abs(S[q][k] - cq)
                if d < bd:
                    bd, bt = d, k
            ts.append(bt)
            outs.append(S[q][bt])
        mx = max(L[q] - outs[q] + outs[(q - 1) % WORLD] for q in range(WORLD))
        key = (mx << 3) | j
        if best is None or key < best[0]:
            best = (key, ts)
    return [MOV_LO + t for t in best[1]], best[0]


def digest(thr: Sequence[int], key: int) -> int:
    """The per-CTA digest word the device writes (stamps slot 23): thresholds in bits 0-31, the winning key in 32-63."""
    return (thr[0] | thr[1] << 8 | thr[2] << 16 | thr[3] << 24) | (key << 32)


def window_slot(e: int, r: int) -> int:
    """Pool slot of expert e on rank r, or -1 (same arithmetic as the device)."""
    ci = (e - (BLOCK * r - (BLOCK - MOV_LO))) & 255
    if ci >= POOL:
        return -1
    return ci - (BLOCK - MOV_LO) if ci >= BLOCK - MOV_LO else BLOCK + ci


def holder(e: int, thr: Sequence[int]) -> int:
    """Rank that runs routed expert e this step."""
    q, i = e // BLOCK, e % BLOCK
    return (q + 1) % WORLD if i >= thr[q] else q


def pool_experts(r: int) -> list[int]:
    """Global expert held at each of rank r's 80 pool slots."""
    own = [BLOCK * r + i for i in range(BLOCK)]
    pred = [BLOCK * ((r - 1) % WORLD) + i for i in range(MOV_LO, BLOCK)]
    return own + pred


def hist_of(packed) -> list[int]:
    """Routed rows per expert from packed ids (expert << 16 | bf16 weight)."""
    import torch

    e = ((packed.to(torch.int64) >> 16) & 0xFFFF).reshape(-1)
    return torch.bincount(e[e < NUM_EXPERTS], minlength=NUM_EXPERTS).tolist()


def virtual_ids(packed, thr: Sequence[int]):
    """Packed ids rewritten to virtual ids holder * 80 + pool slot (the precomputed placement; for the served kernel with
    local_offset = holder * 80 and pool 80)."""
    import torch

    pk = packed.to(torch.int64).cpu()
    e = (pk >> 16) & 0xFFFF
    w = pk & 0xFFFF
    flat_e, flat_v = e.reshape(-1).tolist(), []
    for x in flat_e:
        h = holder(x, thr)
        flat_v.append(h * POOL + window_slot(x, h))
    vid = torch.tensor(flat_v, dtype=torch.int64).reshape(e.shape)
    return ((vid << 16) | w).to(torch.int32).to(packed.device)
