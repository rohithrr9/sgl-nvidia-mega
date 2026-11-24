# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Component ``eager_commit`` (env ``MUSE_VR_EAGER_COMMIT=1``, default off).

After the target verify, SGLang's DFLASH worker (selector path, ``DFlashWorkerV2._accept_block``) samples the accept
lengths and bonus tokens, broadcasts both from rank 0 to the tensor-parallel group (two NCCL broadcasts,
``SpecTpSync``), and builds the committed block (``_commit_accept``: a copy, a fill, a scatter and an add), each as a
small eager launch.

This component keeps the sampling call and performs the broadcast and the commit in one kernel
(``csrc/dflash_commit.cuh``): rank 0's decision travels over the tensor-parallel group's custom all-reduce push plane
(the same plane and phase protocol as SGLang's one-shot push all-reduce; every other rank pushes a marker to every
peer, so each rank waits for every peer in each call as in SGLang's push collectives), and every rank builds the same
committed block. Outputs are identical to the stock sequence (integer copies only; tests/test_eager_commit.py on one
GPU, tests/dist_eager_commit.py across ranks); no work the stock path does is skipped. The component does not change
the synthetic-acceptance benchmark handling: on the branch layout SGLang applies that mode's forcing after
``_accept_block`` and it runs unchanged; on SGLang main the forcing is the last statement of ``_accept_block``, and the
component runs that same statement with the same arguments. On the supported SGLang the selector path also passes
``cutoff_verify_lens`` to the sampling call and broadcasts its ``cap_trim_lens`` before the commit; the component makes
the same call and the same broadcast. That SGLang ships the same fused commit (``SGLANG_OPT_DFLASH_EAGER_COMMIT``,
``_commit_selector_accept``); while this component is engaged it handles the selector path, so the two never both run.

Engaged only for the selector sampling path with accept lengths int32 and bonus tokens int64, contiguous candidates,
at most 1024 requests and, when the decision broadcast is enabled for the group, a custom all-reduce push plane on a
2/4/8-rank group. Every rank decides the same way.
"""

from __future__ import annotations

import functools
import logging
from pathlib import Path

import torch

from . import bed

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "kernels/jit/utils/__init__.py": (
        "is_arch_support_pdl",
        "load_jit",
        "make_cpp_args",
    ),
    "srt/distributed/device_communicators/custom_all_reduce_v2.py": (
        "CustomAllReduceV2",
    ),
    "srt/speculative/spec_tp_sync.py": ("SpecTpSyncSite",),
    "srt/plugins/hook_registry.py": ("HookRegistry", "HookType"),
    "srt/speculative/dflash_worker_v2.py": (
        "_commit_accept",
        "SIMULATE_ACC_LEN",
        "SIMULATE_ACC_METHOD",
        "SIMULATE_ACC_TOKEN_MODE",
        "apply_dflash_simulated_acceptance",
    ),
}


logger = logging.getLogger(__name__)

NAME = "eager_commit"  # env MUSE_VR_EAGER_COMMIT
TARGET_ACCEPT = "sglang.srt.speculative.dflash_worker_v2.DFlashWorkerV2._accept_block"
_CUH = Path(__file__).resolve().parent / "csrc" / "dflash_commit.cuh"

# csrc/dflash_commit.cuh uses the push plane's EpochCounter phase word (load / commit), which the earlier SGLang
# revisions do not have, so only revisions with that communicator are pinned.
PINS = {
    "sglang/srt/speculative/dflash_worker_v2.py": (
        bed.Functions(
            {
                "DFlashWorkerV2._accept_block": "a04a15a5ec8319b8",
                "DFlashWorkerV2._selector_sampling_accept": "ebd8291583a59153",
                "DFlashWorkerV2._commit_selector_accept": "92246006c11bea50",
                "_commit_accept": "6f5e074cf9f76d85",
            }
        ),
    ),
    "sglang/srt/speculative/spec_utils.py": (
        # Nothing here is called or mirrored: the SIMULATE_ACC_* settings are read through the worker module.
        bed.Functions({}),
    ),
    "sglang/srt/speculative/spec_tp_sync.py": (
        bed.Functions(
            {
                "SpecTpSync.__init__": "5a9d15bcab0fc2d9",
                "SpecTpSync.enabled": "b1ac98be791b8b86",
                "SpecTpSync.sync": "286a72c967b47320",
            }
        ),
    ),
    "sglang/srt/distributed/device_communicators/custom_all_reduce_v2.py": (
        bed.Functions(
            {
                "CustomAllReduceV2.__init__": "6871008b7ea3d24e",
                "CustomAllReduceV2.obj": "4eba547724199cb9",
            }
        ),
    ),
    "sglang/kernels/jit/include/sgl_kernel/distributed/communicator.cuh": "43315a3609675cc012be57e88622d19c2fa373126d03de018f87ef371e82f27b",
}


def enabled() -> bool:
    return bed.enabled(NAME)


def pins_match() -> tuple[bool, str]:
    for rel, want in PINS.items():
        if not bed.pinned(rel, want):
            return False, f"{rel} differs from the pinned revision"
    return True, "ok"


@functools.lru_cache(maxsize=None)
def _use_pdl() -> bool:
    from sglang.kernels.jit.utils import is_arch_support_pdl

    return bool(is_arch_support_pdl())


@functools.lru_cache(maxsize=None)
def module(world: int, pdl: bool):
    from sglang.kernels.jit.utils import load_jit, make_cpp_args

    args = make_cpp_args(max(int(world), 1), pdl)
    return load_jit(
        "muse_vr_dflash_commit",
        *args,
        cuda_files=[str(_CUH)],
        cuda_wrappers=[
            ("run", f"MuseVrCommit<{args}>::run"),
            ("run_local", f"MuseVrCommit<{args}>::run_local"),
        ],
    )


def commit(candidates, accept_len, bonus, comm=None):
    """Returns (out_tokens, commit_lens). With ``comm``, rank 0's accept_len / bonus are first delivered to every rank
    of its push plane (collective; ``accept_len`` / ``bonus`` are overwritten with rank 0's values)."""
    bs = candidates.shape[0]
    out_tokens = torch.empty_like(candidates, dtype=torch.int64)
    commit_lens = torch.empty((bs,), dtype=torch.int32, device=candidates.device)
    if comm is not None:
        module(comm.world_size, _use_pdl()).run(
            comm.obj, candidates, accept_len, bonus, out_tokens, commit_lens
        )
    else:
        module(1, _use_pdl()).run_local(
            candidates, accept_len, bonus, out_tokens, commit_lens
        )
    return out_tokens, commit_lens


def _exchange_comm(worker):
    """(needs broadcast, communicator or None, reason)."""
    from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
        CustomAllReduceV2,
    )
    from sglang.srt.speculative.spec_tp_sync import SpecTpSyncSite

    sync = worker._tp_sync
    if not sync.enabled(SpecTpSyncSite.DFLASH_SELECTOR):
        return False, None, None
    group = sync._tp_group
    if group.world_size == 1:
        return False, None, None
    comm = getattr(group, "ca_comm", None)
    if (
        group.world_size not in (2, 4, 8)
        or not isinstance(comm, CustomAllReduceV2)
        or comm.disabled
    ):
        return True, None, "no custom all-reduce for the broadcast"
    return True, comm, None


def _consensus() -> bool:
    """True when ``verify_chain`` produced this step's decision by consensus (consumes the per-step mark)."""
    if not bed.enabled("verify_chain"):
        return False
    from . import verify_chain

    return verify_chain.consensus_taken()


def _result(worker, kw, accept_len, commit_lens, bonus, out_tokens, cap_trim_lens):
    """``_accept_block``'s return value for the installed SGLang. The branch layout returns six values and its caller
    applies the synthetic-acceptance forcing afterwards. SGLang main applies that forcing at the end of
    ``_accept_block`` itself and returns six values, the last two ``new_seq_lens`` (the forcing's, else None on the
    selector path) and ``cap_trim_lens`` (None once forced); on main this runs the same statement with the same
    arguments (the selector path leaves ``target_predict`` as passed in), so what runs after the decision is unchanged
    in both layouts."""
    from . import sgl_compat

    if sgl_compat.layout() != "main":
        return (
            accept_len,
            commit_lens,
            bonus,
            out_tokens,
            None,
            kw.get("target_predict"),
        )
    import sglang.srt.speculative.dflash_worker_v2 as w

    new_seq_lens = None
    if w.SIMULATE_ACC_LEN > 0:
        new_seq_lens = w.apply_dflash_simulated_acceptance(
            candidates=kw["candidates"],
            target_predict=kw.get("target_predict"),
            accept_len=accept_len,
            commit_lens=commit_lens,
            bonus=bonus,
            out_tokens=out_tokens,
            simulate_acc_len=w.SIMULATE_ACC_LEN,
            simulate_acc_method=w.SIMULATE_ACC_METHOD,
            simulate_acc_token_mode=w.SIMULATE_ACC_TOKEN_MODE,
            next_token_logits=kw["next_token_logits"],
            generator=worker._simulated_acc_len_rng,
            prefix_lens=kw["prefix_lens"],
            new_seq_lens=new_seq_lens,
        )
        cap_trim_lens = None
    return accept_len, commit_lens, bonus, out_tokens, new_seq_lens, cap_trim_lens


def accept_block(original, worker, **kw):
    """``DFlashWorkerV2._accept_block`` (pinned revisions), selector path, with broadcast + commit fused."""
    # The stock selector path needs no resolved argmax (a vocab-sharded verify's argmax makes the accept greedy).
    if worker._selector_sample is None or kw.get("target_predict") is not None:
        return original(worker, **kw)
    cand = kw["candidates"]
    if (
        cand.dtype != torch.int64
        or cand.dim() != 2
        or not cand.is_contiguous()
        or cand.shape[0] > 1024
    ):
        bed.count(NAME, "stock:candidates layout")
        return original(worker, **kw)
    need, comm, why = _exchange_comm(worker)
    if need and comm is None:
        bed.count(NAME, "stock:" + why)
        return original(worker, **kw)
    from sglang.srt.speculative.spec_tp_sync import SpecTpSyncSite

    sel_ids, sel_q = worker._selector_sample
    accept_len, bonus, cap_trim_lens = worker._selector_sampling_accept(
        candidates=cand,
        next_token_logits=kw["next_token_logits"],
        candidate_ids=sel_ids,
        q_rows=sel_q,
        sampling_info=kw["sampling_info"],
        draft_input=kw["draft_input"],
        cutoff_verify_lens=kw.get("cutoff_verify_lens"),
    )
    # stock broadcasts the cut counts on their own, before the commit
    worker._tp_sync.sync(SpecTpSyncSite.DFLASH_SELECTOR, cap_trim_lens)
    if (
        accept_len.dtype != torch.int32
        or bonus.dtype != torch.int64
        or not (accept_len.is_contiguous() and bonus.is_contiguous())
    ):
        # the sampled values are ready; finish exactly as stock does from here
        bed.count(NAME, "stock:sampled dtypes")
        from sglang.srt.speculative.dflash_worker_v2 import _commit_accept

        worker._tp_sync.sync(SpecTpSyncSite.DFLASH_SELECTOR, accept_len)
        worker._tp_sync.sync(SpecTpSyncSite.DFLASH_SELECTOR, bonus)
        out_tokens, commit_lens = _commit_accept(cand, accept_len, bonus)
        return _result(
            worker, kw, accept_len, commit_lens, bonus, out_tokens, cap_trim_lens
        )
    if comm is not None and _consensus():
        # verify_chain made every rank compute rank 0's decision (rank 0's draws on identical top-k lists): the
        # broadcast would deliver the values every rank already holds, so the commit runs locally
        comm = None
        bed.count(NAME, "consensus")
    out_tokens, commit_lens = commit(cand, accept_len, bonus, comm)
    bed.count(NAME, f"fused:bs={cand.shape[0]}:exchange={int(comm is not None)}")
    return _result(
        worker, kw, accept_len, commit_lens, bonus, out_tokens, cap_trim_lens
    )


def register() -> bool:
    if not enabled():
        return False
    ok, why = pins_match()
    if not ok:
        bed.report(NAME, False, reason=why)
        return False
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    def hook(original, *args, **kwargs):
        if len(args) != 1:
            return original(*args, **kwargs)
        return accept_block(original, args[0], **kwargs)

    HookRegistry.register(TARGET_ACCEPT, hook, HookType.AROUND)
    bed.report(NAME, True)
    return True
