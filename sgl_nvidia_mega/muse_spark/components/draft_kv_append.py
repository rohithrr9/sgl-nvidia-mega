# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Component ``draft_kv_append`` (env ``MUSE_VR_DRAFT_KV_APPEND=1``, default off).

After every verify, SGLang's DFLASH worker writes the committed tokens' context K/V into the draft KV cache
(``DFlashWorkerV2._append_target_hidden_fused``): one batched projection and one norm + RoPE kernel for all draft
layers (``FusedKVMaterializeHelper.materialize``), then one prefix-valid write kernel per draft layer. This component
keeps the projection and the norm + RoPE and writes every layer's rows in one kernel (``csrc/draft_kv_append.cuh``):
the same rows (block positions below each request's commit length) to the same slots, as plain copies of the same
bf16 rows (tests/test_draft_kv_append.py).

Engaged only for the prefix-valid path (``ctx_cache_loc_2d`` and ``commit_lens`` given), a plain ``MHATokenToKVPool``
draft pool (or ``kv_draftwin``'s sliding-window draft pool: its sliding-window sub-pool, written at the window ids the
writer passes, ``ctx_swa_cache_loc`` on the newer SGLang, else at the slots its full->SWA mapping names) whose cache and
store dtype equal the rows' dtype (bf16, no quantization), equal K/V row widths, and per-layer
rows that come from one workspace at a constant layer stride. Everything else runs stock. Eager only (this region is not
graph-captured).
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
    "srt/mem_cache/memory_pool.py": ("MHATokenToKVPool",),
    "srt/plugins/hook_registry.py": ("HookRegistry", "HookType"),
}


logger = logging.getLogger(__name__)

NAME = "draft_kv_append"  # env MUSE_VR_DRAFT_KV_APPEND
TARGET = (
    "sglang.srt.speculative.dflash_worker_v2.DFlashWorkerV2._append_target_hidden_fused"
)
_CUH = Path(__file__).resolve().parent / "csrc" / "draft_kv_append.cuh"

PINS = {
    "sglang/srt/speculative/dflash_worker_v2.py": (
        "7d462f5f175828bcc3b4d47fcdfa33cb6af4b14669ef016eeb79b1a5191eec8c",
        "8f82de8d666f81f3c1aa43dc651006f4182ff58593fdfe4eea28c5ee2a14bfc0",
        # The hooked writer. The newer one writes through the pool's layer router, which for a plain pool is the
        # same set_kv_buffer_prefix_valid call, and also takes the window ids (ctx_swa_cache_loc).
        bed.Functions(
            {
                "DFlashWorkerV2._append_target_hidden_fused": (
                    "c40ffaf1f6f3074f",
                    "d3353a4a8205b48e",
                )
            }
        ),
    ),
    "sglang/srt/mem_cache/memory_pool.py": (
        "b439441448d1112952876cd5f814ea47fcd561ece1a9aa0e7d2080d5f4874106",
        "f3e12b293f3bdce21604ca5e5b473c07dcb6c8e4a34b582c4498d0bd45a10900",
        # The prefix-valid write the kernel replaces (bf16 rows: plain copies) and that the fallback calls.
        bed.Functions(
            {
                "MHATokenToKVPool.set_kv_buffer_prefix_valid": (
                    "ed389fc54eab86ff",
                    "01df6269dfafbb3b",
                ),
                "_set_mha_kv_buffer_prefix_valid": "0a53bfe8168c95f4",
                "_set_kv_buffer_prefix_valid_impl": "584bfc3c6cbecf35",
            }
        ),
    ),
    "sglang/kernels/ops/speculative/fused_kv_materialize.py": (
        "b8dd985b7764d3212e9e3045b2daa5be0c481215346ab28f2513f8201e9cbfeb",
        # Rows handed to write_layer_kv in layer order from one workspace; the newer projection multiplies by the
        # weight's transposed view instead of a transposed copy.
        bed.Functions(
            {
                "FusedKVMaterializeHelper.materialize": (
                    "ef373ab24550022e",
                    "6ff3c927067b5a1c",
                )
            }
        ),
    ),
}

_tables: dict = {}


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
def module(pdl: bool):
    from sglang.kernels.jit.utils import load_jit, make_cpp_args

    args = make_cpp_args(pdl)
    return load_jit(
        "muse_vr_draft_kv_append",
        *args,
        cuda_files=[str(_CUH)],
        cuda_wrappers=[("run", f"muse_vr::draft_kv_append::KvAppend<{args}>::run")],
    )


def write_all(rows, k_buffers, v_buffers, loc_2d, commit_lens):
    """``rows``: [(k, v)] per layer (bf16 [n, ...] contiguous, one workspace, constant layer stride); buffers: the
    pool's per-layer K / V tensors. Returns False (nothing written) when the layout does not qualify."""
    k0, v0 = rows[0]
    n_layers = len(rows)
    if n_layers > 1:
        stride = rows[1][0].data_ptr() - k0.data_ptr()
        for i, (k, v) in enumerate(rows):
            if (
                k.data_ptr() - k0.data_ptr() != i * stride
                or v.data_ptr() - v0.data_ptr() != i * stride
            ):
                return False
    else:
        stride = 0
    for k, v in rows:
        if (
            not (k.is_contiguous() and v.is_contiguous())
            or k.shape != k0.shape
            or k.dtype != torch.bfloat16
        ):
            return False
    key = tuple(b.data_ptr() for b in k_buffers) + tuple(
        b.data_ptr() for b in v_buffers
    )
    tab = _tables.get(key)
    if tab is None:
        dev = k0.device
        tab = (
            torch.tensor(
                [b.data_ptr() for b in k_buffers], dtype=torch.int64, device=dev
            ),
            torch.tensor(
                [b.data_ptr() for b in v_buffers], dtype=torch.int64, device=dev
            ),
        )
        _tables[key] = tab
    k2 = k0.reshape(k0.shape[0], -1)
    v2 = v0.reshape(v0.shape[0], -1)
    dst_row_bytes = k_buffers[0].stride(0) * k_buffers[0].element_size()
    if k2.shape[1] * 2 != dst_row_bytes:
        return False
    module(_use_pdl()).run(
        k2,
        v2,
        int(stride),
        tab[0],
        tab[1],
        int(dst_row_bytes),
        loc_2d,
        commit_lens,
        n_layers,
    )
    return True


def _draftwin_pool_class():
    """kv_draftwin's pool class when that component built this process's draft pool, else None."""
    from . import kv_draftwin

    return kv_draftwin.pool_class() if kv_draftwin.plan() is not None else None


def append_fused(original, worker, *args, **kw):
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

    loc_2d, commit_lens = kw.get("ctx_cache_loc_2d"), kw.get("commit_lens")
    pool = worker.draft_model_runner.token_to_kv_pool
    helper = worker._fused_kv_helper
    if args or loc_2d is None or commit_lens is None or helper is None:
        bed.count(NAME, "stock:call form")
        return original(worker, *args, **kw)
    # A draft pool kept in the target's sliding-window slots (kv_draftwin) holds its layers in its sliding-window
    # sub-pool, indexed through the full->SWA mapping: write that sub-pool at the translated slots.
    sub, swa = pool, False
    dw_cls = _draftwin_pool_class()
    if dw_cls is not None and isinstance(pool, dw_cls):
        sub, swa = pool.swa_kv_pool, True
    if (
        type(sub) is not MHATokenToKVPool
        or sub.dtype != torch.bfloat16
        or sub.store_dtype != torch.bfloat16
        or (sub.v_row_dim != sub.row_dim)
    ):
        bed.count(NAME, "stock:pool")
        return original(worker, *args, **kw)
    if loc_2d.dtype != torch.int64:
        loc_2d = loc_2d.to(torch.int64)
    if commit_lens.dtype != torch.int32:
        commit_lens = commit_lens.to(torch.int32)
    loc_2d, commit_lens = loc_2d.contiguous(), commit_lens.contiguous()
    rows = []
    helper.materialize(
        ctx_hidden=kw["ctx_hidden"],
        positions=kw["ctx_positions"],
        write_layer_kv=lambda i, k, v: rows.append((i, k, v)),
    )
    layers = worker.draft_model.layers
    if [r[0] for r in rows] != list(range(len(layers))):
        raise RuntimeError("draft_kv_append: unexpected layer order")
    if swa:
        ids = [
            pool.layers_mapping[layers[i].self_attn.attn.layer_id][0]
            for i, _, _ in rows
        ]
        # the newer writer passes the window ids it writes at (ctx_swa_cache_loc); without them, translate
        swa_loc = kw.get("ctx_swa_cache_loc")
        if swa_loc is None:
            write_loc = pool.kvdw_to_swa(loc_2d)
        else:
            write_loc = swa_loc.view_as(loc_2d).to(torch.int64).contiguous()
    else:
        ids = [layers[i].self_attn.attn.layer_id - pool.start_layer for i, _, _ in rows]
        write_loc = loc_2d
    kb = [sub.k_buffer[j] for j in ids]
    vb = [sub.v_buffer[j] for j in ids]
    if write_all([(k, v) for _, k, v in rows], kb, vb, write_loc, commit_lens):
        bed.count(NAME, f"fused{':swa' if swa else ''}:bs={loc_2d.shape[0]}")
        return None
    # the projection already ran: finish with the stock per-layer writes on the same rows
    bed.count(NAME, "stock:row layout")
    for i, k, v in rows:
        attn = layers[i].self_attn.attn
        pool.set_kv_buffer_prefix_valid(
            attn, loc_2d, commit_lens, k, v, attn.k_scale, attn.v_scale
        )
    return None


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
        return append_fused(original, args[0], **kwargs)

    HookRegistry.register(TARGET, hook, HookType.AROUND)
    bed.report(NAME, True)
    return True
