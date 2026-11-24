# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Component ``kv_draftwin`` (env ``MUSE_VR_KV_DRAFTWIN=1``, default off): DFLASH draft KV in the target's
sliding-window slot space.

Every layer of the DFlash draft is a sliding-window layer, yet SGLang gives a DFLASH draft a plain MHA pool indexed by
the target's full-attention token ids: one draft slot per target token, charged per full token in the pool budget,
and a full-indexed HiCache host sidecar. A draft query at position ``p`` reads keys at ``>= p - window_left`` only,
and the target already keeps those tokens' sliding-window slots: it frees a token's slot once the token is more than
``max(window, page)`` behind the request (page-floored; SGLang main rounds that distance up to whole pages, which keeps
at least as many slots), and the draft block's queries sit at the same positions
as the target's verify block in the same iteration. This component stores the draft KV in that slot space:

* The draft pool is an ``SWAKVPool`` with no full-attention layers and every draft layer on its sliding-window side,
  sized to the target's sliding-window token count and registered on the target allocator's full->SWA mapping. This
  is the plumbing SGLang already uses for sliding-window EAGLE/MTP drafts: the FlashAttention backends translate the
  page table and the write locations, the draft runner's ``RunnerIds`` gets a SLIDING_WINDOW id space, and HiCache
  builds an SWA-indexed draft sidecar whose pages follow the target's sliding-window pages (``build_swa_draft_pools``).
* DFLASH's own draft KV writes (prompt materialization by location, the post-verify prefix-valid commit) carry that
  id space's sliding-window ids (``swa_cache_loc``); a write that reaches the pool with full-attention ids only is
  translated through the same mapping, once per materialization call.
* The pool budget charges the draft's bytes per sliding-window token (``x swa_full_tokens_ratio``) instead of per full
  token, so the memory that held never-read draft KV becomes target tokens.

The draft computes exactly what it computed before: the same projections, norm, RoPE and attention kernels over the
same in-window keys, for every token. Only the storage of tokens that left the window is dropped (those tokens have no
sliding-window slot). Draft attention runs over the same sequence lengths and window as stock; keys outside the window
are skipped or masked by the kernel and contribute exact zeros, so the draft's outputs are bit-identical to stock
(tests/test_kv_draftwin.py).

Engaged only when: DFLASH (not DSPARK) with the fa4 draft attention backend, the target is hybrid sliding-window with
a non-unified, paged, non-ring ``SWATokenToKVPoolAllocator``, every draft layer is a sliding-window layer with
window_left <= the target's, no DCP / PP / disaggregation / unified memory / page-major layout, and the SGLang sources
match their pins. The decision is taken once, when the target pool is sized; the draft pool build re-checks it
against the loaded draft model and raises rather than fall back to a full-size pool the budget did not charge. At the
first draft KV write, every attention backend of the draft runner is checked to read the pool through its
sliding-window page table (``kv_draftwin:attn`` engagement record; raises otherwise).
"""

from __future__ import annotations

import functools
import logging
import threading

from . import bed

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "srt/runtime_context.py": (
        "get_disagg",
        "get_memory",
        "get_parallel",
        "get_spec",
        "get_exec",
    ),
    "srt/speculative/dflash_utils.py": (
        "get_dflash_attention_sliding_window_size",
        "get_dflash_layer_types",
    ),
    "srt/mem_cache/memory_pool.py": (
        "KVWriteLoc",
        "unwrap_write_loc",
        "write_loc_is_physical",
        "MHATokenToKVPool",
    ),
    "srt/mem_cache/swa_memory_pool.py": ("SWAKVPool",),
    "srt/layers/radix_attention.py": ("RadixAttention",),
    "srt/mem_cache/allocator/swa.py": ("SWATokenToKVPoolAllocator",),
    "srt/plugins/hook_registry.py": ("HookRegistry", "HookType"),
}


logger = logging.getLogger(__name__)

NAME = "kv_draftwin"  # env MUSE_VR_KV_DRAFTWIN

T_SIZING = (
    "sglang.srt.model_executor.pool_configurator.HybridSWAPoolConfigurator.__init__"
)
T_SWA_CELL = "sglang.srt.model_executor.pool_configurator.HybridSWAPoolConfigurator._swa_cell_bytes"
T_BUILD = "sglang.srt.mem_cache.kv_cache_configurator.KVCacheConfigurator._build_token_to_kv_pool"
T_APPEND = "sglang.srt.speculative.dflash_worker_v2.DFlashWorkerV2._append_target_hidden_to_draft_kv_by_loc"

# The SGLang sources (relative to the directory holding the ``sglang`` package) whose behaviour this relies on, pinned
# by the digests of the functions it hooks, calls or depends on: the DFLASH worker's draft KV writes, the pool /
# allocator / configurator code that builds and sizes the pools, the sliding-window pool and its full->SWA mapping,
# the out-of-window freeing rule, the FlashAttention backend's sliding-window translation, the runner's id spaces and
# the layer router (the index translator's successors), the HiCache sidecar builder and the draft's window convention.
PINS = {
    "sglang/srt/speculative/dflash_worker_v2.py": (
        # SGLang 3117f58 + #48948 with the DFLASH one-launch steps and smx: the draft block and its plan bind in
        # _draft_block; smx's mixed pass writes draft KV in _mixed_verify_extend_result and _mixed_verify_running_result
        bed.Functions(
            {
                "DFlashWorkerV2._append_target_hidden_to_draft_kv_by_loc": "4c2eab09c6743c2d",
                "DFlashWorkerV2._draft_write_ids": "bd6b4184c0c2e851",
                "DFlashWorkerV2._append_target_hidden_decode": "3e8ed52bddd81088",
                "DFlashWorkerV2._append_target_hidden_sequential": "c0f175a24ed32573",
                "DFlashWorkerV2._append_target_hidden_fused": "c40ffaf1f6f3074f",
                "DFlashWorkerV2._draft_kv_full_to_swa_mapping": "8b88568d70254f56",
                "DFlashWorkerV2.forward_batch_generation": "513aab4e868b73a9",
                "DFlashWorkerV2._draft_block": "d0c36d14d397e01d",
                "DFlashWorkerV2.forward_mixed_verify_generation": "8a0eed460a7cf0a0",
                "DFlashWorkerV2._mixed_verify_extend_result": "1e8068b5d304a582",
                "DFlashWorkerV2._mixed_verify_running_result": "0ede791df6525662",
            }
        ),
        bed.Functions(
            {
                "DFlashWorkerV2._append_target_hidden_to_draft_kv_by_loc": "4c2eab09c6743c2d",
                "DFlashWorkerV2._draft_write_ids": "bd6b4184c0c2e851",
                "DFlashWorkerV2._append_target_hidden_decode": "3e8ed52bddd81088",
                "DFlashWorkerV2._append_target_hidden_sequential": "c0f175a24ed32573",
                "DFlashWorkerV2._append_target_hidden_fused": "c40ffaf1f6f3074f",
                "DFlashWorkerV2._draft_kv_full_to_swa_mapping": "8b88568d70254f56",
                "DFlashWorkerV2.forward_batch_generation": "d05d129bac415b00",
            }
        ),
    ),
    "sglang/srt/speculative/dflash_utils.py": (
        bed.Functions(
            {
                "get_dflash_attention_sliding_window_size": "352ff0d705840d21",
                "get_dflash_layer_types": "1476af5b7f476f6a",
            }
        ),
    ),
    "sglang/srt/models/dflash.py": (
        bed.Functions(
            {
                "_get_dflash_layer_attention_params": "e679d4d9479e8970",
                "DFlashAttention.__init__": "00710456bf5598ff",
            }
        ),
    ),
    "sglang/srt/mem_cache/memory_pool.py": (
        bed.Functions(
            {
                "KVWriteLoc.__post_init__": "6eb2d5175b280836",
                "unwrap_write_loc": "85a8f48bd41e6a3e",
                "write_loc_is_physical": "dce2d36b1283495f",
                "layer_window_locs": "3d2af5fc26b11ad3",
                "KVCache.id_spaces": "d3382cd1db764a07",
                "KVCache.layer_router": "a2951e1240c315b6",
                "MHATokenToKVPool.__init__": "19352e82644dfacc",
                "MHATokenToKVPool.set_kv_buffer_prefix_valid": "ed389fc54eab86ff",
            }
        ),
    ),
    "sglang/srt/mem_cache/swa_memory_pool.py": (
        bed.Functions(
            {
                "SWAKVPool.__init__": "3f1c9a5f1d7bc949",
                "SWAKVPool.register_mapping": "6824f04fbe5c103b",
                "SWAKVPool._build_layer_router": "91d33a672b259ae3",
                "SWAKVPool.translate_loc_from_full_to_swa": "70243eb01e970ff6",
                "SWAKVPool._window_loc": "d0c711bd5889df51",
                "SWAKVPool.set_kv_buffer": "cd878f8998497793",
                "SWAKVPool.finalize_backing": "09a05b6f2c207f9a",
            }
        ),
    ),
    "sglang/srt/mem_cache/allocator/swa.py": (
        bed.Functions(
            {
                "SWATokenToKVPoolAllocator.__init__": "c838829da5e8d982",
                "SWATokenToKVPoolAllocator.swa_req_ring": "f099a89ea5170d7a",
                "SWATokenToKVPoolAllocator.alloc_extend": "e1bb9f7e20755438",
                "SWATokenToKVPoolAllocator.alloc_extend_swa_tail": "1aeba2b5126a26c6",
                "SWATokenToKVPoolAllocator.alloc_decode": "39719026a48f194b",
                "SWATokenToKVPoolAllocator.set_full_to_swa_mapping": "660dde1e7986f4bf",
                "SWATokenToKVPoolAllocator.clear_full_to_swa_mapping": "8e6b7b314385284a",
                "SWATokenToKVPoolAllocator.free_swa": "f695877b69b4b551",
                "SWATokenToKVPoolAllocator.free_swa_segment": "8868db8e30129b64",
                "SWATokenToKVPoolAllocator.resize": "c4cd6b3f826728c8",
                "SWATokenToKVPoolAllocator.clear": "38778eb4cfc31bca",
            }
        ),
    ),
    "sglang/srt/mem_cache/common.py": (
        # The out-of-window freeing rule moved to kv/advance.py and kv/alloc/demand.py (pinned below).
        bed.Functions({}),
    ),
    "sglang/srt/mem_cache/kv/advance.py": (
        bed.Functions({"retire_window_rows": "a84251d975d2c05e"}),
    ),
    "sglang/srt/mem_cache/kv/alloc/demand.py": (
        bed.Functions({"retained_window_tokens": "f7f533ae28932995"}),
    ),
    "sglang/srt/mem_cache/kv_cache_configurator.py": (
        bed.Functions(
            {
                "KVCacheConfigurator.__post_init__": "416dc38de735ecce",
                "KVCacheConfigurator.loc_space_scale": "feca89c797adda3c",
                "KVCacheConfigurator.pool_page_size": "ebf2c0867ed42812",
                "KVCacheConfigurator._init_pools": "bbc400d8e1c09b3f",
                "KVCacheConfigurator._build_token_to_kv_pool": "a21976ab0cd6645c",
                "KVCacheConfigurator._build_hybrid_swa_kv_pool": "589c8f0e74b6f106",
                "KVCacheConfigurator._build_mha_kv_pool": "8a7f0e1ccfb18a4f",
                "KVCacheConfigurator._resolve_kv_pool_class": "0aae2026f8fe76b9",
                "KVCacheConfigurator._build_token_to_kv_pool_allocator": "4f981c21c2d830b1",
            }
        ),
    ),
    "sglang/srt/model_executor/pool_configurator.py": (
        bed.Functions(
            {
                "_charges_dflash_draft_kv_per_token": "079d8c7975574169",
                "_dflash_draft_cell_size": "ea8d7ae1397b93d4",
                "dflash_draft_kv_follows_target_swa": "b7ab0d7288ab17b5",
                "HybridSWAPoolConfigurator.__init__": "9d05903465f426e0",
                "HybridSWAPoolConfigurator._full_cell_bytes": "947871da13838ec8",
                "HybridSWAPoolConfigurator._swa_cell_bytes": "dad6782e0d792163",
                "HybridSWAPoolConfigurator._recompute_cell_size": "cf1e80ad7cc0d50b",
                "HybridSWAPoolConfigurator._make_pool_config": "857d0b960e07daf1",
                "HybridSWAPoolConfigurator.calculate_pool_sizes": "2d113f3e80677abc",
                "SWARequestCapPoolConfigurator.calculate_pool_sizes": "c14a363568efbcce",
            }
        ),
    ),
    "sglang/srt/model_executor/model_runner_components/kv_pool_runtime.py": (
        bed.Functions(
            {
                "is_post_capture_kv_active": "4e1ae8e7e44ed0dd",
                "compute_post_capture_kv_resize": "57a608c371c6c6ae",
            }
        ),
    ),
    "sglang/srt/layers/attention/flashattention_backend.py": (
        bed.Functions(
            {
                "FlashAttentionBackend.__init__": "37e0cc6dacc93c70",
                "FlashAttentionBackend._maybe_init_local_attn_builder": "4074c8cc6c8b0727",
                "FlashAttentionBackend.init_forward_metadata_in_graph": "87561051b6eb2b0d",
                "FlashAttentionBackend._in_graph_full_to_swa_index_mapping": "5081e1eda58a99d0",
                "FlashAttentionBackend.draft_extend_metadata_captured_in_graph": "1adaad9a5dc72f6f",
                "FlashAttentionBackend._get_swa_write_locations": "571f61360b236421",
                "FlashAttentionBackend._init_full_cg_decode_metadata": "04e6d11ff21725a4",
                "FlashAttentionBackend._init_full_cg_prefill_metadata": "6e04645cac0d5032",
                "FlashAttentionBackend.init_forward_metadata": "29f40b8f89eee5b9",
                "FlashAttentionBackend.get_fused_kv_write_buffers": "6d75860f4ef08c02",
                "FlashAttentionBackend.forward_extend": "6ef18b40dc548054",
                "FlashAttentionBackend.forward_decode": "d6c4b0eea2d64858",
                "FlashAttentionBackend.init_cuda_graph_state": "1ee66c94c05cdda5",
                "FlashAttentionBackend._bind_metadata_buffers": "1f4d871f060bfebb",
                "FlashAttentionBackend._set_decode_page_metadata": (
                    "78fcc4ffc9e20462",
                    "71bd15b7241803ab",
                ),
                "FlashAttentionBackend._apply_cuda_graph_metadata": "22db4bd7601e038a",
                "FlashAttentionBackend._init_sliding_window_attn_spec_metadata": "94e6d4ea5504df4d",
            }
        ),
    ),
    # Removed upstream: the sliding-window id space is the pool's (BaseSWAKVPool.id_spaces), resolved per runner by
    # RunnerIds / IterationIds and written through the layer router (pinned below).
    "sglang/srt/mem_cache/kv_index_translator.py": ("absent",),
    "sglang/srt/mem_cache/base_swa_memory_pool.py": (
        bed.Functions({"BaseSWAKVPool.id_spaces": "1fb54bf10cedfd05"}),
    ),
    "sglang/srt/mem_cache/kv/alloc/runner_ids.py": (
        bed.Functions(
            {
                "_target_swa_space": "b7728413b9ac11b0",
                "RunnerIds.__init__": "0a3307e760c36cd0",
                "RunnerIds.sub_pool_name": "4f92e59e42267804",
                "RunnerIds.space": "7b390f444acf6684",
                "RunnerIds.ids_in": "5d6bc66bfaad0f9d",
                "RunnerIds.window_storage": "8962dfa20e91fab8",
                "RunnerIds.write_ids": "ef624e6b75bc362c",
                "RunnerIds.read_table": "3b76b6a5230e2c29",
            }
        ),
    ),
    "sglang/srt/mem_cache/kv/alloc/iteration_ids.py": (
        bed.Functions(
            {
                "IterationIds.bind": "8101462174fe0a6c",
                "IterationIds.write_ids": "e58d6011b3096224",
                "IterationIds._window_in": "3da794bc0c20e1cd",
            }
        ),
    ),
    "sglang/srt/mem_cache/kv/router.py": (
        bed.Functions(
            {
                "KVLayer.store": "8421bfb2701a0177",
                "KVLayer.store_prefix": "f943b62840142a9d",
                "KVLayer._write_loc": "a08be60b12a113f2",
                "KVLayerRouter.from_layer_maps": "17b674f74e5c75a4",
                "KVLayerRouter.routes_to": "dbcc41872ab5044a",
            }
        ),
    ),
    "sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py": (
        bed.Functions(
            {
                "build_swa_draft_pools": "a6109f4b22f817bc",
                "build_hicache_draft_sidecars": "f6adaff08539d32f",
            }
        ),
    ),
    "sglang/srt/mem_cache/kv_cache_builder.py": (
        bed.Functions({"maybe_register_hicache_draft": "fa9eebe0c315d774"}),
    ),
    "sglang/srt/speculative/base_spec_worker.py": (
        bed.Functions(
            {
                "BaseSpecWorker._draft_model_runners": "2db3de39d47a040f",
                "BaseSpecWorker._build_hicache_draft_plan": "833cbbcaa7803eb0",
            }
        ),
    ),
}

# plan: the decision taken when the target pool is sized (None until then, or when not engaged)
_state: dict = {"plan": None, "pool": None}
_tls = threading.local()


def enabled() -> bool:
    return bed.enabled(NAME)


def pins_match() -> tuple[bool, str]:
    for rel, want in PINS.items():
        if not bed.pinned(rel, want):
            return False, f"{rel} differs from the pinned revision"
    return True, "ok"


def plan() -> dict | None:
    """The engagement plan of this process (None when not engaged)."""
    return None if _state["plan"] is None else dict(_state["plan"])


# --------------------------------------------------------------------------------------------------------------------
# decision (target pool sizing)
# --------------------------------------------------------------------------------------------------------------------


def decide(kvc) -> tuple[dict | None, str]:
    """(plan, reason) for a target ``KVCacheConfigurator``; plan is None when the component must stay off."""
    from sglang.srt.runtime_context import (
        get_disagg,
        get_memory,
        get_parallel,
        get_spec,
    )

    if kvc.is_draft_worker:
        return None, "draft configurator"
    if not kvc.spec_algorithm.is_dflash():
        return None, "speculative algorithm is not DFLASH"
    backend = get_spec().speculative_draft_attention_backend
    if backend != "fa4":
        return None, f"draft attention backend {backend!r} (checked: fa4)"
    memory = get_memory()
    if memory.enable_unified_memory:
        return None, "unified memory pool"
    if getattr(memory, "enable_page_major_kv_layout", False):
        return None, "page-major KV layout"
    parallel = get_parallel()
    if parallel.attn_dcp_size > 1 or parallel.pp_size > 1:
        return None, "DCP or pipeline parallelism"
    if get_disagg().disaggregation_mode != "null":
        return None, "disaggregation"
    if kvc.page_size <= 1:
        return None, "page size 1"
    info = kvc.layer_info
    if (
        not kvc.is_hybrid_swa
        or not info.swa_attention_layer_ids
        or not info.full_attention_layer_ids
    ):
        return None, "target is not a hybrid sliding-window model"
    target_window = kvc.sliding_window_size
    if target_window is None or target_window <= 0:
        return None, "target sliding window unknown"
    aux = kvc.spec_aux_config
    draft_config = getattr(aux, "draft_model_config", None)
    n_layers = getattr(aux, "dflash_draft_num_layers", None)
    cell = kvc.dflash_draft_cell_size_per_token
    if draft_config is None or not n_layers or not cell:
        return None, "draft geometry unresolved"
    from sglang.srt.speculative.dflash_utils import (
        get_dflash_attention_sliding_window_size,
        get_dflash_layer_types,
    )

    hf = draft_config.hf_config
    types = get_dflash_layer_types(hf)
    if (
        not types
        or len(types) < int(n_layers)
        or any(t != "sliding_attention" for t in types[: int(n_layers)])
    ):
        return None, "draft has non-sliding-window layers"
    draft_window = get_dflash_attention_sliding_window_size(hf)
    if draft_window is None or draft_window <= 0 or draft_window > target_window:
        return (
            None,
            f"draft window_left {draft_window} exceeds the target's {target_window}",
        )
    return {
        "draft_window_left": int(draft_window),
        "target_window_left": int(target_window),
        "draft_layers": int(n_layers),
        "draft_bytes_per_token": int(cell),
    }, "ok"


def _sizing_init(original, self, kvc, *args, **kwargs):
    original(self, kvc, *args, **kwargs)
    if kvc.is_draft_worker:
        return
    p, why = decide(kvc)
    if p is None:
        bed.report(NAME, False, reason=why)
        return
    if (
        int(self._draft_cell_size) != p["draft_bytes_per_token"]
        or getattr(self, "_fused_full_entry", None) is not None
    ):
        bed.report(
            NAME,
            False,
            reason=f"unexpected draft charge {self._draft_cell_size} / fused entry",
        )
        return
    # Charge the draft per sliding-window token instead of per full token.
    self._draft_cell_size = 0
    self._kvdw_draft_swa_bytes = p["draft_bytes_per_token"]
    self._recompute_cell_size()
    p["cell_bytes_per_full_token"] = float(self._cell_size)
    p["swa_full_tokens_ratio"] = float(self._swa_full_tokens_ratio)
    first = _state["plan"] is None
    _state["plan"] = p
    if first:
        bed.report(NAME, True, **p)
        logger.warning(
            "muse_vr_plugin: kv_draftwin: draft KV in the target's sliding-window slots: %d B/token charged per "
            "sliding-window token (ratio %.3f); pool cell %.1f B per full token",
            p["draft_bytes_per_token"],
            p["swa_full_tokens_ratio"],
            p["cell_bytes_per_full_token"],
        )


def _swa_cell_bytes(original, self, *args, **kwargs):
    return original(self, *args, **kwargs) + int(
        getattr(self, "_kvdw_draft_swa_bytes", 0)
    )


# --------------------------------------------------------------------------------------------------------------------
# the draft pool
# --------------------------------------------------------------------------------------------------------------------


class _Scope:
    """Translation cache for one materialization call: every draft layer writes the same locations, so the full->SWA
    gather runs once per call. Keyed by tensor identity and valid only inside the call (no kernel rewrites the
    location tensor between the layers' writes)."""

    def __enter__(self):
        self.prev = getattr(_tls, "scope", None)
        _tls.scope = {}
        return self

    def __exit__(self, *exc):
        _tls.scope = self.prev
        return False


@functools.lru_cache(maxsize=None)
def pool_class():
    import torch
    from sglang.srt.mem_cache.memory_pool import (
        KVWriteLoc,
        unwrap_write_loc,
        write_loc_is_physical,
    )
    from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

    class DraftWindowSWAKVPool(SWAKVPool):
        """DFLASH draft pool: every layer on the sliding-window side, indexed through the target's full->SWA mapping.
        Writes that arrive with full-attention ids only are translated here."""

        kvdw_translations = 0

        def kvdw_to_swa(self, loc):
            scope = getattr(_tls, "scope", None)
            if scope is not None:
                hit = scope.get(id(loc))
                if hit is not None and hit[0] is loc:
                    return hit[1]
            idx = loc if loc.dtype == torch.int64 else loc.to(torch.int64)
            out = self.full_to_swa_index_mapping[idx]
            self.kvdw_translations += 1
            if scope is not None:
                scope[id(loc)] = (
                    loc,
                    out,
                )  # keyed by the caller's tensor; holding it keeps the id unique
            return out

        def set_kv_buffer(
            self, layer, loc_info, cache_k, cache_v, k_scale=1.0, v_scale=1.0
        ):
            loc, window_locs, full_loc = unwrap_write_loc(loc_info)
            if window_locs is None:
                loc_info = KVWriteLoc(
                    loc,
                    window_locs={self.WINDOW_SUB_POOL: self.kvdw_to_swa(loc)},
                    full_loc=full_loc,
                    physical=write_loc_is_physical(loc_info),
                )
            return super().set_kv_buffer(
                layer, loc_info, cache_k, cache_v, k_scale, v_scale
            )

        def set_kv_buffer_prefix_valid(
            self,
            layer,
            loc_2d,
            commit_lens,
            cache_k,
            cache_v,
            k_scale=None,
            v_scale=None,
            layer_id_override=None,
        ):
            layer_id = (
                layer.layer_id if layer_id_override is None else layer_id_override
            )
            index, is_swa = self.layers_mapping[layer_id]
            if not is_swa:
                raise RuntimeError(
                    f"kv_draftwin: draft layer {layer_id} is not on the sliding-window side"
                )
            return self.swa_kv_pool.set_kv_buffer_prefix_valid(
                layer,
                self.kvdw_to_swa(loc_2d),
                commit_lens,
                cache_k,
                cache_v,
                k_scale,
                v_scale,
                layer_id_override=index,
            )

    return DraftWindowSWAKVPool


def _draft_attention_layers(model):
    from sglang.srt.layers.radix_attention import RadixAttention

    return [m for m in model.modules() if isinstance(m, RadixAttention)]


def build_draft_pool(kvc, p: dict):
    """The draft's ``DraftWindowSWAKVPool`` for a draft ``KVCacheConfigurator`` (raises when the plan does not hold)."""
    import torch
    from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
    from sglang.srt.runtime_context import get_exec, get_parallel

    alloc = kvc.token_to_kv_pool_allocator
    if type(alloc) is not SWATokenToKVPoolAllocator or alloc.swa_req_ring:
        raise RuntimeError(
            f"kv_draftwin: target allocator {type(alloc).__name__} is not a paged SWA allocator"
        )
    if kvc.loc_space_scale != 1:
        raise RuntimeError("kv_draftwin: widened location space")
    cfg = kvc.memory_pool_config
    full_tokens, swa_tokens = (
        cfg.full_max_total_num_tokens,
        cfg.swa_max_total_num_tokens,
    )
    if not full_tokens or not swa_tokens:
        raise RuntimeError(
            f"kv_draftwin: pool config has no full/SWA split ({full_tokens}, {swa_tokens})"
        )
    attns = _draft_attention_layers(kvc.model)
    ids = [a.layer_id for a in attns]
    if len(attns) != p["draft_layers"] or sorted(ids) != list(range(len(attns))):
        raise RuntimeError(
            f"kv_draftwin: draft attention layers {ids} do not match the plan ({p['draft_layers']})"
        )
    for a in attns:
        w = a.sliding_window_size
        if (
            w is None
            or w < 0
            or w != p["draft_window_left"]
            or w > p["target_window_left"]
        ):
            raise RuntimeError(
                f"kv_draftwin: draft layer {a.layer_id} window_left {w} does not match the plan"
            )
    mha_cls = kvc._resolve_kv_pool_class(kind="mha", default=MHATokenToKVPool)
    if mha_cls is not MHATokenToKVPool:
        raise RuntimeError(f"kv_draftwin: platform KV pool class {mha_cls.__name__}")
    parallel = get_parallel()
    head_num = kvc.model_config.get_num_kv_heads(
        parallel.attn_tp_size, parallel.attn_dcp_size
    )
    elem = torch._utils._element_size(kvc.kv_cache_dtype)
    per_token = (
        head_num
        * (kvc.model_config.head_dim + kvc.model_config.v_head_dim)
        * len(attns)
        * elem
    )
    if per_token != p["draft_bytes_per_token"]:
        raise RuntimeError(
            f"kv_draftwin: draft pool bytes/token {per_token} differ from the budget's {p['draft_bytes_per_token']}"
        )
    pool = pool_class()(
        size=full_tokens,
        size_swa=swa_tokens,
        page_size=kvc.pool_page_size,
        dtype=kvc.kv_cache_dtype,
        head_num=head_num,
        head_dim=kvc.model_config.head_dim,
        v_head_dim=kvc.model_config.v_head_dim,
        swa_attention_layer_ids=sorted(ids),
        full_attention_layer_ids=[],
        device=kvc.device,
        token_to_kv_pool_class=MHATokenToKVPool,
        enable_memory_saver=get_exec().features.enable_memory_saver,
        post_capture_active=kvc.post_capture_kv_active,
        enable_kv_cache_copy=True,
    )
    pool.register_mapping(alloc.full_to_swa_index_mapping)
    return pool


def _build_pool(original, self, *args, **kwargs):
    p = _state["plan"]
    if not self.is_draft_worker or p is None or not self.spec_algorithm.is_dflash():
        return original(self, *args, **kwargs)
    pool = build_draft_pool(
        self, p
    )  # raises: the budget already left no room for a full-size draft pool
    _state["pool"] = pool
    info = dict(
        swa_tokens=int(pool.size_swa),
        full_tokens=int(pool.size),
        layers=int(pool.swa_layer_nums),
        bytes_per_swa_token=p["draft_bytes_per_token"],
    )
    bed.report(NAME + ":pool", True, **info)
    return pool


def check_draft_backends(runner) -> list[str]:
    """Runtime self-check: every attention backend of the draft runner reads the draft pool through its sliding-window
    page table (the FlashAttention backend decides this once, from the pool type, when it is built). Raises otherwise:
    a backend that indexed this pool with full-attention ids would read the wrong slots."""
    names = []
    for backend in {
        id(b): b
        for b in (
            getattr(runner, "attn_backend", None),
            getattr(runner, "decode_attn_backend", None),
        )
        if b is not None
    }.values():
        if getattr(backend, "use_sliding_window_kv_pool", None) is not True:
            raise RuntimeError(
                f"kv_draftwin: draft attention backend {type(backend).__name__} does not read the "
                "draft pool through the sliding-window page table"
            )
        names.append(type(backend).__name__)
    if not names:
        raise RuntimeError(
            "kv_draftwin: the draft runner has no attention backend to check"
        )
    return names


def _append(original, self, *args, **kwargs):
    if (
        _state["pool"] is None
        or self.draft_model_runner.token_to_kv_pool is not _state["pool"]
    ):
        return original(self, *args, **kwargs)
    if not _state.get("backends"):
        _state["backends"] = check_draft_backends(self.draft_model_runner)
        bed.report(NAME + ":attn", True, backends=_state["backends"])
    with _Scope():
        return original(self, *args, **kwargs)


def register() -> bool:
    if not enabled():
        return False
    ok, why = pins_match()
    if not ok:
        bed.report(NAME, False, reason=why)
        return False
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(T_SIZING, _sizing_init, HookType.AROUND)
    HookRegistry.register(T_SWA_CELL, _swa_cell_bytes, HookType.AROUND)
    HookRegistry.register(T_BUILD, _build_pool, HookType.AROUND)
    HookRegistry.register(T_APPEND, _append, HookType.AROUND)
    bed.report(NAME + ":hooks", True)
    return True
