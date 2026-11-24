# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Component ``ringbal`` (env ``MUSE_VR_RINGBAL=1``, default off; needs ``moe_ld`` with ``moe_ldw``): per-step placement of the
routed NVFP4 experts over a fixed ring of replicas for MoE-LD-W calls at expert-parallel world 4.

Replica set (fixed arithmetic, ``ring_place``): rank r holds its own experts 64 r .. 64 r + 63 in pool slots 0..63 and its
predecessor's (rank r - 1 mod 4) local experts 48..63 in slots 64..79. Each step, the MoE-LD-W prologue of the ring build
(``-DMUSE_VR_LDW_RBAL=5``) computes the placement from the routed ids it already reads: every one of those 16 experts per rank
runs on exactly one of its two holders, and every other expert on its home rank.

Hooks (AROUND, by attribute replacement):

* ``MuseSpark1xFp4Experts.__init__``: an eligible layer's per-expert parameters get storage for 80 slots; the parameters stay
  views of the first 64, so every path other than the ring call (FlashInfer, MoE-LD, prefill, the board consumers) sees the
  stock experts, count and offset.
* ``_copy_expert_stack`` of the causal-LM model: after the stock copy of this rank's experts from the full checkpoint stack,
  the predecessor's experts 48..63 go to slots 64..79.
* ``MuseSparkNvFp4FusedMoEMethod.process_weights_after_loading`` (around ``moe_ld``'s wrapper): the method prepares all 80
  slots (under per-token activation scaling each expert's preparation depends on that expert only), then every per-expert
  parameter returns to a view of its first 64 slots. A load-time check over the EP group digests every prepared per-expert
  tensor: each replica slot must equal its home expert's prepared bytes on the predecessor. Any mismatch turns the component
  off for the process (every MoE-LD-W call then runs the stock build on slots 0..63), loudly.
* ``full(w13, ...)``: MoE-LD-W's call reads the 80-slot tensors of the layer whose ``gate_up_proj`` starts at ``w13``.
* ``MuseSpark1xModel.forward``: the forward batch is kept in a context variable for the forward's duration, so a ring call
  can hand the kernel the batch's real-token count (``real_rows``): SGLang's ``num_token_non_padded``, an int32 device scalar
  that a CUDA-graph replay rewrites every step. The placement is computed from the routed ids of the real tokens only: the
  padding rows a replay adds (their token ids are left from earlier steps) are still placed and computed, but never move the
  placement of a real row. A ring call inside a capture without that count keeps its static placement (the plain MoE-LD-W
  build on slots 0..63).

Engagement pass (``MUSE_VR_RINGBAL_CHECK=1``, diagnostic, never on a timed run): every ring call made outside graph capture
also hands the kernel a stamps buffer, synchronizes, and compares every CTA's placement digest (thresholds + winning key)
with the host reference computed from the same routed ids (tallies ``check:digest_ok`` / ``check:digest_MISMATCH``; a
mismatch turns the component off for the process). Every rank computes the same reference from the same ids, so equal
digests on every rank also mean equal placements across ranks.

Numerics: each routed (token, expert) row is bitwise the served row; which rank adds it to its partial before the cross-rank
reduction can change (the grouping of that sum).

Steps aside (reported): expert-parallel world other than 4 or expert tensor parallelism, global experts other than 256, local
experts other than 64, Marlin or MegaMoE experts, per-token activation scaling off, the wide MoE-LD-W build (1024 rows),
``moe_ld`` / ``moe_ldw`` not enabled.
"""

from __future__ import annotations

import contextvars
import logging

from . import bed

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "srt/models/muse_spark.py": (
        "MuseSpark1xModel",
        "MuseSparkTextForCausalLM",
        "get_parallel",
    ),
    "srt/layers/muse_spark/muse_spark_fp4_moe.py": ("MuseSpark1xFp4Experts",),
    "srt/layers/muse_spark/muse_spark_nvfp4_moe.py": ("MuseSparkNvFp4FusedMoEMethod",),
    "srt/distributed/*.py": ("get_moe_ep_group",),
}

logger = logging.getLogger("muse_vr_plugin")

NAME = "ringbal"
# The SGLang sources this component hooks or relies on, at 3117f58 + #48948: the expert-stack copy and the loader that
# calls it, the model forward it wraps, the expert layer whose parameters it grows, the quant method that creates and
# prepares them
PINS = {
    "sglang/srt/models/muse_spark.py": (
        bed.Functions(
            {
                "MuseSparkTextForCausalLM._copy_expert_stack": "88ca5c9c6082e1bb",
                "MuseSparkTextForCausalLM.load_weights": "4f22831eb2ce54d3",
                "MuseSpark1xModel.forward": "5a17d5920ad718b9",
            }
        ),
    ),
    "sglang/srt/layers/muse_spark/muse_spark_fp4_moe.py": (
        bed.Functions({"MuseSpark1xFp4Experts.__init__": "997db56fae18e78f"}),
    ),
    "sglang/srt/layers/muse_spark/muse_spark_nvfp4_moe.py": (
        bed.Functions(
            {
                "MuseSparkNvFp4FusedMoEMethod.create_weights": "af134d51a5c94cb7",
                "MuseSparkNvFp4FusedMoEMethod.process_weights_after_loading": "b60324c6c91216a2",
            }
        ),
    ),
}
FLAGS = (
    "-DMUSE_VR_LDW_RBAL=5",
)  # the MoE-LD-W build with the in-prologue ring placement
WORLD, BLOCK, MOV_LO, POOL, NUM_EXPERTS = 4, 64, 48, 80, 256
# the prepared per-expert tensors the ring call reads (FlashInfer argument -> layer attribute)
CALL_TENSORS = (
    "gate_up_proj",
    "gate_up_proj_weight_scale",
    "down_proj",
    "down_proj_weight_scale",
    "g1_alphas",
    "g1_scale_c",
    "g2_alphas",
)

_state: dict = {
    "engaged": None,
    "storage": {},
    "full": {},
    "layers": 0,
    "stacks": 0,
    "verified": 0,
    "skipped": "",
    "off": "",
}


_FORWARD_BATCH = contextvars.ContextVar("muse_vr_ringbal_forward_batch", default=None)


def real_rows(device):
    """The running forward's real-token count as an int32 [1] tensor on ``device`` (its ``num_token_non_padded``), or None:
    outside a CUDA-graph capture an unpadded call then counts all its rows; inside a capture the caller keeps the static
    placement."""
    import torch

    fb = _FORWARD_BATCH.get()
    t = getattr(fb, "num_token_non_padded", None) if fb is not None else None
    if (
        torch.is_tensor(t)
        and t.dtype == torch.int32
        and t.numel() == 1
        and t.device == device
    ):
        return t.reshape(1)
    return None


def _parallel():
    import sglang.srt.models.muse_spark as ms

    return ms.get_parallel()


def _eligible(experts) -> str:
    """Empty when this routed-expert layer takes the ring placement, else why not."""
    par = _parallel()
    if int(par.moe_ep_size) != WORLD or int(getattr(par, "moe_tp_size", 1)) != 1:
        return f"needs expert-parallel world {WORLD} without expert TP (ep {par.moe_ep_size})"
    if (
        int(getattr(experts, "global_num_experts", -1)) != NUM_EXPERTS
        or int(experts.num_local_experts) != BLOCK
    ):
        return "expert counts differ from 256 / 64"
    if int(experts.local_expert_offset) != BLOCK * int(par.moe_ep_rank):
        return "local expert offset is not 64 x rank"
    if getattr(experts, "use_marlin", False):
        return "marlin expert path"
    if not bool(
        getattr(
            getattr(experts, "quant_config", None), "use_per_token_activation", False
        )
    ):
        return "per-token activation scaling is off"
    return ""


def _pred_rows(rank: int) -> slice:
    """Checkpoint rows of the predecessor's experts 48..63 (slots 64..79 of ``rank``)."""
    pred = (rank + WORLD - 1) % WORLD
    return slice(BLOCK * pred + MOV_LO, BLOCK * pred + BLOCK)


def _grow(experts) -> int:
    """Every direct parameter whose leading dimension is the local expert count gets 80-slot storage; the parameter becomes a
    view of the first 64 slots. Returns the count."""
    import torch

    n = 0
    with torch.no_grad():
        for _, p in list(experts.named_parameters(recurse=False)):
            if p.dim() >= 1 and p.shape[0] == BLOCK:
                store = torch.empty(
                    (POOL,) + tuple(p.shape[1:]), dtype=p.dtype, device=p.device
                )
                store[:BLOCK].copy_(p.data)
                store[BLOCK:].copy_(
                    p.data[MOV_LO:]
                )  # placeholders until the loader fills the replica slots
                p.data = store[:BLOCK]
                _state["storage"][id(p)] = store
                n += 1
    return n


def _copy_replicas(param, loaded_weight, name: str) -> bool:
    """Slots 64..79 of a grown parameter from the full checkpoint stack; False when ``param`` is not grown."""
    import torch

    store = _state["storage"].get(id(param))
    if store is None:
        return False
    if loaded_weight.shape[0] != NUM_EXPERTS or tuple(loaded_weight.shape[1:]) != tuple(
        store.shape[1:]
    ):
        raise RuntimeError(
            f"ringbal: unexpected expert stack shape for {name}: slots {tuple(store.shape)}, checkpoint "
            f"{tuple(loaded_weight.shape)}"
        )
    with torch.no_grad():
        store[BLOCK:].copy_(loaded_weight[_pred_rows(int(_parallel().moe_ep_rank))])
    _state["stacks"] += 1
    return True


def _digest_rows(t, rows: slice):
    """int64 [rows, 2] digests of the bytes of the given leading-dimension slots: (byte sum, sum of byte x (position mod
    65521 + 1)). Equal bytes give equal digests on every rank. One slot at a time (bounded temporary memory)."""
    import torch

    b = t.detach().reshape(t.shape[0], -1)
    out = []
    w = None
    for i in range(*rows.indices(t.shape[0])):
        v = b[i].contiguous().view(torch.uint8).to(torch.int64)
        if w is None or w.numel() != v.numel():
            w = torch.arange(v.numel(), device=v.device, dtype=torch.int64) % 65521 + 1
        out.append(torch.stack([v.sum(), (v * w).sum()]))
    return torch.stack(out)


def local_table(full: dict, rank: int):
    """This rank's rows of the replica-check table, int64 [4, 2, 16, 2 x tensors]: [rank, 0] digests the replica slots 64..79,
    [rank, 1] the own experts 48..63; every other entry is 0 (the EP group sums the tables)."""
    import torch

    names = sorted(full)
    k = BLOCK - MOV_LO
    table = torch.zeros(
        WORLD, 2, k, 2 * len(names), dtype=torch.int64, device=full[names[0]].device
    )
    for i, n in enumerate(names):
        table[rank, 0, :, 2 * i : 2 * i + 2] = _digest_rows(full[n], slice(BLOCK, POOL))
        table[rank, 1, :, 2 * i : 2 * i + 2] = _digest_rows(
            full[n], slice(MOV_LO, BLOCK)
        )
    return table


def tables_ok(table) -> bool:
    """The summed table: every rank's replica digests equal its predecessor's own-expert digests."""
    import torch

    t = table.cpu()
    return all(
        torch.equal(t[r, 0], t[(r + WORLD - 1) % WORLD, 1]) for r in range(WORLD)
    )


def _verify(full: dict) -> bool:
    """Every rank together: each replica slot's prepared bytes equal its home expert's on the predecessor."""
    import torch
    from sglang.srt.distributed import get_moe_ep_group

    table = local_table(full, int(_parallel().moe_ep_rank))
    torch.distributed.all_reduce(table, group=get_moe_ep_group().device_group)
    return tables_ok(table)


def _prepare(orig_pwal, method, layer):
    """The wrapped preparation for a grown layer (see the module docstring)."""
    import torch

    grown = [
        (n, p)
        for n, p in layer.named_parameters(recurse=False)
        if id(p) in _state["storage"]
    ]
    if not grown:
        return orig_pwal(method, layer)
    if getattr(layer, "use_megamoe", False) or getattr(layer, "use_marlin", False):
        with torch.no_grad():
            for _, p in grown:  # back to 64-slot storage of their own
                _state["storage"].pop(id(p))
                p.data = p.data.clone()
        _state["skipped"] = "marlin / megamoe expert path"
        return orig_pwal(method, layer)
    for _, p in grown:
        p.data = _state["storage"].pop(id(p))
    out = orig_pwal(method, layer)
    full = {}
    for n, p in list(layer.named_parameters(recurse=False)):
        if p.dim() >= 1 and p.shape[0] == POOL:
            full[n] = p.data
            p.data = p.data[:BLOCK]
    missing = [n for n in CALL_TENSORS if n not in full]
    if missing:
        _state["off"] = f"prepared tensors missing: {missing}"
    elif not _verify(full):
        _state["off"] = "replica digest mismatch"
        logger.error(
            "muse_vr_plugin: ringbal replica digest MISMATCH; the ring placement is off for this process"
        )
    else:
        _state["full"][full["gate_up_proj"].data_ptr()] = {
            n: full[n] for n in CALL_TENSORS
        }
        _state["verified"] += 1
    _report()
    return out


def _report() -> None:
    bed.report(
        NAME,
        _state["verified"] > 0 and not _state["off"],
        layers=_state["layers"],
        stacks=_state["stacks"],
        verified=_state["verified"],
        off=_state["off"] or "-",
        skipped=_state["skipped"] or "-",
    )


def check_stamps(device):
    """A stamps buffer for an engagement-pass call (32 words per SM), or None outside the pass / during graph capture."""
    import torch

    if not bed.flag("ringbal_check") or torch.cuda.is_current_stream_capturing():
        return None
    from . import ring_place as rp

    nsm = torch.cuda.get_device_properties(device).multi_processor_count
    return torch.zeros(nsm * rp.STAMP_WORDS, dtype=torch.int64, device=device)


def check_digest(packed, extra: tuple, stamps, n_real=None) -> bool:
    """Engagement pass: every CTA that ran wrote the host reference's placement digest (the reference counts the routed ids of
    the call's real tokens, as the kernel does)."""
    import torch

    from . import ring_place as rp
    from .ops import moe_ldw

    pr = moe_ldw.PASS_ROWS
    for f in extra:
        if f.startswith("-DMUSE_VR_LDW_PASS_ROWS="):
            pr = int(f.split("=", 1)[1])
    torch.cuda.synchronize()
    if n_real is not None:
        packed = packed[
            : max(0, min(int(n_real.reshape(-1)[0].item()), packed.shape[0]))
        ]
    want = rp.digest(*rp.place_form(rp.hist_of(packed), pr, 5))
    words = [
        w & 0xFFFFFFFFFFFFFFFF
        for w in stamps.view(-1, rp.STAMP_WORDS)[:, 23].tolist()
        if w != 0
    ]
    ok = bool(words) and all(w == want for w in words)
    bed.count(NAME, "check:digest_ok" if ok else "check:digest_MISMATCH")
    if not ok:
        _state["off"] = "engagement-pass digest mismatch"
        logger.error(
            "muse_vr_plugin: ringbal placement digest MISMATCH; the ring placement is off for this process"
        )
    return ok


def full(w13, p: int, local_expert_offset: int):
    """The 80-slot tensors for a MoE-LD-W call on the layer whose gate_up_proj view is ``w13``, or None (stock call)."""
    if not _state["engaged"] or _state["off"] or p != BLOCK or w13.shape[0] != BLOCK:
        return None
    t = _state["full"].get(w13.data_ptr())
    if (
        t is None
        or local_expert_offset % BLOCK
        or not 0 <= local_expert_offset < NUM_EXPERTS
    ):
        return None
    return t


def register() -> bool:
    if _state["engaged"] is not None:
        return _state["engaged"]
    if not bed.enabled(NAME):
        _state["engaged"] = False
        return False
    if not (bed.enabled("moe_ld") and bed.enabled("moe_ldw")):
        bed.report(NAME, False, reason="needs MUSE_VR_MOE_LD=1 and MUSE_VR_MOE_LDW=1")
        _state["engaged"] = False
        return False
    from .moe_ld_ops import form_l

    if not form_l():  # the ring build and the 80-slot preparation belong to the MoE-LD of MUSE_VR_MOE_LD_FORM=l
        bed.report(NAME, False, reason="needs MUSE_VR_MOE_LD_FORM=l")
        _state["engaged"] = False
        return False
    if bed.env_int("MUSE_VR_MOE_LDW_MAX_M", 512) != 512:
        bed.report(
            NAME,
            False,
            reason="the ring build takes at most 512 rows (MUSE_VR_MOE_LDW_MAX_M)",
        )
        _state["engaged"] = False
        return False
    for rel, sha in PINS.items():
        if not bed.pinned(rel, sha):
            bed.report(NAME, False, reason=f"pin mismatch: {rel}")
            _state["engaged"] = False
            return False
    import sglang.srt.layers.muse_spark.muse_spark_fp4_moe as fp4
    import sglang.srt.layers.muse_spark.muse_spark_nvfp4_moe as q
    import sglang.srt.models.muse_spark as ms

    lm = getattr(ms, "MuseSparkTextForCausalLM", None)
    if lm is None or not hasattr(lm, "_copy_expert_stack"):
        bed.report(NAME, False, reason="no _copy_expert_stack on the causal-LM model")
        _state["engaged"] = False
        return False
    model = getattr(ms, "MuseSpark1xModel", None)
    if model is None or not hasattr(model, "forward"):
        bed.report(NAME, False, reason="no MuseSpark1xModel.forward")
        _state["engaged"] = False
        return False
    orig_forward = model.forward
    orig_init = fp4.MuseSpark1xFp4Experts.__init__
    orig_copy = lm._copy_expert_stack
    orig_pwal = q.MuseSparkNvFp4FusedMoEMethod.process_weights_after_loading

    def experts_init(self, *a, **kw):
        orig_init(self, *a, **kw)
        why = _eligible(self)
        if why:
            _state["skipped"] = why
            return
        if _grow(self):
            _state["layers"] += 1

    def copy_expert_stack(self, param, loaded_weight, name):
        out = orig_copy(self, param, loaded_weight, name)
        _copy_replicas(param, loaded_weight, name)
        return out

    def process_weights_after_loading(self, layer):
        return _prepare(orig_pwal, self, layer)

    def model_forward(self, input_ids, positions, forward_batch, *a, **kw):
        tok = _FORWARD_BATCH.set(forward_batch)
        try:
            return orig_forward(self, input_ids, positions, forward_batch, *a, **kw)
        finally:
            _FORWARD_BATCH.reset(tok)

    fp4.MuseSpark1xFp4Experts.__init__ = experts_init
    lm._copy_expert_stack = copy_expert_stack
    q.MuseSparkNvFp4FusedMoEMethod.process_weights_after_loading = (
        process_weights_after_loading
    )
    model.forward = model_forward
    _state["engaged"] = True
    logger.warning(
        "muse_vr_plugin: ringbal hooks installed (pool %d, build %s)", POOL, FLAGS[0]
    )
    return True
