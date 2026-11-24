# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU contracts for fp8_prefill's mixed-pass form (``MUSE_VR_FP8_PREFILL_MIXED``, ``components/fp8_prefill.py``) on
SGLang's DFLASH mixed verify layout (``ForwardBatch.mixed_verify``); the pass rules are those of
``tests/test_fp8_prefill_mixed.py``.

- The pass gate is the same for pure and mixed passes (rows > MIN_TOKENS, EXTEND): a mixed pass above the threshold is an
  FP8 pass, counted ``pass:mixed:<bucket>`` with its block rows (``bs x block``) in ``rows:block``; below it, it is not.
- Outside a mixed pass (no layout, or an SGLang without the field) or with the knob off, no mixed key is counted.
- Decode / target-verify forwards never set the pass, whatever their row count.
- Install: mixed verify on (SGLang's switch or ``smx``) without the knob refuses; the knob needs MIN_TOKENS >= 1024.
"""

from __future__ import annotations

import types

import pytest
import torch
from sglang.srt.model_executor.forward_batch_info import ForwardMode as FM

from sgl_nvidia_mega.muse_spark.components import bed
from sgl_nvidia_mega.muse_spark.components import fp8_prefill as fp
from sgl_nvidia_mega.muse_spark.components import smx

# a forward batch without the field (an SGLang without DFLASH mixed verify)
ABSENT = object()
REFUSED_NEXT_TO_SMX = (
    "refused next to smx (mixed passes carry verify rows through the prefill graphs)"
)


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    """Fresh engagement records; fp8_prefill's state restored after the test; no knob from the environment."""
    monkeypatch.setattr(bed, "_installs", {})
    monkeypatch.setattr(bed, "_counts", {})
    for key in ("ctx", "min_tokens", "pass", "mixed"):
        monkeypatch.setitem(fp._state, key, fp._state.get(key, False))
    for var in (
        "MUSE_VR_ENGAGE_DIR",
        "MUSE_VR_PLUGIN_DISABLE",
        "MUSE_VR_SMX",
        "MUSE_VR_FP8_PREFILL",
        "MUSE_VR_FP8_PREFILL_MIXED",
        "MUSE_VR_FP8_PREFILL_MIN_TOKENS",
        "MUSE_VR_FP8_PREFILL_GROUPS",
        "MUSE_VR_FP8_PREFILL_WEIGHTS",
        "MUSE_VR_FP8_PREFILL_RESIDENT_LAYERS",
        "MUSE_VR_FP8_PREFILL_B3",
        smx.SWITCH,
    ):
        monkeypatch.delenv(var, raising=False)
    bed.env_int.cache_clear()
    yield
    bed.env_int.cache_clear()


def _layout(n_p, P, bs, block):
    """The row counts of SGLang's MixedVerifyLayout the component reads."""
    return types.SimpleNamespace(
        num_extend_reqs=n_p, num_extend_tokens=P, num_verify_reqs=bs, block_size=block
    )


def _run(n, mode, layout, mixed=True, min_tokens=2048):
    seen = {}

    def orig(self, input_ids, positions, forward_batch, input_embeds=None):
        seen["pass"] = fp._state["pass"]
        return "out"

    fwd = fp._wrap_forward(orig)
    fp._state.update(
        ctx=types.SimpleNamespace(pre={}), min_tokens=min_tokens, mixed=mixed
    )
    fb = types.SimpleNamespace(forward_mode=mode)
    if layout is not ABSENT:
        fb.mixed_verify = layout
    before = dict(bed.counts().get(fp.NAME, {}))
    out = fwd(None, torch.zeros(n, dtype=torch.long), None, fb)
    after = bed.counts().get(fp.NAME, {})
    delta = {
        k: after.get(k, 0) - before.get(k, 0)
        for k in after
        if after.get(k, 0) != before.get(k, 0)
    }
    assert out == "out" and fp._state["pass"] is False  # reset after the forward
    return seen["pass"], delta


# n_p extend REQUESTS, P extend rows, bs x block block rows
SPLIT = (3, 3072, 64, 16)


def test_mixed_pass_above_threshold_is_fp8_and_counted():
    on, d = _run(4096, FM.EXTEND, _layout(*SPLIT))
    assert on and d.get("pass:m<=4096") == 1 and d.get("pass:mixed:m<=4096") == 1
    assert d.get("rows:block") == 64 * 16


def test_block_rows_are_not_rows_minus_requests():
    # n_p (requests) != P (rows): the counter must read block x bs, not rows - n_p
    on, d = _run(4096, FM.EXTEND, _layout(2, 2048, 32, 16))
    assert on and d.get("rows:block") == 512


def test_mixed_pass_without_block_rows_counts_zero():
    on, d = _run(4096, FM.EXTEND, _layout(4, 4096, 0, 16))
    assert on and d.get("pass:mixed:m<=4096") == 1 and d.get("rows:block", 0) == 0


def test_mixed_pass_below_threshold_is_not_fp8():
    on, d = _run(1500, FM.EXTEND, _layout(2, 1372, 8, 16))
    assert not on and not d


@pytest.mark.parametrize("layout", [None, ABSENT], ids=["no layout", "no field"])
def test_pure_extend_counts_no_mixed_keys(layout):
    on, d = _run(4096, FM.EXTEND, layout)
    assert on and "pass:mixed:m<=4096" not in d and "rows:block" not in d


def test_knob_off_counts_no_mixed_keys():
    on, d = _run(4096, FM.EXTEND, _layout(*SPLIT), mixed=False)
    assert on and "pass:mixed:m<=4096" not in d and "rows:block" not in d


@pytest.mark.parametrize("mode", [FM.DECODE, FM.TARGET_VERIFY])
def test_decode_and_verify_never_fp8(mode):
    on, d = _run(8192, mode, None)
    assert not on and not d


def test_sglang_layout_is_read_where_sglang_has_it():
    mv = pytest.importorskip(
        "sglang.srt.speculative.dflash_mixed_verify",
        reason="this SGLang has no DFLASH mixed verify",
    )
    layout = mv.MixedVerifyLayout(
        num_extend_reqs=3,
        num_extend_tokens=3072,
        extend_lens=[1024] * 3,
        num_verify_reqs=64,
        num_padded_verify_reqs=64,
        block_size=16,
        prune_lens=torch.ones(3 + 64 * 16, dtype=torch.int32),
    )
    on, d = _run(4096, FM.EXTEND, layout)
    assert on and d.get("pass:mixed:m<=4096") == 1 and d.get("rows:block") == 1024


# ---------------------------------------------------------------------------------------------------------- install
def _register(monkeypatch, **env):
    """register() up to the B3 check: an invalid B3 position refuses right after the mixed checks, before any pin
    is read or hook installed."""
    monkeypatch.setenv("MUSE_VR_FP8_PREFILL", "1")
    monkeypatch.setenv("MUSE_VR_FP8_PREFILL_B3", "X")
    for var, value in env.items():
        monkeypatch.setenv(var, value)
    bed.env_int.cache_clear()
    assert fp.register() is False
    return bed.installs()[fp.NAME]["reason"]


def test_refused_next_to_smx_without_the_knob(monkeypatch):
    monkeypatch.setattr(smx, "mixed_verify_reason", lambda: "no mixed verify")
    assert _register(monkeypatch, MUSE_VR_SMX="1") == REFUSED_NEXT_TO_SMX


def test_refused_next_to_sglang_mixed_verify_without_the_knob(monkeypatch):
    monkeypatch.setattr(smx, "mixed_verify_reason", lambda: None)  # SGLang's switch on
    assert _register(monkeypatch) == REFUSED_NEXT_TO_SMX


def test_without_mixed_verify_the_knob_is_not_needed(monkeypatch):
    # this SGLang's own answer: no mixed verify module, or its switch unset
    assert smx.mixed_verify_reason() is not None
    assert _register(monkeypatch).startswith("bad config b3=")
    assert fp._state["mixed"] is False


@pytest.mark.parametrize("min_tokens", ["512", "1023"])
def test_the_knob_needs_the_floor(monkeypatch, min_tokens):
    monkeypatch.setattr(smx, "mixed_verify_reason", lambda: None)
    reason = _register(
        monkeypatch,
        MUSE_VR_FP8_PREFILL_MIXED="1",
        MUSE_VR_FP8_PREFILL_MIN_TOKENS=min_tokens,
    )
    assert reason == (
        f"MUSE_VR_FP8_PREFILL_MIXED needs MIN_TOKENS >= {fp.MIXED_MIN_TOKENS_FLOOR} "
        f"(got {min_tokens})"
    )


@pytest.mark.parametrize("min_tokens", ["1024", "2048", None])
def test_the_knob_lifts_the_refusal(monkeypatch, min_tokens):
    monkeypatch.setattr(smx, "mixed_verify_reason", lambda: None)
    env = {"MUSE_VR_FP8_PREFILL_MIXED": "1", "MUSE_VR_SMX": "1"}
    if min_tokens is not None:  # default 1280
        env["MUSE_VR_FP8_PREFILL_MIN_TOKENS"] = min_tokens
    assert _register(monkeypatch, **env).startswith("bad config b3=")
    assert fp._state["mixed"] is True
