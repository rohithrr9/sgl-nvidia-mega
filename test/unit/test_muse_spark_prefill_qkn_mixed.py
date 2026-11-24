# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU contracts for prefill_qkn's install next to DFLASH mixed verify (``components/prefill_qkn.py``): prefill graphs
replayed for mixed passes carry verify block rows, which must reach the verify forms with normalized q / k and a bf16 q.

- Mixed verify on (SGLang's switch, or ``smx`` requested): store and q8 stay off with a reason; norm-only installs.
- Mixed verify off: store and q8 engage on matching pins as before; a pin mismatch keeps its own reason.
"""

from __future__ import annotations

import pytest

from sgl_nvidia_mega.muse_spark.components import bed
from sgl_nvidia_mega.muse_spark.components import prefill_qkn as pq
from sgl_nvidia_mega.muse_spark.components import smx
from sgl_nvidia_mega.muse_spark.components.ops import swa_verify_attn


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    """Fresh engagement records and mode state; every switch from the test; hooks recorded, not installed."""
    monkeypatch.setattr(bed, "_installs", {})
    monkeypatch.setattr(bed, "_counts", {})
    monkeypatch.setitem(pq._store_state, "engaged", False)
    monkeypatch.setitem(pq._q8_state, "engaged", False)
    for var in (
        "MUSE_VR_PLUGIN_DISABLE",
        "MUSE_VR_SMX",
        "MUSE_VR_PREFILL_QKN",
        "MUSE_VR_PREFILL_QKN_STORE",
        "MUSE_VR_PREFILL_QKN_Q8",
        smx.SWITCH,
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("MUSE_VR_PREFILL_QKN", "1")
    monkeypatch.setattr(swa_verify_attn, "qk_arith", lambda: "fp32")
    registered = []
    from sglang.srt.plugins.hook_registry import HookRegistry

    monkeypatch.setattr(
        HookRegistry,
        "register",
        staticmethod(lambda target, fn, kind: registered.append(target)),
    )
    yield registered


def _install(monkeypatch, *, mixed_reason, pins=True, **env):
    monkeypatch.setattr(smx, "mixed_verify_reason", lambda: mixed_reason)
    monkeypatch.setattr(pq.bed, "pinned", lambda rel, sha: pins or rel in pq.PINS)
    for var, value in env.items():
        monkeypatch.setenv(var, value)
    assert pq.register() is True
    return bed.installs()[pq.NAME]


MODES = {"MUSE_VR_PREFILL_QKN_STORE": "1", "MUSE_VR_PREFILL_QKN_Q8": "1"}


def test_mixed_verify_off_engages_store_and_q8(monkeypatch, _isolated):
    rec = _install(monkeypatch, mixed_reason="switch off", **MODES)
    assert rec["mode"] == "qk_norm+store+q8"
    assert "store" not in rec and "q8" not in rec
    assert pq._store_state["engaged"] and pq._q8_state["engaged"]
    assert pq.TARGET_EXTEND in _isolated


def test_sglang_mixed_verify_keeps_norm_only(monkeypatch, _isolated):
    rec = _install(monkeypatch, mixed_reason=None, **MODES)
    assert rec["mode"] == "qk_norm"
    assert rec["store"] == "DFLASH mixed verify: norm-only"
    assert rec["q8"] == "DFLASH mixed verify: bf16 q"
    assert not pq._store_state["engaged"] and not pq._q8_state["engaged"]
    assert _isolated == [pq.TARGET]


def test_smx_requested_keeps_norm_only(monkeypatch, _isolated):
    rec = _install(monkeypatch, mixed_reason="switch off", MUSE_VR_SMX="1", **MODES)
    assert rec["mode"] == "qk_norm"
    assert rec["store"] == "DFLASH mixed verify: norm-only"
    assert rec["q8"] == "DFLASH mixed verify: bf16 q"


def test_pin_mismatch_keeps_its_reason(monkeypatch, _isolated):
    rec = _install(monkeypatch, mixed_reason=None, pins=False, **MODES)
    assert rec["store"] == "store pins mismatch: norm-only"
    assert rec["q8"] == "q8 pins mismatch: bf16 q"
