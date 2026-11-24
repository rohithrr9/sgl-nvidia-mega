# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU contracts for Muse Spark's smx component (``components/smx.py``): the attention form of the verify (block) rows
of SGLang's DFLASH mixed verify passes, an AROUND hook on ``mixed_verify_attention.forward_verify_rows``.

The routing tests are the SWA-T mirror checks (``tests/test_smx_swat_mirror.py``) on the hook SGLang binds: the
attention ops, the FP8 view and the backend are stand-ins that record their calls (no GPU). The binding tests check the
refusal on an SGLang without DFLASH mixed verify, the hooks ``register()`` installs, the SGLang names the guarded reads
use, and, where SGLang has the module, its own split handing the padded verify rows to the hook.
"""

from __future__ import annotations

import dataclasses
import importlib
import importlib.util
import inspect
import types

import pytest
import sglang.srt.layers.radix_attention as ra
import torch

from sgl_nvidia_mega.muse_spark import components
from sgl_nvidia_mega.muse_spark.components import bed
from sgl_nvidia_mega.muse_spark.components import nvfp4_kv_mixed as km
from sgl_nvidia_mega.muse_spark.components import smx
from sgl_nvidia_mega.muse_spark.components import verify_attn as va
from sgl_nvidia_mega.muse_spark.components.ops import swa_t as op_t
from sgl_nvidia_mega.muse_spark.components.ops import swa_verify_attn as op_s

E4M3, BF16 = torch.float8_e4m3fn, torch.bfloat16
FP4 = "nvfp4 (hybrid pool)"  # the backend's data type outside the FP8 view of a mixed pool
W, D, PAGE, SLOTS = 2047, 64, 128, 4096
HAS_MIXED_VERIFY = importlib.util.find_spec(smx.MODULE) is not None
needs_mixed_verify = pytest.mark.skipif(
    not HAS_MIXED_VERIFY, reason="this SGLang has no DFLASH mixed verify"
)


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    """Fresh engagement records and smx state; no record file; no SGLang switch or smx knob from the environment."""
    monkeypatch.setattr(bed, "_installs", {})
    monkeypatch.setattr(bed, "_counts", {})
    monkeypatch.setattr(smx, "_tally", {})
    monkeypatch.setattr(smx, "_logged", set())
    for key in ("on", "swat_mirror", "swat_view"):
        monkeypatch.setitem(smx._state, key, smx._state[key])
    for var in (
        "MUSE_VR_ENGAGE_DIR",
        "MUSE_VR_PLUGIN_DISABLE",
        "MUSE_VR_SMX",
        "MUSE_VR_SMX_SWAT_MIRROR",
        "MUSE_VR_NVFP4_KV_SWA_FP8",
        smx.SWITCH,
    ):
        monkeypatch.delenv(var, raising=False)


class FakeT:
    """Shape / dtype / device / contiguity carrier with the few methods the routing calls."""

    def __init__(self, shape, dtype=None, device="cpu", contiguous=True, tag="t"):
        self.shape, self.dtype, self.device = tuple(shape), dtype, device
        self._contig, self.tag = contiguous, tag

    def reshape(self, *shape):
        return FakeT(shape, self.dtype, self.device, self._contig, self.tag)

    def view(self, *shape):
        if len(shape) == 1 and not isinstance(shape[0], int):
            return FakeT(self.shape, shape[0], self.device, self._contig, self.tag)
        n = 1
        for s in self.shape:
            n *= s
        shape = list(shape)
        if -1 in shape:
            known = 1
            for s in shape:
                if s != -1:
                    known *= s
            shape[shape.index(-1)] = n // known
        return FakeT(shape, self.dtype, self.device, self._contig, self.tag)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            start, stop, _ = idx.indices(self.shape[0])
            return FakeT(
                (stop - start,) + self.shape[1:],
                self.dtype,
                self.device,
                self._contig,
                self.tag,
            )
        raise TypeError(idx)

    def is_contiguous(self):
        return self._contig

    def data_ptr(self):
        return id(self)


class Sub:  # an FP8 sliding-window sub-pool
    quant_method = None


class Pool:
    def __init__(self, hkv, contiguous=True):
        self.hkv, self.contig = hkv, contiguous

    def get_kv_buffer(self, layer_id):
        return (
            FakeT((SLOTS, self.hkv, D), E4M3, contiguous=self.contig, tag="k_pool"),
            FakeT((SLOTS, self.hkv, D), E4M3, contiguous=self.contig, tag="v_pool"),
        )


class Backend:
    """The backend with the verify view's metadata set (as SGLang sets it before ``forward_verify_rows``)."""

    def __init__(self, hkv, bs_pad, mixed=False, contiguous=True):
        self.data_type = FP4 if mixed else E4M3
        self.is_nvfp4_kvcache = bool(mixed)
        self.is_xqa_impl = False
        self.q_data_type = BF16
        self.page_size = PAGE
        self.native_fp4 = False
        self.bmm = (0.125, 1.0)
        self.in_view = False
        self.token_to_kv_pool = Pool(hkv, contiguous)
        self.forward_metadata = types.SimpleNamespace(
            is_ragged_verify=False,
            max_seq_len_q=8,
            cache_seqlens_int32=FakeT((bs_pad,), torch.int32, tag="lens"),
        )
        self.events = []

    def _forward_extend_uses_native_fp4(self, fb):
        return self.native_fp4

    def _get_bmm_scales(self, layer):
        return self.bmm

    def _fused_fp8_qkv_kv_cache(self, q, k, v, layer, ft):
        self.events.append(("store", self.data_type is E4M3, self.in_view))
        return FakeT((q.shape[0], q.shape[1]), E4M3, tag="q8")

    def _get_layer_page_table(self, layer, ft):
        return FakeT((ft.batch_size, 32), torch.int32, tag="swa_page_table")


def layer_of(hkv, window=W, decoder=True):
    return types.SimpleNamespace(
        sliding_window_size=window,
        tp_k_head_num=hkv,
        tp_q_head_num=8 * hkv,
        head_dim=D,
        layer_id=3,
        attn_type=(ra.AttentionType.DECODER if decoder else list(ra.AttentionType)[-1]),
        k_scale=None,
        v_scale=None,
    )


class Rig:
    """Stand-ins for the ops, the FP8 view and the SGLang helpers; records every op call."""

    def __init__(
        self,
        mixed=False,
        kc=None,
        loaded=True,
        capturing=False,
        skip_softmax=False,
        cp=False,
    ):
        self.mixed, self.kc, self.loaded, self.capturing = (
            mixed,
            kc or {},
            loaded,
            capturing,
        )
        self.skip_softmax, self.cp = skip_softmax, cp
        self.calls, self.preloads = [], []

    def kc_for(self, reqs, device, gate=True, use_pdl=True, kv_heads=2):
        self.calls.append(("kc_for", reqs, gate, use_pdl, kv_heads))
        return self.kc.get(reqs, 4)

    def swat_preload(
        self,
        max_reqs,
        device,
        gate=True,
        use_pdl=True,
        min_reqs=1,
        qk=False,
        kv_heads=2,
    ):
        self.preloads.append(
            ("swa_t", max_reqs, str(device), gate, use_pdl, min_reqs, qk, kv_heads)
        )

    def is_loaded(self, reqs, device, gate, use_pdl, kv_heads=2):
        return self.loaded

    def swa_t_verify_attn(
        self,
        q,
        k_pool,
        v_pool,
        page_table,
        seq_lens,
        window_left,
        bmm1,
        bmm2,
        out=None,
        use_pdl=False,
        gate=None,
        norm_eps=0.0,
        kc=None,
        ng_vec=8,
    ):
        self.calls.append(
            (
                "swa_t",
                dict(
                    q=(q.shape, q.dtype, q.tag),
                    k=(k_pool.shape, k_pool.tag),
                    v=(v_pool.shape, v_pool.tag),
                    pt=(page_table.shape, page_table.tag),
                    lens=(seq_lens.shape, seq_lens.tag),
                    window=window_left,
                    bmm=(bmm1, bmm2),
                    out=tuple(out.shape),
                    use_pdl=use_pdl,
                    gate=gate,
                    kc=kc,
                    in_view=self.backend.in_view,
                ),
            )
        )
        return out

    def swa_preload(self, *a, **k):
        self.preloads.append(("swa",) + tuple(a[:2]))

    def swa_verify_attn(self, q, *a, **k):
        self.calls.append(("swa", q.shape, self.backend.in_view))
        return k.get("out")

    def requested(self):
        return self.mixed

    def sub_of(self, pool, layer_id):
        return self.sub if self.mixed else None

    def fp8_view(self, backend, sub):
        rig = self

        class _View:
            def __enter__(self):
                self.saved = (
                    backend.data_type,
                    backend.is_nvfp4_kvcache,
                    backend.in_view,
                )
                backend.data_type, backend.is_nvfp4_kvcache, backend.in_view = (
                    E4M3,
                    False,
                    True,
                )
                rig.views += 1

            def __exit__(self, *exc):
                backend.data_type, backend.is_nvfp4_kvcache, backend.in_view = (
                    self.saved
                )
                return False

        return _View()

    def install(self, monkeypatch, backend, va_state=None, mirror=True, view=True):
        self.backend, self.sub, self.views = backend, Sub(), 0
        vs = dict(
            swa=True,
            swa_t=True,
            swa_t6=False,
            swa_t_min=12,
            swa_t_max=32,
            swa_t_min4=6,
            swa_t_max4=48,
            max_reqs=None,
        )
        vs.update(va_state or {})
        monkeypatch.setattr(bed, "_counts", {})
        monkeypatch.setattr(smx, "_logged", set())
        for key, value in vs.items():
            monkeypatch.setitem(va._state, key, value)
        monkeypatch.setitem(smx._state, "swat_mirror", mirror)
        monkeypatch.setitem(smx._state, "swat_view", view)
        monkeypatch.setitem(smx._state, "swat_loaded", set())
        monkeypatch.setitem(smx._state, "swa_loaded", set())
        monkeypatch.setattr(op_t, "kc_for", self.kc_for)
        monkeypatch.setattr(op_t, "preload", self.swat_preload)
        monkeypatch.setattr(op_t, "is_loaded", self.is_loaded)
        monkeypatch.setattr(op_t, "swa_t_verify_attn", self.swa_t_verify_attn)
        monkeypatch.setattr(op_s, "preload", self.swa_preload)
        monkeypatch.setattr(op_s, "swa_verify_attn", self.swa_verify_attn)
        monkeypatch.setattr(km, "requested", self.requested)
        monkeypatch.setattr(km, "_fp8_sub_pool_of", self.sub_of)
        monkeypatch.setattr(km, "_fp8_view", self.fp8_view)
        monkeypatch.setattr(smx, "_capturing", lambda: self.capturing)
        monkeypatch.setattr(smx, "_skip_softmax_set", lambda: self.skip_softmax)
        monkeypatch.setattr(smx, "_cp_active", lambda ft: self.cp)
        return self


def run_block(rig, backend, layer, bs_pad, **kwargs):
    """The hook on a verify view of bs_pad requests (rows already padded by SGLang); returns the route taken."""
    T = 8 * bs_pad
    hq = layer.tp_q_head_num
    q = FakeT((T, hq * D), BF16, tag="q")
    k = FakeT((T, layer.tp_k_head_num * D), BF16, tag="k")
    v = FakeT((T, layer.tp_k_head_num * D), BF16, tag="v")
    ft = types.SimpleNamespace(batch_size=bs_pad, dcp_kv_mask=None)

    def original(be, q_, k_, v_, layer_, ft_, **kw):  # SGLang's verify chain
        rig.calls.append(("chain", ft_.batch_size, kw))
        return FakeT((q_.shape[0], hq * D), BF16)

    n0 = len(rig.calls)
    smx._around_rows(original, backend, q, k, v, layer, ft, **kwargs)
    kinds = [c[0] for c in rig.calls[n0:] if c[0] in ("swa_t", "swa", "chain")]
    assert len(kinds) == 1, kinds
    return kinds[0], rig.calls[n0:]


def counts():
    return bed.counts().get(smx.NAME, {})


def base_rule(n, hkv, swa_on):
    return "swa" if swa_on and n <= {2: 16, 4: 6}[hkv] else "chain"


def expect_on(n, hkv, lo, hi, swa_on=True):
    if lo <= n <= hi:
        return "swa_t"
    return base_rule(n, hkv, swa_on)


# ---------------------------------------------------------------------------------------------- routing (checks 1-7)
def test_mirror_off_routes_as_the_base_form(monkeypatch):
    assert smx._state["swat_mirror"] is False and smx._state["swat_view"] is True
    for hkv in (2, 4):
        for swa_t in (True, False):
            for swa_on in (True, False):
                rig = Rig().install(
                    monkeypatch,
                    Backend(hkv, 1),
                    {"swa_t": swa_t, "swa": swa_on},
                    mirror=False,
                )
                for n in range(1, 65):
                    rig.backend = Backend(hkv, n)
                    route, _ = run_block(rig, rig.backend, layer_of(hkv), n)
                    assert route == base_rule(n, hkv, swa_on), (hkv, swa_t, swa_on, n)
                assert not any(c[0] == "swa_t" for c in rig.calls)
                assert not any(key.startswith("block:swa_t") for key in counts())
                swa = sum(base_rule(n, hkv, swa_on) == "swa" for n in range(1, 65))
                assert counts().get("block:swa_verify_attn", 0) == swa
                assert counts().get("block:verify chain", 0) == 64 - swa


def test_the_verify_chain_gets_the_call_unchanged(monkeypatch):
    rig = Rig().install(monkeypatch, Backend(4, 20), {"swa": False}, mirror=False)
    route, calls = run_block(rig, rig.backend, layer_of(4), 20, sinks="s", flag=1)
    assert route == "chain"
    assert [c for c in calls if c[0] == "chain"] == [
        ("chain", 20, {"sinks": "s", "flag": 1})
    ]


def test_order(monkeypatch):
    for hkv, extra, lo, hi in (
        (4, {}, 6, 48),
        (2, {}, 12, 32),
        (4, {"swa_t_min4": 5, "swa_t_max4": 52}, 5, 52),
    ):
        rig = Rig().install(monkeypatch, Backend(hkv, 1), extra)
        for n in range(1, 65):
            rig.backend = Backend(hkv, n)
            route, _ = run_block(rig, rig.backend, layer_of(hkv), n)
            assert route == expect_on(n, hkv, lo, hi), (hkv, lo, hi, n, route)
        c = counts()
        assert c.get("block:swa_t", 0) == hi - lo + 1
        assert all(c.get(f"block:swa_t:reqs{n}", 0) == 1 for n in range(lo, hi + 1))
        # every block-row call is counted: no silent fallback
        assert all(c.get(f"block:attn:reqs{n}", 0) == 1 for n in range(1, 65))
        assert not any(key.startswith("block:swa_t:aside") for key in c)
    # swa_verify_attn off: below the window the chain
    rig = Rig().install(monkeypatch, Backend(4, 1), {"swa": False})
    for n in (1, 5, 6, 48, 49):
        rig.backend = Backend(4, n)
        route, _ = run_block(rig, rig.backend, layer_of(4), n)
        assert route == ("swa_t" if 6 <= n <= 48 else "chain"), n
    # full-attention layer of a plain pool: the chain
    rig = Rig().install(monkeypatch, Backend(4, 16))
    route, _ = run_block(rig, rig.backend, layer_of(4, window=-1), 16)
    assert route == "chain"


def test_call_arguments(monkeypatch):
    for mixed in (False, True):
        for n, kc in ((6, 7), (24, 5), (48, 3)):
            be = Backend(4, n, mixed=mixed)
            rig = Rig(mixed=mixed, kc={n: kc}).install(monkeypatch, be)
            route, calls = run_block(rig, be, layer_of(4), n)
            assert route == "swa_t", (mixed, n)
            a = [c for c in calls if c[0] == "swa_t"][0][1]
            m = 8 * n
            assert a["q"] == ((m, 32, D), E4M3, "q8")
            assert a["k"] == ((SLOTS, 4 * D), "k_pool")
            assert a["v"] == ((SLOTS, 4 * D), "v_pool")
            assert a["pt"] == ((n, 32), "swa_page_table")
            assert a["lens"] == ((n,), "lens")
            assert a["window"] == W and a["bmm"] == (0.125, 1.0)
            assert a["out"] == (m, 32, D)
            assert a["use_pdl"] is False and a["gate"] is None and a["kc"] == kc
            kcs = [c for c in calls if c[0] == "kc_for"]
            assert kcs and all(c[1] == n and c[3] is False and c[4] == 4 for c in kcs)
            # the FP8 store first, inside the layer's FP8 view on a mixed pool
            assert be.events == [("store", True, mixed)]
            assert a["in_view"] is mixed
            assert be.in_view is False and (be.data_type is FP4) == mixed


def test_step_asides(monkeypatch):
    cases = [
        ("no cluster size fits", dict(kc={7: 0, 30: 0}), None),
        ("tensor scales", {}, lambda be, ly: setattr(be, "bmm", (object(), 1.0))),
        ("output dtype", {}, lambda be, ly: setattr(be, "q_data_type", torch.float16)),
        ("no fused FP8 KV store", {}, lambda be, ly: setattr(ly, "k_scale", 1.0)),
        (
            "kernel not loaded before graph capture",
            dict(loaded=False, capturing=True),
            None,
        ),
    ]
    for why, rig_kw, mutate in cases:
        for n in (5, 7, 30):
            be, ly = Backend(4, n), layer_of(4)
            if mutate:
                mutate(be, ly)
            rig = Rig(**rig_kw).install(monkeypatch, be)
            route, _ = run_block(rig, be, ly, n)
            if n == 5:  # below the window: SWA-T not asked; the op's own checks decide
                continue
            want = "swa" if n <= 6 else "chain"
            if (
                why in ("tensor scales", "output dtype", "no fused FP8 KV store")
                and n <= 6
            ):
                want = "chain"
            assert route == want, (why, n)
            assert counts().get("block:swa_t:aside:" + why, 0) == 1, (why, counts())
            assert "swat:" + why in smx._logged
    # n 6 with no cluster size: the op takes it (its limit is 6 at 4 KV heads), as the served step does
    rig = Rig(kc={6: 0}).install(monkeypatch, Backend(4, 6))
    assert run_block(rig, rig.backend, layer_of(4), 6)[0] == "swa"
    # non-contiguous buffers / skip-softmax: SWA-T steps aside (the base op's own checks then apply)
    rig = Rig().install(monkeypatch, Backend(4, 20, contiguous=False))
    assert run_block(rig, rig.backend, layer_of(4), 20)[0] == "chain"
    assert counts().get("block:swa_t:aside:KV buffers not contiguous") == 1
    rig = Rig(skip_softmax=True).install(monkeypatch, Backend(4, 20))
    assert run_block(rig, rig.backend, layer_of(4), 20)[0] == "chain"
    assert counts().get("block:swa_t:aside:skip-softmax threshold set") == 1
    # by design: not counted
    rig = Rig().install(monkeypatch, Backend(4, 1))
    for n in (1, 3, 49, 64):
        rig.backend = Backend(4, n)
        run_block(rig, rig.backend, layer_of(4), n)
    assert not any(key.startswith("block:swa_t:aside") for key in counts())


def test_served_common_checks_and_the_view(monkeypatch):
    muts = {
        "verify layout": lambda be, ly: setattr(
            be.forward_metadata, "max_seq_len_q", 16
        ),
        "ragged": lambda be, ly: setattr(be.forward_metadata, "is_ragged_verify", True),
        "attention type": lambda be, ly: setattr(
            ly, "attn_type", list(ra.AttentionType)[-1]
        ),
        "kv cache format": lambda be, ly: setattr(be, "is_xqa_impl", True),
        "native fp4": lambda be, ly: setattr(be, "native_fp4", True),
    }
    for name, mut in muts.items():
        for n in (3, 6, 30):
            be, ly = Backend(4, n), layer_of(4)
            mut(be, ly)
            rig = Rig().install(monkeypatch, be)
            assert run_block(rig, be, ly, n)[0] == "chain", (name, n)
    for n in (3, 6, 30):
        rig = Rig(cp=True).install(monkeypatch, Backend(4, n))
        assert run_block(rig, rig.backend, layer_of(4), n)[0] == "chain", n
    # mixed pool without the served FP8 view (install: no kv4_dec): the chain at every count
    rig = Rig(mixed=True).install(monkeypatch, Backend(4, 1, mixed=True), view=False)
    for n in range(1, 65):
        rig.backend = Backend(4, n, mixed=True)
        assert run_block(rig, rig.backend, layer_of(4), n)[0] == "chain", n
    # an NVFP4 layer of a mixed pool (not in the FP8 sub-pool): the chain
    rig = Rig(mixed=True).install(monkeypatch, Backend(4, 20, mixed=True))
    monkeypatch.setattr(km, "_fp8_sub_pool_of", lambda pool, layer_id: None)
    assert run_block(rig, rig.backend, layer_of(4), 20)[0] == "chain"
    # mixed pool with the view: SWA-T inside the window, the op below it (both in the view)
    rig = Rig(mixed=True).install(monkeypatch, Backend(4, 1, mixed=True))
    for n in (3, 6, 48, 49):
        rig.backend = Backend(4, n, mixed=True)
        route, calls = run_block(rig, rig.backend, layer_of(4), n)
        assert route == expect_on(n, 4, 6, 48), n
        if route == "swa":
            assert [c for c in calls if c[0] == "swa"][0][2] is True


def test_install_decision():
    f = smx.swat_mirror_install
    base = dict(swa_t=True, swa_t6=False)
    r = f(dict(base, swa_t=False), False, False)
    assert r[0] is False and "not engaged" in r[1]
    r = f(dict(base, swa_t6=True), True, True)
    assert r[0] is False and "swa_t6" in r[1]
    assert f(base, True, True) == (True, None, True)
    assert f(dict(base, mixed_swa=True), True, False) == (True, None, True)
    r = f(base, True, False)
    assert r[0] is True and r[2] is False and "verify chain" in r[1]
    assert f(base, False, False) == (True, None, True)
    window = {"swa_t_min": 12, "swa_t_max": 32, "swa_t_min4": 6, "swa_t_max4": 48}
    assert smx.swat_window(window, 4) == (6, 48)
    assert smx.swat_window(window, 2) == (12, 32)


def test_preload(monkeypatch):
    rig = Rig().install(monkeypatch, Backend(4, 1))
    for n in (6, 7, 48, 20, 6):
        rig.backend = Backend(4, n)
        run_block(rig, rig.backend, layer_of(4), n)
    sw = [p for p in rig.preloads if p[0] == "swa_t"]
    # once per window, KV heads, request range and device: ungated, PDL off, no QK prologue
    assert sw == [("swa_t", 48, "cpu", False, False, 6, False, 4)]
    rig = Rig(capturing=True, loaded=True).install(monkeypatch, Backend(4, 20))
    route, _ = run_block(rig, rig.backend, layer_of(4), 20)
    assert route == "swa_t" and not rig.preloads


# --------------------------------------------------------------------------------------------------- the binding
def test_build_views_hook_drops_the_pass_plan_from_both_views():
    pass_batch = types.SimpleNamespace(_muse_vr_prefill_dq_plan=(9, 9))
    views = types.SimpleNamespace(
        forward_batch=pass_batch,
        extend_batch=types.SimpleNamespace(**vars(pass_batch)),
        verify_batch=types.SimpleNamespace(**vars(pass_batch)),
    )
    seen = []

    def original(**kwargs):
        seen.append(kwargs)
        return views

    assert (
        smx._around_views(original, backend="b", forward_batch=pass_batch, layout="l")
        is views
    )
    assert seen == [{"backend": "b", "forward_batch": pass_batch, "layout": "l"}]
    assert not hasattr(views.extend_batch, "_muse_vr_prefill_dq_plan")
    assert not hasattr(views.verify_batch, "_muse_vr_prefill_dq_plan")
    assert pass_batch._muse_vr_prefill_dq_plan == (9, 9)  # the pass keeps its own


def test_components_register_smx_after_the_served_verify_attention():
    order = components.COMPONENTS
    assert order.index("smx") > max(order.index("verify_attn"), order.index("kv4_dec"))
    assert "smx" not in components.ENTRY  # entry point: register()


@pytest.fixture
def hooks(monkeypatch):
    """HookRegistry.register calls, recorded instead of applied."""
    from sglang.srt.plugins.hook_registry import HookRegistry

    calls = []
    monkeypatch.setattr(
        HookRegistry,
        "register",
        lambda target, hook, hook_type=None, **kw: calls.append(
            (target, hook, hook_type)
        ),
    )
    return calls


def test_register_is_off_without_its_switch(hooks):
    assert smx.register() is False
    assert smx.NAME not in bed.installs() and hooks == []


def test_register_refuses_an_sglang_without_the_switch(monkeypatch, hooks):
    from sglang.srt.environ import Envs

    monkeypatch.setenv("MUSE_VR_SMX", "1")
    monkeypatch.setenv(smx.SWITCH, "1")
    monkeypatch.delattr(Envs, smx.SWITCH, raising=False)
    assert smx.register() is False
    rec = bed.installs()[smx.NAME]
    assert rec["engaged"] is False
    assert rec["reason"] == f"this SGLang has no DFLASH mixed verify ({smx.SWITCH})"
    assert hooks == [] and smx._state["on"] is False


def test_register_refuses_an_sglang_without_the_module(monkeypatch, hooks):
    find_spec = importlib.util.find_spec
    monkeypatch.setenv("MUSE_VR_SMX", "1")
    monkeypatch.setenv(smx.SWITCH, "1")
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a: None if name == smx.MODULE else find_spec(name, *a),
    )
    assert smx.register() is False
    assert bed.installs()[smx.NAME]["reason"] == (
        f"this SGLang has no DFLASH mixed verify ({smx.SWITCH})"
    )
    assert hooks == []


def test_mixed_verify_reason_on_this_sglang(monkeypatch):
    if not HAS_MIXED_VERIFY:
        monkeypatch.setenv(smx.SWITCH, "1")
        assert smx.mixed_verify_reason() == (
            f"this SGLang has no DFLASH mixed verify ({smx.SWITCH})"
        )
        return
    assert smx.mixed_verify_reason() == (
        f"{smx.SWITCH} is off: SGLang runs no mixed verify pass"
    )
    monkeypatch.setenv(smx.SWITCH, "1")
    assert smx.mixed_verify_reason() is None


@needs_mixed_verify
def test_register_refuses_with_the_switch_off(monkeypatch, hooks):
    monkeypatch.setenv("MUSE_VR_SMX", "1")
    assert smx.register() is False
    assert bed.installs()[smx.NAME]["reason"] == (
        f"{smx.SWITCH} is off: SGLang runs no mixed verify pass"
    )
    assert hooks == []


def test_register_refuses_a_pin_mismatch(monkeypatch, hooks):
    monkeypatch.setenv("MUSE_VR_SMX", "1")
    monkeypatch.setattr(smx, "mixed_verify_reason", lambda: None)
    monkeypatch.setattr(bed, "pinned", lambda rel, want: False)
    assert smx.register() is False
    assert bed.installs()[smx.NAME]["reason"] == (
        "sglang/srt/layers/attention/mixed_verify_attention.py differs from the pinned revision"
    )
    assert hooks == []


@needs_mixed_verify
def test_register_installs_two_around_hooks(monkeypatch, hooks):
    from sglang.srt.plugins.hook_registry import HookType

    monkeypatch.setenv("MUSE_VR_SMX", "1")
    monkeypatch.setenv(smx.SWITCH, "1")
    assert smx.pins_match() == (True, "ok")  # the pinned functions of this SGLang
    assert smx.register() is True
    assert hooks == [
        (smx.T_VIEWS, smx._around_views, HookType.AROUND),
        (smx.T_ROWS, smx._around_rows, HookType.AROUND),
    ]
    rec = bed.installs()[smx.NAME]
    assert rec["engaged"] is True
    assert rec["block_rows"] == "served verify call (graph-padded view)"
    assert "swat_mirror" not in rec and smx._state["on"] is True


def test_register_records_the_swat_mirror(monkeypatch, hooks):
    monkeypatch.setenv("MUSE_VR_SMX", "1")
    monkeypatch.setenv("MUSE_VR_SMX_SWAT_MIRROR", "1")
    monkeypatch.setattr(smx, "mixed_verify_reason", lambda: None)
    monkeypatch.setattr(smx, "pins_match", lambda: (True, "ok"))
    monkeypatch.setitem(va._state, "swa_t", False)
    assert smx.register() is True
    rec = bed.installs()[smx.NAME]
    assert rec["swat_mirror"] is False and rec["swat_mirror_view"] is True
    assert "not engaged" in rec["swat_mirror_note"]
    assert smx._state["swat_mirror"] is False and len(hooks) == 2
    monkeypatch.setattr(bed, "_installs", {})
    monkeypatch.setitem(va._state, "swa_t", True)
    monkeypatch.setitem(va._state, "swa_t6", False)
    assert smx.register() is True
    rec = bed.installs()[smx.NAME]
    assert rec["swat_mirror"] is True and rec["swat_mirror_view"] is True
    assert "swat_mirror_note" not in rec and smx._state["swat_mirror"] is True


@needs_mixed_verify
def test_the_names_the_guarded_reads_use():
    """getattr(envs, SWITCH) and ForwardBatch.mixed_verify cannot be declared (their files exist on every SGLang)."""
    from sglang.srt.environ import EnvBool, envs
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.speculative.dflash_mixed_verify import MixedVerifyLayout

    mva = importlib.import_module(smx.MODULE)
    assert isinstance(getattr(envs, smx.SWITCH), EnvBool)
    assert "mixed_verify" in {f.name for f in dataclasses.fields(ForwardBatch)}
    rows = inspect.signature(mva.forward_verify_rows).parameters
    assert list(rows) == ["backend", "q", "k", "v", "layer", "forward_batch", "kwargs"]
    views = inspect.signature(mva.build_views).parameters
    assert all(views[n].kind is inspect.Parameter.KEYWORD_ONLY for n in views)
    assert {"extend_batch", "verify_batch"} <= set(
        mva.MixedVerifyViews.__struct_fields__
    )
    assert {
        "num_extend_reqs",
        "num_extend_tokens",
        "num_verify_reqs",
        "block_size",
    } <= set(MixedVerifyLayout.__struct_fields__)


# ------------------------------------------------------------------- SGLang's own split with the hooks (CPU tensors)
class SplitBackend:
    """A TRT-LLM MHA backend stand-in for SGLang's mixed verify split: forward_extend records the view it gets."""

    def __init__(self, md, hkv):
        self.forward_metadata = md
        self.q_data_type = BF16
        self.data_type = E4M3
        self.is_nvfp4_kvcache = self.is_xqa_impl = False
        self.page_size = PAGE
        self.token_to_kv_pool = Pool(hkv)
        self.calls, self.stores = [], []

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def forward_extend(self, q, k, v, layer, forward_batch, save_kv_cache=True, **kw):
        md = self.forward_metadata
        self.calls.append(
            dict(
                mode=forward_batch.forward_mode,
                bs=forward_batch.batch_size,
                q=q.clone(),
                save=save_kv_cache,
                lens=md.cache_seqlens_int32.tolist(),
                plan="_muse_vr_prefill_dq_plan" in vars(forward_batch),
                kw=kw,
            )
        )
        return q * 2

    def _forward_extend_uses_native_fp4(self, fb):
        return False

    def _get_bmm_scales(self, layer):
        return 0.125, 1.0

    def _fused_fp8_qkv_kv_cache(self, q, k, v, layer, ft):
        self.stores.append((tuple(q.shape), ft.forward_mode, ft.batch_size))
        return q.to(E4M3)

    def _get_layer_page_table(self, layer, ft):
        return self.forward_metadata.page_table


def _split_pass(monkeypatch):
    """A mixed verify pass of 2 extend requests (2 + 4 rows) and 3 running requests (8 rows each, padded to 4), with
    SGLang's views, build and verify-row binding hooked as HookRegistry would hook them."""
    from sglang.srt.layers.attention.trtllm_mha_backend import TRTLLMMHAMetadata
    from sglang.srt.model_executor.forward_mode import ForwardMode
    from sglang.srt.plugins.hook_registry import HookType, _wrap_fn
    from sglang.srt.speculative.dflash_mixed_verify import MixedVerifyLayout

    mva = importlib.import_module(smx.MODULE)
    ext, bs, bs_pad, block, hkv = [2, 4], 3, 4, 8, 4
    n_p, P, T = len(ext), sum(ext), block * bs
    committed = [30, 40, 50]
    md = TRTLLMMHAMetadata()
    md.cache_seqlens_int32 = torch.tensor(
        [10, 20] + [c + block for c in committed], dtype=torch.int32
    )
    md.cu_seqlens_q = torch.nn.functional.pad(
        torch.cumsum(torch.tensor(ext + [block] * bs), 0, dtype=torch.int32), (1, 0)
    )
    md.cu_seqlens_k = torch.nn.functional.pad(
        torch.cumsum(md.cache_seqlens_int32, 0, dtype=torch.int32), (1, 0)
    )
    md.max_seq_len_q = block
    md.page_table = (
        torch.arange((n_p + bs) * 4, dtype=torch.int32).view(n_p + bs, 4) + 7
    )
    layout = MixedVerifyLayout(
        num_extend_reqs=n_p,
        num_extend_tokens=P,
        extend_lens=ext,
        num_verify_reqs=bs,
        num_padded_verify_reqs=bs_pad,
        block_size=block,
        prune_lens=torch.ones(P + T, dtype=torch.int32),
    )
    fb = types.SimpleNamespace(
        mixed_verify=layout,
        forward_mode=ForwardMode.EXTEND,
        batch_size=n_p + bs,
        spec_info=None,
        _attn_output=None,
        out_cache_loc=torch.arange(P + T) + 900,
        seq_lens=md.cache_seqlens_int32.to(torch.int64),
        req_pool_indices=torch.arange(n_p + bs) + 3,
        positions=torch.arange(P + T) + 11,
        extend_seq_lens=None,
        extend_prefix_lens=None,
        extend_start_loc=None,
        seq_lens_cpu=None,
        extend_seq_lens_cpu=ext + [block] * bs,
        extend_prefix_lens_cpu=[8, 16] + committed,
        extend_logprob_start_lens_cpu=None,
        _muse_vr_prefill_dq_plan="the pass's plan",
    )
    for name, hook in (
        ("build_views", smx._around_views),
        ("forward_verify_rows", smx._around_rows),
    ):
        monkeypatch.setattr(
            mva, name, _wrap_fn(getattr(mva, name), hook, HookType.AROUND)
        )
    layer = layer_of(hkv)
    g = torch.Generator().manual_seed(5)
    q = torch.randn(P + T, layer.tp_q_head_num * D, generator=g).to(BF16)
    k = torch.randn(P + T, hkv * D, generator=g).to(BF16)
    v = torch.randn(P + T, hkv * D, generator=g).to(BF16)
    return (
        mva,
        SplitBackend(md, hkv),
        layer,
        fb,
        (q, k, v),
        (P, T, bs_pad, block, committed),
    )


@needs_mixed_verify
def test_sglang_split_hands_the_padded_verify_rows_to_the_chain(monkeypatch):
    from sglang.srt.model_executor.forward_mode import ForwardMode

    mva, backend, layer, fb, (q, k, v), (P, T, bs_pad, block, committed) = _split_pass(
        monkeypatch
    )
    md = backend.forward_metadata
    monkeypatch.setitem(
        va._state, "swa", False
    )  # no op form: every block row takes the chain
    out = mva.forward_mixed_verify(backend, q, k, v, layer, fb, marker="m")
    extend, verify = backend.calls
    assert extend["mode"] == ForwardMode.EXTEND and extend["bs"] == 2
    assert extend["lens"] == [10, 20] and torch.equal(extend["q"], q[:P])
    assert verify["mode"] == ForwardMode.TARGET_VERIFY and verify["bs"] == bs_pad
    assert verify["save"] is True and verify["kw"] == {"marker": "m"}
    # committed + block per running request, then the padding request (length 1 before its block)
    assert verify["lens"] == [c + block for c in committed] + [1 + block]
    assert torch.equal(verify["q"][:T], q[P : P + T])
    assert not verify["q"][T:].any() and verify["q"].shape[0] == block * bs_pad
    assert (
        not extend["plan"] and not verify["plan"]
    )  # the hooked build dropped the pass's plan
    assert fb._muse_vr_prefill_dq_plan == "the pass's plan"
    assert torch.equal(out[: P + T], (q * 2)[: P + T])
    assert backend.forward_metadata is md
    assert counts() == {"block:verify chain": 1, "block:verify chain>=1": 1}


@needs_mixed_verify
def test_sglang_split_hands_the_verify_view_to_the_swa_op(monkeypatch):
    from sglang.srt.model_executor.forward_mode import ForwardMode

    mva, backend, layer, fb, (q, k, v), (P, T, bs_pad, block, committed) = _split_pass(
        monkeypatch
    )
    md = backend.forward_metadata
    for key, value in dict(swa=True, max_reqs=None).items():
        monkeypatch.setitem(va._state, key, value)
    monkeypatch.setitem(smx._state, "swat_mirror", False)
    monkeypatch.setitem(smx._state, "swa_loaded", set())
    ops = []

    def swa_verify_attn(
        q8, k_pool, v_pool, table, lens, window, page, bmm1, bmm2, out=None
    ):
        ops.append(
            dict(
                q8=(tuple(q8.shape), q8.dtype),
                table=table.clone(),
                lens=lens.tolist(),
                window=window,
                page=page,
                bmm=(bmm1, bmm2),
            )
        )
        out.fill_(1.5)
        return out

    monkeypatch.setattr(op_s, "preload", lambda *a, **kw: ops.append("preload"))
    monkeypatch.setattr(op_s, "swa_verify_attn", swa_verify_attn)
    out = mva.forward_mixed_verify(backend, q, k, v, layer, fb)
    assert [c["mode"] for c in backend.calls] == [
        ForwardMode.EXTEND
    ]  # the extend view only
    assert backend.stores == [
        ((block * bs_pad, layer.tp_q_head_num * D), ForwardMode.TARGET_VERIFY, bs_pad)
    ]
    assert ops[0] == "preload" and len(ops) == 2
    op = ops[1]
    assert op["q8"] == ((block * bs_pad, layer.tp_q_head_num, D), E4M3)
    assert op["lens"] == [c + block for c in committed] + [1 + block]
    assert torch.equal(op["table"][:3], md.page_table[2:5]) and not op["table"][3].any()
    assert op["window"] == W and op["page"] == PAGE and op["bmm"] == (0.125, 1.0)
    assert torch.equal(out[:P], (q * 2)[:P])
    assert bool((out[P : P + T] == 1.5).all())
    assert counts().get("block:swa_verify_attn") == 1
