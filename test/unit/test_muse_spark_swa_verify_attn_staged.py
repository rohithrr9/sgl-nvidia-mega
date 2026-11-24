# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The exit cluster barrier of the staged form of ``ops.swa_verify_attn`` (``csrc/swa_verify_attn_d.cuh``).

The cluster form ends with an exit cluster barrier by default; MUSE_VR_SWA_NO_EXIT_BARRIER_UNSAFE=1 selects the form
without it, which is not guaranteed by the PTX memory model. The tests pin the barrier leg (``exit_bar=True``) instead
of inheriting it, and check that the two legs use different builds before they compare outputs.

- CPU: staged_key's exit-barrier field (after the ticket field) follows the knob and is set by default; the two keys
  load two modules whose JIT markers and kernel template arguments differ only in the last one (kExitBar). SGLang's JIT
  loader is replaced by a stand-in that records its calls.
- GPU (skipped without CUDA): per case, launch-bounds minimum (1, 2) and trigger placement, the cluster form with the
  barrier is bitwise equal to the kernel on e4m3 queries it replaces (staged form off) and to a repeat launch, and the
  form without the barrier, from its own build, is bitwise equal to it. The queries and the new tokens' K/V rows come
  from SGLang's fused QK-norm Triton kernel with the FP8 KV-cache store. Cases: 1-8, 12, 16, 24 and 32 requests,
  sequence lengths from 8 tokens (new tokens only) to 165,766, page and block-range boundaries, values that saturate
  the e4m3 range, padded requests (cache location 0), with and without the fused output norm + gate.
"""

from __future__ import annotations

import functools
import os

import pytest
import torch

from sgl_nvidia_mega.muse_spark.components import sgl_compat
from sgl_nvidia_mega.muse_spark.components.ops import swa_verify_attn as op

HKV = 4
HQ, D, PAGE, T = 8 * HKV, 64, 128, 8
WINDOW = 2047
BMM1 = 64**-0.5
F8 = torch.float8_e4m3fn
DEV = torch.device("cuda")
QKV_W = (2 * HQ + 2 * HKV) * D  # q, k, v, gate
EPS = 1e-5
KC = op.cluster_size(WINDOW, op.STAGED_CFG[0])

needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a GPU (JIT-built CUDA kernels)"
)


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    """No sliding-window attention knob from the environment: every form at its default."""
    for var in [k for k in os.environ if k.startswith("MUSE_VR_SWA_")]:
        monkeypatch.delenv(var)


def staged_mod(minb, early, norm_gate, ticket):
    # the default form (exit cluster barrier) pinned, not inherited
    return op._module_staged(*op.STAGED_CFG, KC, minb, early, norm_gate, ticket, True)


def exit_keys(num_requests, norm_gate):
    """staged_key's exit-barrier field with MUSE_VR_SWA_NO_EXIT_BARRIER_UNSAFE "0" and "1" (the env restored)."""
    saved = os.environ.get("MUSE_VR_SWA_NO_EXIT_BARRIER_UNSAFE")
    try:
        os.environ["MUSE_VR_SWA_NO_EXIT_BARRIER_UNSAFE"] = "0"
        on = op.staged_key(num_requests, WINDOW, norm_gate)
        os.environ["MUSE_VR_SWA_NO_EXIT_BARRIER_UNSAFE"] = "1"
        off = op.staged_key(num_requests, WINDOW, norm_gate)
    finally:
        if saved is None:
            os.environ.pop("MUSE_VR_SWA_NO_EXIT_BARRIER_UNSAFE", None)
        else:
            os.environ["MUSE_VR_SWA_NO_EXIT_BARRIER_UNSAFE"] = saved
    return on, off


@pytest.mark.parametrize("norm_gate", [False, True])
@pytest.mark.parametrize("num_requests", [1, 3])
def test_exit_barrier_key_field(num_requests, norm_gate):
    k_on, k_off = exit_keys(num_requests, norm_gate)
    assert k_on[9] is True and k_off[9] is False and k_on != k_off
    assert k_on[:9] == k_off[:9]
    assert op.staged_key(num_requests, WINDOW, norm_gate) == k_on  # knob unset


def test_exit_barrier_two_builds(monkeypatch):
    """The two keys load two modules; the knob reaches only the last template argument (kExitBar)."""
    import sglang.kernels.jit.utils as jit

    calls = []

    def load_jit(name, *args, **kwargs):  # records the build; builds nothing
        calls.append((name, args, dict(kwargs["cuda_wrappers"])))
        return object()

    monkeypatch.setattr(jit, "load_jit", load_jit)
    # its own build cache, so that no stand-in module reaches the op's
    monkeypatch.setattr(
        op,
        "_module_staged",
        functools.lru_cache(maxsize=None)(op._module_staged.__wrapped__),
    )
    k_on, k_off = exit_keys(3, True)
    mod_on, mod_off = op._module_staged(*k_on[1:]), op._module_staged(*k_off[1:])
    assert mod_off is not mod_on and op._module_staged(*k_on[1:]) is mod_on
    assert len(calls) == 2
    (n_on, a_on, w_on), (n_off, a_off, w_off) = calls
    assert n_on == n_off == "muse_vr_swa_verify_attn_staged"
    assert a_on[-1] == "true" and a_off[-1] == "false" and a_on[:-1] == a_off[:-1]
    for args, wrappers in ((a_on, w_on), (a_off, w_off)):
        assert (
            wrappers["swa_verify_attn"]
            == f"muse_vr::SwaVerifyAttnStagedKernel<{', '.join(args)}>::run"
        )


def make_case(seq_lens, seed, scale=1.0, pad_reqs=()):
    g = torch.Generator(device=DEV).manual_seed(seed)
    bs = len(seq_lens)
    cols = max((L + PAGE - 1) // PAGE for L in seq_lens) + 1
    need = [
        [p for p in range((L + PAGE - 1) // PAGE) if (p + 1) * PAGE > L - T - WINDOW]
        for L in seq_lens
    ]
    npages = sum(len(n) for n in need) + 2
    k_pool = torch.randn(npages * PAGE, HKV * D, generator=g, device=DEV).to(F8)
    v_pool = torch.randn(npages * PAGE, HKV * D, generator=g, device=DEV).to(F8)
    table = torch.full((bs, cols), npages - 1, dtype=torch.int32)
    perm = torch.randperm(
        npages - 2, generator=torch.Generator().manual_seed(seed)
    ).tolist()
    for b, pages in enumerate(need):
        for p in pages:
            table[b, p] = perm.pop() + 1  # page 0 holds slot 0 (reserved)
    table = table.to(DEV)
    qkv = (torch.randn(bs * T, QKV_W, generator=g, device=DEV) * scale).to(
        torch.bfloat16
    )
    qkv[0, HQ * D + HKV * D + 5] = 3000.0
    qkv[0, HQ * D + HKV * D + 6] = -1.0e4
    pos = torch.cat([torch.arange(L - T, L) for L in seq_lens]).to(torch.int64).to(DEV)
    for b in pad_reqs:
        pos[b * T : (b + 1) * T] = 0
    keys = pos.clone()
    loc = (
        table[torch.arange(bs, device=DEV).repeat_interleave(T), (keys // PAGE)].long()
        * PAGE
        + keys % PAGE
    )
    for b in pad_reqs:
        loc[b * T : (b + 1) * T] = 0
    seq = torch.tensor(seq_lens, dtype=torch.int32, device=DEV)
    return k_pool, v_pool, table, seq, qkv, pos, loc.contiguous()


def call_staged(mod, q, kp, vp, table, seq, out, gate):
    """One call of the cluster form (the ticket form's workspace arguments are unused)."""
    rec, ws = op._staged_ws(DEV, (q.shape[0] // T) * HKV, KC)
    mod.swa_verify_attn(
        rec,
        ws,
        0,
        q,
        kp,
        vp,
        table,
        seq,
        out,
        WINDOW,
        PAGE,
        BMM1,
        1.0,
        gate if gate is not None else out.view(out.shape[0], -1),
        1e-6,
    )


def run_staged(q, kp, vp, table, seq, gate, minb, early):
    out = torch.empty(q.shape[0], HQ, D, dtype=torch.bfloat16, device=DEV)
    mod = staged_mod(minb, early, gate is not None, False)
    call_staged(mod, q, kp, vp, table, seq, out, gate)
    return out


def eq(a, b):
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


def run_case(name, seq_lens, seed, scale=1.0, pad_reqs=(), norm_gate=False):
    k_pool, v_pool, table, seq, qkv, _, loc = make_case(seq_lens, seed, scale, pad_reqs)
    qkn = sgl_compat.module(
        "sglang.kernels.ops.muse_spark_v12.fused_qk_norm"
    ).FusedQKNorm(head_dim=D, num_q_heads=HQ, num_kv_heads=HKV, eps=EPS)
    gate = qkv[:, (HQ + 2 * HKV) * D :] if norm_gate else None
    # T: Triton QK-norm (stores K/V, returns e4m3 q), then the e4m3-query kernel
    kt, vt = k_pool.clone(), v_pool.clone()
    q, _ = qkn._forward_triton(qkv, None, None, (kt, vt, loc))
    q = q.view(-1, HQ, D)
    if q.dtype != F8:
        q = q.to(F8)
    ref_t = op.swa_verify_attn(
        q, kt, vt, table, seq, WINDOW, PAGE, BMM1, 1.0, gate=gate, norm_eps=1e-6
    )
    torch.cuda.synchronize()
    ok = True
    for minb in (1, 2):
        for early in (False, True):
            out = run_staged(q, kt, vt, table, seq, gate, minb, early)
            again = run_staged(q, kt, vt, table, seq, gate, minb, early)
            torch.cuda.synchronize()
            e_t = eq(out, ref_t)
            e_r = eq(out, again)
            n_t = (out.view(torch.int16) != ref_t.view(torch.int16)).sum().item()
            fin = torch.isfinite(out.float()).all().item()
            good = e_t and e_r and fin
            ok &= good
            print(
                f"{name} form=cluster minb={minb} early={early}: {'PASS' if good else 'FAIL'} | reqs "
                f"{len(seq_lens)} gate {gate is not None} | ==T {e_t} (diff {n_t}) repeat {e_r} finite {fin}",
                flush=True,
            )
            # the cluster form without the exit cluster barrier (MUSE_VR_SWA_NO_EXIT_BARRIER_UNSAFE=1): bitwise
            # equal, from a different build (its own module; the knob flips staged_key's exit-barrier field)
            off = torch.empty_like(out)
            mod_on = staged_mod(minb, early, gate is not None, False)
            mod_off = op._module_staged(
                *op.STAGED_CFG, KC, minb, early, gate is not None, False, False
            )
            call_staged(mod_off, q, kt, vt, table, seq, off, gate)
            torch.cuda.synchronize()
            k_on, k_off = exit_keys(len(seq_lens), gate is not None)
            two = (
                mod_off is not mod_on
                and k_on[9] is True
                and k_off[9] is False
                and k_on != k_off
            )
            e_x = (
                two
                and eq(off, out)
                and torch.unique(out.view(torch.int16)).numel() >= 2
            )
            ok &= e_x
            print(
                f"{name} form=cluster minb={minb} early={early} exit barrier on / off: "
                f"{'PASS' if e_x else 'FAIL'} bitwise | two builds {two} (key field {k_on[9]} / {k_off[9]})",
                flush=True,
            )
    return ok


CASES = [
    ("r1 ctx 8", [8], 1, {}),
    ("r1 ctx 9", [9], 2, {}),
    ("r1 ctx 135 (page boundary in the window)", [135], 3, {}),
    ("r2 ctx 2055 / 2056", [2055, 2056], 4, {}),
    ("r3 ctx 165766 x3", [165766] * 3, 5, {}),
    ("r3 ctx 165766 gate", [165766] * 3, 6, {"norm_gate": True}),
    ("r3 mixed", [8, 2057, 100003], 7, {"norm_gate": True}),
    ("r4 ctx 4099..", [4099, 4227, 4355, 5000], 8, {}),
    ("r4 saturating", [70001, 70002, 70003, 70004], 9, {"scale": 64.0}),
    (
        "r4 padded",
        [165766, 165766, 165766, 165766],
        10,
        {"pad_reqs": (3,), "norm_gate": True},
    ),
    ("r5", [30000 + 7 * i for i in range(5)], 11, {}),
    ("r6", [20000 + 13 * i for i in range(6)], 12, {"norm_gate": True}),
    ("r7", [2100 + 191 * i for i in range(7)], 13, {}),
    ("r8", [165766 - i for i in range(8)], 14, {"norm_gate": True}),
    ("r12", [9000 + 3 * i for i in range(12)], 15, {}),
    ("r16", [1000 + 64 * i for i in range(16)], 16, {"norm_gate": True}),
    ("r24", [50000 + i for i in range(24)], 17, {}),
    ("r32", [8 + 4000 * i for i in range(32)], 18, {"norm_gate": True}),
]


@needs_cuda
@pytest.mark.parametrize("name, seq_lens, seed, kw", CASES, ids=[c[0] for c in CASES])
def test_exit_barrier_legs_bitwise(name, seq_lens, seed, kw):
    assert run_case(name, seq_lens, seed, **kw)
