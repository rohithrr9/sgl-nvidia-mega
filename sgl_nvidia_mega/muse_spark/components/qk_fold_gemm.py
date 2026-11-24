# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# Modified by NVIDIA: copied from FlashInfer's flashinfer/gemm/kernels/dense_bf16_gemm_sm100_splitk.py (FlashInfer 0.7.2);
# adds an optional epilogue mode (constructor argument ``qk_fold``, default None = unchanged) in which the owner CTA of
# each query / key / value head tile, after its unchanged reduction, rounding and output store, also computes the
# per-head RMS norm of the query and key heads with the arithmetic of SGLang's fused QK-norm Triton kernel (see
# ``qk_fold_row_ptx``), and stores the e4m3 queries and the e4m3 key / value cache rows; in that mode the epilogue warps
# issue their own grid-dependency wait before they read the cache locations. The main loop, the reduction and the output
# store are unchanged. Public entry points are renamed (``run_splitk_dense`` -> ``run_qk_fold_gemm``).
# Modified by NVIDIA: an optional RoPE form of the epilogue (positions and a cos / sin table) applies the Triton kernel's
# interleaved rotation to the normalized q and k heads of the layers that have RoPE.
# Modified by NVIDIA: the plain GEMM is no longer a copy: without the epilogue, ``run_qk_fold_gemm`` calls FlashInfer's
# own ``run_splitk_dense``; the epilogue kernel class subclasses FlashInfer's ``SplitKDenseGemmKernel`` and inherits its
# constructor tiling and main loop, overriding ``kernel`` and ``epilog_warp`` only.
# Modified by NVIDIA: two optional epilogue-kernel forms (``QkFold.grouped_mailbox``, ``QkFold.deep_ring``; default off =
# unchanged): the split-K partials exchanged in groups of four values per thread (contiguous peer stores per warp, the
# same sum order), and a deeper operand ring in the device's oversized shared-memory mode where the driver reports it.
# Modified by NVIDIA: a plain-GEMM kernel class with that grouped exchange (``SplitKGroupedGemmKernel``, entry point
# ``run_splitk_grouped_gemm``): FlashInfer's main loop, reduction order and output store, no epilogue rows, its own name.
"""Blackwell low-M BF16/FP16 GEMM with an in-kernel cluster split-K reduction.

Each cluster rank accumulates an exact K slice in FP32. Peers publish partials
to rank 0 through DSMEM; rank 0 reduces, casts, and stores once. The public
``A[M, K] @ B[K, N]`` problem is swapped internally, so tile dimensions below
use kernel coordinates: kernel-M carries public N and kernel-N carries public M.

Optional QK-norm epilogue (``qk_fold``): for a projection output laid out as
``[q heads | k heads | v heads | other]`` with 64-column heads and kernel-M tile
64 (one head per tile), the owner CTA of a q / k / v head tile restages the
bf16 tile in shared memory and, per (token, head) row, computes what SGLang's
fused QK-norm kernel computes for that row (RMS norm of q and k heads, v
unchanged, e4m3 with saturation), storing q rows to a query buffer and k / v
rows to the key / value cache rows given by per-token cache locations
(location 0 is not stored). The bf16 output is stored unchanged.
"""

from __future__ import annotations

import dataclasses
import functools
import struct

import cuda.bindings.driver as _cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import Int32
from cutlass._mlir.dialects import llvm
from cutlass.cute import experimental as cute_ext
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op
from flashinfer.gemm.kernels import dense_bf16_gemm_sm100_splitk as _fi_splitk

#: Per-CTA SMEM capacity reported by CuTeDSL on SM100/SM103.
_SMEM_CAPACITY_BYTES = utils.get_smem_capacity_in_bytes("sm_100")

#: K extent of one CTA tile.
_CTA_K = 128

#: Kernel-M tiles; 64 increases CTA count for low-M decode shapes.
_SUPPORTED_MMA_M = (64, 128)

#: Kernel-N carries public M, which is limited to 32.
_SUPPORTED_MMA_N = (8, 16, 32)

#: Physical cluster-K sizes; split 1 compiles out the DSMEM path.
_SUPPORTED_SPLIT_K = (1, 2, 3, 4)

#: Largest public M this low-M policy serves.
_MAX_M = 32

#: Bytes per FP32 partial exchanged through DSMEM.
_FP32_BYTES = 4

#: DSMEM mailbox base alignment, in bytes.
_MAILBOX_ALIGN_BYTES = 128

#: Size and alignment of one mbarrier, in bytes.
_MBARRIER_BYTES = 8

#: Size and alignment of the TMEM base pointer slot.
_TMEM_POINTER_BYTES = 4

#: Bytes per BF16/FP16 element.
_AB_ELEMENT_BYTES = 2

#: Alignment of the A/B shared-memory buffers.
_AB_BUFFER_ALIGN_BYTES = 1024

#: A/B pipeline stage bounds. One stage is useful only for a single K tile.
_MIN_AB_STAGES = 1
_DEFAULT_MIN_AB_STAGES = 2
_MAX_AB_STAGES = 12

#: Stage cap of the deep-ring form (``QkFold.deep_ring``), which may use the device's oversized shared memory.
_MAX_DEEP_AB_STAGES = 20


@dataclasses.dataclass(frozen=True)
class SplitKTactic:
    """One specialization; mma_m carries public N and mma_n carries public M."""

    mma_m: int
    mma_n: int
    split_k: int
    ab_stages: int


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _smem_bytes(
    tactic: SplitKTactic,
    ab_stages: int,
) -> int:
    """Mirror the device allocator's shared-memory layout."""
    cursor = (
        _align_up(
            tactic.mma_m * _CTA_K * _AB_ELEMENT_BYTES * ab_stages,
            _AB_BUFFER_ALIGN_BYTES,
        )
        + tactic.mma_n * _CTA_K * _AB_ELEMENT_BYTES * ab_stages
    )

    cursor = _align_up(cursor, _MBARRIER_BYTES)
    cursor += 2 * ab_stages * _MBARRIER_BYTES
    cursor += 3 * _MBARRIER_BYTES
    cursor = _align_up(cursor, _TMEM_POINTER_BYTES)
    cursor += _TMEM_POINTER_BYTES

    if tactic.split_k == 1:
        return cursor

    return (
        _align_up(
            _align_up(cursor, _MAILBOX_ALIGN_BYTES)
            + (tactic.split_k - 1) * tactic.mma_m * tactic.mma_n * _FP32_BYTES,
            _MBARRIER_BYTES,
        )
        + _MBARRIER_BYTES
    )


def _max_ab_stages_for(
    tactic: SplitKTactic,
    smem_capacity: int,
) -> int:
    return next(
        (
            stages
            for stages in range(_MAX_AB_STAGES, -1, -1)
            if _smem_bytes(tactic, stages) <= smem_capacity
        ),
        0,
    )


def validate_tactic(
    tactic: SplitKTactic,
    m: int,
    n: int,
    k: int,
    *,
    smem_capacity: int = _SMEM_CAPACITY_BYTES,
) -> None:
    """Reject a tactic that cannot serve ``(m, n, k)``."""
    if tactic.mma_m not in _SUPPORTED_MMA_M:
        raise ValueError(f"unsupported mma_m={tactic.mma_m}")
    if tactic.mma_n not in _SUPPORTED_MMA_N:
        raise ValueError(f"unsupported mma_n={tactic.mma_n}")
    if tactic.split_k not in _SUPPORTED_SPLIT_K:
        raise ValueError(f"unsupported split_k={tactic.split_k}")
    if not _MIN_AB_STAGES <= tactic.ab_stages <= _MAX_AB_STAGES:
        raise ValueError(
            f"ab_stages must be in [{_MIN_AB_STAGES}, {_MAX_AB_STAGES}], "
            f"got {tactic.ab_stages}"
        )
    if not 1 <= m <= _MAX_M:
        raise ValueError(f"this low-M policy requires 1 <= M <= {_MAX_M}, got {m}")
    if n <= 0:
        raise ValueError(f"N must be positive, got {n}")
    if k <= 0 or k % _CTA_K or (k // _CTA_K) % tactic.split_k:
        raise ValueError(
            f"K={k} with CTA_K={_CTA_K} does not divide evenly across "
            f"split_k={tactic.split_k}"
        )
    smem_bytes = _smem_bytes(tactic, tactic.ab_stages)
    if smem_bytes > smem_capacity:
        raise ValueError(
            f"tactic {tactic} needs {smem_bytes} B of shared memory but only "
            f"{smem_capacity} B are available; max ab_stages is "
            f"{_max_ab_stages_for(tactic, smem_capacity)}"
        )


def autotune_tactics(
    m: int,
    n: int,
    k: int,
    *,
    smem_capacity: int = _SMEM_CAPACITY_BYTES,
) -> list[SplitKTactic]:
    """Return valid tactics in the shape-specific stage window."""
    tactics: list[SplitKTactic] = []
    for mma_m in _SUPPORTED_MMA_M:
        for mma_n in _SUPPORTED_MMA_N:
            for split_k in _SUPPORTED_SPLIT_K:
                min_stages = _MIN_AB_STAGES if k == _CTA_K else _DEFAULT_MIN_AB_STAGES
                base = SplitKTactic(mma_m, mma_n, split_k, min_stages)
                try:
                    validate_tactic(base, m, n, k, smem_capacity=smem_capacity)
                except ValueError:
                    continue
                max_stages = _max_ab_stages_for(base, smem_capacity)
                # Short K favors shallow pipelines; long K stays near the cap.
                if k <= 4 * _CTA_K:
                    max_stages = min(max_stages, 6)
                else:
                    min_stages = min(max(5, max_stages - 2), max_stages)
                # Retain a non-wrapping power-of-two ring for cheap stage indexing.
                stage_limit = 1 << (k // _CTA_K // split_k).bit_length()
                tactics.extend(
                    dataclasses.replace(base, ab_stages=ab_stages)
                    for ab_stages in range(
                        min(min_stages, stage_limit),
                        min(max_stages, stage_limit) + 1,
                    )
                )
    return tactics


def default_tactic(m: int, n: int, k: int) -> SplitKTactic:
    """Choose the default occupancy-oriented tactic."""
    if n <= 512:
        mma_m = 64
        mma_n = 16 if n == 512 and m > 24 else 8
        requested_split = 4
    else:
        mma_n = 8 if m <= 8 else 16 if m <= 16 else 32
        if n <= 3072:
            mma_m = 128 if m <= 16 else 64
            requested_split = 4 if m <= 16 else 2
        elif n < 8192:
            mma_m = 64
            requested_split = 2
        else:
            mma_m = 128 if k <= 1024 and m <= 24 else 64
            requested_split = 1

    if k <= 4 * _CTA_K:
        requested_split = 1
    split_k = next(
        split_k
        for split_k in reversed(_SUPPORTED_SPLIT_K)
        if split_k <= requested_split and (k // _CTA_K) % split_k == 0
    )
    tactic = SplitKTactic(mma_m, mma_n, split_k, _MIN_AB_STAGES)
    max_stages = _max_ab_stages_for(tactic, _SMEM_CAPACITY_BYTES)
    tactic = dataclasses.replace(
        tactic,
        ab_stages=(
            _MIN_AB_STAGES
            if k == _CTA_K
            else _DEFAULT_MIN_AB_STAGES
            if k <= 2 * _CTA_K and m > 8
            else min(max_stages, 6)
            if k <= 4 * _CTA_K
            else max_stages
        ),
    )
    validate_tactic(tactic, m, n, k)
    return tactic


#: Per device index: (per-block opt-in, the largest per-block capacity the deep-ring form may use). The second is the
#: driver's oversized per-block shared memory where the driver and the bindings report that mode, else the opt-in.
#: Resolved on the first deep-ring call, never at import.
_DEEP_CAPACITY: dict = {}


def _deep_capacity(device_index: int) -> tuple[int, int]:
    got = _DEEP_CAPACITY.get(device_index)
    if got is not None:
        return got
    optin = _SMEM_CAPACITY_BYTES
    capacity = optin
    try:
        _cuda.cuInit(0)
        err, dev = _cuda.cuDeviceGet(device_index)
        if int(err) == 0:
            err, value = _cuda.cuDeviceGetAttribute(
                _cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
                dev,
            )
            if int(err) == 0:
                optin = max(optin, int(value))
                capacity = optin
            # the oversized mode: the device attribute, the function attribute and its value must all exist
            attr = _cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_OVERSIZED_SHARED_MEMORY_PER_BLOCK
            _ = _cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_SHARED_MEMORY_MODE
            _ = _cuda.CUsharedMemoryMode.CU_SHARED_MEMORY_MODE_ALLOW_OVERSIZED_SHARED_MEMORY
            err, value = _cuda.cuDeviceGetAttribute(attr, dev)
            if int(err) == 0:
                capacity = max(optin, int(value))
    except (
        AttributeError
    ):  # a driver or bindings without the oversized mode: the opt-in only
        pass
    _DEEP_CAPACITY[device_index] = (optin, capacity)
    return optin, capacity


#: Names of the functions the hook below switched to the oversized shared-memory mode, in order (for engagement records).
OVERSIZED_FUNCTIONS: list = []


def _function_name(kernel, device) -> str:
    try:
        err, name = (
            _cuda.cuFuncGetName(kernel)
            if device is None
            else _cuda.cuKernelGetName(kernel)
        )
        if int(err) == 0:
            return name.decode() if isinstance(name, bytes) else str(name)
    except (AttributeError, TypeError):
        pass
    return repr(kernel)


def _install_oversized_mode(optin: int) -> None:
    """CuTe DSL sets a kernel's maximum dynamic shared memory from its launch request through
    ``cutlass.base_dsl.runtime.cuda.set_kernel_attribute``. That attribute does not accept a value above the per-block
    opt-in; such a request selects the function's oversized shared-memory mode instead. Requests within the opt-in
    (every other kernel) pass through unchanged. Installed once per process, on the first deep-ring call that needs it."""
    from cutlass.base_dsl.runtime import cuda as dsl_cuda

    if (
        getattr(dsl_cuda.set_kernel_attribute, "_muse_vr_oversized_optin", None)
        is not None
    ):
        return
    original = dsl_cuda.set_kernel_attribute
    dynamic_size = (
        _cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
    )
    mode_attr = _cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_SHARED_MEMORY_MODE
    mode = int(
        _cuda.CUsharedMemoryMode.CU_SHARED_MEMORY_MODE_ALLOW_OVERSIZED_SHARED_MEMORY
    )

    def set_kernel_attribute(kernel, attribute, value, device=None):
        if attribute == dynamic_size and int(value) > optin:
            if device is None:
                result = _cuda.cuFuncSetAttribute(kernel, mode_attr, mode)
            else:
                result = _cuda.cuKernelSetAttribute(mode_attr, mode, kernel, device)
            err = result[0] if isinstance(result, tuple) else result
            if int(err) != 0:
                raise RuntimeError(
                    f"the oversized shared-memory mode was refused ({err})"
                )
            OVERSIZED_FUNCTIONS.append(_function_name(kernel, device))
            return None
        return original(kernel, attribute, value, device)

    set_kernel_attribute._muse_vr_oversized_optin = optin
    dsl_cuda.set_kernel_attribute = set_kernel_attribute


def deep_ring_tactic(
    tactic: SplitKTactic, k: int, device_index: int
) -> tuple[SplitKTactic, int]:
    """The deep-ring form of ``tactic`` for a K extent ``k``: the most stages (at most ``_MAX_DEEP_AB_STAGES`` and the
    K tiles of one CTA) whose shared memory fits the device's deep-ring capacity, and the shared-memory request of the
    launch. Returns ``tactic`` and the served request unchanged where no deeper ring fits. The stage count changes no
    arithmetic (each CTA's K slice and the accumulation order are the same)."""
    optin, capacity = _deep_capacity(device_index)
    k_tiles = k // _CTA_K // tactic.split_k
    cap = min(_MAX_DEEP_AB_STAGES, k_tiles)
    stages = next(
        (s for s in range(cap, 0, -1) if _smem_bytes(tactic, s) <= capacity), 0
    )
    if stages <= tactic.ab_stages:
        return tactic, _SMEM_CAPACITY_BYTES
    deep = dataclasses.replace(tactic, ab_stages=stages)
    request = max(_SMEM_CAPACITY_BYTES, _smem_bytes(deep, stages))
    if request > optin:
        _install_oversized_mode(optin)
    return deep, request


__all__ = [
    "SplitKTactic",
    "autotune_tactics",
    "default_tactic",
    "qk_fold_cols",
    "run_qk_fold_gemm",
    "run_splitk_grouped_gemm",
]


@dsl_user_op
def _map_shared_rank(
    smem_ptr: cute.Pointer,
    peer_cta_rank_in_cluster: Int32,
    *,
    loc=None,
    ip=None,
) -> Int32:
    """Map an SMEM pointer into a peer CTA's address space."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [
                smem_ptr.toint(loc=loc, ip=ip).ir_value(),
                peer_cta_rank_in_cluster.ir_value(),
            ],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _store_shared_remote_v4(
    value0,
    value1,
    value2,
    value3,
    smem_ptr: cute.Pointer,
    mbar_ptr: cute.Pointer,
    peer_cta_rank_in_cluster: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Publish four FP32 partials into a peer's SMEM, crediting 16 bytes."""
    llvm.inline_asm(
        None,
        [
            _map_shared_rank(
                smem_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip
            ).ir_value(),
            value0.bitcast(Int32).ir_value(loc=loc, ip=ip),
            value1.bitcast(Int32).ir_value(loc=loc, ip=ip),
            value2.bitcast(Int32).ir_value(loc=loc, ip=ip),
            value3.bitcast(Int32).ir_value(loc=loc, ip=ip),
            _map_shared_rank(
                mbar_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip
            ).ir_value(),
        ],
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 "
        "[$0], {$1, $2, $3, $4}, [$5];",
        "r,r,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


#: Head dimension the QK-norm epilogue serves (one head per kernel-M tile of 64).
QK_HEAD_DIM = 64

#: Row pitch (elements) of the restaged bf16 tile: 64 columns plus 8 of padding per token row.
_FOLD_PITCH = QK_HEAD_DIM + 8

#: Saturation bound of the e4m3 conversion, as the QK-norm kernel clamps before it.
_E4M3_MAX = 448.0


def _f32_hex(value: float) -> str:
    """PTX immediate (0fXXXXXXXX) of ``value`` rounded to fp32 (round to nearest even)."""
    return "0f%08X" % struct.unpack("<I", struct.pack("<f", float(value)))[0]


def qk_fold_cols(num_tokens: int, num_q_heads: int, num_kv_heads: int) -> int | None:
    """Columns per lane of SGLang's fused QK-norm Triton kernel for ``num_tokens`` rows with the K / V store: its block
    size follows the row count tokens x (q heads + 2 kv heads) (4 / 8 rows per block below 1024 / 8192 rows, 4 warps,
    64-column heads, so 2 / 4 columns per lane). None: a row count this epilogue does not cover."""
    n_rows = num_tokens * (num_q_heads + 2 * num_kv_heads)
    if n_rows < 1024:
        return 2
    if n_rows < 8192:
        return 4
    return None


def qk_fold_row_ptx(
    cols: int,
    unfused: bool,
    num_q_heads: int,
    num_kv_heads: int,
    eps: float,
    rope: bool = False,
) -> str:
    """PTX of one (token, head) row's share for one lane: the lane's ``cols`` consecutive bf16 values of the restaged tile
    -> the row's RMS norm with the arithmetic of SGLang's fused QK-norm Triton kernel -> e4m3 -> the q buffer or the k / v
    cache row.

    Operands: $0 shared address of the lane's values, $1 q row address of the lane's bytes, $2 / $3 key / value cache
    base addresses plus the lane's byte offset, $4 cache location of the token, $5 token, $6 token count, $7 head index;
    with ``rope``: $8 .. the lane's cos values for its column pairs, then its sin values (cos_sin[position][pair] and
    cos_sin[position][32 + pair]).

    With ``rope`` (the layers with interleaved RoPE), for each column pair (2i, 2i + 1) of q and k heads, on a = the bf16-
    rounded normalized values, as the Triton kernel computes it: y_2i = bf16_rn(a_2i * c - a_2i+1 * s),
    y_2i+1 = bf16_rn(a_2i * s + a_2i+1 * c), each product rounded and then the sum (``unfused``), or
    y_2i = bf16_rn(fma(a_2i, c, -(a_2i+1 * s))), y_2i+1 = bf16_rn(fma(a_2i+1, c, a_2i * s)) (the build with FP contraction).

    Arithmetic, per row in fp32 (the 64 / cols lanes of a row call it together; the butterfly stays inside them):
      ss  = the lane's sum of squares (cols 2: fma(x0, x0, x1 * x1), or x0 * x0 + x1 * x1 when ``unfused``;
            cols 4: (x0 * x0 + x2 * x2) + (x1 * x1 + x3 * x3)), then shfl.bfly xor 32 / cols, ..., 1 with fp32 adds
      var = div.full(ss, 64), or div.approx(1, 64) * ss when ``unfused``
      r   = rsqrt.approx.ftz(eps + var);  y = f32(bf16_rn(r * x)) for q and k heads, y = x for v heads
      out = e4m3 (rn, satfinite) of min(max(y, -448), 448)
    Stores: q heads to $1 when the token is below the count; k / v heads to cache row $4 (byte offset $4 x kv_heads x 64 +
    kv head x 64) when the token is below the count and $4 != 0."""
    hq, hqk = num_q_heads, num_q_heads + num_kv_heads
    row_bytes = num_kv_heads * QK_HEAD_DIM
    lines = [
        "{",
        ".reg .b32 w<2>, u<4>, k32;",
        ".reg .f32 x<4>, z<4>, y<4>, p<4>, ss, o, var, t, r, sixty4, inv, one, epsr, lo448, hi448;",
        ".reg .b16 h<4>, e<2>;",
        ".reg .b64 off, kvh64, ga, kvb;",
        ".reg .pred pok, pq, pv, pk, pl, pst;",
    ]
    if rope:
        lines.append(".reg .f32 rc<2>, rs<2>, t0, t1, t2, t3;")
    lines += [
        "setp.lt.s32 pok, $5, $6;",
        f"setp.lt.s32 pq, $7, {hq};",
        f"setp.ge.s32 pv, $7, {hqk};",
    ]
    if cols == 2:
        lines.append("ld.shared.b32 w0, [$0];")
    else:
        lines.append("ld.shared.v2.b32 {w0, w1}, [$0];")
    for i in range(cols // 2):
        lines += [
            f"shl.b32 u{2 * i}, w{i}, 16;",
            f"and.b32 u{2 * i + 1}, w{i}, 0xFFFF0000;",
            f"mov.b32 x{2 * i}, u{2 * i};",
            f"mov.b32 x{2 * i + 1}, u{2 * i + 1};",
        ]
    if cols == 2:
        if unfused:
            lines += [
                "mul.rn.f32 p0, x0, x0;",
                "mul.rn.f32 p1, x1, x1;",
                "add.rn.f32 ss, p0, p1;",
            ]
        else:
            lines += ["mul.rn.f32 p1, x1, x1;", "fma.rn.f32 ss, x0, x0, p1;"]
    else:
        lines += [
            "mul.rn.f32 p0, x0, x0;",
            "mul.rn.f32 p2, x2, x2;",
            "mul.rn.f32 p1, x1, x1;",
            "mul.rn.f32 p3, x3, x3;",
            "add.rn.f32 p0, p0, p2;",
            "add.rn.f32 p1, p1, p3;",
            "add.rn.f32 ss, p0, p1;",
        ]
    m = 32 // cols
    while m >= 1:
        lines += [
            f"shfl.sync.bfly.b32 o, ss, {m}, 0x1f, 0xffffffff;",
            "add.rn.f32 ss, ss, o;",
        ]
        m //= 2
    lines += [f"mov.f32 sixty4, {_f32_hex(64.0)};", f"mov.f32 epsr, {_f32_hex(eps)};"]
    if unfused:
        lines += [
            f"mov.f32 one, {_f32_hex(1.0)};",
            "div.approx.f32 inv, one, sixty4;",
            "mul.rn.f32 var, inv, ss;",
        ]
    else:
        lines.append("div.full.f32 var, ss, sixty4;")
    lines += [
        "add.rn.f32 t, epsr, var;",
        "rsqrt.approx.ftz.f32 r, t;",
        f"mov.f32 lo448, {_f32_hex(-_E4M3_MAX)};",
        f"mov.f32 hi448, {_f32_hex(_E4M3_MAX)};",
    ]

    def bf16_round(dst, src, c):  # dst = f32(bf16_rn(src)), through h / u register c
        return [
            f"cvt.rn.bf16.f32 h{c}, {src};",
            f"cvt.u32.u16 u{c}, h{c};",
            f"shl.b32 u{c}, u{c}, 16;",
            f"mov.b32 {dst}, u{c};",
        ]

    if not rope:  # the layers without RoPE: the per-column sequence, no rotation
        for c in range(cols):
            lines += (
                [f"mul.rn.f32 y{c}, r, x{c};"]
                + bf16_round(f"z{c}", f"y{c}", c)
                + [
                    f"selp.f32 z{c}, x{c}, z{c}, pv;",
                    f"max.f32 z{c}, z{c}, lo448;",
                    f"min.f32 z{c}, z{c}, hi448;",
                ]
            )
    else:
        for c in range(cols):
            lines += [f"mul.rn.f32 y{c}, r, x{c};"] + bf16_round(f"z{c}", f"y{c}", c)
        pairs = cols // 2
        for i in range(pairs):
            lines += [f"mov.f32 rc{i}, ${8 + i};", f"mov.f32 rs{i}, ${8 + pairs + i};"]
        for i in range(pairs):
            a0, a1 = f"z{2 * i}", f"z{2 * i + 1}"
            if unfused:
                lines += [
                    f"mul.rn.f32 t0, {a0}, rc{i};",
                    f"mul.rn.f32 t1, {a1}, rs{i};",
                    "sub.rn.f32 t2, t0, t1;",
                    f"mul.rn.f32 t0, {a0}, rs{i};",
                    f"mul.rn.f32 t1, {a1}, rc{i};",
                    "add.rn.f32 t3, t0, t1;",
                ]
            else:
                lines += [
                    f"mul.rn.f32 t1, {a1}, rs{i};",
                    "neg.f32 t1, t1;",
                    f"fma.rn.f32 t2, {a0}, rc{i}, t1;",
                    f"mul.rn.f32 t0, {a0}, rs{i};",
                    f"fma.rn.f32 t3, {a1}, rc{i}, t0;",
                ]
            lines += bf16_round(a0, "t2", 2 * i) + bf16_round(a1, "t3", 2 * i + 1)
        for c in range(cols):
            lines += [
                f"selp.f32 z{c}, x{c}, z{c}, pv;",
                f"max.f32 z{c}, z{c}, lo448;",
                f"min.f32 z{c}, z{c}, hi448;",
            ]
    for i in range(cols // 2):
        lines.append(f"cvt.rn.satfinite.e4m3x2.f32 e{i}, z{2 * i + 1}, z{2 * i};")
    lines += [
        # key / value cache address: row $4, kv head (head - q heads, or head - q heads - kv heads for values)
        f"selp.b32 k32, {hqk}, {hq}, pv;",
        "sub.s32 k32, $7, k32;",
        f"mul.wide.s32 kvh64, k32, {QK_HEAD_DIM};",
        f"mul.lo.s64 off, $4, {row_bytes};",
        "add.s64 off, off, kvh64;",
        "selp.b64 kvb, $3, $2, pv;",
        "add.s64 ga, kvb, off;",
        "setp.ne.s64 pl, $4, 0;",
        "and.pred pst, pok, pl;",
        "not.pred pk, pq;",
        "and.pred pst, pst, pk;",
        "and.pred pq, pq, pok;",
    ]
    if cols == 2:
        lines += ["@pq st.global.b16 [$1], e0;", "@pst st.global.b16 [ga], e0;"]
    else:
        lines += [
            "mov.b32 w0, {e0, e1};",
            "@pq st.global.b32 [$1], w0;",
            "@pst st.global.b32 [ga], w0;",
        ]
    lines.append("}")
    return "\n".join(lines)


def qk_fold_rows_ptx(
    cols: int,
    unfused: bool,
    num_q_heads: int,
    num_kv_heads: int,
    eps: float,
    rows: int,
    s_stride: int,
    q_stride: int,
    tok_stride: int,
    tree: int | None = None,
) -> str:
    """PTX of ALL of one lane's rows (the form without RoPE), with the rows' work interleaved stage by stage: every row's
    load, then every row's sum of squares, then each butterfly level across all rows, then every row's divide /
    reciprocal square root, rounding, conversion and stores. Per row the operations and their order are exactly those of
    ``qk_fold_row_ptx``; only the interleaving across independent rows differs, so the results are the same bytes.

    ``tree`` (cols 4 only) selects the sum-of-squares tree of a half-warp row: None or 4 = the Triton kernel's 4-column
    tree (as ``qk_fold_row_ptx``); 2 = the Triton kernel's 2-column tree computed on a half-warp: per row the lane forms
    the two 2-column leaves A = sq(x0, x1) and B = sq(x2, x3) (the 2-column leaf arithmetic), runs the butterfly
    xor 8, 4, 2, 1 on A and on B separately and then adds ss = A + B. Lane u holds the leaves of the 2-column lanes 2u and
    2u + 1, so the 2-column butterfly levels xor 16, 8, 4, 2 are the half-warp levels xor 8, 4, 2, 1 applied to A and to B,
    and its last level (xor 1) is the lane's own A + B: the same additions in the same order, so the same bytes as the
    2-column form. 0 = A + B first, then the butterfly on the sum (a different association, used by the tests to show
    that they detect one).

    Operands: $0 shared address of row 0's values, $1 q address of row 0's bytes, $2 / $3 key / value cache bases plus
    the lane's byte offset, $4 row 0's token, $5 token count, $6 head index, $7 .. $(6 + rows) the rows' cache locations.
    Row j is at shared address $0 + j * s_stride, q address $1 + j * q_stride, token $4 + j * tok_stride."""
    hq, hqk = num_q_heads, num_q_heads + num_kv_heads
    row_bytes = num_kv_heads * QK_HEAD_DIM
    R, C = rows, cols
    if tree not in (None, C) and not (C == 4 and tree in (2, 0)):
        raise ValueError("the 2-column tree runs on 4 columns per lane only")
    pair_tree = C == 4 and tree in (2, 0)
    lines = [
        "{",
        f".reg .b32 w<{2 * R}>, u<{4 * R}>, kh<{R}>, sa<{R}>, tk<{R}>;",
        f".reg .f32 x<{4 * R}>, z<{4 * R}>, y<{4 * R}>, p<{4 * R}>, ss<{R}>, o<{R}>, var<{R}>, t<{R}>, r<{R}>;",
        ".reg .f32 sixty4, inv, one, epsr, lo448, hi448;",
        f".reg .b16 h<{4 * R}>, e<{2 * R}>;",
        f".reg .b64 off<{R}>, kvo<{R}>, ga<{R}>, kvb<{R}>, qa<{R}>;",
        f".reg .pred pok<{R}>, pq<{R}>, pv, pk<{R}>, pl<{R}>, pst<{R}>;",
    ]
    if pair_tree:
        lines.append(f".reg .f32 lfa<{R}>, lfb<{R}>, ofb<{R}>;")
    lines.append(f"setp.ge.s32 pv, $6, {hqk};")
    for j in range(R):
        lines += [
            f"add.u32 sa{j}, $0, {j * s_stride};",
            f"add.s32 tk{j}, $4, {j * tok_stride};",
            f"setp.lt.s32 pok{j}, tk{j}, $5;",
            f"setp.lt.s32 pq{j}, $6, {hq};",
        ]
        if C == 2:
            lines.append(f"ld.shared.b32 w{2 * j}, [sa{j}];")
        else:
            lines.append(f"ld.shared.v2.b32 {{w{2 * j}, w{2 * j + 1}}}, [sa{j}];")
    for j in range(R):
        for i in range(C // 2):
            lines += [
                f"shl.b32 u{4 * j + 2 * i}, w{2 * j + i}, 16;",
                f"and.b32 u{4 * j + 2 * i + 1}, w{2 * j + i}, 0xFFFF0000;",
                f"mov.b32 x{4 * j + 2 * i}, u{4 * j + 2 * i};",
                f"mov.b32 x{4 * j + 2 * i + 1}, u{4 * j + 2 * i + 1};",
            ]

    def leaf2(
        dst, i0, i1
    ):  # the 2-column leaf: x_i0 * x_i0 + x_i1 * x_i1 (or its contracted form)
        if unfused:
            return [
                f"mul.rn.f32 p{i0}, x{i0}, x{i0};",
                f"mul.rn.f32 p{i1}, x{i1}, x{i1};",
                f"add.rn.f32 {dst}, p{i0}, p{i1};",
            ]
        return [
            f"mul.rn.f32 p{i1}, x{i1}, x{i1};",
            f"fma.rn.f32 {dst}, x{i0}, x{i0}, p{i1};",
        ]

    for j in range(R):
        b = 4 * j
        if C == 2:
            lines += leaf2(f"ss{j}", b, b + 1)
        elif pair_tree:
            lines += leaf2(f"lfa{j}", b, b + 1) + leaf2(f"lfb{j}", b + 2, b + 3)
        else:
            lines += [
                f"mul.rn.f32 p{b}, x{b}, x{b};",
                f"mul.rn.f32 p{b + 2}, x{b + 2}, x{b + 2};",
                f"mul.rn.f32 p{b + 1}, x{b + 1}, x{b + 1};",
                f"mul.rn.f32 p{b + 3}, x{b + 3}, x{b + 3};",
                f"add.rn.f32 p{b}, p{b}, p{b + 2};",
                f"add.rn.f32 p{b + 1}, p{b + 1}, p{b + 3};",
                f"add.rn.f32 ss{j}, p{b}, p{b + 1};",
            ]
    if pair_tree and tree == 0:
        for j in range(R):
            lines.append(f"add.rn.f32 ss{j}, lfa{j}, lfb{j};")
    m = 32 // C
    while m >= 1:
        if (
            pair_tree and tree == 2
        ):  # the 2-column butterfly levels xor 2m on the lane's two leaves
            for j in range(R):
                lines += [
                    f"shfl.sync.bfly.b32 o{j}, lfa{j}, {m}, 0x1f, 0xffffffff;",
                    f"shfl.sync.bfly.b32 ofb{j}, lfb{j}, {m}, 0x1f, 0xffffffff;",
                ]
            for j in range(R):
                lines += [
                    f"add.rn.f32 lfa{j}, lfa{j}, o{j};",
                    f"add.rn.f32 lfb{j}, lfb{j}, ofb{j};",
                ]
        else:
            for j in range(R):
                lines.append(f"shfl.sync.bfly.b32 o{j}, ss{j}, {m}, 0x1f, 0xffffffff;")
            for j in range(R):
                lines.append(f"add.rn.f32 ss{j}, ss{j}, o{j};")
        m //= 2
    if pair_tree and tree == 2:  # the 2-column last level (xor 1): the lane's own A + B
        for j in range(R):
            lines.append(f"add.rn.f32 ss{j}, lfa{j}, lfb{j};")
    lines += [
        f"mov.f32 sixty4, {_f32_hex(64.0)};",
        f"mov.f32 epsr, {_f32_hex(eps)};",
        f"mov.f32 lo448, {_f32_hex(-_E4M3_MAX)};",
        f"mov.f32 hi448, {_f32_hex(_E4M3_MAX)};",
    ]
    if unfused:
        lines += [f"mov.f32 one, {_f32_hex(1.0)};", "div.approx.f32 inv, one, sixty4;"]
    for j in range(R):
        if unfused:
            lines.append(f"mul.rn.f32 var{j}, inv, ss{j};")
        else:
            lines.append(f"div.full.f32 var{j}, ss{j}, sixty4;")
        lines += [
            f"add.rn.f32 t{j}, epsr, var{j};",
            f"rsqrt.approx.ftz.f32 r{j}, t{j};",
        ]
    for j in range(R):
        for c in range(C):
            n = 4 * j + c
            lines += [
                f"mul.rn.f32 y{n}, r{j}, x{n};",
                f"cvt.rn.bf16.f32 h{n}, y{n};",
                f"cvt.u32.u16 u{n}, h{n};",
                f"shl.b32 u{n}, u{n}, 16;",
                f"mov.b32 z{n}, u{n};",
                f"selp.f32 z{n}, x{n}, z{n}, pv;",
                f"max.f32 z{n}, z{n}, lo448;",
                f"min.f32 z{n}, z{n}, hi448;",
            ]
        for i in range(C // 2):
            lines.append(
                f"cvt.rn.satfinite.e4m3x2.f32 e{2 * j + i}, z{4 * j + 2 * i + 1}, z{4 * j + 2 * i};"
            )
    for j in range(R):
        lines += [
            f"selp.b32 kh{j}, {hqk}, {hq}, pv;",
            f"sub.s32 kh{j}, $6, kh{j};",
            f"mul.wide.s32 kvo{j}, kh{j}, {QK_HEAD_DIM};",
            f"mul.lo.s64 off{j}, ${7 + j}, {row_bytes};",
            f"add.s64 off{j}, off{j}, kvo{j};",
            f"selp.b64 kvb{j}, $3, $2, pv;",
            f"add.s64 ga{j}, kvb{j}, off{j};",
            f"add.s64 qa{j}, $1, {j * q_stride};",
            f"setp.ne.s64 pl{j}, ${7 + j}, 0;",
            f"and.pred pst{j}, pok{j}, pl{j};",
            f"not.pred pk{j}, pq{j};",
            f"and.pred pst{j}, pst{j}, pk{j};",
            f"and.pred pq{j}, pq{j}, pok{j};",
        ]
        if C == 2:
            lines += [
                f"@pq{j} st.global.b16 [qa{j}], e{2 * j};",
                f"@pst{j} st.global.b16 [ga{j}], e{2 * j};",
            ]
        else:
            lines += [
                f"mov.b32 w{2 * j}, {{e{2 * j}, e{2 * j + 1}}};",
                f"@pq{j} st.global.b32 [qa{j}], w{2 * j};",
                f"@pst{j} st.global.b32 [ga{j}], w{2 * j};",
            ]
    lines.append("}")
    return "\n".join(lines)


def _qk_fold_rows(ptx, s_addr, q_addr, k_addr, v_addr, tok, num_tokens, head, locs):
    llvm.inline_asm(
        None,
        [
            cutlass.Int32(s_addr).ir_value(),
            cutlass.Int64(q_addr).ir_value(),
            cutlass.Int64(k_addr).ir_value(),
            cutlass.Int64(v_addr).ir_value(),
            cutlass.Int32(tok).ir_value(),
            cutlass.Int32(num_tokens).ir_value(),
            cutlass.Int32(head).ir_value(),
        ]
        + [cutlass.Int64(v).ir_value() for v in locs],
        ptx,
        "r,l,l,l,r,r,r," + "l," * len(locs) + "~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


def _qk_fold_row(
    ptx, s_addr, q_addr, k_addr, v_addr, loc, tok, num_tokens, head, rope_cs=()
):
    llvm.inline_asm(
        None,
        [
            cutlass.Int32(s_addr).ir_value(),
            cutlass.Int64(q_addr).ir_value(),
            cutlass.Int64(k_addr).ir_value(),
            cutlass.Int64(v_addr).ir_value(),
            cutlass.Int64(loc).ir_value(),
            cutlass.Int32(tok).ir_value(),
            cutlass.Int32(num_tokens).ir_value(),
            cutlass.Int32(head).ir_value(),
        ]
        + [cutlass.Float32(v).ir_value() for v in rope_cs],
        ptx,
        "r,l,l,l,l,r,r,r," + "f," * len(rope_cs) + "~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


def _load_f32(addr):
    """One fp32 load issued after the epilogue warp's own grid-dependency wait (side effects: not hoisted above it)."""
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [cutlass.Int64(addr).ir_value()],
            "ld.global.f32 $0, [$1];",
            "=f,l,~{memory}",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


def _load_cache_loc(loc_base, tok, last):
    """Cache location of token min(tok, last) (an in-range load for every lane; rows past the count are not stored).
    Issued after the epilogue warp's own grid-dependency wait; it has side effects so that it is not moved above it."""
    return cutlass.Int64(
        llvm.inline_asm(
            T.i64(),
            [
                cutlass.Int64(loc_base).ir_value(),
                cutlass.Int32(tok).ir_value(),
                cutlass.Int32(last).ir_value(),
            ],
            "{\n.reg .b32 t;\n.reg .b64 a;\nmin.s32 t, $2, $3;\nmul.wide.s32 a, t, 8;\nadd.s64 a, a, $1;\n"
            "ld.global.b64 $0, [a];\n}",
            "=l,l,r,r,~{memory}",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


#: Rank that gathers partials and stores the output.
OWNER_RANK = 0


def _fi_tactic(tactic: SplitKTactic) -> _fi_splitk.SplitKTactic:
    """The same tactic as FlashInfer's own tactic type."""
    return _fi_splitk.SplitKTactic(
        tactic.mma_m, tactic.mma_n, tactic.split_k, tactic.ab_stages
    )


class SplitKQkFoldGemmKernel(_fi_splitk.SplitKDenseGemmKernel):
    """FlashInfer's split-K kernel with the QK-norm epilogue. The constructor's tiling and the main loop (``dma_warp``,
    ``mma_warp``, ``_drain_producer``) are FlashInfer's own, inherited unchanged; ``kernel`` and ``epilog_warp`` are
    FlashInfer's with the epilogue operands and the QK-norm rows added."""

    def __init__(
        self,
        *,
        tactic: SplitKTactic,
        use_pdl: bool,
        has_bias: bool,
        qk_fold: tuple,
        grouped_mailbox: int = 0,
        smem_request: int = _SMEM_CAPACITY_BYTES,
    ) -> None:
        # qk_fold: (num_q_heads, num_kv_heads, cols per lane, unfused, eps[, rope[, tree]]) for the QK-norm epilogue (rope:
        # the interleaved RoPE of the layers that have it, from per-token positions and a cos / sin table; tree: the
        # sum-of-squares tree of qk_fold_rows_ptx, 2 = the 2-column tree on a half-warp row).
        # grouped_mailbox 1: the split-K partials are exchanged value group by value group (four values per thread per
        # group, the threads' groups contiguous), so a warp's peer stores cover one contiguous range; the owner adds the
        # same values in the same order. 2: peer stores grouped but the owner reads the ungrouped layout (a mismatched
        # exchange: the self-test's plant, never served). smem_request: the launch's dynamic shared memory (the
        # deep-ring form's).
        if (
            tactic.mma_m != QK_HEAD_DIM
            or has_bias
            or qk_fold[2] not in (2, 4)
            or grouped_mailbox not in (0, 1, 2)
        ):
            raise ValueError(
                "the QK-norm epilogue needs mma_m 64, no bias and 2 or 4 columns per lane"
            )
        super().__init__(tactic=_fi_tactic(tactic), use_pdl=use_pdl, has_bias=has_bias)
        self.qk_fold = qk_fold
        self.grouped_store = grouped_mailbox in (1, 2)
        self.grouped_read = grouped_mailbox == 1
        self.smem_request = int(smem_request)
        self.fold_rope = len(qk_fold) > 5 and bool(qk_fold[5])
        self.fold_heads = qk_fold[0] + 2 * qk_fold[1]
        self.fold_ptx = qk_fold_row_ptx(
            qk_fold[2],
            bool(qk_fold[3]),
            qk_fold[0],
            qk_fold[1],
            qk_fold[4],
            self.fold_rope,
        )
        # the form without RoPE runs all of a lane's rows in one stage-interleaved block (qk_fold_rows_ptx)
        step = (
            2 * qk_fold[2]
        )  # rows per step across the 4 epilogue warps: 128 lanes / (64 / cols lanes per row)
        self.fold_rows_ptx = qk_fold_rows_ptx(
            qk_fold[2],
            bool(qk_fold[3]),
            qk_fold[0],
            qk_fold[1],
            qk_fold[4],
            (tactic.mma_n + step - 1) // step,
            step * _FOLD_PITCH * 2,
            step * qk_fold[0] * QK_HEAD_DIM,
            step,
            qk_fold[6] if len(qk_fold) > 6 else None,
        )

    @cute.experimental.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        bias: cute.Tensor,
        q_out: cute.Tensor,
        k_cache: cute.Tensor,
        v_cache: cute.Tensor,
        cache_loc: cute.Tensor,
        positions: cute.Tensor,
        cos_sin: cute.Tensor,
        stream: _cuda.CUstream,
    ):
        # Grid-y packs output-N tile and cluster rank.
        self.kernel(
            a, b, c, bias, q_out, k_cache, v_cache, cache_loc, positions, cos_sin
        ).launch(
            grid=(
                cute.ceil_div(c.layout.shape[0], self.cta_m),
                cute.ceil_div(c.layout.shape[1], self.cta_n) * self.split_k,
                c.layout.shape[2],
            ),
            block=(self.threads_per_cta, 1, 1),
            cluster=self.cluster_shape,
            smem=cute.Int64(self.smem_request),
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.experimental.kernel
    def kernel(
        self,
        mA: cute.Tensor,  # (Gemm_M, Gemm_K, Gemm_L), K-major
        mB: cute.Tensor,  # (Gemm_N, Gemm_K, Gemm_L), K-major
        mC: cute.Tensor,  # (Gemm_M, Gemm_N, Gemm_L), M-major
        mBias: cute.Tensor,  # Broadcast bias; dead when has_bias=False
        mQ: cute.Tensor,  # QK-norm epilogue: e4m3 queries [tokens, q heads x 64] as bytes
        mKc: cute.Tensor,  # e4m3 key cache rows [slots, kv heads x 64] as bytes
        mVc: cute.Tensor,  # e4m3 value cache rows, same shape
        mLoc: cute.Tensor,  # int64 cache location per token
        mPos: cute.Tensor,  # int64 position per token; dead without the RoPE form
        mCs: cute.Tensor,  # fp32 [positions, 64] cos (columns 0..31) / sin (32..63); dead without the RoPE form
    ):
        """Allocate storage and dispatch the specialized warps."""
        stages = self.num_ab_stage

        ab_dtype = mA.element_type
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            ab_dtype,
            ab_dtype,
            utils.LayoutEnum.from_tensor(mA).mma_major_mode(),
            utils.LayoutEnum.from_tensor(mB).mma_major_mode(),
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler_mn,
        )

        mnk_tiler = (self.mma_tiler_mn[0], self.mma_tiler_mn[1], self.cta_k)
        block_idx = cute.arch.block_idx()
        bidx = block_idx[0]
        split_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        n_idx = block_idx[1] // self.split_k
        l_idx = block_idx[2]
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        sA = cute_ext.allocate(
            ab_dtype,
            cute.AddressSpace.smem,
            sm100_utils.make_smem_layout_a(tiled_mma, mnk_tiler, ab_dtype, stages),
            alignment=_AB_BUFFER_ALIGN_BYTES,
        )
        sB = cute_ext.allocate(
            ab_dtype,
            cute.AddressSpace.smem,
            sm100_utils.make_smem_layout_b(tiled_mma, mnk_tiler, ab_dtype, stages),
            alignment=_AB_BUFFER_ALIGN_BYTES,
        )

        acc_layout = cute_ext.make_tmem_layout_acc(
            tiled_mma, self.mma_tiler_mn, acc_stage=1
        )
        c_tiler_mn = (self.cta_m, self.cta_n)

        bar_full = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(stages),
            alignment=_MBARRIER_BYTES,
        ).iterator
        bar_empty = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(stages),
            alignment=_MBARRIER_BYTES,
        ).iterator
        bar_tma_epilog = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(1),
            alignment=_MBARRIER_BYTES,
        ).iterator
        bar_mma_epilog = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(1),
            alignment=_MBARRIER_BYTES,
        ).iterator
        bar_tmem_alloc = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(1),
            alignment=_MBARRIER_BYTES,
        ).iterator
        tmem_base_ptr = cute_ext.allocate(
            cutlass.Int32,
            cute.AddressSpace.smem,
            cute.make_layout(1),
            alignment=_TMEM_POINTER_BYTES,
        ).iterator

        if cutlass.const_expr(self.split_k > 1):
            mailbox = cute_ext.allocate(
                cutlass.Float32,
                cute.AddressSpace.smem,
                cute.make_layout(self.mailbox_elements),
                alignment=_MAILBOX_ALIGN_BYTES,
            )
            bar_reduce = cute_ext.allocate(
                cutlass.Int64,
                cute.AddressSpace.smem,
                cute.make_layout(1),
                alignment=_MBARRIER_BYTES,
            ).iterator
        else:
            # Dummy operands for the compile-time-elided reduction.
            mailbox = sA
            bar_reduce = bar_mma_epilog

        if warp_idx == 0:
            with cute.arch.elect_one():
                for i in range(stages):
                    cute.arch.mbarrier_init(bar_full + i, 2)
                    cute.arch.mbarrier_init(bar_empty + i, 1)
                cute.arch.mbarrier_init(bar_tma_epilog, 32)
                cute.arch.mbarrier_init(bar_mma_epilog, 1)
                cute.arch.mbarrier_init(bar_tmem_alloc, 160)

                if cutlass.const_expr(self.split_k > 1):
                    # Owner arrival plus peer transaction-byte credits.
                    cute.arch.mbarrier_init(bar_reduce, 1)

        cute.arch.mbarrier_init_fence()
        if cutlass.const_expr(self.split_k > 1):
            # Publish peer barriers before cross-CTA stores.
            cute.arch.cluster_arrive_relaxed()
        else:
            cute.arch.barrier()

        # Host validation guarantees an equal, tail-free K partition.
        k_tile_count = cute.size(mA, mode=[1]) // self.cta_k // self.split_k
        k_tile_start = split_rank * k_tile_count

        if cutlass.const_expr(self.split_k > 1):
            cute.arch.cluster_wait()

        # Warp 3 is idle; warps 4-7 run the epilogue.
        if warp_idx == 0:
            self.dma_warp(
                bar_full,
                bar_empty,
                bar_tma_epilog,
                cute.local_tile(mA, (self.cta_m, self.cta_k), (bidx, None, l_idx)),
                sA,
                cute_ext.get_cta_v_map_ab(mA, mnk_tiler, tiled_mma, "A"),
                k_tile_start,
                k_tile_count,
                True,
            )
        elif warp_idx == 1:
            self.dma_warp(
                bar_full,
                bar_empty,
                bar_tma_epilog,
                cute.local_tile(mB, (self.cta_n, self.cta_k), (n_idx, None, l_idx)),
                sB,
                cute_ext.get_cta_v_map_ab(mB, mnk_tiler, tiled_mma, "B"),
                k_tile_start,
                k_tile_count,
                False,
            )
        elif warp_idx == 2:
            self.mma_warp(
                bar_full,
                bar_empty,
                bar_mma_epilog,
                bar_tmem_alloc,
                tiled_mma,
                sA,
                sB,
                tmem_base_ptr,
                acc_layout,
                self.cta_k // cute.size(tiled_mma.shape_mnk, mode=[2]),
                k_tile_count,
            )
        elif warp_idx >= 4:
            self.epilog_warp(
                bar_tma_epilog,
                bar_mma_epilog,
                bar_tmem_alloc,
                tmem_base_ptr,
                acc_layout,
                cute.local_tile(mC, c_tiler_mn, (bidx, n_idx, l_idx)),
                cute.local_tile(mBias, c_tiler_mn, (bidx, n_idx, l_idx)),
                cute.arch.thread_idx()[0] - 128,
                mC.element_type,
                utils.LayoutEnum.from_tensor(mC),
                mailbox,
                bar_reduce,
                split_rank,
                sA,
                bidx,
                n_idx,
                cute.size(mC, mode=[1]),
                mQ,
                mKc,
                mVc,
                mLoc,
                mPos,
                mCs,
            )

    @cute.experimental.jit
    def epilog_warp(
        self,
        bar_tma_epilog,
        bar_mma_epilog,
        bar_tmem_alloc,
        tmem_base_ptr,
        acc_layout: cutlass.Constexpr,
        gD_tile: cute.Tensor,
        gBias_tile: cute.Tensor,
        epi_tid: cutlass.Int32,
        c_dtype: cutlass.Constexpr,
        d_layout: cutlass.Constexpr,
        mailbox,
        bar_reduce,
        split_rank: cutlass.Int32,
        sA: cute.Tensor,
        head: cutlass.Int32,
        n_idx: cutlass.Int32,
        num_tokens: cutlass.Int32,
        mQ: cute.Tensor,
        mKc: cute.Tensor,
        mVc: cute.Tensor,
        mLoc: cute.Tensor,
        mPos: cute.Tensor,
        mCs: cute.Tensor,
    ):
        # QK-norm epilogue lane map (the Triton kernel's): cols 2 = one warp per token row, cols 4 = one half-warp.
        if cutlass.const_expr(self.qk_fold is not None):
            fold_cols = self.qk_fold[2]
            fold_lanes = QK_HEAD_DIM // fold_cols
            fold_step = 128 // fold_lanes
            fold_rows = (self.cta_n + fold_step - 1) // fold_step
            fold_warp = epi_tid // cutlass.Int32(32)
            fold_lane = epi_tid - fold_warp * cutlass.Int32(32)
            fold_sub = fold_lane // cutlass.Int32(fold_lanes)
            fold_li = fold_lane - fold_sub * cutlass.Int32(fold_lanes)
            fold_first = fold_warp * cutlass.Int32(
                32 // fold_lanes
            )  # first row of the warp (warp-uniform)
            fold_row0 = fold_first + fold_sub
            tok0 = n_idx * cutlass.Int32(self.cta_n)
            # Cache locations of this lane's rows, read after this warp's own grid-dependency wait; the loads overlap
            # the main loop.
            if cutlass.const_expr(self.use_pdl):
                cute.arch.griddepcontrol_wait()
            loc_base = cutlass.Int64(mLoc.iterator.toint())
            fold_locs = []
            for j in cutlass.range_constexpr(fold_rows):
                fold_locs.append(
                    _load_cache_loc(
                        loc_base,
                        tok0 + fold_row0 + cutlass.Int32(fold_step * j),
                        num_tokens - 1,
                    )
                )
            # RoPE form: each row's position, then the lane's cos values (cos_sin[pos][pair]) and sin values
            # (cos_sin[pos][32 + pair]) for its column pairs, also ahead of the main loop's end.
            fold_cs = []
            if cutlass.const_expr(self.fold_rope):
                pos_base = cutlass.Int64(mPos.iterator.toint())
                cs_base = cutlass.Int64(mCs.iterator.toint())
                pairs = fold_cols // 2
                for j in cutlass.range_constexpr(fold_rows):
                    pos = _load_cache_loc(
                        pos_base,
                        tok0 + fold_row0 + cutlass.Int32(fold_step * j),
                        num_tokens - 1,
                    )
                    row = (
                        cs_base
                        + pos * cutlass.Int64(QK_HEAD_DIM * 4)
                        + cutlass.Int64(fold_li * cutlass.Int32(pairs * 4))
                    )
                    vals = [_load_f32(row + cutlass.Int64(4 * i)) for i in range(pairs)]
                    vals += [
                        _load_f32(row + cutlass.Int64(QK_HEAD_DIM // 2 * 4 + 4 * i))
                        for i in range(pairs)
                    ]
                    fold_cs.append(vals)

        # Wait until MMA publishes the TMEM base pointer.
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 0)

        acc_view = cute.make_tensor(
            cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr),
            acc_layout,
        )[((None, None), 0, 0, 0)]

        epi_tile = (self.cta_m, self.cta_n)
        tiled_copy_t2r = cute.nvgpu.tcgen05.make_tmem_copy(
            sm100_utils.get_tmem_load_op(
                (self.cta_m, self.cta_n, self.cta_k),
                d_layout,
                c_dtype,
                self.acc_dtype,
                epi_tile,
                False,
            ),
            acc_view,
        )
        gD_epi = cute.flat_divide(gD_tile, epi_tile)

        # Match each epilogue thread's TMEM partition in RMEM.
        rmem_layout = cute_ext.make_t2r_rmem_layout(tiled_copy_t2r, gD_epi, epi_tid)
        rAcc = cute_ext.allocate(
            self.acc_dtype,
            cute.AddressSpace.rmem,
            rmem_layout,
            alignment=32,
        )
        rD = cute_ext.allocate(
            c_dtype,
            cute.AddressSpace.rmem,
            rmem_layout,
            alignment=32,
        )
        thr_t2r = tiled_copy_t2r.get_slice(epi_tid)

        if cutlass.const_expr(self.has_bias):
            bias_dtype = gBias_tile.element_type
            rBias = cute_ext.allocate(
                bias_dtype,
                cute.AddressSpace.rmem,
                rmem_layout,
                alignment=32,
            )
            rBiasAcc = cute_ext.allocate(
                self.acc_dtype,
                cute.AddressSpace.rmem,
                rmem_layout,
                alignment=32,
            )
            if split_rank == OWNER_RANK:
                cute.arch.mbarrier_wait(bar_tma_epilog, 0)
                cute_ext.partition_and_copy(
                    cute.make_tiled_copy_D(
                        cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), bias_dtype),
                        tiled_copy_t2r,
                    ).get_slice(epi_tid),
                    cute.flat_divide(gBias_tile, epi_tile)[None, None, 0, 0],
                    rBias,
                )
                rBiasAcc.store(rBias.load().to(self.acc_dtype))

        cute.arch.mbarrier_wait(bar_mma_epilog, 0)
        cute_ext.partition_and_copy(thr_t2r, acc_view, rAcc)
        # Make tcgen05.ld visible before TMEM release and RMEM use.
        cute.arch.fence_view_async_tmem_load()
        cute.arch.mbarrier_arrive(bar_tmem_alloc)

        # Peers publish FP32 partials; only rank 0 reduces and stores.
        if cutlass.const_expr(self.split_k > 1):
            assert cute.size(rmem_layout) == self.mailbox_elements // (
                (self.split_k - 1) * self.epilog_threads
            )
            values_per_thread = cutlass.const_expr(cute.size(rmem_layout))
            values_per_peer = cutlass.const_expr(
                self.epilog_threads * values_per_thread
            )
            if split_rank != OWNER_RANK:
                for value_idx in cutlass.range_constexpr(0, values_per_thread, 4):
                    if cutlass.const_expr(self.grouped_store):
                        # group value_idx / 4 of every thread is contiguous: this thread's four values at
                        # (value_idx / 4) * 4 * threads + 4 * thread
                        slot = Int32(value_idx * self.epilog_threads) + epi_tid * Int32(
                            4
                        )
                    else:
                        slot = epi_tid * values_per_thread + value_idx
                    _store_shared_remote_v4(
                        rAcc[value_idx],
                        rAcc[value_idx + 1],
                        rAcc[value_idx + 2],
                        rAcc[value_idx + 3],
                        mailbox.iterator
                        + (split_rank - Int32(1)) * values_per_peer
                        + slot,
                        bar_reduce,
                        Int32(OWNER_RANK),
                    )
            else:
                if epi_tid == 0:
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        bar_reduce, self.expected_transaction_bytes
                    )
                cute.arch.mbarrier_wait(bar_reduce, 0)
                for peer in cutlass.range_constexpr(self.split_k - 1):
                    for value_idx in cutlass.range_constexpr(values_per_thread):
                        if cutlass.const_expr(self.grouped_read):
                            slot = Int32(
                                (value_idx // 4) * 4 * self.epilog_threads
                                + value_idx % 4
                            ) + epi_tid * Int32(4)
                        else:
                            slot = epi_tid * values_per_thread + value_idx
                        rAcc[value_idx] = (
                            rAcc[value_idx] + mailbox[peer * values_per_peer + slot]
                        )

        if split_rank == OWNER_RANK:
            if cutlass.const_expr(self.has_bias):
                rAcc.store(rAcc.load() + rBiasAcc.load())

            rD.store(rAcc.load().to(c_dtype))
            # Preserve TMEM coordinates; the copy predicates output tails.
            cute_ext.partition_and_copy(thr_t2r, rD, gD_epi[None, None, 0, 0])

            if cutlass.const_expr(self.qk_fold is not None):
                if head < cutlass.Int32(
                    self.fold_heads
                ):  # q / k / v head tile (CTA-uniform)
                    # Restage the bf16 tile in the A stage buffers (token-major, padded pitch). They were last accessed
                    # by the async proxy (TMA writes, MMA reads, all complete once bar_mma_epilog completed); order
                    # those accesses before the generic-proxy writes (PTX memory model: proxies, fence.proxy.async).
                    s_base = cutlass.Int32(sA.iterator.toint())
                    sD = cute.make_tensor(
                        cute.make_ptr(
                            c_dtype, s_base, cute.AddressSpace.smem, assumed_align=128
                        ),
                        cute.make_layout(
                            (self.cta_m, self.cta_n), stride=(1, _FOLD_PITCH)
                        ),
                    )
                    sD_epi = cute.flat_divide(sD, epi_tile)
                    cute.arch.fence_proxy("async.shared", space="cta")
                    cute_ext.partition_and_copy(thr_t2r, rD, sD_epi[None, None, 0, 0])
                    cute.arch.barrier(barrier_id=1, number_of_threads=128)
                    col = fold_li * cutlass.Int32(fold_cols)
                    q_base = cutlass.Int64(mQ.iterator.toint()) + cutlass.Int64(
                        head * cutlass.Int32(QK_HEAD_DIM) + col
                    )
                    k_base = cutlass.Int64(mKc.iterator.toint()) + cutlass.Int64(col)
                    v_base = cutlass.Int64(mVc.iterator.toint()) + cutlass.Int64(col)
                    q_row_bytes = self.qk_fold[0] * QK_HEAD_DIM
                    # Every row of the tile runs, branch-free (rows past the token count are computed on the staged
                    # zeros and not stored: the row PTX predicates its stores on token < count). Without RoPE, all of
                    # the lane's rows are one stage-interleaved PTX block; with RoPE, one block per row.
                    if cutlass.const_expr(not self.fold_rope):
                        _qk_fold_rows(
                            self.fold_rows_ptx,
                            s_base
                            + (fold_row0 * cutlass.Int32(_FOLD_PITCH) + col)
                            * cutlass.Int32(2),
                            q_base
                            + cutlass.Int64(tok0 + fold_row0)
                            * cutlass.Int64(q_row_bytes),
                            k_base,
                            v_base,
                            tok0 + fold_row0,
                            num_tokens,
                            head,
                            fold_locs,
                        )
                    for j in cutlass.range_constexpr(
                        fold_rows if self.fold_rope else 0
                    ):
                        r_loc = fold_row0 + cutlass.Int32(
                            fold_step * j
                        )  # row within the tile
                        tok = tok0 + r_loc
                        _qk_fold_row(
                            self.fold_ptx,
                            s_base
                            + (r_loc * cutlass.Int32(_FOLD_PITCH) + col)
                            * cutlass.Int32(2),
                            q_base + cutlass.Int64(tok) * cutlass.Int64(q_row_bytes),
                            k_base,
                            v_base,
                            fold_locs[j],
                            tok,
                            num_tokens,
                            head,
                            fold_cs[j] if self.fold_rope else (),
                        )

        # The reduction mbarrier covers remote stores; no cluster barrier needed.


class SplitKGroupedGemmKernel(SplitKQkFoldGemmKernel):
    """The plain GEMM (reduction, rounding and output store; no QK-norm rows) with the grouped split-K exchange and the
    launch's shared-memory request of ``SplitKQkFoldGemmKernel``. FlashInfer's constructor tiling, main loop, sum order
    and output store; a separate class, so that its kernel has its own name."""

    def __init__(
        self,
        *,
        tactic: SplitKTactic,
        use_pdl: bool,
        grouped_mailbox: int = 1,
        smem_request: int = _SMEM_CAPACITY_BYTES,
    ) -> None:
        # grouped_mailbox: as SplitKQkFoldGemmKernel's (0 = FlashInfer's exchange layout, 1 = grouped, 2 = the
        # self-test's mismatched exchange, never served).
        if grouped_mailbox not in (0, 1, 2):
            raise ValueError("grouped_mailbox must be 0, 1 or 2")
        _fi_splitk.SplitKDenseGemmKernel.__init__(
            self, tactic=_fi_tactic(tactic), use_pdl=use_pdl, has_bias=False
        )
        self.qk_fold = None
        self.fold_rope = False
        self.grouped_store = grouped_mailbox in (1, 2)
        self.grouped_read = grouped_mailbox == 1
        self.smem_request = int(smem_request)


import torch as _torch

_SUPPORTED_TORCH_DTYPES = (_torch.bfloat16, _torch.float16)


@cute.experimental.jit
def _bmm_qk_fold(
    gemm_op: cutlass.Constexpr,
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
    q_out: cute.Tensor,
    k_cache: cute.Tensor,
    v_cache: cute.Tensor,
    cache_loc: cute.Tensor,
    positions: cute.Tensor,
    cos_sin: cute.Tensor,
    stream: _cuda.CUstream,
):
    c = cute.make_tensor(c.iterator, cute.select(c.layout, mode=[1, 2, 0]))
    gemm_op(
        cute.make_tensor(a.iterator, cute.select(a.layout, mode=[1, 2, 0])),
        cute.make_tensor(b.iterator, cute.select(b.layout, mode=[2, 1, 0])),
        c,
        cute.make_tensor(c.iterator, cute.select(c.layout, mode=[0, 1, 2])),
        q_out,
        k_cache,
        v_cache,
        cache_loc,
        positions,
        cos_sin,
        stream,
    )


def _from_dlpack_dynamic(tensor, leading_dim: int, assumed_align: int = 32):
    return from_dlpack(
        tensor.detach(), assumed_align=assumed_align
    ).mark_layout_dynamic(leading_dim=leading_dim)


def _detect_leading_dim(tensor: _torch.Tensor) -> int:
    # Ignore synthetic batch stride, including 1x1.
    for dim, stride in enumerate(tensor.stride()[1:], start=1):
        if stride == 1:
            return dim
    raise ValueError("tensor has no stride-1 dimension")


def _make_layout_tensor(
    shape: tuple[int, ...], dtype: _torch.dtype, leading_dim: int
) -> _torch.Tensor:
    permutation = [dim for dim in range(len(shape)) if dim != leading_dim] + [
        leading_dim
    ]
    return _torch.empty(
        tuple(shape[dim] for dim in permutation), dtype=dtype, device="cuda"
    ).permute([permutation.index(dim) for dim in range(len(shape))])


def _make_compile_repr_tensors(
    dtype: _torch.dtype,
    has_bias: bool,
    a_leading: int,
    b_leading: int,
    c_leading: int,
):
    m, n, k, batch = 64, 8, _CTA_K, 1
    tensors = tuple(
        _from_dlpack_dynamic(
            _make_layout_tensor(shape, dtype, leading_dim), leading_dim
        )
        for shape, leading_dim in (
            ((batch, n, k), a_leading),
            ((batch, k, m), b_leading),
            ((batch, n, m), c_leading),
        )
    )
    if not has_bias:
        return (*tensors, None)
    return (
        *tensors,
        _from_dlpack_dynamic(
            _torch.empty((n,), dtype=dtype, device="cuda").as_strided(
                size=(batch, n, m), stride=(0, 1, 0)
            ),
            1,
            2,
        ),
    )


def _to_cute_swap(a, b, out, bias):
    tensors = (
        b.unsqueeze(0).transpose(-2, -1),
        a.unsqueeze(0).transpose(-2, -1),
        out.unsqueeze(0).transpose(-2, -1),
    )
    leading_dims = tuple(_detect_leading_dim(tensor) for tensor in tensors)
    return (
        *(
            _from_dlpack_dynamic(tensor, leading_dim)
            for tensor, leading_dim in zip(tensors, leading_dims, strict=True)
        ),
        _from_dlpack_dynamic(
            bias.as_strided(size=tensors[2].shape, stride=(0, 1, 0)), 1, 2
        )
        if bias is not None
        else None,
        leading_dims,
    )


@functools.cache
def _get_compiled_qk_fold_kernel(
    device_index: int,
    dtype,
    tactic: SplitKTactic,
    use_pdl: bool,
    leading_dims: tuple[int, int, int],
    qk_fold: tuple,
    grouped_mailbox: int = 0,
    smem_request: int = _SMEM_CAPACITY_BYTES,
):
    if dtype not in _SUPPORTED_TORCH_DTYPES:
        raise ValueError(
            f"split-K dense GEMM supports {_SUPPORTED_TORCH_DTYPES}; got {dtype}"
        )

    with _torch.cuda.device(device_index):
        compile_tensors = _make_compile_repr_tensors(dtype, False, *leading_dims)
        kernel = SplitKQkFoldGemmKernel(
            tactic=tactic,
            use_pdl=use_pdl,
            has_bias=False,
            qk_fold=qk_fold,
            grouped_mailbox=grouped_mailbox,
            smem_request=smem_request,
        )
        stream = _cuda.CUstream(_torch.cuda.current_stream().cuda_stream)
        rope = len(qk_fold) > 5 and bool(qk_fold[5])
        return cute_ext.compile(
            _bmm_qk_fold,
            kernel,
            *compile_tensors[:3],
            *_make_fold_repr_tensors(rope),
            stream,
        )


@cute.experimental.jit
def _bmm_grouped(
    gemm_op: cutlass.Constexpr,
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
    stream: _cuda.CUstream,
):
    # the plain kernel: the bias and the QK-norm operands are dead (the output view stands in for them)
    c = cute.make_tensor(c.iterator, cute.select(c.layout, mode=[1, 2, 0]))
    gemm_op(
        cute.make_tensor(a.iterator, cute.select(a.layout, mode=[1, 2, 0])),
        cute.make_tensor(b.iterator, cute.select(b.layout, mode=[2, 1, 0])),
        c,
        cute.make_tensor(c.iterator, cute.select(c.layout, mode=[0, 1, 2])),
        c,
        c,
        c,
        c,
        c,
        c,
        stream,
    )


@functools.cache
def _get_compiled_grouped_kernel(
    device_index: int,
    dtype,
    tactic: SplitKTactic,
    use_pdl: bool,
    leading_dims: tuple[int, int, int],
    grouped_mailbox: int,
    smem_request: int,
):
    if dtype not in _SUPPORTED_TORCH_DTYPES:
        raise ValueError(
            f"split-K dense GEMM supports {_SUPPORTED_TORCH_DTYPES}; got {dtype}"
        )

    with _torch.cuda.device(device_index):
        compile_tensors = _make_compile_repr_tensors(dtype, False, *leading_dims)
        kernel = SplitKGroupedGemmKernel(
            tactic=tactic,
            use_pdl=use_pdl,
            grouped_mailbox=grouped_mailbox,
            smem_request=smem_request,
        )
        stream = _cuda.CUstream(_torch.cuda.current_stream().cuda_stream)
        return cute_ext.compile(_bmm_grouped, kernel, *compile_tensors[:3], stream)


def _make_fold_repr_tensors(rope: bool = False):
    """Representative QK-norm epilogue operands for compilation: e4m3 bytes [tokens, columns] (q buffer, key / value
    cache rows), int64 cache locations, int64 positions and the fp32 [positions, 64] cos / sin table (the last two only
    read by the RoPE form; int64 stand-ins otherwise), dynamic layouts (only their base addresses are read)."""
    byte_rows = tuple(
        _from_dlpack_dynamic(
            _torch.empty((8, 256), dtype=_torch.uint8, device="cuda"), 1, 16
        )
        for _ in range(3)
    )
    loc = _from_dlpack_dynamic(
        _torch.empty((8,), dtype=_torch.int64, device="cuda"), 0, 8
    )
    if not rope:
        return (*byte_rows, loc, loc, loc)
    pos = _from_dlpack_dynamic(
        _torch.empty((8,), dtype=_torch.int64, device="cuda"), 0, 8
    )
    cs = _from_dlpack_dynamic(
        _torch.empty((8, QK_HEAD_DIM), dtype=_torch.float32, device="cuda"), 1, 16
    )
    return (*byte_rows, loc, pos, cs)


def _validate_runtime_tensors(a, b, bias, out) -> tuple[int, int, int]:
    tensors = (a, b, out) + ((bias,) if bias is not None else ())
    if any(not isinstance(tensor, _torch.Tensor) for tensor in tensors):
        raise ValueError("a, b, out, and bias must be torch tensors")
    if a.ndim != 2 or b.ndim != 2 or out.ndim != 2:
        raise ValueError("split-K dense GEMM accepts only 2D tensors")
    if a.device.type != "cuda" or any(tensor.device != a.device for tensor in tensors):
        raise ValueError("all tensors must be on the same CUDA device")
    if a.dtype not in _SUPPORTED_TORCH_DTYPES or any(
        tensor.dtype != a.dtype for tensor in tensors
    ):
        raise ValueError("a, b, out, and bias must share BF16 or FP16 dtype")
    if any(
        not (tensor.is_contiguous() or tensor.t().is_contiguous())
        for tensor in (a, b, out)
    ):
        raise ValueError(
            "a, b, and out must be dense row-major or column-major matrices"
        )
    if any(tensor.data_ptr() % 32 for tensor in (a, b, out)):
        raise ValueError("a, b, and out must be 32-byte aligned")

    m, k = a.shape
    if b.shape[0] != k:
        raise ValueError(
            f"incompatible shapes: a is {tuple(a.shape)}, b is {tuple(b.shape)}"
        )
    n = b.shape[1]
    if out.shape != (m, n):
        raise ValueError(f"out must have shape {(m, n)}, got {tuple(out.shape)}")
    if bias is not None and (
        bias.ndim != 1 or bias.shape[0] != n or not bias.is_contiguous()
    ):
        raise ValueError(
            f"bias must be contiguous with shape {(n,)}, "
            f"got shape {tuple(bias.shape)} and stride {bias.stride()}"
        )

    return m, n, k


@dataclasses.dataclass(frozen=True)
class QkFold:
    """Operands of the QK-norm epilogue.

    q_out: e4m3 [tokens, num_q_heads * 64], contiguous; k_cache / v_cache: e4m3 [slots, num_kv_heads * 64], contiguous;
    cache_loc: int64 [>= tokens], contiguous (row 0 is not stored); eps: the norm's epsilon (rounded to fp32);
    unfused: the arithmetic form of the Triton kernel compiled without floating-point contraction; positions /
    cos_sin (both or neither): int64 [>= tokens] positions and the fp32 [positions, 64] cos / sin table (cos in columns
    0..31, sin in 32..63), both contiguous, for the layers with interleaved RoPE; half_warp_tree: for a row count where
    the Triton kernel uses 2 columns per lane (and no RoPE), run each row on a half-warp (4 columns per lane, half the
    rows per lane) with the 2-column sum-of-squares tree (``qk_fold_rows_ptx`` tree 2), which gives the same bytes.
    grouped_mailbox: the split-K partials exchanged in groups of four values per thread (the same values added in the
    same order: the same bytes); the value 2 is the self-test's plant (peer stores grouped, the owner reading the
    ungrouped layout), never served. deep_ring: the most operand-ring stages that fit the device's per-block shared memory
    (its oversized mode where the driver reports it), so more of the weight is staged ahead of the dependency wait (the
    same K slices and accumulation order: the same bytes)."""

    q_out: _torch.Tensor
    k_cache: _torch.Tensor
    v_cache: _torch.Tensor
    cache_loc: _torch.Tensor
    num_q_heads: int
    num_kv_heads: int
    eps: float
    unfused: bool = False
    positions: _torch.Tensor | None = None
    cos_sin: _torch.Tensor | None = None
    half_warp_tree: bool = False
    grouped_mailbox: int = 0
    deep_ring: bool = False


def _validate_fold(fold: QkFold, m: int, n: int, tactic: SplitKTactic) -> tuple:
    hq, hkv = int(fold.num_q_heads), int(fold.num_kv_heads)
    cols = qk_fold_cols(m, hq, hkv)
    if cols is None:
        raise ValueError("row count not covered by the QK-norm epilogue")
    if tactic.mma_m != QK_HEAD_DIM:
        raise ValueError("the QK-norm epilogue needs mma_m 64 (one head per tile)")
    if n < (hq + 2 * hkv) * QK_HEAD_DIM:
        raise ValueError("output narrower than the q / k / v heads")
    e4m3 = _torch.float8_e4m3fn
    q, kc, vc, loc = fold.q_out, fold.k_cache, fold.v_cache, fold.cache_loc
    if (
        q.dtype != e4m3
        or kc.dtype != e4m3
        or vc.dtype != e4m3
        or loc.dtype != _torch.int64
    ):
        raise ValueError("q_out / caches must be e4m3 and cache_loc int64")
    if not (
        q.is_contiguous()
        and kc.is_contiguous()
        and vc.is_contiguous()
        and loc.is_contiguous()
    ):
        raise ValueError("QK-norm epilogue operands must be contiguous")
    if tuple(q.shape) != (m, hq * QK_HEAD_DIM):
        raise ValueError(f"q_out must have shape {(m, hq * QK_HEAD_DIM)}")
    if kc.ndim != 2 or kc.shape != vc.shape or kc.shape[1] != hkv * QK_HEAD_DIM:
        raise ValueError(f"caches must have shape [slots, {hkv * QK_HEAD_DIM}]")
    if loc.ndim != 1 or loc.shape[0] < m:
        raise ValueError("cache_loc must hold at least one location per token")
    if any(t.device != q.device for t in (kc, vc, loc)):
        raise ValueError("QK-norm epilogue operands must share one device")
    if (
        q.data_ptr() % 16
        or kc.data_ptr() % 16
        or vc.data_ptr() % 16
        or loc.data_ptr() % 8
    ):
        raise ValueError("QK-norm epilogue operand alignment")
    pos, cs = fold.positions, fold.cos_sin
    if (pos is None) != (cs is None):
        raise ValueError("positions and cos_sin go together")
    if pos is None:
        if fold.half_warp_tree and cols == 2:
            return (hq, hkv, 4, bool(fold.unfused), float(fold.eps), False, 2)
        return (hq, hkv, cols, bool(fold.unfused), float(fold.eps))
    if (
        pos.dtype != _torch.int64
        or pos.ndim != 1
        or pos.shape[0] < m
        or not pos.is_contiguous()
        or pos.data_ptr() % 8
    ):
        raise ValueError(
            "positions must be contiguous int64 with at least one entry per token"
        )
    if (
        cs.dtype != _torch.float32
        or cs.ndim != 2
        or cs.shape[1] != QK_HEAD_DIM
        or not cs.is_contiguous()
        or cs.data_ptr() % 16
    ):
        raise ValueError(f"cos_sin must be contiguous fp32 [positions, {QK_HEAD_DIM}]")
    if pos.device != q.device or cs.device != q.device:
        raise ValueError("QK-norm epilogue operands must share one device")
    return (hq, hkv, cols, bool(fold.unfused), float(fold.eps), True)


def run_qk_fold_gemm(
    a,
    b,
    bias,
    out,
    pdl: bool,
    tactic: SplitKTactic,
    fold: QkFold | None = None,
):
    """Run ``A[M,K] @ B[K,N]`` with the ``mm_bf16`` layouts; with ``fold``, also the QK-norm epilogue (no bias).
    Without ``fold`` this is FlashInfer's own ``run_splitk_dense`` (its kernel, its compile cache)."""
    if fold is None:
        return _fi_splitk.run_splitk_dense(a, b, bias, out, pdl, _fi_tactic(tactic))
    m, n, k = _validate_runtime_tensors(a, b, bias, out)
    validate_tactic(tactic, m, n, k)
    if bias is not None:
        raise ValueError("the QK-norm epilogue takes no bias")
    qk = _validate_fold(fold, m, n, tactic)
    smem_request = _SMEM_CAPACITY_BYTES
    if fold.deep_ring:
        tactic, smem_request = deep_ring_tactic(tactic, k, a.get_device())
    with _torch.cuda.device(a.device):
        tensors = _to_cute_swap(a, b, out, bias)
        compiled = _get_compiled_qk_fold_kernel(
            device_index=a.get_device(),
            dtype=a.dtype,
            tactic=tactic,
            use_pdl=pdl,
            leading_dims=tensors[4],
            qk_fold=qk,
            grouped_mailbox=int(fold.grouped_mailbox),
            smem_request=smem_request,
        )
        stream = _cuda.CUstream(_torch.cuda.current_stream(a.device).cuda_stream)
        u8 = _torch.uint8
        loc = _from_dlpack_dynamic(fold.cache_loc, 0, 8)
        if fold.positions is not None:
            extra = (
                _from_dlpack_dynamic(fold.positions, 0, 8),
                _from_dlpack_dynamic(fold.cos_sin, 1, 16),
            )
        else:
            extra = (loc, loc)
        compiled(
            *tensors[:3],
            _from_dlpack_dynamic(fold.q_out.view(u8), 1, 16),
            _from_dlpack_dynamic(fold.k_cache.view(u8), 1, 16),
            _from_dlpack_dynamic(fold.v_cache.view(u8), 1, 16),
            loc,
            *extra,
            stream,
        )
    return out


def grouped_tactic_fits(
    tactic: SplitKTactic, m: int, n: int, k: int, device_index: int
) -> bool:
    """Whether ``run_splitk_grouped_gemm`` takes ``tactic`` for ``(m, n, k)`` on the device: a valid tiling whose shared
    memory fits the device's per-block capacity (runtime query: the opt-in, or the oversized mode where reported)."""
    capacity = max(_SMEM_CAPACITY_BYTES, _deep_capacity(device_index)[1])
    try:
        validate_tactic(tactic, m, n, k, smem_capacity=capacity)
    except ValueError:
        return False
    return True


def run_splitk_grouped_gemm(
    a, b, out, pdl: bool, tactic: SplitKTactic, grouped_mailbox: int = 1
):
    """Run ``A[M,K] @ B[K,N]`` (the ``mm_bf16`` layouts, no bias) on ``SplitKGroupedGemmKernel``: FlashInfer's split-K
    kernel class with the split-K partials exchanged in groups of four values per thread (``grouped_mailbox`` 1; 0 keeps
    FlashInfer's exchange layout; 2 is the self-test's mismatched exchange, never served). The sum order is FlashInfer's
    (own partial, then the peers in cluster-rank order, one rounding), so 0 and 1 give the same bytes as FlashInfer's own
    ``run_splitk_dense`` at the same tactic. The tactic may use more shared memory than CuTe DSL's sm_100 capacity where
    the device's per-block capacity allows it (``grouped_tactic_fits``); a tactic that does not fit is rejected."""
    m, n, k = _validate_runtime_tensors(a, b, None, out)
    device_index = a.get_device()
    optin, capacity = _deep_capacity(device_index)
    validate_tactic(tactic, m, n, k, smem_capacity=max(_SMEM_CAPACITY_BYTES, capacity))
    smem_request = max(_SMEM_CAPACITY_BYTES, _smem_bytes(tactic, tactic.ab_stages))
    if smem_request > optin:
        _install_oversized_mode(optin)
    with _torch.cuda.device(a.device):
        tensors = _to_cute_swap(a, b, out, None)
        compiled = _get_compiled_grouped_kernel(
            device_index=device_index,
            dtype=a.dtype,
            tactic=tactic,
            use_pdl=bool(pdl),
            leading_dims=tensors[4],
            grouped_mailbox=int(grouped_mailbox),
            smem_request=smem_request,
        )
        stream = _cuda.CUstream(_torch.cuda.current_stream(a.device).cuda_stream)
        compiled(*tensors[:3], stream)
    return out
