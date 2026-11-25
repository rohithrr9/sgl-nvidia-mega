# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU contract for Muse Spark's bounded pull kernel (components/pull_bounded.py): the sha256 pins of its two CUDA
sources name the files the package ships, so the component and its row-layout entry can engage."""

from __future__ import annotations

import hashlib

import pytest

from sgl_nvidia_mega.muse_spark.components import pull_bounded


@pytest.mark.parametrize(
    "path, pin",
    [
        (pull_bounded._CUH, pull_bounded.SOURCE_SHA256),
        (pull_bounded._ROWS_CUH, pull_bounded.ROWS_SOURCE_SHA256),
    ],
    ids=["bounded", "rows"],
)
def test_source_pins_name_the_shipped_files(path, pin):
    assert hashlib.sha256(path.read_bytes()).hexdigest() == pin
