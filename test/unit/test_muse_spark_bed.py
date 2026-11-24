# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU contracts for Muse Spark's source pins (components/bed.py): a pin tuple's ``Functions`` entry accepts a revision
whose sha256 is not listed when every named function has a listed token digest, the digest ``SGLANG_MIRRORS`` uses."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pytest
import sglang
import test_contract  # same directory: its mirror_digest is the reference digest

from sgl_nvidia_mega.muse_spark.components import bed

SOURCE = '''
import functools


def helper(x):
    """Docstring, ignored."""
    return x + 1  # comment, ignored


class Owner:
    @functools.lru_cache
    def method(self, y):
        return y * 2
'''


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """A fake installed SGLang root holding sglang/srt/fake.py."""
    path = tmp_path / "sglang" / "srt" / "fake.py"
    path.parent.mkdir(parents=True)
    path.write_text(SOURCE)
    monkeypatch.setattr(bed, "sglang_root", lambda: tmp_path)
    from sgl_nvidia_mega.muse_spark.components import sgl_compat

    monkeypatch.setattr(sgl_compat, "_root", lambda: tmp_path)
    bed.source_text.cache_clear()
    yield path
    bed.source_text.cache_clear()


def _digest(path, qualname):
    return test_contract.mirror_digest(path, qualname)


def test_function_digest_is_the_contract_digest(tree):
    for qualname in ("helper", "Owner.method"):
        assert bed.function_digest(tree.read_text(), qualname) == _digest(
            tree, qualname
        )
    assert bed.function_digest(tree.read_text(), "Owner.missing") is None
    assert bed.function_digest(tree.read_text(), "Missing.method") is None


def test_function_digest_matches_on_installed_sglang():
    path = Path(sglang.__file__).resolve().parent / "srt" / "utils" / "common.py"
    names = [
        n.name
        for n in ast.parse(path.read_text()).body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ][:20]
    assert names
    for name in names:
        assert bed.function_digest(path.read_text(), name) == _digest(path, name)


def test_pinned_accepts_a_matching_functions_entry(tree):
    rel = "sglang/srt/fake.py"
    sha = hashlib.sha256(tree.read_bytes()).hexdigest()
    good = bed.Functions(
        {
            "helper": _digest(tree, "helper"),
            "Owner.method": (_digest(tree, "Owner.method"), "0" * 16),
        }
    )
    assert bed.pinned(rel, sha)
    assert not bed.pinned(rel, "0" * 64)
    assert bed.pinned(rel, ("0" * 64, good))
    assert not bed.pinned(rel, ("0" * 64, bed.Functions({"helper": "0" * 16})))
    assert not bed.pinned(
        rel, ("0" * 64, bed.Functions({"Owner.gone": _digest(tree, "helper")}))
    )
    # an empty entry leaves the source to its declared reads
    assert bed.pinned(rel, ("0" * 64, bed.Functions({})))
    # a body change takes the pin out; a change elsewhere in the file does not
    tree.write_text(SOURCE + "\n\nCONSTANT = 3\n")
    bed.source_text.cache_clear()
    assert bed.pinned(rel, ("0" * 64, good))
    tree.write_text(SOURCE.replace("return y * 2", "return y * 3"))
    bed.source_text.cache_clear()
    assert not bed.pinned(rel, ("0" * 64, good))


def test_pinned_rejects_a_missing_source(tree):
    rel = "sglang/srt/absent.py"
    assert not bed.pinned(rel, ("0" * 64, bed.Functions({})))
    assert bed.pinned(rel, ("0" * 64, "absent"))


def test_sha_label_names_functions_entries():
    label = bed.sha_label(("a" * 64, bed.Functions({"f": "0" * 16, "g": "1" * 16})))
    assert label == "a" * 16 + "|functions:2"
