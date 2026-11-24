# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Shared helpers for plugin components: env gating, SGLang source pins and engagement records.

``enabled(name)``
    A component is opted in with ``MUSE_VR_<NAME>=1`` (default off). ``MUSE_VR_PLUGIN_DISABLE=<name>[,<name>...]``
    forces it off even when opted in.
``pinned(relpath, sha256_hex)``
    True when the installed SGLang source file (``relpath`` relative to the directory that holds the ``sglang``
    package, e.g. ``sglang/srt/models/muse_spark.py``) has exactly this sha256. The package is located without
    importing it. A pin tuple may also hold ``Functions({qualname: digest})``: a revision whose sha256 is not listed
    then passes when each named function (the ones the component hooks, mirrors or calls there) has a listed token
    digest, the same digest as ``SGLANG_MIRRORS`` in ``test/unit/test_contract.py``.
``report(name, engaged, **info)``
    Once per process and component: a WARNING-level log line ``muse_vr_plugin: <name> engaged=<bool> ...`` and, when
    ``MUSE_VR_ENGAGE_DIR`` is set, ``<dir>/install_<pid>.json`` holding every decision this process made. WARNING,
    because SGLang loads plugins before it configures logging in its worker processes, so INFO records emitted from
    ``register()`` are dropped by the logging module's last-resort handler.
``flag(name)`` / ``env_int(var, default)``
    Cached forms for per-call (hot eager path) checks: the environment is read once per process and name, since a
    server's environment does not change while it runs.
``count(name, key)``
    Host-side tally of fast-path call sites (e.g. per CUDA-graph capture shape), written to the same JSON record.
    Calls replayed from a captured graph do not reach the host and are not counted.
"""

from __future__ import annotations

import ast
import functools
import hashlib
import importlib.util
import io
import json
import logging
import os
import threading
import time
import tokenize
from pathlib import Path

logger = logging.getLogger("muse_vr_plugin")

_lock = threading.Lock()
_installs: dict[str, dict] = {}
_counts: dict[str, dict[str, int]] = {}


def disabled(name: str) -> bool:
    return name in {
        x.strip()
        for x in os.environ.get("MUSE_VR_PLUGIN_DISABLE", "").split(",")
        if x.strip()
    }


def enabled(name: str) -> bool:
    """Opt-in gate: MUSE_VR_<NAME>=1 and not listed in MUSE_VR_PLUGIN_DISABLE."""
    v = os.environ.get("MUSE_VR_" + name.upper(), "0").strip().lower()
    return v in ("1", "true", "yes", "on") and not disabled(name)


@functools.lru_cache(maxsize=None)
def flag(name: str) -> bool:
    """``enabled(name)``, read once per process."""
    return enabled(name)


@functools.lru_cache(maxsize=None)
def env_int(var: str, default: int) -> int:
    """Integer environment value (``default`` when unset or malformed), read once per process."""
    try:
        return int(os.environ.get(var, str(default)))
    except ValueError:
        return default


def sglang_root() -> Path | None:
    """Directory that contains the ``sglang`` package (found without importing it)."""
    spec = importlib.util.find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(list(spec.submodule_search_locations)[0]).resolve().parent


def file_sha256(relpath: str) -> str | None:
    """sha256 of the pinned source ``relpath``, read at its installed location (``sgl_compat.relpath``)."""
    from . import sgl_compat

    root = sglang_root()
    if root is None:
        return None
    try:
        return hashlib.sha256(
            (root / sgl_compat.relpath(relpath)).read_bytes()
        ).hexdigest()
    except OSError:
        return None


def sha_ok(got: str | None, want) -> bool:
    """``want``: one sha256 hex digest, or a tuple of the digests of every validated release of that source; a tuple
    entry ``"absent"`` accepts a release that does not have the source at all (a reference the component does not
    import, removed upstream)."""
    if got is None:
        return not isinstance(want, str) and "absent" in want
    return got == want if isinstance(want, str) else got in want


def sha_label(want) -> str:
    if isinstance(want, str):
        return want[:16]
    return "|".join(
        f"functions:{len(w)}" if isinstance(w, Functions) else w[:16] for w in want
    )


class Functions(dict):
    """A pin tuple entry, ``{qualname: digest}`` (``Class.method`` or ``function``; a digest or a tuple of digests): a
    revision of the source whose sha256 the tuple does not list passes when every function named here has a listed
    digest. Name the functions the component hooks, mirrors or calls in that source; an empty entry leaves the source
    to the names its ``SGLANG_READS`` declares."""


@functools.lru_cache(maxsize=None)
def source_text(relpath: str) -> str | None:
    """Text of the pinned source ``relpath`` at its installed location, read once per process."""
    from . import sgl_compat

    root = sglang_root()
    if root is None:
        return None
    try:
        return (root / sgl_compat.relpath(relpath)).read_text()
    except (OSError, UnicodeDecodeError):
        return None


def function_digest(text: str, qualname: str) -> str | None:
    """Token digest of ``qualname`` in the Python source ``text`` (None when it does not define it): decorators,
    signature and body without comments, blank lines or docstring. ``test/unit/test_contract.py``'s
    ``mirror_digest`` computes the same digest for ``SGLANG_MIRRORS``."""
    try:
        scope = ast.parse(text).body
    except SyntaxError:
        return None
    *owners, name = qualname.split(".")
    for owner in owners:
        cls = next(
            (n for n in scope if isinstance(n, ast.ClassDef) and n.name == owner), None
        )
        if cls is None:
            return None
        scope = cls.body
    node = next(
        (
            n
            for n in scope
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name
        ),
        None,
    )
    if node is None:
        return None
    first = min([node.lineno, *(d.lineno for d in node.decorator_list)])
    lines = text.splitlines(keepends=True)[first - 1 : node.end_lineno]
    doc = node.body[0]
    has_doc = (
        isinstance(doc, ast.Expr)
        and isinstance(doc.value, ast.Constant)
        and isinstance(doc.value.value, str)
    )
    skip = (doc.lineno - first + 1, doc.end_lineno - first + 1) if has_doc else None
    digest = hashlib.sha256()
    for tok in tokenize.generate_tokens(io.StringIO("".join(lines)).readline):
        if tok.type in (tokenize.COMMENT, tokenize.NL):
            continue
        if skip and skip[0] <= tok.start[0] and tok.end[0] <= skip[1]:
            continue
        # indentation depth is structure, not text
        tok_text = "" if tok.type in (tokenize.INDENT, tokenize.NEWLINE) else tok.string
        digest.update(f"{tok.type}:{tok_text}\n".encode())
    return digest.hexdigest()[:16]


def functions_ok(relpath: str, fns: Functions) -> bool:
    """True when every function ``fns`` names has one of its digests in the installed source ``relpath``."""
    text = source_text(relpath)
    if text is None:
        return False
    for qualname, want in fns.items():
        got = function_digest(text, qualname)
        if got is None or got not in ((want,) if isinstance(want, str) else want):
            logger.warning(
                "muse_vr_plugin: %s %s digest %s does not match the pin %s",
                relpath,
                qualname,
                got or "<missing>",
                want if isinstance(want, str) else "|".join(want),
            )
            return False
    return True


def pinned(relpath: str, sha256_hex) -> bool:
    """``sha256_hex``: one digest, or a tuple of the digests of every validated revision of that source (a tuple may
    also hold ``Functions`` entries, checked when the installed sha256 is not listed)."""
    got = file_sha256(relpath)
    if sha_ok(got, sha256_hex):
        return True
    if got is not None and not isinstance(sha256_hex, str):
        if any(
            isinstance(w, Functions) and functions_ok(relpath, w) for w in sha256_hex
        ):
            return True
    logger.warning(
        "muse_vr_plugin: %s sha256 %s does not match the pin %s",
        relpath,
        (got or "<missing>")[:16],
        sha_label(sha256_hex),
    )
    return False


def _dump() -> None:
    d = os.environ.get("MUSE_VR_ENGAGE_DIR")
    if not d:
        return
    try:
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"install_{os.getpid()}.json")
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(
                {
                    "pid": os.getpid(),
                    "t": time.time(),
                    "installs": _installs,
                    "counts": _counts,
                },
                f,
                indent=1,
            )
        os.replace(tmp, path)
    except OSError as e:
        logger.warning(
            "muse_vr_plugin: cannot write engagement record to %s (%s)", d, e
        )


def report(name: str, engaged: bool, **info) -> None:
    """Record an install decision; logs once per process and component."""
    with _lock:
        first = name not in _installs
        _installs[name] = dict(info, engaged=bool(engaged), t=time.time())
        _dump()
    if first:
        extra = " ".join(f"{k}={v}" for k, v in info.items())
        logger.warning(
            "muse_vr_plugin: %s engaged=%s pid=%d %s",
            name,
            bool(engaged),
            os.getpid(),
            extra,
        )


def count(name: str, key: str = "calls", n: int = 1) -> None:
    with _lock:
        c = _counts.setdefault(name, {})
        new_key = key not in c
        c[key] = c.get(key, 0) + n
        if new_key:  # write on new keys only: no file I/O on hot eager paths
            _dump()


def shadow_result(name: str, ok: bool) -> None:
    """Tally a shadow (debug) comparison. Mismatches are tallied under a new key on the first one; matches also leave a
    ``match>=2^k`` key each time the tally reaches a power of two, so the engagement record shows progress."""
    key = "shadow:match" if ok else "shadow:MISMATCH"
    count(name, key)
    n = _counts.get(name, {}).get(key, 0)
    if ok and n & (n - 1) == 0:
        count(name, f"shadow:match>={n}")


def installs() -> dict:
    return {k: dict(v) for k, v in _installs.items()}


def counts() -> dict:
    return {k: dict(v) for k, v in _counts.items()}
