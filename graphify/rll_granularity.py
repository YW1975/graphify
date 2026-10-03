"""Finer-grained code-graph facts for change-impact questions.

Copyright 2026 RLL project contributors.
Licensed under the Apache License, Version 2.0 (see LICENSE). This module is
not part of upstream Graphify; it was added by the RLL project (2026-10).

A graph that answers "which tests exercise the code this diff touched" needs
four facts the base extractor does not record:

* ``source_range`` — the first and last line of every function / method /
  class node, so a diff hunk can be mapped onto the callable that contains it
  (``source_location`` only names the first line).
* ``role`` — whether a node is product code, a test case, a test helper, a
  script or documentation, so a product-only traversal can stay out of test
  code. See :data:`DEFAULT_ROLE_RULES`.
* test-case nodes — one node per ``describe`` / ``it`` / ``test`` call, owning
  the calls written inside its callback (previously they were attributed to
  the file node, where they competed with the file's import edges).
* ``invokes_cli`` edges — from the code that launches the project's CLI as a
  subprocess to the command handler the subcommand dispatches to. See
  :func:`resolve_cli_invocations`.
"""
from __future__ import annotations

import fnmatch
import json
import os
import re
from typing import Any, Iterable

# ── 1. Source ranges ─────────────────────────────────────────────────────────


def _line_of(node: dict) -> int | None:
    loc = node.get("source_location")
    if isinstance(loc, str) and loc.startswith("L"):
        try:
            return int(loc[1:].split("-")[0])
        except ValueError:
            return None
    if isinstance(loc, int):
        return loc
    return None


def annotate_source_ranges(
    nodes: list[dict],
    def_nodes: dict[str, Any],
    function_bodies: Iterable[tuple[str, Any]],
    skip_ids: Iterable[str],
) -> None:
    """Stamp ``source_range: [start, end]`` (1-based, inclusive) on callables.

    ``def_nodes`` maps an id to its defining AST node (function / method /
    class declaration): its own extent is the range. An id known only through
    a tracked body (``const f = () => …``, a nested declaration, a CommonJS
    member) takes its start from its ``source_location`` and its end from the
    furthest body owned by it. File / module-scope owners are skipped.
    """
    skip = set(skip_ids)
    ranges: dict[str, list[int]] = {}
    for nid, ts_node in def_nodes.items():
        if nid in skip:
            continue
        ranges[nid] = [ts_node.start_point[0] + 1, ts_node.end_point[0] + 1]
    body_end: dict[str, int] = {}
    for nid, body in function_bodies:
        if nid in skip or nid in ranges or body is None:
            continue
        end = body.end_point[0] + 1
        if end > body_end.get(nid, 0):
            body_end[nid] = end
    for n in nodes:
        nid = n.get("id")
        if nid in ranges:
            start, end = ranges[nid]
            loc = _line_of(n)
            # Keep the range consistent with the reported start line (a
            # decorator or `export` keyword may precede the declaration node).
            if loc is not None and loc < start:
                start = loc
            n["source_range"] = [start, max(start, end)]
        elif nid in body_end:
            start = _line_of(n)
            if start is None:
                continue
            n["source_range"] = [start, max(start, body_end[nid])]


# ── 2. Roles ─────────────────────────────────────────────────────────────────
#
# Every node gets ``role``: one of
#   product      shipped code (the default)
#   test-case    code the test runner executes as a test: a test-case /
#                test-suite node, and a test file's own file / module-scope node
#   test-helper  reusable test support: a named function/class/const declared
#                in a test file, or anything under a test-support directory
#   script       repository tooling under scripts/
#   doc          documentation (.md/.mdx/.rst/.txt, docs/)
#   external     a stub for a symbol defined outside the corpus (no source file)
#   sentinel     a synthetic node standing for "cannot be resolved statically"
#
# Rules are ordered (first match wins) and matched against the POSIX path
# relative to the scan root. ``**`` spans directories, ``*`` stays inside one.
# Override: set GRAPHIFY_ROLE_RULES to a JSON file holding
#   {"rules": [{"role": "test-helper", "globs": ["e2e/support/**"]}, ...],
#    "replace": false}
# Extra rules are consulted BEFORE the defaults; "replace": true drops them.
# The pseudo-role "test-file" marks a file as a test file (its file / module
# node and test cases become test-case, its named declarations test-helper).

DEFAULT_ROLE_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("test-file", (
        "**/*.test.*", "**/*.spec.*", "**/__tests__/**",
        "**/test_*.py", "**/*_test.py", "**/*_test.go",
    )),
    ("test-helper", (
        "**/test-lib/**", "**/test-utils/**", "**/test-helpers/**",
        "**/fixtures/**", "**/__fixtures__/**", "**/__mocks__/**",
        "**/test/**", "**/tests/**", "**/testdata/**", "**/conftest.py",
    )),
    ("script", ("**/scripts/**",)),
    ("doc", ("**/docs/**", "**/*.md", "**/*.mdx", "**/*.rst", "**/*.txt")),
]

ROLES = ("product", "test-case", "test-helper", "script", "doc", "external", "sentinel")


def _glob_to_regex(glob: str) -> re.Pattern:
    out = []
    i = 0
    while i < len(glob):
        if glob.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif glob.startswith("**", i):
            out.append(".*")
            i += 2
        elif glob[i] == "*":
            out.append("[^/]*")
            i += 1
        elif glob[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(glob[i]))
            i += 1
    return re.compile("^" + "".join(out) + "$")


def load_role_rules(env: dict | None = None) -> list[tuple[str, list[re.Pattern]]]:
    env = os.environ if env is None else env
    rules: list[tuple[str, tuple[str, ...]]] = list(DEFAULT_ROLE_RULES)
    cfg_path = env.get("GRAPHIFY_ROLE_RULES")
    if cfg_path:
        try:
            with open(cfg_path, encoding="utf-8") as fh:
                cfg = json.load(fh)
            extra = [(r["role"], tuple(r["globs"])) for r in cfg.get("rules", [])]
            rules = extra if cfg.get("replace") else extra + rules
        except (OSError, ValueError, KeyError, TypeError):
            pass  # a broken override must not break extraction; defaults apply
    return [(role, [_glob_to_regex(g) for g in globs]) for role, globs in rules]


def path_role(source_file: str, rules) -> str:
    sf = source_file.replace("\\", "/") if source_file else ""
    while sf.startswith("./"):
        sf = sf[2:]
    for role, pats in rules:
        if any(p.match(sf) for p in pats):
            return role
    return "product"


def assign_roles(nodes: list[dict], rules=None) -> None:
    """Set ``role`` on every node that does not already carry one."""
    rules = load_role_rules() if rules is None else rules
    cache: dict[str, str] = {}
    for n in nodes:
        if n.get("role"):
            continue
        if n.get("test_kind"):
            n["role"] = "test-case"
            continue
        sf = n.get("source_file") or ""
        if not sf:
            n["role"] = "external"
            continue
        role = cache.get(sf)
        if role is None:
            role = cache[sf] = path_role(sf, rules)
        if role == "test-file":
            label = n.get("label") or ""
            is_file_node = label == sf.rsplit("/", 1)[-1] or label == sf
            role = "test-case" if (is_file_node or label == "<module scope>") else "test-helper"
        n["role"] = role
