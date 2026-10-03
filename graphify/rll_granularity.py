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
