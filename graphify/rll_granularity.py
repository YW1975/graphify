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
            if n.get("source_file") and n.get("test_name"):
                n["qualified_name"] = f"{n['source_file']}::{n['test_name']}"
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


# ── 3. Test-case nodes (JS / TS) ─────────────────────────────────────────────
#
# One node per `describe` / `it` / `test` call (and their `.skip` / `.only` /
# `.todo` / `.each(...)` / `.concurrent` forms, Playwright `test.describe`, and
# node:test subtests `t.test(...)` on the callback's context parameter). The
# callback body is owned by that node, so calls written inside it are
# attributed to the case instead of to the file node, where they collided with
# the file's own import edges (one edge per node pair) and were lost.
#
# Node fields: test_kind ("suite" | "case"), test_name ("Outer > inner >
# title"), qualified_name ("<source_file>::<test_name>", set once source_file
# is relative), test_modifiers, test_title_dynamic, source_range = callback.
# Id: make_id(file stem, "test", test_name), suffixed _2, _3, ... on collision
# in source order, so it is deterministic and unique within the file.

_SUITE_FUNCS = frozenset({"describe", "suite", "context", "fdescribe", "xdescribe"})
_CASE_FUNCS = frozenset({"it", "test", "specify", "fit", "xit", "xtest"})
_TEST_MODIFIERS = frozenset({
    "skip", "only", "todo", "concurrent", "sequential", "serial", "failing",
    "fixme", "slow", "each", "fails", "runIf", "skipIf",
})
_TEST_MODULES = (
    "node:test", "test", "vitest", "@jest/globals", "mocha", "@playwright/test",
    "jest", "ava", "tap", "uvu", "bun:test",
)
_TEST_FILE_RE = re.compile(r"(?:^|[/\\])(?:[^/\\]+\.(?:test|spec)\.[^/\\]+|__tests__[/\\].*)$")
_JS_FN_TYPES = frozenset({"arrow_function", "function_expression", "function", "generator_function"})


def _text(node, source: bytes) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _file_uses_test_framework(root, source: bytes, str_path: str) -> bool:
    if _TEST_FILE_RE.search(str_path):
        return True
    for c in root.children:
        if c.type == "import_statement":
            src = c.child_by_field_name("source")
            if src is not None and _text(src, source).strip("'\"`") in _TEST_MODULES:
                return True
    head = source[:4000]
    return any(f"require('{m}')".encode() in head or f'require("{m}")'.encode() in head
               for m in _TEST_MODULES)


def _string_value(node, source: bytes) -> tuple[str, bool] | None:
    """(text, dynamic) for a string / template literal, else None."""
    if node.type == "string":
        return _text(node, source)[1:-1], False
    if node.type == "template_string":
        dynamic = any(c.type == "template_substitution" for c in node.children)
        return _text(node, source)[1:-1], dynamic
    return None


def _test_callee(fn, source: bytes, ctx_params: frozenset[str]):
    """Classify a call's function expression: (kind, base, modifiers) or None."""
    modifiers: list[str] = []
    cur = fn
    if cur.type == "call_expression":  # it.each(table)(title, fn)
        inner = cur.child_by_field_name("function")
        if inner is None or inner.type != "member_expression":
            return None
        prop = inner.child_by_field_name("property")
        if prop is None or _text(prop, source) != "each":
            return None
        modifiers.append("each")
        cur = inner.child_by_field_name("object")
    while cur is not None and cur.type == "member_expression":
        prop = cur.child_by_field_name("property")
        obj = cur.child_by_field_name("object")
        name = _text(prop, source) if prop is not None else ""
        if obj is not None and obj.type == "identifier":
            base = _text(obj, source)
            # t.test(...) / t.describe(...) on a node:test context parameter
            if base in ctx_params and name in ("test", "it", "describe", "suite"):
                return ("suite" if name in ("describe", "suite") else "case"), name, modifiers
            # test.describe(...) (Playwright), test.step(...) is NOT a case
            if base in _CASE_FUNCS | _SUITE_FUNCS and name == "describe":
                return "suite", base, modifiers
        if name not in _TEST_MODIFIERS:
            return None
        modifiers.insert(0, name)
        cur = obj
    if cur is None or cur.type != "identifier":
        return None
    base = _text(cur, source)
    if base in _SUITE_FUNCS:
        return "suite", base, modifiers
    if base in _CASE_FUNCS:
        return "case", base, modifiers
    return None


def js_collect_test_cases(
    root, source: bytes, *, stem: str, str_path: str, file_nid: str,
    nodes: list, edges: list, seen_ids: set, function_bodies: list,
    closure_locals_by_body: dict, make_id, local_names_of,
    direct_lexical_names_of,
) -> list[str]:
    """Create test-suite / test-case nodes and give each its callback body.

    Must run before the call walk registers tracked bodies, so the walk
    attributes a callback's calls to its case and does not re-descend into it
    from the enclosing suite. Returns the new node ids.
    """
    if not _file_uses_test_framework(root, source, str_path):
        return []
    created: list[str] = []
    claimed: dict[Any, str] = {}

    def add(kind, base, title, dynamic, modifiers, call, callback, parent_nid, path, locals_):
        name = " > ".join(path + [title])
        base_id = make_id(stem, "test", name) or make_id(stem, "test")
        nid, k = base_id, 2
        while nid in seen_ids:
            nid, k = f"{base_id}_{k}", k + 1
        seen_ids.add(nid)
        line = call.start_point[0] + 1
        span = callback if callback is not None else call
        node = {
            # `it: title` — the prefix keeps a title from ever reading as an
            # identifier to the label-based resolvers (a test titled `Foo`
            # must not absorb a stub for the class `Foo`).
            "id": nid, "label": f"{base}: {title}", "file_type": "code", "source_file": str_path,
            "source_location": f"L{line}",
            "source_range": [span.start_point[0] + 1, span.end_point[0] + 1],
            "test_kind": kind, "test_name": name,
        }
        if modifiers:
            node["test_modifiers"] = list(modifiers)
        if dynamic:
            node["test_title_dynamic"] = True
        nodes.append(node)
        edges.append({
            "source": parent_nid, "target": nid, "relation": "contains",
            "confidence": "EXTRACTED", "source_file": str_path,
            "source_location": f"L{line}", "weight": 1.0,
        })
        created.append(nid)
        if callback is not None:
            body = callback.child_by_field_name("body")
            if body is not None:
                claimed[body] = nid
                closure_locals_by_body[id(body)] = set(locals_)
        return nid

    def scan(node, parent_nid, path, ctx_params, locals_):
        for c in node.children:
            if c.type == "call_expression":
                fn = c.child_by_field_name("function")
                hit = _test_callee(fn, source, ctx_params) if fn is not None else None
                args = c.child_by_field_name("arguments")
                if hit is not None and args is not None:
                    kind, base, modifiers = hit
                    argv = [a for a in args.named_children if a.type != "comment"]
                    callback = next((a for a in reversed(argv) if a.type in _JS_FN_TYPES), None)
                    if argv and (callback is not None or "todo" in modifiers):
                        sv = _string_value(argv[0], source)
                        if sv is None:
                            title, dynamic = "<" + _text(argv[0], source)[:60] + ">", True
                        else:
                            title, dynamic = sv
                        title = " ".join(title.split()) or "<empty>"
                        cb_locals = set(locals_)
                        cb_params: frozenset[str] = frozenset()
                        if callback is not None:
                            own = local_names_of(callback, source)
                            cb_locals |= own
                            body = callback.child_by_field_name("body")
                            if body is not None and body.type == "statement_block":
                                cb_locals |= direct_lexical_names_of(body, source)
                            first = _first_param_name(callback, source)
                            cb_params = frozenset({first}) if first else frozenset()
                        nid = add(kind, base, title, dynamic, modifiers, c, callback,
                                  parent_nid, path, cb_locals)
                        # options object / other args may hold closures too, but
                        # only the callback body is the test; recurse into it.
                        if callback is not None:
                            body = callback.child_by_field_name("body")
                            if body is not None:
                                scan(body, nid, path + [title], cb_params, cb_locals)
                        continue
            scan(c, parent_nid, path, ctx_params, locals_)

    scan(root, file_nid, [], frozenset(), set())
    if claimed:
        # A top-level callback was registered under the FILE node by the
        # module-call branch (#3124); it now belongs to its case.
        function_bodies[:] = [(o, b) for (o, b) in function_bodies if b not in claimed]
        function_bodies.extend((nid, body) for body, nid in claimed.items())
    return created


def _first_param_name(fn_node, source: bytes) -> str | None:
    solo = fn_node.child_by_field_name("parameter")
    if solo is not None and solo.type == "identifier":
        return _text(solo, source)
    params = fn_node.child_by_field_name("parameters")
    if params is None:
        return None
    for p in params.named_children:
        target = p
        if p.type in ("required_parameter", "optional_parameter"):
            target = p.child_by_field_name("pattern") or p
        if target.type == "identifier":
            return _text(target, source)
        return None
    return None
