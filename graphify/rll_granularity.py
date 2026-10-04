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


# ── 1b. Module-level declarations (TS/JS) ────────────────────────────────────
#
# Symbol-level impact (RLL design 2026-10-04 §3.2): a change at module scope must be
# attributable to the declaration it touches. The base extractor gives a node and a
# range only to callables and types; data constants have a node only when exported or
# object/array/call-initialised, and never a range; scalars, `let` and `var` have no
# node at all. This pass, run after the base walk, gives EVERY top-level declarator a
# node with ``source_range``, ``decl_kind``, ``decl_name`` and ``load_effect``, and
# records the ranges of top-level statements that run code on load as
# ``effect_ranges`` on the ``<module scope>`` node. It only adds facts: no existing
# node or edge is renamed, and new nodes are not entered into any resolution table.

_JS_FUNCTION_VALUES = {"arrow_function", "function_expression", "function", "generator_function"}
_JS_FUNCTION_BODIES = _JS_FUNCTION_VALUES | {"method_definition", "class_body"}
_JS_EFFECT_NODES = {"call_expression", "new_expression", "await_expression"}
_JS_TYPE_DECLS = {"interface_declaration", "type_alias_declaration"}
_JS_EFFECT_STATEMENTS = {
    "expression_statement", "if_statement", "for_statement", "for_in_statement",
    "while_statement", "do_statement", "switch_statement", "try_statement",
    "labeled_statement", "throw_statement", "statement_block",
}


def _runs_code(node) -> bool:
    """Whether evaluating ``node`` at load time can run code: a call, ``new``, ``await``
    (a tagged template is a call). Function and class bodies are not entered — they run
    when called, not when the module loads."""
    if node is None:
        return False
    if node.type in _JS_EFFECT_NODES:
        return True
    if node.type in _JS_FUNCTION_BODIES:
        return False
    return any(_runs_code(c) for c in node.children)


def _is_import_binding(value) -> bool:
    """`require('./x')`, `require('./x').y`, `await import('./x')`: an import, not a declaration.
    Giving such a binding its own node would collide with the definition it only imports
    (the base extractor documents this trap for destructured requires, #2604)."""
    v = value
    # `require('./x') as T` is still the import (a cast is how TS types a CommonJS require).
    while v is not None and v.type in ("await_expression", "member_expression", "parenthesized_expression",
                                       "as_expression", "satisfies_expression", "non_null_expression"):
        v = next((c for c in v.children if c.is_named), None)
    if v is None or v.type != "call_expression":
        return False
    fn = v.child_by_field_name("function")
    return fn is not None and fn.type in ("identifier", "import") and fn.text in (b"require", b"import")


def _class_runs_code(cls) -> bool:
    body = cls.child_by_field_name("body")
    if any(c.type == "decorator" for c in cls.children):
        return True
    if body is None:
        return False
    for member in body.children:
        if member.type in ("class_static_block", "static_block"):
            return True
        if member.type == "decorator":
            return True
        if member.type in ("public_field_definition", "field_definition"):
            is_static = any(c.type == "static" for c in member.children)
            if is_static and _runs_code(member.child_by_field_name("value")):
                return True
    return False


def annotate_module_declarations(root, source: bytes, nodes: list, edges: list, seen_ids: set, *,
                                 file_nid: str, module_nid: str, stem: str, str_path: str,
                                 make_id, nid_of) -> None:
    if root is None or root.type != "program":
        return

    def text(n) -> str:
        return source[n.start_byte:n.end_byte].decode("utf-8", errors="replace")

    def find(label: str, line: int) -> dict | None:
        for n in nodes:
            if n.get("source_file") == str_path and n.get("label") == label and _line_of(n) == line:
                return n
        return None

    def stamp(node: dict, *, rng: list[int], kind: str, name: str, effect: bool) -> None:
        node.setdefault("source_range", rng)
        node["decl_kind"] = kind
        node["decl_name"] = name
        node["load_effect"] = "impure" if effect else "pure"

    def declare(name: str, line: int, rng: list[int], kind: str, effect: bool, *, create: bool = True) -> None:
        existing = find(name, line) or find(f"{name}()", line)
        if existing is not None:
            stamp(existing, rng=rng, kind=kind, name=name, effect=effect)
            return
        # A name that normalizes to nothing (minified `$`, `_`) would collapse the id to the
        # file stem and leak the scan path (#1899): no node, as the base extractor does.
        if not create or not make_id(name):
            return
        nid = nid_of(make_id(stem, name), name)
        if nid in seen_ids:
            # Same name as another declaration of this file (interface + const merging):
            # a distinct, deterministic id; ``decl_name`` keeps the name for lookups.
            nid = make_id(stem, f"{name}__value")
            if nid in seen_ids:
                return
        seen_ids.add(nid)
        node = {"id": nid, "label": name, "file_type": "code", "source_file": str_path,
                "source_location": f"L{line}"}
        stamp(node, rng=rng, kind=kind, name=name, effect=effect)
        nodes.append(node)
        edges.append({"source": file_nid, "target": nid, "relation": "contains", "context": "contains",
                      "confidence": "EXTRACTED", "source_file": str_path, "source_location": f"L{line}",
                      "weight": 1.0})

    effect_ranges: list[list[int]] = []
    for stmt in root.children:
        decl = stmt
        if stmt.type == "export_statement":
            inner = stmt.child_by_field_name("declaration")
            if inner is not None:
                decl = inner
            else:
                value = stmt.child_by_field_name("value")
                # `export default <expression>`: declarative unless evaluating it runs code.
                if value is not None and _runs_code(value):
                    effect_ranges.append([stmt.start_point[0] + 1, stmt.end_point[0] + 1])
                continue
        s_line, e_line = stmt.start_point[0] + 1, stmt.end_point[0] + 1
        t = decl.type
        if t in ("lexical_declaration", "variable_declaration"):
            declarators = [c for c in decl.children if c.type == "variable_declarator"]
            for d in declarators:
                name_node = d.child_by_field_name("name")
                value = d.child_by_field_name("value")
                rng = [s_line, e_line] if len(declarators) == 1 else [d.start_point[0] + 1, d.end_point[0] + 1]
                kind = "function" if value is not None and value.type in _JS_FUNCTION_VALUES else "data"
                effect = _runs_code(value) if kind == "data" else False
                line = d.start_point[0] + 1
                if _is_import_binding(value):
                    continue
                if name_node is not None and name_node.type == "identifier":
                    declare(text(name_node), line, rng, kind, effect)
                elif name_node is not None and name_node.type in ("object_pattern", "array_pattern"):
                    if stmt.type == "export_statement":
                        # Exported destructuring is named by the exported KEY and skips
                        # `...rest` / defaulted entries (#2604): annotate those nodes only.
                        for ident in _decl_pattern_keys(name_node, text):
                            declare(ident, line, rng, "data", effect, create=False)
                    else:
                        for ident in _decl_pattern_names(name_node, text):
                            declare(ident, line, rng, "data", effect)
                if effect:
                    effect_ranges.append(rng)
        elif t in ("function_declaration", "generator_function_declaration"):
            name_node = decl.child_by_field_name("name")
            if name_node is not None:
                declare(text(name_node), decl.start_point[0] + 1, [s_line, e_line], "function", False)
        elif t in ("class_declaration", "abstract_class_declaration"):
            name_node = decl.child_by_field_name("name")
            effect = _class_runs_code(decl)
            if name_node is not None:
                declare(text(name_node), decl.start_point[0] + 1, [s_line, e_line], "class", effect)
            if effect:
                effect_ranges.append([s_line, e_line])
        elif t in _JS_TYPE_DECLS or t == "enum_declaration":
            name_node = decl.child_by_field_name("name")
            if name_node is not None:
                declare(text(name_node), decl.start_point[0] + 1, [s_line, e_line],
                        "enum" if t == "enum_declaration" else "type", False)
        elif t in ("internal_module", "module"):
            name_node = decl.child_by_field_name("name")
            if name_node is not None:
                declare(text(name_node), decl.start_point[0] + 1, [s_line, e_line], "namespace", _runs_code(decl))
        elif t == "import_statement":
            # A bare `import './x'` exists only to run that module's top level.
            if not any(c.type in ("import_clause", "import_require_clause") for c in decl.children):
                effect_ranges.append([s_line, e_line])
        elif t in _JS_EFFECT_STATEMENTS:
            effect_ranges.append([s_line, e_line])

    if not effect_ranges:
        return
    module = next((n for n in nodes if n.get("id") == module_nid), None)
    if module is None:
        seen_ids.add(module_nid)
        module = {"id": module_nid, "label": "<module scope>", "file_type": "code",
                  "source_file": str_path, "source_location": "L1"}
        nodes.append(module)
        edges.append({"source": file_nid, "target": module_nid, "relation": "contains", "context": "contains",
                      "confidence": "EXTRACTED", "source_file": str_path, "source_location": "L1", "weight": 1.0})
    module["effect_ranges"] = effect_ranges


def _decl_pattern_keys(pattern, text) -> list[str]:
    """Exported names of an object pattern: shorthand names and pair KEYS."""
    out = []
    for prop in pattern.named_children:
        if prop.type == "shorthand_property_identifier_pattern":
            out.append(text(prop))
        elif prop.type == "pair_pattern":
            key = prop.child_by_field_name("key")
            if key is not None:
                out.append(text(key))
    return out


def _decl_pattern_names(pattern, text) -> list[str]:
    """Binding names in a destructuring pattern (defaults and nested patterns included)."""
    out: list[str] = []
    stack = [pattern]
    while stack:
        n = stack.pop()
        if n.type in ("shorthand_property_identifier_pattern", "identifier"):
            out.append(text(n))
            continue
        if n.type == "pair_pattern":
            v = n.child_by_field_name("value")
            if v is not None:
                stack.append(v)
            continue
        if n.type in ("assignment_pattern", "object_assignment_pattern"):
            left = n.child_by_field_name("left")
            if left is not None:
                stack.append(left)
            continue
        stack.extend(c for c in n.children if c.is_named)
    return sorted(set(out))


# ── 1c. Same-file read references (TS/JS) ────────────────────────────────────
#
# Symbol-level impact (RLL design 2026-10-04 §3.2 item 2, same-file half): when a
# module-level declaration changes, the code that READS it is affected, and a call
# graph cannot say which code that is — `const B = [...A]` or `return LIMIT * 2`
# calls nothing. This pass, run after the S2 declaration pass, emits
#
#     <owner> -references-> <module-level declaration>   context: read | write
#
# once per (owner, declaration, context), at the first such line. The owner is the
# innermost TRACKED callable (the base extractor's ``function_bodies``: functions,
# methods, test cases, const-assigned arrows), a field initializer's owner, the
# declaration whose initializer holds the read (`const B = [...A]` → B→A), the file
# for `export default X`, and `<module scope>` for statements that run on load.
#
# Shadowing reuses the base walk's collectors (`_js_local_bound_names` for the
# function-wide set, `_js_direct_lexical_names` per block), plus what only a read
# needs: a nested function/class declaration name, a named function expression's
# own name, catch and for-in/of bindings. Doubt resolves towards NOT shadowing — an
# extra edge selects a test too many, a wrongly shadowed one misses a test.
#
# Not reads: property names (`obj.X`, `{X: 1}`), type positions (`: X`, `typeof X`
# in a type), import/export specifier lists, labels, intrinsic JSX tags. Reads:
# value identifiers, shorthand `{X}`, defaults inside patterns, `export default X`,
# capitalised JSX tags (`<Foo/>` is `createElement(Foo)`). `X = …` is a write,
# `X += …` / `X++` both.

_JS_FN_SCOPES = frozenset({
    "function_declaration", "generator_function_declaration", "function_expression", "function",
    "generator_function", "arrow_function", "method_definition",
})
_JS_NOT_CODE = frozenset({
    "type_annotation", "type_arguments", "type_parameters", "type_query", "implements_clause",
    "nested_type_identifier", "interface_declaration", "type_alias_declaration", "opting_type_annotation",
    "omitting_type_annotation", "asserts_annotation", "type_predicate_annotation", "ambient_declaration",
    "abstract_method_signature", "method_signature", "function_signature", "index_signature",
    "import_statement", "export_clause", "statement_identifier", "jsx_closing_element", "comment",
})
_JS_NAMED_DECLS = frozenset({
    "function_declaration", "generator_function_declaration", "class_declaration", "abstract_class_declaration",
})


def annotate_same_file_reads(root, source: bytes, nodes: list, edges: list, *, function_bodies,
                             initializer_nodes, file_nid: str, module_nid: str, str_path: str,
                             import_refs: list | None = None) -> None:
    from graphify.extractors.engine import (
        _js_collect_pattern_idents, _js_direct_lexical_names, _js_local_bound_names,
    )
    if root is None or root.type != "program":
        return

    def text(n) -> str:
        return source[n.start_byte:n.end_byte].decode("utf-8", errors="replace")

    # Targets: this file's value declarations, by name. Not `label_to_nid`, which strips
    # leading dots and is last-write-wins. A type shares its name with a value only in
    # type position, which is never a read.
    decls: dict[str, str] = {}
    for n in nodes:
        if n.get("source_file") == str_path and n.get("decl_name") and n.get("decl_kind") != "type":
            decls.setdefault(n["decl_name"], n["id"])
    present = {n.get("id") for n in nodes}
    module_owner = module_nid if module_nid in present else file_nid

    def key(n):
        return (n.start_byte, n.end_byte, n.type)

    owner_of: dict[tuple, tuple[str, ...]] = {}
    for nid, body in function_bodies:
        if body is None or (nid != file_nid and nid not in present):
            continue
        owner = (module_owner if nid == file_nid else nid,)
        owner_of.setdefault(key(body), owner)
        if body.parent is not None and body.parent.type in _JS_FN_SCOPES:
            owner_of.setdefault(key(body.parent), owner)  # parameter defaults belong to it too
    for nid, init in initializer_nodes:
        if init is not None and nid in present:
            owner_of.setdefault(key(init), (nid,))

    found: dict[tuple[str, str, str], int] = {}
    imported: dict[tuple[str, str, str, str], int] = {}

    def import_ref(owners, spec: str, name: str, ctx: str, line: int, via: str) -> None:
        for o in owners:
            k = (o, spec, name, ctx)
            if k not in imported or line < imported[k][0]:
                imported[k] = (line, via)

    def binding_of(name: str, scopes):
        """The innermost binding of ``name``: None for a plain local, (spec, imported, via)
        for an import / namespace / require binding, or ``False`` when nothing binds it."""
        for s in reversed(scopes):
            if name in s:
                return s.get(name) if isinstance(s, dict) else None
        return False

    def ref(ident, owners, scopes, ctx: str) -> None:
        name = text(ident)
        line = ident.start_point[0] + 1
        bound = binding_of(name, scopes)
        if bound is not False:
            if bound and ctx == "read":
                import_ref(owners, bound[0], bound[1], ctx, line, bound[2])  # resolved across files later
            return
        target = decls.get(name)
        if target is None:
            return
        for o in owners:
            if o != target:
                k = (o, target, ctx)
                if k not in found or line < found[k]:
                    found[k] = line

    def names_of(pattern) -> frozenset[str]:
        bound: set[str] = set()
        _js_collect_pattern_idents(pattern, source, bound)
        return frozenset(bound)

    def require_spec(n) -> str | None:
        """`require('./m')` -> './m'."""
        if n is None or n.type != "call_expression":
            return None
        fn, args = n.child_by_field_name("function"), n.child_by_field_name("arguments")
        if fn is None or fn.type != "identifier" or text(fn) != "require" or args is None:
            return None
        first = next((a for a in args.named_children if a.type != "comment"), None)
        if first is None or first.type != "string":
            return None
        return text(first)[1:-1] or None

    def require_value(value):
        """(spec, member) for `require('./m')` (member None) or `require('./m').x`."""
        v = value
        while v is not None and v.type in ("parenthesized_expression", "await_expression", "as_expression",
                                           "satisfies_expression", "non_null_expression"):
            v = next((c for c in v.named_children if c.type != "comment"), None)
        member = None
        if v is not None and v.type == "member_expression":
            prop = v.child_by_field_name("property")
            if prop is None or prop.type != "property_identifier":
                return None
            member, v = text(prop), v.child_by_field_name("object")
        spec = require_spec(v)
        return (spec, member) if spec else None

    def require_bindings(holder) -> dict:
        """Names bound by `const|let|var <pattern> = require(...)` directly in ``holder``."""
        out: dict = {}
        for decl in holder.named_children:
            if decl.type not in ("lexical_declaration", "variable_declaration"):
                continue
            for d in decl.named_children:
                if d.type != "variable_declarator":
                    continue
                req = require_value(d.child_by_field_name("value"))
                name = d.child_by_field_name("name")
                if req is None or name is None:
                    continue
                spec, member = req
                if name.type == "identifier":
                    out[text(name)] = (spec, member or "*", "require")
                elif name.type == "object_pattern" and member is None:
                    for prop in name.named_children:
                        if prop.type == "shorthand_property_identifier_pattern":
                            out[text(prop)] = (spec, text(prop), "require")
                        elif prop.type == "object_assignment_pattern":
                            left = prop.child_by_field_name("left")
                            if left is not None:
                                out[text(left)] = (spec, text(left), "require")
                        elif prop.type == "pair_pattern":
                            k, v = prop.child_by_field_name("key"), prop.child_by_field_name("value")
                            if k is not None and v is not None and v.type == "identifier" \
                                    and k.type == "property_identifier":
                                out[text(v)] = (spec, text(k), "require")
        return out

    def import_bindings() -> dict:
        """Top-level `import` clauses and `require` declarators of this file."""
        out = require_bindings(root)
        for stmt in root.named_children:
            if stmt.type != "import_statement":
                continue
            src = stmt.child_by_field_name("source")
            clause = next((c for c in stmt.named_children if c.type == "import_clause"), None)
            if src is None or clause is None:
                continue
            spec = text(src)[1:-1]
            for c in clause.named_children:
                if c.type == "identifier":
                    out[text(c)] = (spec, "default", "import")
                elif c.type == "namespace_import":
                    ident = next((x for x in c.named_children if x.type == "identifier"), None)
                    if ident is not None:
                        out[text(ident)] = (spec, "*", "namespace")
                elif c.type == "named_imports":
                    for sp in c.named_children:
                        if sp.type != "import_specifier":
                            continue
                        name, alias = sp.child_by_field_name("name"), sp.child_by_field_name("alias")
                        if name is not None:
                            out[text(alias if alias is not None else name)] = (spec, text(name), "import")
        return out

    def pattern(n, owners, scopes, mode: str) -> None:
        """A binding (mode "bind") or assignment target (mode "write"): its names are
        bound or written, its defaults and computed keys are read."""
        t = n.type
        if t in ("identifier", "shorthand_property_identifier_pattern"):
            if mode == "write":
                ref(n, owners, scopes, "write")
            return
        if t in _JS_NOT_CODE:
            return
        if t in ("assignment_pattern", "object_assignment_pattern"):
            left, right = n.child_by_field_name("left"), n.child_by_field_name("right")
            if left is not None:
                pattern(left, owners, scopes, mode)
            if right is not None:
                visit(right, owners, scopes)
            return
        if t == "pair_pattern":
            k, v = n.child_by_field_name("key"), n.child_by_field_name("value")
            if k is not None and k.type == "computed_property_name":
                visit(k, owners, scopes)
            if v is not None:
                pattern(v, owners, scopes, mode)
            return
        if t in ("required_parameter", "optional_parameter"):
            p, v = n.child_by_field_name("pattern"), n.child_by_field_name("value")
            if p is not None:
                pattern(p, owners, scopes, mode)
            if v is not None:
                visit(v, owners, scopes)
            return
        if t in ("member_expression", "subscript_expression"):
            visit(n, owners, scopes)  # `X.y = …` reads the binding X
            return
        for c in n.named_children:
            pattern(c, owners, scopes, mode)

    def visit(n, owners, scopes) -> None:
        t = n.type
        if t in _JS_NOT_CODE:
            return
        owners = owner_of.get(key(n), owners)
        if t in ("identifier", "shorthand_property_identifier"):
            ref(n, owners, scopes, "read")
            return
        if t in _JS_FN_SCOPES:
            local = set(_js_local_bound_names(n, source))
            name = n.child_by_field_name("name")
            if name is not None and t in ("function_expression", "function", "generator_function"):
                local.add(text(name))  # a named function expression sees its own name
            inner = scopes + (frozenset(local),)
            for i, c in enumerate(n.children):
                field = n.field_name_for_child(i)
                if not c.is_named or field in ("name", "return_type", "type_parameters"):
                    continue
                if field in ("parameters", "parameter"):
                    pattern(c, owners, inner, "bind")
                else:
                    visit(c, owners, inner)
            return
        if t in ("statement_block", "for_statement", "switch_body"):
            local = set()
            for holder in ([n] if t != "switch_body" else n.named_children):
                local |= _js_direct_lexical_names(holder, source)
                for c in holder.named_children:
                    if c.type in _JS_NAMED_DECLS and c.child_by_field_name("name") is not None:
                        local.add(text(c.child_by_field_name("name")))
            required = {}
            for holder in ([n] if t != "switch_body" else n.named_children):
                required.update(require_bindings(holder))
            if local:
                scopes = scopes + (frozenset(local - required.keys()),)
            if required:
                scopes = scopes + (required,)  # a lazy require binds its import, not a local
        elif t == "for_in_statement":
            left, right, body = (n.child_by_field_name(f) for f in ("left", "right", "body"))
            if right is not None:
                visit(right, owners, scopes)
            if any(c.type in ("let", "const", "var") for c in n.children):
                inner = scopes + (names_of(left),) if left is not None else scopes
                if left is not None:
                    pattern(left, owners, inner, "bind")
            else:
                inner = scopes
                if left is not None:
                    pattern(left, owners, scopes, "write")
            if body is not None:
                visit(body, owners, inner)
            return
        elif t == "catch_clause":
            param, body = n.child_by_field_name("parameter"), n.child_by_field_name("body")
            inner = scopes + (names_of(param),) if param is not None else scopes
            if param is not None:
                pattern(param, owners, inner, "bind")
            if body is not None:
                visit(body, owners, inner)
            return
        elif t == "variable_declarator":
            name, value = n.child_by_field_name("name"), n.child_by_field_name("value")
            if name is not None:
                pattern(name, owners, scopes, "bind")
            if value is not None:
                visit(value, owners, scopes)
            return
        elif t in ("assignment_expression", "augmented_assignment_expression", "update_expression"):
            target = n.child_by_field_name("argument" if t == "update_expression" else "left")
            right = n.child_by_field_name("right")
            if target is not None:
                if t == "assignment_expression" and target.type != "identifier":
                    if target.type in ("object_pattern", "array_pattern"):
                        pattern(target, owners, scopes, "write")
                    else:
                        visit(target, owners, scopes)
                elif target.type == "identifier":
                    ref(target, owners, scopes, "write")
                    if t != "assignment_expression":
                        ref(target, owners, scopes, "read")
                else:
                    visit(target, owners, scopes)
            if right is not None:
                visit(right, owners, scopes)
            return
        elif t in ("jsx_opening_element", "jsx_self_closing_element"):
            name = n.child_by_field_name("name")
            if name is not None:
                if name.type == "identifier":
                    if text(name)[:1].isupper():
                        ref(name, owners, scopes, "read")
                else:
                    visit(name, owners, scopes)  # `<Lib.Item>` reads Lib
            for c in n.named_children:
                if name is None or key(c) != key(name):
                    visit(c, owners, scopes)
            return
        elif t == "member_expression":
            obj, prop = n.child_by_field_name("object"), n.child_by_field_name("property")
            if prop is not None and prop.type == "property_identifier" and obj is not None:
                line = prop.start_point[0] + 1
                if obj.type == "identifier":
                    bound = binding_of(text(obj), scopes)
                    if bound and bound[1] == "*":
                        import_ref(owners, bound[0], text(prop), "read", line, bound[2])  # `ns.X` reads X
                        return
                spec = require_spec(obj)
                if spec:
                    import_ref(owners, spec, text(prop), "read", line, "require")  # `require('./m').X`
                    return
        elif t == "pair":
            k, v = n.child_by_field_name("key"), n.child_by_field_name("value")
            if k is not None and k.type == "computed_property_name":
                visit(k, owners, scopes)
            if v is not None:
                visit(v, owners, scopes)
            return
        elif t in ("class_declaration", "abstract_class_declaration", "class", "enum_declaration",
                   "internal_module", "module"):
            name = n.child_by_field_name("name")
            if t == "class" and name is not None:
                scopes = scopes + (frozenset({text(name)}),)
            for c in n.named_children:
                if name is None or key(c) != key(name):
                    visit(c, owners, scopes)
            return
        for c in n.named_children:
            visit(c, owners, scopes)

    base = (import_bindings(),)
    for stmt in root.named_children:
        inner = stmt
        if stmt.type == "import_statement":
            continue
        if stmt.type == "export_statement":
            decl = stmt.child_by_field_name("declaration")
            if decl is None:
                value = stmt.child_by_field_name("value")
                if value is not None:
                    visit(value, (file_nid,), base)  # `export default X` reads X
                continue  # `export { X }` / `export * from`: names, not reads
            inner = decl
        if inner.type in ("lexical_declaration", "variable_declaration"):
            for d in inner.named_children:
                if d.type != "variable_declarator":
                    continue
                name, value = d.child_by_field_name("name"), d.child_by_field_name("value")
                if name is None:
                    continue
                own = tuple(decls[x] for x in sorted(names_of(name)) if x in decls) or (module_owner,)
                pattern(name, own, base, "bind")
                if value is not None:
                    visit(value, own, base)  # a local `require` (createRequire) is still read
            continue
        name = inner.child_by_field_name("name") if inner.type in _JS_NAMED_DECLS | {
            "enum_declaration", "internal_module", "module"} else None
        owner = decls.get(text(name)) if name is not None else None
        visit(inner, (owner,) if owner else (module_owner,), base)

    if import_refs is not None:
        for (o, spec, name, ctx), (line, via) in sorted(imported.items()):
            import_refs.append({"owner": o, "spec": spec, "imported": name, "ctx": ctx, "line": line,
                                "via": via})
    for (o, target, ctx), line in sorted(found.items()):
        edges.append({"source": o, "target": target, "relation": "references", "context": ctx,
                      "confidence": "EXTRACTED", "source_file": str_path, "source_location": f"L{line}",
                      "weight": 1.0})


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
    from the enclosing suite. Returns the new node ids and the callback bodies.
    """
    if not _file_uses_test_framework(root, source, str_path):
        return [], set()
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
        # A callback was registered under the FILE node by the module-call
        # branch (#3124); it now belongs to its case. So does any other
        # closure that branch parked on the file node from INSIDE a case
        # (`assert.throws(() => f())`): its calls run as part of that case.
        rest: list = []
        for owner, body in function_bodies:
            if body in claimed:
                continue
            if owner == file_nid and body is not None:
                cur = body.parent
                while cur is not None and cur not in claimed:
                    cur = cur.parent
                if cur is not None:
                    owner = claimed[cur]
            rest.append((owner, body))
        function_bodies[:] = rest
        function_bodies.extend((nid, body) for body, nid in claimed.items())
    return created, set(claimed)


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


# ── 4. CLI invocations (JS / TS) ─────────────────────────────────────────────
#
# Tests launch the CLI as a subprocess — `spawnSync(process.execPath, [CLI,
# 'whose-turn'])`, or through helpers `run(dir, ['whose-turn'])` that forward a
# parameter into such an argv — so there is no call edge from the test to the
# command handler. This pass recovers it as an `invokes_cli` edge.
#
# Per file (syntactic, cached with the file): each owner node (function,
# method, test case, module scope) gets `_rll_cli` facts — its parameters, the
# spawn-family calls it makes with their program and argv described as value
# templates, and the call sites it makes with their arguments described the
# same way. A `switch` over string literals becomes a dispatch table
# (`_rll_dispatch`) on its owner.
#
# Corpus-wide (resolve_cli_invocations): a function whose spawn depends on its
# parameters gets a summary; call sites substitute their arguments into the
# callee's summary, to a fixpoint over the resolved call edges (no depth
# limit, across files). Where the program, CLI entry and subcommand are all
# known, an `invokes_cli` edge goes from the calling node to the handler(s)
# that subcommand dispatches to. Anything not statically decidable (computed /
# template / loop / callback values, a script path instead of the CLI, a
# parameter nobody binds) still produces an edge, marked `uncertain: true`
# with a `reason`, to the dispatcher or to a sentinel node — never dropped.
#
# Value descriptors (JSON lists, so they survive the AST cache):
#   ["lit", s]  ["node"]  ["cli", base, rel]  ["path", tail]  ["param", i]
#   ["spread", i] (array param spread into an argv)  ["array", [elems]]
#   ["import", name]  ["unknown", reason] / ["unknown", reason, 1] (variadic)
#
# Configuration (environment):
#   GRAPHIFY_CLI_PATH_PATTERN  regex a path's literal tail must match to be the
#                              CLI entry (default: basename cli.js/.mjs/.cjs)
#   GRAPHIFY_CLI_ENTRY         source file of the dispatcher (default: derived
#                              from the CLI path, e.g. dist/cli.js -> src/cli.ts)
#   GRAPHIFY_CLI_BIN_NAMES     comma-separated executable names that ARE the CLI

_SPAWN_ARGV = frozenset({"spawn", "spawnSync", "execFile", "execFileSync", "execa", "execaSync"})
_SPAWN_FORK = frozenset({"fork", "execaNode"})
_SPAWN_SHELL = frozenset({"exec", "execSync", "execaCommand", "execaCommandSync"})
_CP_OBJECTS = frozenset({"cp", "childProcess", "child_process", "child", "proc", "execa"})
_PATH_FUNCS = frozenset({"join", "resolve", "normalize", "fileURLToPath", "URL", "relative"})
_SHELLS = frozenset({"bash", "sh", "zsh", "/bin/sh", "/bin/bash"})
_NODE_NAMES = frozenset({"node", "nodejs"})
_FN_LIKE = frozenset({
    "arrow_function", "function_expression", "function", "generator_function",
    "function_declaration", "generator_function_declaration", "method_definition",
})
_MAX_DEPTH = 10


def cli_path_pattern() -> re.Pattern:
    return re.compile(os.environ.get("GRAPHIFY_CLI_PATH_PATTERN") or r"(?:^|/)cli\.[cm]?js$")


def _param_list(fn_node, source: bytes) -> tuple[list[str | None], int]:
    """Positional parameter names (None for destructuring) and the rest index."""
    names: list[str | None] = []
    rest = -1
    solo = fn_node.child_by_field_name("parameter")
    if solo is not None:
        return [_text(solo, source) if solo.type == "identifier" else None], -1
    params = fn_node.child_by_field_name("parameters")
    if params is None:
        return names, rest
    for p in params.named_children:
        if p.type == "comment":
            continue
        target = p
        if p.type in ("required_parameter", "optional_parameter"):
            target = p.child_by_field_name("pattern") or p
        if target.type == "assignment_pattern":
            target = target.child_by_field_name("left") or target
        if target.type == "rest_pattern":
            rest = len(names)
            inner = next((c for c in target.named_children if c.type == "identifier"), None)
            names.append(_text(inner, source) if inner is not None else None)
            continue
        names.append(_text(target, source) if target.type == "identifier" else None)
    return names, rest


class _Facts:
    """Syntactic CLI facts of one file."""

    def __init__(self, source: bytes, tracked: dict, pattern: re.Pattern):
        self.source = source
        self.tracked = tracked  # body node -> owner nid
        self.pattern = pattern
        # the owner being visited: its body, its function node, is it a test case
        self.cur_body = None
        self.cur_is_case = False
        # False for a closure owned by a node it is not the definition of
        # (a callback inside a test case, `const h = wrap((req) => ...)`):
        # its parameters are bound by whoever calls the closure, not by the
        # owner's callers, so they must not enter the owner's summary.
        self.cur_bindable = True
        self.locals: dict[str, dict] = {}  # local closure key -> {name, rest}

    # -- binding resolution ------------------------------------------------
    def resolve_ident(self, ident, owner_fn, depth: int):
        name = _text(ident, self.source)
        if name == "__dirname":
            return ["path", "__dirname"]
        cur = ident
        passed_body = False
        while cur.parent is not None:
            cur = cur.parent
            t = cur.type
            if self.cur_body is not None and cur == self.cur_body:
                passed_body = True
            if t in _FN_LIKE:
                pnames, _rest = _param_list(cur, self.source)
                if name in pnames:
                    if not passed_body:
                        local = _local_closure_name(cur, self.source)
                        if local is not None:
                            # a `const run = (args) => spawn(...)` closure inside
                            # the owner: bound at the owner's own call sites
                            key = str(cur.start_byte)
                            self.locals[key] = {"name": local, "rest": _param_list(cur, self.source)[1]}
                            return ["lparam", key, pnames.index(name)]
                        return ["unknown", "callback-param"]
                    if owner_fn is not None and cur == owner_fn:
                        if self.cur_is_case:
                            return ["unknown", "test-context-param"]
                        if not self.cur_bindable:
                            return ["unknown", "callback-param"]
                        return ["param", pnames.index(name)]
                    return ["unknown", "outer-param"]
                continue
            if t in ("for_in_statement",):
                left = cur.child_by_field_name("left")
                if left is not None and name in _pattern_names(left, self.source):
                    return ["unknown", "loop-variable"]
            if t == "for_statement":
                init = cur.child_by_field_name("initializer")
                if init is not None and name in _decl_names(init, self.source):
                    return ["unknown", "loop-variable"]
            if t == "catch_clause":
                param = cur.child_by_field_name("parameter")
                if param is not None and name in _pattern_names(param, self.source):
                    return ["unknown", "catch-param"]
            if t in ("statement_block", "program", "switch_case", "switch_default", "class_body"):
                found = self._block_binding(cur, name, owner_fn, depth)
                if found is not None:
                    return found
        return ["unknown", "free-identifier"]

    def _block_binding(self, block, name, owner_fn, depth):
        for st in block.named_children:
            decl = st
            if st.type == "export_statement":
                decl = next((c for c in st.named_children
                             if c.type in ("lexical_declaration", "variable_declaration")), None)
                if decl is None:
                    continue
            if decl.type in ("lexical_declaration", "variable_declaration"):
                is_const = any(c.type == "const" for c in decl.children)
                for d in decl.named_children:
                    if d.type != "variable_declarator":
                        continue
                    nm = d.child_by_field_name("name")
                    if nm is None:
                        continue
                    if nm.type == "identifier":
                        if _text(nm, self.source) != name:
                            continue
                        val = d.child_by_field_name("value")
                        if val is None:
                            return ["unknown", "uninitialised"]
                        if not is_const and _reassigned(block, name, self.source):
                            return ["unknown", "reassigned"]
                        return self.value(val, owner_fn, depth + 1)
                    if name in _pattern_names(nm, self.source):
                        return ["unknown", "destructured"]
            elif decl.type == "import_statement" and block.type == "program":
                if name in _import_names(decl, self.source):
                    return ["import", name]
            elif decl.type in ("function_declaration", "class_declaration"):
                nm = decl.child_by_field_name("name")
                if nm is not None and _text(nm, self.source) == name:
                    return ["unknown", "function-reference"]
        return None

    # -- expression values -------------------------------------------------
    def value(self, node, owner_fn, depth: int = 0):
        if depth > _MAX_DEPTH:
            return ["unknown", "too-deep"]
        t = node.type
        src = self.source
        while t in ("parenthesized_expression", "as_expression", "satisfies_expression",
                    "non_null_expression", "type_assertion"):
            inner = node.named_children[0] if node.named_children else None
            if inner is None:
                break
            node, t = inner, inner.type
        if t == "string":
            return ["lit", _text(node, src)[1:-1]]
        if t == "template_string":
            raw = _text(node, src)[1:-1]
            if not any(c.type == "template_substitution" for c in node.children):
                return ["lit", raw]
            tail = re.sub(r"\$\{[^}]*\}", "", raw.rsplit("}", 1)[-1]) if "}" in raw else raw
            if self.pattern.search(raw.replace("\\", "/")):
                return ["cli", "unknown", tail.lstrip("/")]
            return ["unknown", "template-literal"]
        if t == "identifier":
            return self.resolve_ident(node, owner_fn, depth)
        if t == "member_expression":
            txt = _text(node, src).replace(" ", "")
            if txt in ("process.execPath", "process.argv0", "process.argv[0]"):
                return ["node"]
            if txt in ("import.meta.dirname",):
                return ["path", "__dirname"]
            return ["unknown", "computed"]
        if t == "subscript_expression":
            return ["unknown", "computed"]
        if t == "array":
            elems = []
            for c in node.named_children:
                if c.type == "comment":
                    continue
                if c.type == "spread_element":
                    inner = c.named_children[0] if c.named_children else None
                    v = self.value(inner, owner_fn, depth + 1) if inner is not None else ["unknown", "computed"]
                    elems.extend(_spread_of(v))
                else:
                    elems.append(self.value(c, owner_fn, depth + 1))
            return ["array", elems]
        if t in ("call_expression", "new_expression"):
            fn = node.child_by_field_name("function") or node.child_by_field_name("constructor")
            fname = _callee_name(fn, src) if fn is not None else None
            if fname in _PATH_FUNCS:
                return self._path_value(node, owner_fn, depth)
            if fname in ("slice", "concat", "map", "filter", "flatMap"):
                return ["unknown", "computed-array"]
            return ["unknown", "computed"]
        if t == "binary_expression":
            return ["unknown", "computed"]
        return ["unknown", "computed"]

    def _path_value(self, call, owner_fn, depth):
        args = call.child_by_field_name("arguments")
        parts: list[str] = []
        base = "unknown"
        if args is not None:
            for i, a in enumerate(x for x in args.named_children if x.type != "comment"):
                if a.type in ("string", "template_string") and not any(
                        c.type == "template_substitution" for c in a.children):
                    parts.append(_text(a, self.source)[1:-1])
                elif a.type in ("call_expression", "new_expression"):
                    inner = self._path_value(a, owner_fn, depth + 1)
                    if inner[0] in ("cli", "path") and len(inner) > 2:
                        parts.append(inner[2])
                    elif inner[0] == "path" and inner[1] == "__dirname" and i == 0:
                        base = "dirname"
                elif i == 0:
                    v = self.value(a, owner_fn, depth + 1)
                    if v == ["path", "__dirname"]:
                        base = "dirname"
                    elif v[0] == "lit":
                        parts.append(v[1])
                    elif v[0] in ("path", "cli") and len(v) > 2:
                        parts.append(v[2])
                        base = v[1] if v[0] == "cli" else base
                else:
                    v = self.value(a, owner_fn, depth + 1)
                    if v[0] == "lit":
                        parts.append(v[1])
                    else:
                        parts.append("*")
        rel = "/".join(p.strip("/") for p in parts if p not in ("", "."))
        if parts and self.pattern.search(rel):
            return ["cli", base, rel]
        if base == "dirname" and not parts:
            return ["path", "__dirname"]
        return ["path", base, rel]

    # -- spawn sites -----------------------------------------------------------
    def spawn(self, call, owner_fn):
        fn = call.child_by_field_name("function")
        if fn is None:
            return None
        name = _callee_name(fn, self.source)
        if name is None:
            return None
        if fn.type == "member_expression":
            obj = fn.child_by_field_name("object")
            oname = _text(obj, self.source) if obj is not None and obj.type == "identifier" else None
            if name in ("exec",) and oname not in _CP_OBJECTS:
                return None
        if name not in _SPAWN_ARGV | _SPAWN_FORK | _SPAWN_SHELL:
            return None
        args = call.child_by_field_name("arguments")
        argv_nodes = [a for a in (args.named_children if args is not None else []) if a.type != "comment"]
        if not argv_nodes:
            return None
        if name in _SPAWN_SHELL:
            return {"prog": ["shellcmd"], "argv": self._command_tokens(argv_nodes[0], owner_fn)}
        first = self.value(argv_nodes[0], owner_fn)
        rest = []
        if len(argv_nodes) > 1 and argv_nodes[1].type != "object":
            v = self.value(argv_nodes[1], owner_fn)
            rest = _spread_of(v)
        if name in _SPAWN_FORK:
            return {"prog": ["node"], "argv": [first] + rest}
        return {"prog": first, "argv": rest}

    def _command_tokens(self, node, owner_fn):
        """Split a shell command string into argv-like descriptors."""
        if node.type == "string":
            return [["lit", tok] for tok in _text(node, self.source)[1:-1].split()]
        if node.type == "template_string":
            toks: list = []
            glue = False
            for c in node.children:
                if c.type == "template_substitution":
                    inner = c.named_children[0] if c.named_children else None
                    v = self.value(inner, owner_fn) if inner is not None else ["unknown", "computed"]
                    if glue and toks:
                        toks[-1] = ["unknown", "template-literal"]
                    else:
                        toks.append(v)
                    glue = True
                elif c.type in ("string_fragment", "escape_sequence") or c.is_named:
                    txt = _text(c, self.source)
                    if txt and not txt[0].isspace() and glue and toks:
                        toks[-1] = ["unknown", "template-literal"]
                        txt = txt.split(None, 1)[1] if len(txt.split(None, 1)) > 1 else ""
                    toks.extend(["lit", tok] for tok in txt.split())
                    glue = bool(txt) and not txt[-1].isspace()
            return toks
        v = self.value(node, owner_fn)
        if v[0] == "lit":
            return [["lit", tok] for tok in v[1].split()]
        return [["unknown", v[1] if v[0] == "unknown" else "computed", 1]]

    def call_args(self, call, owner_fn):
        args = call.child_by_field_name("arguments")
        out = []
        for a in (args.named_children if args is not None else []):
            if a.type == "comment":
                continue
            if a.type == "spread_element":
                inner = a.named_children[0] if a.named_children else None
                v = self.value(inner, owner_fn) if inner is not None else ["unknown", "computed"]
                out.append(["spreadarg", v])
            else:
                out.append(self.value(a, owner_fn))
        return out


def _spread_of(v):
    """Elements contributed by spreading value v into an array."""
    if v[0] == "array":
        return list(v[1])
    if v[0] == "param":
        return [["spread", v[1]]]
    if v[0] == "lparam":
        return [["lspread", v[1], v[2]]]
    if v[0] == "unknown":
        return [["unknown", v[1], 1]]
    return [["unknown", "spread-of-scalar", 1]]


def _callee_name(fn, source: bytes) -> str | None:
    if fn.type == "identifier":
        return _text(fn, source)
    if fn.type == "member_expression":
        prop = fn.child_by_field_name("property")
        return _text(prop, source) if prop is not None else None
    return None


def _pattern_names(node, source: bytes) -> set[str]:
    out: set[str] = set()

    def rec(n):
        if n.type in ("identifier", "shorthand_property_identifier_pattern"):
            out.add(_text(n, source))
            return
        if n.type == "pair_pattern":
            v = n.child_by_field_name("value")
            if v is not None:
                rec(v)
            return
        for c in n.named_children:
            rec(c)
    rec(node)
    return out


def _decl_names(node, source: bytes) -> set[str]:
    out: set[str] = set()
    for d in node.named_children:
        if d.type == "variable_declarator":
            nm = d.child_by_field_name("name")
            if nm is not None:
                out |= _pattern_names(nm, source)
    return out


def _import_names(imp, source: bytes) -> set[str]:
    out: set[str] = set()
    for c in imp.named_children:
        if c.type == "import_clause":
            out |= _pattern_names(c, source)
    return out


def _reassigned(block, name: str, source: bytes) -> bool:
    stack = [block]
    while stack:
        n = stack.pop()
        if n.type in ("assignment_expression", "augmented_assignment_expression"):
            left = n.child_by_field_name("left")
            if left is not None and left.type == "identifier" and _text(left, source) == name:
                return True
        stack.extend(n.children)
    return False


def js_collect_cli_facts(
    root, source: bytes, *, function_bodies: list, nodes: list,
    file_nid: str, module_nid: str | None, case_bodies: Iterable[Any] = (),
) -> None:
    """Attach `_rll_cli` / `_rll_dispatch` / `_rll_cli_consts` facts to nodes."""
    pattern = cli_path_pattern()
    tracked: dict[Any, str] = {}
    for nid, body in function_bodies:
        if body is not None:
            tracked.setdefault(body, nid)
    facts = _Facts(source, tracked, pattern)
    case_body_set = set(case_bodies)
    per_owner: dict[str, dict] = {}
    dispatch: dict[str, list] = {}

    def owner_entry(nid):
        e = per_owner.get(nid)
        if e is None:
            e = per_owner[nid] = {"spawns": [], "calls": []}
        return e

    def visit(node, owner_nid, owner_fn, is_case=False, bindable=True):
        facts.cur_body = node
        facts.cur_is_case = is_case
        facts.cur_bindable = bindable
        stack = [node]
        while stack:
            n = stack.pop()
            t = n.type
            if t in _FN_LIKE or t in ("class_declaration", "class"):
                if n != node:
                    body = n.child_by_field_name("body")
                    if body is not None and body in tracked:
                        continue  # owned by its own node
            if t == "switch_statement":
                table = _dispatch_table(n, source)
                if table:
                    dispatch.setdefault(owner_nid, []).append(table)
            if t == "call_expression":
                line = n.start_point[0] + 1
                sp = facts.spawn(n, owner_fn)
                if sp is not None:
                    sp["line"] = line
                    owner_entry(owner_nid)["spawns"].append(sp)
                else:
                    fn = n.child_by_field_name("function")
                    name = _callee_name(fn, source) if fn is not None else None
                    if name and name not in _PATH_FUNCS:
                        args = facts.call_args(n, owner_fn)
                        if args and any(a[0] != "unknown" or a[1] != "computed" for a in args):
                            site = {"line": line, "name": name, "args": args}
                            if fn.type == "member_expression":
                                site["member"] = True
                            owner_entry(owner_nid)["calls"].append(site)
            stack.extend(reversed(n.children))

    for nid, body in function_bodies:
        if body is None or tracked.get(body) != nid:
            continue
        fn = body.parent if body.parent is not None and body.parent.type in _FN_LIKE else None
        is_case = body in case_body_set
        bindable = fn is not None and _bindable(fn)
        visit(body, nid, fn, is_case, bindable)
        e = per_owner.get(nid)
        if e is not None and bindable and not is_case and "params" not in e:
            pn, rest = _param_list(fn, source)
            e["params"] = pn
            e["rest"] = rest
    mod_owner = module_nid or file_nid
    visit(root, mod_owner, None)
    facts.cur_body = None
    facts.cur_is_case = False
    for e in per_owner.values():
        _bind_local_closures(e, facts.locals)  # also scrubs unbound local refs

    consts: list[str] = []
    for st in root.named_children:
        decl = st
        if st.type == "export_statement":
            decl = next((c for c in st.named_children if c.type == "lexical_declaration"), None)
            if decl is None:
                continue
        if decl.type != "lexical_declaration":
            continue
        for d in decl.named_children:
            nm = d.child_by_field_name("name") if d.type == "variable_declarator" else None
            val = d.child_by_field_name("value") if d.type == "variable_declarator" else None
            if nm is not None and nm.type == "identifier" and val is not None:
                v = facts.value(val, None)
                if v[0] == "cli":
                    consts.append([_text(nm, source), v[1], v[2]])

    by_id = {n["id"]: n for n in nodes}
    for nid, e in per_owner.items():
        if not e["spawns"] and not e["calls"]:
            continue
        tgt = by_id.get(nid) or by_id.get(file_nid)
        if tgt is None:
            continue
        prev = tgt.get("_rll_cli")
        if prev:
            prev["spawns"] += e["spawns"]
            prev["calls"] += e["calls"]
        else:
            tgt["_rll_cli"] = e
    for nid, tables in dispatch.items():
        tgt = by_id.get(nid) or by_id.get(file_nid)
        if tgt is not None:
            tgt.setdefault("_rll_dispatch", []).extend(tables)
    if consts and file_nid in by_id:
        by_id[file_nid]["_rll_cli_consts"] = consts


def _local_closure_name(fn, source: bytes) -> str | None:
    """`const run = (args) => ...` -> "run" (an arrow / function expression
    bound directly to a const/let name)."""
    if fn.type not in ("arrow_function", "function_expression"):
        return None
    p = fn.parent
    if p is None or p.type != "variable_declarator":
        return None
    nm = p.child_by_field_name("name")
    return _text(nm, source) if nm is not None and nm.type == "identifier" else None


def _has_local(v) -> bool:
    if v[0] in ("lparam", "lspread"):
        return True
    if v[0] == "array":
        return any(_has_local(x) for x in v[1])
    if v[0] == "spreadarg":
        return _has_local(v[1])
    return False


def _drop_local(v):
    if v[0] == "lparam":
        return ["unknown", "callback-param"]
    if v[0] == "lspread":
        return ["unknown", "callback-param", 1]
    if v[0] == "array":
        return ["array", [_drop_local(x) for x in v[1]]]
    if v[0] == "spreadarg":
        return ["spreadarg", _drop_local(v[1])]
    return v


def _local_key(v) -> str | None:
    if v[0] in ("lparam", "lspread"):
        return v[1]
    if v[0] == "array":
        for x in v[1]:
            k = _local_key(x)
            if k:
                return k
    if v[0] == "spreadarg":
        return _local_key(v[1])
    return None


def _bind_local_closures(entry: dict, locals_: dict) -> None:
    """Substitute an owner's own calls to its local runner closures.

    A spawn (or a helper call) inside `const run = (args) => spawnSync(node,
    [CLI, ...args])` refers to the closure's parameters (lparam / lspread).
    Each call `run([..])` the owner makes binds them, yielding a concrete spawn
    / call at that call's line. Whatever stays unbound becomes callback-param
    (uncertain).
    """
    local_spawns: dict[str, list] = {}
    local_calls: dict[str, list] = {}
    spawns, calls = [], []
    for sp in entry["spawns"]:
        k = next((x for x in map(_local_key, [sp["prog"], *sp["argv"]]) if x), None)
        (local_spawns.setdefault(k, []) if k else spawns).append(sp)
    for site in entry["calls"]:
        k = next((x for x in map(_local_key, site["args"]) if x), None)
        (local_calls.setdefault(k, []) if k else calls).append(site)
    by_name: dict[str, list[str]] = {}
    for key in set(local_spawns) | set(local_calls):
        info = locals_.get(key)
        if info:
            by_name.setdefault(info["name"], []).append(key)
    bound: set[str] = set()
    for site in list(calls):
        if site.get("member") or site["name"] not in by_name:
            continue
        for key in by_name[site["name"]]:
            rest_idx = locals_[key].get("rest", -1)
            for sp in local_spawns.get(key, ()):
                spawns.append({
                    "prog": _drop_local(_subst_local_value(sp["prog"], key, site["args"], rest_idx)),
                    "argv": [_drop_local(x) for x in
                             _subst_local_elems(sp["argv"], key, site["args"], rest_idx)],
                    "line": site["line"], "local": site["name"],
                })
            for inner in local_calls.get(key, ()):
                calls.append({
                    **inner, "line": site["line"],
                    "args": [_drop_local(_subst_local_arg(a, key, site["args"], rest_idx))
                             for a in inner["args"]],
                })
            bound.add(key)
    for key, sps in local_spawns.items():
        if key not in bound:
            spawns.extend({**sp, "prog": _drop_local(sp["prog"]),
                           "argv": [_drop_local(x) for x in sp["argv"]]} for sp in sps)
    for key, cs in local_calls.items():
        if key not in bound:
            calls.extend({**c, "args": [_drop_local(a) for a in c["args"]]} for c in cs)
    entry["spawns"], entry["calls"] = spawns, calls


def _subst_local_arg(a, key, args, rest_idx):
    if a[0] == "lparam" and a[1] == key:
        return _subst_value(["param", a[2]], args, rest_idx) if _arg_at(args, a[2], rest_idx)[0] != "array" \
            else _arg_at(args, a[2], rest_idx)
    if a[0] == "array":
        return ["array", _subst_local_elems(a[1], key, args, rest_idx)]
    if a[0] == "spreadarg":
        return ["spreadarg", _subst_local_arg(a[1], key, args, rest_idx)]
    return a


def _subst_local_value(v, key, args, rest_idx):
    if v[0] == "lparam" and v[1] == key:
        return _subst_value(["param", v[2]], args, rest_idx)
    return v


def _subst_local_elems(elems, key, args, rest_idx):
    mapped = []
    for el in elems:
        if el[0] == "lspread" and el[1] == key:
            mapped.append(["spread", el[2]])
        elif el[0] == "lparam" and el[1] == key:
            mapped.append(["param", el[2]])
        elif el[0] in ("param", "spread"):
            mapped.append(["__outer__", el])  # the owner's own params: keep as-is
        else:
            mapped.append(el)
    out = _subst_elems([m for m in mapped], args, rest_idx)
    return [x[1] if x[0] == "__outer__" else x for x in out]


def _bindable(fn) -> bool:
    """Is fn the definition of a named callable (so its callers bind its params)?"""
    if fn.type in ("function_declaration", "generator_function_declaration", "method_definition"):
        return True
    p = fn.parent
    return p is not None and p.type in (
        "variable_declarator", "public_field_definition", "field_definition",
        "assignment_expression", "pair")


def _dispatch_table(sw, source: bytes) -> dict | None:
    """`switch (x) { case "a": case "b": f(); break; ... }` -> {cases: {a: [...]}}."""
    body = sw.child_by_field_name("body")
    if body is None:
        return None
    cases: dict[str, list[str]] = {}
    pending: list[str] = []
    n_lit = 0
    for c in body.named_children:
        if c.type not in ("switch_case", "switch_default"):
            continue
        label = None
        val = c.child_by_field_name("value") if c.type == "switch_case" else None
        if c.type == "switch_case":
            if val is not None and val.type == "string":
                label = _text(val, source)[1:-1]
                n_lit += 1
        else:
            label = "<default>"
        if label is not None:
            pending.append(label)
        stmts = [s for s in c.named_children if (val is None or s != val)
                 and s.type != "comment"]
        if not stmts:
            continue  # falls through to the next case's body
        names: list[str] = []
        stack = list(stmts)
        while stack:
            n = stack.pop()
            if n.type == "call_expression":
                fn = n.child_by_field_name("function")
                nm = _callee_name(fn, source) if fn is not None else None
                if nm and nm not in names:
                    names.append(nm)
            stack.extend(n.children)
        for lab in pending:
            cases[lab] = list(names)
        pending = []  # fallthrough past a non-empty body is not followed
    if n_lit < 3:
        return None
    return {"line": sw.start_point[0] + 1, "cases": cases}


# ── 4b. Corpus-level resolution ──────────────────────────────────────────────

SENTINEL_ID = "rll_unresolved_process_invocation"
_NODE_VALUE_FLAGS = frozenset({
    "-r", "--require", "--import", "--loader", "--experimental-loader", "-C",
    "--conditions", "--env-file", "--input-type", "--inspect-port", "--max-old-space-size",
})
_NODE_EVAL_FLAGS = frozenset({"-e", "--eval", "-p", "--print"})
_SUMMARY_CAP = 256  # per-function bound on distinct summary entries (termination guard)
_ARGV_CAP = 64


def _norm_label(label: str) -> str:
    return (label or "").strip().rstrip(")").rstrip("(").lstrip(".")


def _is_pathish(s: str) -> bool:
    return "/" in s or bool(re.search(r"\.(?:[cm]?js|ts|sh|py|rb)$", s))


def _cli_entry_candidates(hint, caller_sf: str) -> list[str]:
    _kind, base, rel = hint
    rel = rel.replace("\\", "/")
    cands: list[str] = []

    def variants(p: str) -> list[str]:
        p = os.path.normpath(p).replace("\\", "/")
        stem = re.sub(r"\.[cm]?js$", "", p)
        out = []
        for s in (stem, stem.replace("/dist/", "/src/"), re.sub(r"^dist/", "src/", stem),
                  stem.replace("/build/", "/src/"), stem.replace("/lib/", "/src/")):
            for ext in (".ts", ".mts", ".tsx", ".js", ".mjs", ".cjs"):
                out.append(s + ext)
        return out

    if base == "dirname" and caller_sf:
        cands += variants(os.path.join(os.path.dirname(caller_sf), rel))
    cands += ["*/" + v.lstrip("./") for v in variants(rel.lstrip("./").lstrip("../") or rel)]
    return cands


class _Resolver:
    def __init__(self, nodes: list[dict], edges: list[dict]):
        self.nodes = nodes
        self.by_id = {n["id"]: n for n in nodes}
        self.out: dict[str, list[dict]] = {}
        for e in edges:
            self.out.setdefault(e["source"], []).append(e)
        self.dispatch_files: dict[str, list[tuple[str, dict]]] = {}
        for n in nodes:
            for table in n.get("_rll_dispatch") or ():
                self.dispatch_files.setdefault(n.get("source_file") or "", []).append((n["id"], table))
        self.file_consts: dict[str, dict[str, list]] = {}
        for n in nodes:
            for name, base, rel in n.get("_rll_cli_consts") or ():
                self.file_consts.setdefault(n.get("source_file") or "", {})[name] = ["cli", base, rel]
        self.label_index: dict[str, list[str]] = {}
        for n in nodes:
            if n.get("_callable") and not n.get("test_kind"):
                self.label_index.setdefault(_norm_label(n.get("label", "")), []).append(n["id"])
        self.bin_names = {b.strip() for b in os.environ.get("GRAPHIFY_CLI_BIN_NAMES", "").split(",") if b.strip()}
        self.forced_entry = os.environ.get("GRAPHIFY_CLI_ENTRY") or None
        self._entry_cache: dict[tuple, str | None] = {}
        self._handler_cache: dict[tuple, tuple] = {}

    # -- resolution helpers --------------------------------------------------
    def callee_targets(self, owner: str, name: str) -> list[str]:
        out = []
        for e in self.out.get(owner, ()):
            if e.get("relation") in ("calls", "indirect_call"):
                t = self.by_id.get(e["target"])
                if t is not None and _norm_label(t.get("label", "")) == name and e["target"] not in out:
                    out.append(e["target"])
        return out

    def resolve_import(self, owner_sf: str, owner: str, name: str):
        """An imported identifier: a CLI-path const exported by the imported file?"""
        file_ids = [n["id"] for n in self.nodes
                    if n.get("source_file") == owner_sf and n.get("label") == owner_sf.rsplit("/", 1)[-1]]
        for fid in file_ids:
            for e in self.out.get(fid, ()):
                if e.get("relation") != "imports":
                    continue
                t = self.by_id.get(e["target"])
                if t is None or t.get("label") != name:
                    continue
                c = self.file_consts.get(t.get("source_file") or "", {}).get(name)
                if c is not None:
                    return c
        return ["unknown", "imported-value"]

    def entry_for(self, hint, caller_sf: str) -> str | None:
        key = (tuple(hint), caller_sf if hint[1] == "dirname" else "")
        if key in self._entry_cache:
            return self._entry_cache[key]
        result = None
        if self.forced_entry and self.forced_entry in self.dispatch_files:
            result = self.forced_entry
        else:
            files = list(self.dispatch_files)
            for c in _cli_entry_candidates(hint, caller_sf):
                if c.startswith("*/"):
                    hit = [f for f in files if f == c[2:] or f.endswith("/" + c[2:])]
                else:
                    hit = [f for f in files if f == c]
                if len(hit) == 1:
                    result = hit[0]
                    break
            if result is None:
                stem = re.sub(r"\.[cm]?js$", "", hint[2].rsplit("/", 1)[-1])
                same = [f for f in files if re.sub(r"\.[cm]?[jt]sx?$", "", f.rsplit("/", 1)[-1]) == stem]
                if len(same) == 1:
                    result = same[0]
        self._entry_cache[key] = result
        return result

    def dispatcher(self, entry: str) -> tuple[str, dict] | None:
        tables = self.dispatch_files.get(entry) or []
        if not tables:
            return None

        def score(item):
            nid, table = item
            n = self.by_id.get(nid) or {}
            top = n.get("label") in ("<module scope>", entry.rsplit("/", 1)[-1])
            return (top, len(table.get("cases") or {}))
        return max(tables, key=score)

    def handlers(self, entry: str, sub: str) -> tuple[list[str], str | None]:
        d = self.dispatcher(entry)
        if d is None:
            return [], "no-dispatch-table"
        owner, table = d
        key = (entry, sub)
        if key in self._handler_cache:
            return self._handler_cache[key]
        names = (table.get("cases") or {}).get(sub)
        if names is None:
            return [], "unknown-subcommand"
        targets: list[str] = []
        for nm in names:
            hits = self.callee_targets(owner, nm)
            if not hits:
                cand = [i for i in self.label_index.get(nm, ())
                        if (self.by_id[i].get("role") or "product") == "product"]
                hits = cand if len(cand) == 1 else []
            for h in hits:
                if h not in targets:
                    targets.append(h)
        self._handler_cache[key] = (targets, None if targets else "handler-unresolved")
        return self._handler_cache[key]


def _arg_at(args: list, i: int, rest_idx: int):
    """The call-site value bound to callee parameter i."""
    if rest_idx >= 0 and i == rest_idx:
        tail = args[i:]
        elems: list = []
        for a in tail:
            if a[0] == "spreadarg":
                elems.extend(_spread_of(a[1]))
            else:
                elems.append(a)
        return ["array", elems]
    for j, a in enumerate(args[: i + 1]):
        if a[0] == "spreadarg":
            if j == i and a[1][0] == "param":
                return ["unknown", "spread-call"]
            return ["unknown", "spread-call"]
    if i >= len(args):
        return ["missing"]
    return args[i]


def _subst_value(v, args, rest_idx):
    if v[0] == "param":
        a = _arg_at(args, v[1], rest_idx)
        if a[0] == "missing":
            return ["unknown", "missing-argument"]
        if a[0] == "array":
            return ["unknown", "array-as-value"]
        return a
    return v


def _subst_elems(elems, args, rest_idx):
    out = []
    for el in elems:
        if el[0] == "spread":
            a = _arg_at(args, el[1], rest_idx)
            if a[0] == "array":
                out.extend(a[1])
            elif a[0] == "missing":
                continue
            elif a[0] == "param":
                out.append(["spread", a[1]])
            elif a[0] == "unknown":
                out.append(["unknown", a[1], 1])
            else:
                out.append(["unknown", "spread-of-scalar", 1])
        else:
            out.append(_subst_value(el, args, rest_idx))
    return out[:_ARGV_CAP]


def _decisive(inv):
    """Classify an invocation template.

    Returns ("needs", None) when a decisive position depends on a parameter,
    ("skip", None) when it is not a CLI invocation, or
    ("cli", (hint, sub_desc)) / ("uncertain", (reason, hint_or_None)).
    """
    prog, argv = inv["prog"], inv["argv"]
    k = prog[0]
    if k == "shellcmd":
        if not argv:
            return ("skip", None)
        prog, argv = argv[0], argv[1:]
        k = prog[0]
    if k in ("param", "spread"):
        return ("needs", None)
    if k == "lit" and prog[1] in _NODE_NAMES or k == "node":
        i = 0
        while i < len(argv):
            el = argv[i]
            if el[0] in ("param", "spread"):
                return ("needs", None)
            if el[0] == "lit" and el[1].startswith("-"):
                if el[1] in _NODE_EVAL_FLAGS:
                    code = argv[i + 1] if i + 1 < len(argv) else None
                    if code and code[0] == "lit" and "cli" in code[1]:
                        return ("uncertain", ("node-eval", None))
                    if code and code[0] in ("param", "spread"):
                        return ("needs", None)
                    return ("skip", None)
                i += 2 if el[1] in _NODE_VALUE_FLAGS else 1
                continue
            break
        if i >= len(argv):
            return ("skip", None)
        entry, rest = argv[i], argv[i + 1:]
        if entry[0] == "cli":
            return ("cli", (entry, rest[0] if rest else ["missing"]))
        if entry[0] == "lit":
            if cli_path_pattern().search(entry[1]):
                return ("cli", (["cli", "unknown", entry[1]], rest[0] if rest else ["missing"]))
            return ("uncertain", ("script-path", None)) if _is_pathish(entry[1]) else ("skip", None)
        if entry[0] == "path":
            return ("uncertain", ("script-path", None))
        if entry[0] == "import":
            return ("import-entry", (entry, rest[0] if rest else ["missing"]))
        if entry[0] == "unknown":
            return ("uncertain", ("script-" + entry[1], None))
        return ("skip", None)
    if k == "cli":
        return ("cli", (prog, argv[0] if argv else ["missing"]))
    if k == "lit":
        name = prog[1]
        if name in _SHELLS:
            if argv and argv[0][0] in ("param", "spread"):
                return ("needs", None)
            if argv and argv[0][0] == "lit" and argv[0][1] == "-c":
                return ("uncertain", ("shell-command", None)) if any(
                    a[0] == "cli" or (a[0] == "lit" and "cli.js" in a[1]) for a in argv[1:]) else ("skip", None)
            if argv and (argv[0][0] == "path" or (argv[0][0] == "lit" and _is_pathish(argv[0][1]))):
                return ("uncertain", ("script-path", None))
            return ("skip", None)
        if name in _BIN_NAMES_PLACEHOLDER:
            return ("bin", (argv[0] if argv else ["missing"]))
        if _is_pathish(name):
            return ("uncertain", ("script-path", None))
        return ("skip", None)
    if k == "path":
        return ("uncertain", ("script-path", None))
    if k in ("unknown", "import"):
        if any(a[0] == "cli" for a in argv):
            return ("uncertain", ("program-" + (prog[1] if k == "unknown" else "imported"), None))
        if any(a[0] in ("param", "spread") for a in argv):
            return ("needs", None)
        return ("skip", None)
    return ("skip", None)


_BIN_NAMES_PLACEHOLDER: set[str] = set()


def _has_params(inv) -> bool:
    def rec(v):
        if v[0] in ("param", "spread"):
            return True
        if v[0] == "array":
            return any(rec(x) for x in v[1])
        return False
    return rec(inv["prog"]) or any(rec(a) for a in inv["argv"])


def resolve_cli_invocations(
    nodes: list[dict], edges: list[dict], *,
    context_nodes: Iterable[dict] = (), context_edges: Iterable[dict] = (),
) -> dict:
    """Emit `invokes_cli` edges (appended to ``edges``); return stats.

    ``context_nodes`` / ``context_edges`` are read-only resolution context
    (an incremental rebuild's unchanged corpus): their persisted summaries and
    dispatch tables are used, but no edge is emitted FROM them.
    """
    global _BIN_NAMES_PLACEHOLDER
    fresh_ids = {n["id"] for n in nodes}
    ctx = [n for n in context_nodes if n.get("id") and n["id"] not in fresh_ids]
    all_nodes = list(nodes) + ctx
    R = _Resolver(all_nodes, list(edges) + list(context_edges))
    _BIN_NAMES_PLACEHOLDER = R.bin_names
    stats = {"resolved": 0, "uncertain": {}, "summaries": 0}

    facts: dict[str, dict] = {n["id"]: n["_rll_cli"] for n in nodes if n.get("_rll_cli")}
    # summaries: nid -> list of (inv, chain) ; inv = {prog, argv, line}
    summaries: dict[str, list] = {}
    keys: dict[str, set] = {}
    for n in ctx:
        for item in n.get("_rll_cli_summary") or ():
            summaries.setdefault(n["id"], []).append((item["inv"], item.get("chain", [])))
            keys.setdefault(n["id"], set()).add(json.dumps(item["inv"], sort_keys=True))

    emitted: dict[tuple, dict] = {}
    sentinel_needed = False

    def emit(src, tgt, *, line, sub, via, chain, uncertain=False, reason=None, entry=None):
        nonlocal sentinel_needed
        key = (src, tgt, sub or "", reason or "")
        if key in emitted:
            return
        n = R.by_id.get(src) or {}
        e = {
            "source": src, "target": tgt, "relation": "invokes_cli",
            "confidence": "AMBIGUOUS" if uncertain else "EXTRACTED",
            "source_file": n.get("source_file") or "", "source_location": f"L{line}",
            "weight": 1.0, "context": "cli-invocation", "via": via,
        }
        if sub is not None:
            e["subcommand"] = sub
        if chain:
            e["helper_chain"] = list(chain)
        if entry:
            e["cli_entry"] = entry
        if uncertain:
            e["uncertain"] = True
            e["reason"] = reason
            stats["uncertain"][reason] = stats["uncertain"].get(reason, 0) + 1
            if tgt == SENTINEL_ID:
                sentinel_needed = True
        else:
            stats["resolved"] += 1
        emitted[key] = e

    def conclude(owner, inv, chain, line):
        """Decide one invocation at `owner`; True when it needs a summary entry."""
        verdict, data = _decisive(inv)
        via = "helper" if chain else "direct"
        sf = (R.by_id.get(owner) or {}).get("source_file") or ""
        if verdict == "needs":
            return True
        if verdict == "skip":
            return False
        if verdict == "import-entry":
            entry_desc = R.resolve_import(sf, owner, data[0][1])
            if entry_desc[0] != "cli":
                emit(owner, SENTINEL_ID, line=line, sub=None, via=via, chain=chain,
                     uncertain=True, reason="script-imported-value")
                return False
            verdict, data = "cli", (entry_desc, data[1])
        if verdict == "bin":
            verdict, data = "cli", (["cli", "unknown", "cli.js"], data)
        if verdict == "uncertain":
            reason = data[0]
            emit(owner, SENTINEL_ID, line=line, sub=None, via=via, chain=chain,
                 uncertain=True, reason=reason)
            return False
        hint, sub = data
        entry = R.entry_for(hint, sf)
        if entry is None:
            emit(owner, SENTINEL_ID, line=line, sub=None, via=via, chain=chain,
                 uncertain=True, reason="cli-entry-unresolved")
            return False
        disp = R.dispatcher(entry)
        disp_id = disp[0] if disp else SENTINEL_ID
        if sub[0] in ("param", "spread"):
            return True
        if sub[0] == "missing":
            emit(owner, disp_id, line=line, sub=None, via=via, chain=chain,
                 uncertain=True, reason="no-subcommand", entry=entry)
            return False
        if sub[0] == "import":
            sub = ["unknown", "imported-value"]
        if sub[0] != "lit":
            emit(owner, disp_id, line=line, sub=None, via=via, chain=chain,
                 uncertain=True, reason=sub[1] if sub[0] == "unknown" else "computed", entry=entry)
            return False
        targets, why = R.handlers(entry, sub[1])
        if sub[1].startswith("-") and not targets:
            # a global flag (`--json status`) or a flag the dispatcher handles
            # before its switch: the subcommand is not where we looked
            emit(owner, disp_id, line=line, sub=sub[1], via=via, chain=chain,
                 uncertain=True, reason="flag-before-subcommand", entry=entry)
            return False
        if not targets:
            emit(owner, disp_id, line=line, sub=sub[1], via=via, chain=chain,
                 uncertain=True, reason=why, entry=entry)
            return False
        for t in targets:
            emit(owner, t, line=line, sub=sub[1], via=via, chain=chain, entry=entry)
        return False

    def add_summary(owner, inv, chain) -> bool:
        k = json.dumps({"prog": inv["prog"], "argv": inv["argv"]}, sort_keys=True)
        seen = keys.setdefault(owner, set())
        if k in seen or len(seen) >= _SUMMARY_CAP:
            return False
        seen.add(k)
        summaries.setdefault(owner, []).append((inv, chain))
        return True

    # Seed: every spawn site, decided at its owner.
    for owner, f in facts.items():
        for sp in f.get("spawns", ()):
            inv = {"prog": sp["prog"], "argv": sp["argv"]}
            if conclude(owner, inv, [], sp["line"]) and _has_params(inv):
                add_summary(owner, inv, [])
    # Call-site index: callee nid -> [(caller, site)]
    callers: dict[str, list] = {}
    for owner, f in facts.items():
        for site in f.get("calls", ()):
            for tgt in R.callee_targets(owner, site["name"]):
                callers.setdefault(tgt, []).append((owner, site))
    # Fixpoint over resolved call edges: propagate each summary entry to callers.
    work = [(fid, i) for fid, lst in summaries.items() for i in range(len(lst))]
    bound: set[str] = set()
    while work:
        fid, i = work.pop()
        inv, chain = summaries[fid][i]
        cf = facts.get(fid) or next((n.get("_rll_cli_params") for n in ctx if n["id"] == fid), None) or {}
        rest_idx = cf.get("rest", -1) if isinstance(cf, dict) else -1
        for owner, site in callers.get(fid, ()):
            bound.add(fid)
            new = {"prog": _subst_value(inv["prog"], site["args"], rest_idx),
                   "argv": _subst_elems(inv["argv"], site["args"], rest_idx)}
            new_chain = [fid] + list(chain)
            if owner in new_chain:
                continue  # recursion: the cycle adds nothing new
            if conclude(owner, new, new_chain, site["line"]) and _has_params(new):
                if add_summary(owner, new, new_chain):
                    work.append((owner, len(summaries[owner]) - 1))
    # A parameter-dependent spawn nobody binds is still an invocation: say so.
    for fid, lst in summaries.items():
        if fid in bound or fid not in facts:
            continue
        for inv, chain in lst:
            line = next((sp["line"] for sp in facts[fid].get("spawns", ())), 1)
            emit(fid, SENTINEL_ID, line=line, sub=None, via="helper" if chain else "direct",
                 chain=chain, uncertain=True, reason="unbound-parameter")
    stats["summaries"] = sum(len(v) for v in summaries.values())

    # Persist what an incremental rebuild needs from this batch, drop the rest.
    for n in nodes:
        fid = n["id"]
        if fid in summaries and summaries[fid]:
            n["_rll_cli_summary"] = [{"inv": inv, "chain": ch} for inv, ch in summaries[fid]]
            f = facts.get(fid) or {}
            if f.get("rest", -1) >= 0:
                n["_rll_cli_params"] = {"rest": f["rest"]}
        n.pop("_rll_cli", None)
    if sentinel_needed and SENTINEL_ID not in R.by_id:
        nodes.append({
            "id": SENTINEL_ID, "label": "<unresolved process invocation>",
            "file_type": "concept", "type": "sentinel", "source_file": "",
            "role": "sentinel",
        })
    edges.extend(emitted.values())
    return stats


# ── 6. Case-only id collisions (JS / TS) ─────────────────────────────────────
#
# make_id case-folds, so `interface WorktreeFingerprint` and `function
# worktreeFingerprint()` in one file mint the same id and add_node silently
# drops whichever is walked second. Pre-scan the file's top-level declarations;
# in a group of DISTINCT names sharing one id, the unique highest-priority one
# (callable > class > other value > type-only) keeps the plain id — that is
# what calls and value imports target — and every other member gets a stable
# name-derived salt. No unique winner: all are salted (order never matters).

_DECL_PRIORITY = {
    "function_declaration": 3, "generator_function_declaration": 3,
    "class_declaration": 2, "abstract_class_declaration": 2,
    "enum_declaration": 1, "interface_declaration": 0, "type_alias_declaration": 0,
}


def js_pre_scan_case_collisions(root, source: bytes, stem: str, make_id) -> dict[str, dict[str, int]]:
    groups: dict[str, dict[str, int]] = {}

    def record(name: str, prio: int) -> None:
        if not name:
            return
        g = groups.setdefault(make_id(stem, name), {})
        g[name] = max(prio, g.get(name, -1))

    for st in root.named_children:
        decl = st
        if st.type == "export_statement":
            decl = next((c for c in st.named_children
                         if c.type in _DECL_PRIORITY or c.type in ("lexical_declaration", "variable_declaration")), None)
            if decl is None:
                continue
        if decl.type in _DECL_PRIORITY:
            nm = decl.child_by_field_name("name")
            if nm is not None:
                record(_text(nm, source), _DECL_PRIORITY[decl.type])
        elif decl.type in ("lexical_declaration", "variable_declaration"):
            for d in decl.named_children:
                if d.type != "variable_declarator":
                    continue
                nm = d.child_by_field_name("name")
                val = d.child_by_field_name("value")
                if nm is not None and nm.type == "identifier":
                    record(_text(nm, source), 3 if val is not None and val.type in _JS_FN_TYPES else 1)
    return {nid: names for nid, names in groups.items() if len(names) >= 2}


def js_salted_nid(plain_nid: str, name: str, groups: dict[str, dict[str, int]], make_id) -> str:
    names = groups.get(plain_nid)
    if not names or name not in names:
        return plain_nid
    top = max(names.values())
    winners = [n for n, p in names.items() if p == top]
    if len(winners) == 1 and winners[0] == name:
        return plain_nid
    import hashlib
    return make_id(plain_nid, hashlib.sha1(name.encode("utf-8")).hexdigest()[:6])
