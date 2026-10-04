# Copyright 2026 RLL project contributors. Licensed under the Apache License, Version 2.0.
# Added by the RLL project (2026-10); not part of upstream Graphify.
"""Module-level declarations, import kinds and load-time effects (symbol-level impact S2).

A change at module scope has to be attributable to the declaration it touches before a
selector can follow that declaration's uses instead of widening to the whole file and
everything importing it. These pin the three facts the extractor now records.
"""
from __future__ import annotations

from pathlib import Path

from graphify.extract import extract

SRC = """import './side';
import { x } from './x';
export const A = [1, 2] as const;
const S = 'scalar';
let counter = 0;
let pending;
var legacy = 1, other = 2;
const B = compute();
const { p, q: r } = makePair();
export const { e1, e2: alias } = makePair();
const { lib } = require('./lib');
export function f() { return A.length + S.length; }
export interface I { a: number }
export type T = string;
export enum E { One }
class K { static { init(); } }
class Plain { m() { return go(); } }
registerX();
if (process.env.X) { counter++; }
const I = 3;
const $ = 1;
"""


def _nodes(tmp_path: Path, src: str = SRC) -> dict:
    (tmp_path / "m.ts").write_text(src)
    (tmp_path / "x.ts").write_text("export const x = 1;\n")
    (tmp_path / "lib.ts").write_text("export const lib = 1;\n")
    r = extract([tmp_path / "m.ts", tmp_path / "x.ts", tmp_path / "lib.ts"], root=tmp_path,
                cache_root=tmp_path / "out", parallel=False)
    out: dict = {}
    for n in r["nodes"]:
        if n.get("source_file", "").endswith("m.ts"):
            out.setdefault(n["label"], []).append(n)
    return {"by_label": out, "edges": r["edges"]}


def _one(by_label, label, kind=None):
    cands = [n for n in by_label[label] if kind is None or n.get("decl_kind") == kind]
    assert len(cands) == 1, by_label.get(label)
    return cands[0]


def test_every_top_level_declarator_has_a_node_range_and_kind(tmp_path):
    g = _nodes(tmp_path)["by_label"]
    expect = {"A": [3, 3], "S": [4, 4], "counter": [5, 5], "pending": [6, 6], "legacy": [7, 7],
              "other": [7, 7], "B": [8, 8], "p": [9, 9], "r": [9, 9]}
    for label, rng in expect.items():
        n = _one(g, label)
        assert n["source_range"] == rng, (label, n)
        assert n["decl_kind"] == "data" and n["decl_name"] == label
    assert _one(g, "f()")["decl_kind"] == "function"
    assert _one(g, "E")["decl_kind"] == "enum"
    assert _one(g, "T")["decl_kind"] == "type"
    # interface + const of the same name: both are declarations of their own
    assert _one(g, "I", "type")["source_range"] == [13, 13]
    assert _one(g, "I", "data")["source_range"] == [20, 20]


def test_exported_destructuring_keeps_its_keys_and_imports_get_no_node(tmp_path):
    g = _nodes(tmp_path)["by_label"]
    assert "e1" in g and "e2" in g and "alias" not in g, "exported patterns are named by key (#2604)"
    assert _one(g, "e2")["decl_kind"] == "data"
    assert "lib" not in g, "a require() binding is an import, not a declaration"
    assert "$" not in g, "an unnameable minified name gets no node (#1899)"


def test_load_effects_are_classified(tmp_path):
    g = _nodes(tmp_path)["by_label"]
    assert _one(g, "A")["load_effect"] == "pure"
    assert _one(g, "S")["load_effect"] == "pure"
    assert _one(g, "f()")["load_effect"] == "pure"
    assert _one(g, "B")["load_effect"] == "impure"
    assert _one(g, "p")["load_effect"] == "impure"
    assert _one(g, "K")["load_effect"] == "impure", "a static block runs on load"
    assert _one(g, "Plain")["load_effect"] == "pure", "a method body runs when called, not on load"
    module = _one(g, "<module scope>")
    # bare import, B, the two destructurings from makePair(), K's static block, registerX(), the if
    assert module["effect_ranges"] == [[1, 1], [8, 8], [9, 9], [10, 10], [16, 16], [18, 18], [19, 19]]


def test_a_file_without_effects_has_no_effect_ranges(tmp_path):
    g = _nodes(tmp_path, "export const A = 1;\nexport function f() { return A; }\n")["by_label"]
    assert all("effect_ranges" not in n for nodes in g.values() for n in nodes)


def test_import_kind_per_binding(tmp_path):
    (tmp_path / "b.ts").write_text("export interface A { x: number }\nexport const B = 1;\nexport type C = string;\n")
    (tmp_path / "a.ts").write_text(
        "import type { A } from './b';\n"
        "import { type C, B } from './b';\n"
        "import { type C as D } from './b';\n"
        "export type { A as AA } from './b';\n"
        "export function f(a: A): number { return B; }\n")
    r = extract([tmp_path / "a.ts", tmp_path / "b.ts"], root=tmp_path, cache_root=tmp_path / "out", parallel=False)
    got = sorted((e["relation"], e["source_location"], e["target"].split("_")[-1], e.get("import_kind"))
                 for e in r["edges"] if e["relation"] in ("imports", "imports_from", "re_exports")
                 and e.get("source_file", "").endswith("a.ts"))
    assert got == [
        ("imports", "L1", "a", "type"),
        ("imports", "L2", "b", "value"),
        ("imports", "L2", "c", "type"),
        ("imports", "L3", "c", "type"),
        ("imports_from", "L1", "b", "type"),
        ("imports_from", "L2", "b", "value"),
        ("imports_from", "L3", "b", "type"),
        ("imports_from", "L4", "b", "type"),
        ("re_exports", "L4", "a", "type"),
        ("re_exports", "L4", "b", "type"),
    ]


def test_require_bindings_are_value_imports(tmp_path):
    (tmp_path / "lib.ts").write_text("export function cmd() { return 1; }\n")
    (tmp_path / "use.ts").write_text("export function run() {\n  const { cmd } = require('./lib');\n  return cmd();\n}\n")
    r = extract([tmp_path / "use.ts", tmp_path / "lib.ts"], root=tmp_path, cache_root=tmp_path / "out", parallel=False)
    kinds = {(e["relation"], e.get("import_kind")) for e in r["edges"]
             if e["relation"] in ("imports", "imports_from") and e.get("source_file", "").endswith("use.ts")}
    assert kinds and all(k == "value" for _, k in kinds), kinds
