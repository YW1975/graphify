"""RLL S3 (symbol-level impact design 2026-10-04 §3.2 item 2): same-file reads of
module-level declarations, `<owner> -references-> <declaration>` with context read|write.

A read here is what makes a change to a constant reach the code that uses it: the call
graph alone cannot, because reading `LIMIT` calls nothing. Shadowing is where this goes
wrong — a name wrongly treated as local loses its edge, which later means a test not
selected — so most cases below are about scopes.
"""
import os
from pathlib import Path

import pytest

from graphify.build import build_from_json
from graphify.extract import extract


def _extract(tmp_path, files: dict[str, str]):
    base = tmp_path / "src"
    base.mkdir()
    for name, body in files.items():
        (base / name).write_text(body)
    old = os.getcwd()
    try:
        os.chdir(tmp_path)
        return extract([Path("src") / n for n in files], cache_root=Path(".cache"), parallel=False)
    finally:
        os.chdir(old)


def _refs(tmp_path, body: str, name: str = "m.ts") -> set[tuple[str, str, str]]:
    """(owner, declaration, context), each id shortened by its `src_<stem>_` prefix."""
    r = _extract(tmp_path, {name: body})
    prefix = "src_" + name.split(".")[0] + "_"
    short = lambda nid: nid[len(prefix):] if nid.startswith(prefix) else nid  # noqa: E731
    return {(short(e["source"]), short(e["target"]), e["context"])
            for e in r["edges"] if e["relation"] == "references"}


def _reads(tmp_path, body: str, name: str = "m.ts") -> set[tuple[str, str]]:
    return {(s, t) for s, t, c in _refs(tmp_path, body, name) if c == "read"}


# ── reads ─────────────────────────────────────────────────────────────────────

def test_a_function_reading_a_constant(tmp_path):
    assert ("f", "limit") in _reads(tmp_path, "const LIMIT = 3;\nexport function f() { return LIMIT * 2; }\n")


def test_member_receiver_argument_and_template_are_reads(tmp_path):
    reads = _reads(tmp_path, (
        "const A = ['x'];\nconst B = 'b';\nconst C = 'c';\n"
        "function sink(v: unknown) { return v; }\n"
        "export function f() { A.join(','); sink(B); return `${C}`; }\n"
    ))
    assert {("f", "a"), ("f", "b"), ("f", "c")} <= reads


def test_shorthand_property_is_a_read(tmp_path):
    assert ("f", "a") in _reads(tmp_path, "const A = 1;\nexport function f() { return { A }; }\n")


def test_default_parameter_value_is_a_read_of_the_function(tmp_path):
    reads = _reads(tmp_path, "const X = 1;\nexport const f = (a = X) => a;\n")
    assert ("f", "x") in reads


def test_defaults_inside_patterns_are_reads(tmp_path):
    """`{a = X}` and TS `c: number = Y`: the default is read, only a / c are bound."""
    reads = _reads(tmp_path, (
        "const X = 1;\nconst Y = 2;\n"
        "export function f({ a = X }: { a?: number }, c: number = Y) { return a + c; }\n"
    ))
    assert {("f", "x"), ("f", "y")} <= reads


def test_class_method_and_field_initializer(tmp_path):
    reads = _reads(tmp_path, (
        "const X = 1;\nconst Y = 2;\n"
        "export class K {\n  y = Y;\n  m() { return X; }\n}\n"
    ))
    assert ("k_m", "x") in reads
    assert any(t == "y" for _s, t in reads), reads


def test_initializer_reads_belong_to_the_declaration(tmp_path):
    reads = _reads(tmp_path, "const A = [1];\nconst B = [...A];\nexport const C = B.length;\n")
    assert {("b", "a"), ("c", "b")} <= reads


def test_load_time_statement_reads_belong_to_module_scope(tmp_path):
    assert ("module", "a") in _reads(tmp_path, "const A = 1;\nconsole.log(A);\n")


def test_export_default_reads_the_value(tmp_path):
    assert ("ts", "a") in _reads(tmp_path, "const A = 1;\nexport default A;\n")


def test_an_untracked_closure_reads_for_its_enclosing_function(tmp_path):
    assert ("f", "x") in _reads(tmp_path, "const X = 1;\nexport function f() { return [1].map(() => X); }\n")


def test_jsx_expression_and_component_tag_are_reads_intrinsic_tag_is_not(tmp_path):
    reads = _reads(tmp_path, (
        "const V = 1;\nconst Comp = () => null;\nconst div = 0;\n"
        "export function R() { return <section><Comp a={V} /><div /></section>; }\n"
    ), name="m.tsx")
    assert {("r", "v"), ("r", "comp")} <= reads
    assert ("r", "div") not in reads


def test_a_read_inside_a_test_case_belongs_to_the_case(tmp_path):
    r = _extract(tmp_path, {"m.test.ts": (
        "const FIXTURE = [1];\n"
        "declare function it(n: string, f: () => void): void;\n"
        "it('uses the fixture', () => { FIXTURE.length; });\n"
    )})
    label = {n["id"]: n["label"] for n in r["nodes"]}
    owners = {label[e["source"]] for e in r["edges"]
              if e["relation"] == "references" and label.get(e["target"]) == "FIXTURE"}
    assert any("uses the fixture" in o for o in owners), owners


# ── not reads ─────────────────────────────────────────────────────────────────

def test_property_names_types_labels_and_export_lists_are_not_reads(tmp_path):
    refs = _refs(tmp_path, (
        "const X = 1;\n"
        "function id<T>(v: unknown): T { return v as T; }\n"
        "export function f(o: { X: number }, t: typeof X) {\n"
        "  const p = { X: 2 };\n  X: for (;;) { break X; }\n"
        "  const q = (o as unknown as typeof X) + id<typeof X>(1);\n  return o.X + p.X + q;\n}\n"
        "export { X };\n"
    ))
    assert not any(t == "x" for _s, t, _c in refs), refs


# ── shadowing ─────────────────────────────────────────────────────────────────

SHADOWS = {
    "parameter": "export function f(X: number) { return X; }\n",
    "single arrow parameter": "export const f = X => X;\n",
    "destructured parameter": "export const f = ({ X }: { X: number }) => X;\n",
    "catch binding": "export function f() { try { g(); } catch (X) { return X; } }\nfunction g() {}\n",
    "for-of binding": "export function f(xs: number[]) { for (const X of xs) { use(X); } }\nfunction use(v: number) {}\n",
    "hoisted var": "export function f() { use(X); var X = 2; }\nfunction use(v: number) {}\n",
    "named function expression": "export const f = function X() { return X; };\n",
    "nested function declaration": "export function f() { function X() {} return X; }\n",
    "nested class declaration": "export function f() { class X {} return X; }\n",
    "closure over a shadowing local": "export function f(X: number) { return () => X; }\n",
}


@pytest.mark.parametrize("form", sorted(SHADOWS))
def test_each_local_binding_shadows_the_module_declaration(tmp_path, form):
    refs = _refs(tmp_path, "const X = 1;\n" + SHADOWS[form])
    assert not any(t == "x" for _s, t, _c in refs), refs


def test_a_block_let_shadows_inside_its_block_only(tmp_path):
    r = _extract(tmp_path, {"m.ts": (
        "const X = 1;\n"
        "export function f() {\n  { let X = 2; use(X); }\n  return X;\n}\n"
        "function use(v: number) {}\n"
    )})
    lines = [e["source_location"] for e in r["edges"]
             if e["relation"] == "references" and e["target"] == "src_m_x"]
    assert lines == ["L4"], lines


def test_a_parameter_does_not_shadow_outside_its_function(tmp_path):
    reads = _reads(tmp_path, (
        "const X = 1;\n"
        "export function f(X: number) { return X; }\n"
        "export function g() { return X; }\n"
    ))
    assert ("g", "x") in reads and ("f", "x") not in reads


# ── writes ────────────────────────────────────────────────────────────────────

def test_assignment_is_a_write_update_is_both(tmp_path):
    refs = _refs(tmp_path, (
        "let n = 0;\nlet m = 0;\n"
        "export function set() { n = 1; }\n"
        "export function bump() { m++; }\n"
    ))
    assert ("set", "n", "write") in refs and ("set", "n", "read") not in refs
    assert {("bump", "m", "write"), ("bump", "m", "read")} <= refs


# ── on the built graph ────────────────────────────────────────────────────────

def test_a_read_on_a_called_pair_is_listed_beside_the_call(tmp_path):
    r = _extract(tmp_path, {"m.ts": "function g() {}\nexport function f() { g(); return g; }\n"})
    d = build_from_json(r, directed=True).edges["src_m_f", "src_m_g"]
    assert d["relation"] == "calls"
    assert ("references", "read") in {(x["relation"], x.get("context")) for x in d["relations"]}


def test_a_default_value_no_longer_hides_a_by_name_callback(tmp_path):
    """Collector fix: `{a = k}` used to bind k itself, so passing k on emitted no
    indirect_call to the module function k."""
    r = _extract(tmp_path, {"m.js": (
        "function k(x) { return x; }\n"
        "export function run(pool, { a = k } = {}) { pool.submit(k); return a; }\n"
    )})
    pairs = {(e["source"], e["target"]) for e in r["edges"] if e["relation"] == "indirect_call"}
    assert ("src_m_run", "src_m_k") in pairs
