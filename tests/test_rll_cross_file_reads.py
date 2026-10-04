"""RLL S4 (symbol-level impact design 2026-10-04 §3.2 item 2, cross-file half): a read
of an IMPORTED name becomes `<owner> -references-> <definition>` in the module that
defines it, through re-exports, `export *`, default exports, namespace imports and
CommonJS `require` (cli code in this repository is mostly lazy requires).

The per-file pass records each read with its owner and import source; the corpus pass
resolves the source and follows the name to its declaration.
"""
from pathlib import Path

from graphify.build import build_from_json
from graphify.extract import extract

DEFS = (
    "export const LIMIT = 3;\n"
    "export const NAMES = ['a'];\n"
    "const D = 7;\n"
    "export default D;\n"
    "export function helper() { return 1; }\n"
)


def _extract(tmp_path: Path, files: dict[str, str], cache: str = ".cache"):
    for name, body in files.items():
        p = tmp_path / "src" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)
    return extract([tmp_path / "src" / n for n in files], root=tmp_path,
                   cache_root=tmp_path / cache, parallel=False)


def _reads(r) -> set[tuple[str, str, str]]:
    """(owner label, target label, target file name) of every cross-file read."""
    node = {n["id"]: n for n in r["nodes"]}
    out = set()
    for e in r["edges"]:
        if e["relation"] != "references" or e.get("context") != "read":
            continue
        s, t = node.get(e["source"]), node.get(e["target"])
        if s is None or t is None or s.get("source_file") == t.get("source_file"):
            continue
        out.add((s["label"].rstrip("()"), t["label"].rstrip("()"), Path(t["source_file"]).name))
    return out


def _use(tmp_path, body: str, extra: dict[str, str] | None = None):
    return _reads(_extract(tmp_path, {"defs.ts": DEFS, **(extra or {}), "use.ts": body}))


# ── ES imports ────────────────────────────────────────────────────────────────

def test_named_and_aliased_imports(tmp_path):
    reads = _use(tmp_path, (
        "import { LIMIT, NAMES as N } from './defs';\n"
        "export function a() { return LIMIT + N.length; }\n"
    ))
    assert {("a", "LIMIT", "defs.ts"), ("a", "NAMES", "defs.ts")} <= reads


def test_default_import_reaches_the_declaration_behind_export_default(tmp_path):
    assert ("a", "D", "defs.ts") in _use(tmp_path, "import Dflt from './defs';\nexport function a() { return Dflt; }\n")


def test_export_as_default_is_followed(tmp_path):
    reads = _use(tmp_path, "import V from './v';\nexport function a() { return V; }\n",
                 {"v.ts": "const VALUE = 1;\nexport { VALUE as default };\n"})
    assert ("a", "VALUE", "v.ts") in reads


def test_namespace_member_and_whole_namespace(tmp_path):
    reads = _use(tmp_path, (
        "import * as ns from './defs';\n"
        "export function a() { return ns.LIMIT; }\n"
        "export function b() { return ns; }\n"
    ))
    assert ("a", "LIMIT", "defs.ts") in reads
    assert ("b", "defs.ts", "defs.ts") in reads  # the module object read whole


def test_type_only_use_of_an_import_is_not_a_read(tmp_path):
    reads = _use(tmp_path, (
        "import type { LIMIT } from './defs';\n"
        "export function a(x: typeof LIMIT) { return x as unknown as typeof LIMIT; }\n"
    ))
    assert not reads


# ── CommonJS ──────────────────────────────────────────────────────────────────

def test_destructured_require_at_top_level_and_inside_a_function(tmp_path):
    reads = _use(tmp_path, (
        "const { LIMIT } = require('./defs');\n"
        "export function a() { return LIMIT; }\n"
        "export function b() { const { NAMES } = require('./defs'); return NAMES; }\n"
    ))
    assert {("a", "LIMIT", "defs.ts"), ("b", "NAMES", "defs.ts")} <= reads


def test_whole_module_require_member_and_inline_require_member(tmp_path):
    reads = _use(tmp_path, (
        "export function a() { const m = require('./defs'); return m.LIMIT; }\n"
        "export function b() { return require('./defs').NAMES; }\n"
    ))
    assert {("a", "LIMIT", "defs.ts"), ("b", "NAMES", "defs.ts")} <= reads


def test_a_lazy_require_does_not_leak_into_a_sibling(tmp_path):
    """b's NAMES is this file's own declaration, not a's lazily required one."""
    r = _extract(tmp_path, {"defs.ts": DEFS, "use.ts": (
        "const NAMES = 0;\n"
        "export function a() { const { NAMES } = require('./defs'); return NAMES; }\n"
        "export function b() { return NAMES; }\n"
    )})
    assert ("a", "NAMES", "defs.ts") in _reads(r)
    assert ("b", "NAMES", "defs.ts") not in _reads(r)
    node = {n["id"]: n for n in r["nodes"]}
    same_file = {(node[e["source"]]["label"], node[e["target"]]["label"]) for e in r["edges"]
                 if e["relation"] == "references" and node[e["source"]]["source_file"] == node[e["target"]]["source_file"]}
    assert ("b()", "NAMES") in same_file


# ── re-exports ────────────────────────────────────────────────────────────────

def test_barrel_rename_and_two_level_export_star(tmp_path):
    reads = _use(tmp_path, (
        "import { CAP } from './barrel';\n"
        "import { NAMES } from './outer';\n"
        "export function a() { return CAP; }\n"
        "export function b() { return NAMES; }\n"
    ), {
        "barrel.ts": "export { LIMIT as CAP } from './defs';\n",
        "inner.ts": "export * from './defs';\n",
        "outer.ts": "export * from './inner';\n",
    })
    assert {("a", "LIMIT", "defs.ts"), ("b", "NAMES", "defs.ts")} <= reads


# ── shadowing and ordinary objects ────────────────────────────────────────────

def test_a_parameter_shadows_an_import_and_a_namespace(tmp_path):
    reads = _use(tmp_path, (
        "import { LIMIT } from './defs';\nimport * as ns from './defs';\n"
        "export function a(LIMIT: number) { return LIMIT; }\n"
        "export function b(ns: { LIMIT: number }) { return ns.LIMIT; }\n"
    ))
    assert not any(owner in ("a", "b") for owner, _t, _f in reads), reads


def test_a_member_of_an_ordinary_object_is_not_an_import_read(tmp_path):
    reads = _use(tmp_path, (
        "import { helper } from './defs';\n"
        "export function a() { const cfg = { LIMIT: 1 }; return cfg.LIMIT + helper(); }\n"
    ))
    assert ("a", "LIMIT", "defs.ts") not in reads
    assert ("a", "helper", "defs.ts") in reads


# ── owners ────────────────────────────────────────────────────────────────────

def test_owners_method_module_scope_and_initializer(tmp_path):
    reads = _use(tmp_path, (
        "import { LIMIT, NAMES } from './defs';\n"
        "export class K { m() { return LIMIT; } }\n"
        "console.log(NAMES);\n"
    ))
    assert (".m", "LIMIT", "defs.ts") in reads, reads  # method nodes are labelled `.m()`
    assert ("<module scope>", "NAMES", "defs.ts") in reads


def test_an_initializer_read_in_a_file_without_load_effects(tmp_path):
    """No effect statement, so no `<module scope>` node: the read belongs to the
    declaration whose initializer holds it and must not vanish at the owned check."""
    assert ("TWICE", "LIMIT", "defs.ts") in _use(
        tmp_path, "import { LIMIT } from './defs';\nexport const TWICE = LIMIT * 2;\n")


def test_a_read_inside_a_test_case_belongs_to_the_case(tmp_path):
    reads = _reads(_extract(tmp_path, {"defs.ts": DEFS, "use.test.ts": (
        "import { LIMIT } from './defs';\n"
        "declare function it(n: string, f: () => void): void;\n"
        "it('respects the limit', () => { LIMIT; });\n"
    )}))
    assert any("respects the limit" in o and t == "LIMIT" for o, t, _f in reads), reads


# ── unresolvable, merge, cache ────────────────────────────────────────────────

def test_an_external_package_gives_no_edge_and_no_phantom_node(tmp_path):
    r = _extract(tmp_path, {"use.ts": "import { chunk } from 'lodash';\nexport function a() { return chunk; }\n"})
    assert not _reads(r)
    assert not any(n["label"] == "chunk" for n in r["nodes"])


def test_an_imported_function_called_and_passed_keeps_one_call_edge(tmp_path):
    r = _extract(tmp_path, {"defs.ts": DEFS, "use.ts": (
        "import { helper } from './defs';\nexport function f() { helper(); return [helper]; }\n"
    )})
    ids = {n["label"]: n["id"] for n in r["nodes"]}
    d = build_from_json(r, directed=True).edges[ids["f()"], ids["helper()"]]
    assert d["relation"] == "calls"
    assert ("references", "read") in {(x["relation"], x.get("context")) for x in d["relations"]}


def test_repeated_extraction_gives_the_same_reads(tmp_path):
    """JS/TS bypass the AST cache (`_JS_CACHE_BYPASS_SUFFIXES`), so this is not a cache
    round-trip test; it pins that a second run into the same cache dir agrees."""
    files = {"defs.ts": DEFS, "use.ts": (
        "import * as ns from './defs';\n"
        "export function a() { const { NAMES } = require('./defs'); return ns.LIMIT + NAMES.length; }\n"
    )}
    cold = _reads(_extract(tmp_path, files))
    warm = _reads(_extract(tmp_path, files))
    assert cold == warm and {("a", "LIMIT", "defs.ts"), ("a", "NAMES", "defs.ts")} <= cold


def test_a_ref_whose_owner_has_no_node_is_dropped(tmp_path):
    """The corpus pass only links owners that exist: an id remapped or merged away must
    not leave an edge from a node-less source (the #2262 producer guard)."""
    from graphify.extractors.models import _SymbolResolutionFacts
    from graphify.extractors.resolution import _apply_symbol_resolution_facts
    src = tmp_path / "src"
    src.mkdir()
    (src / "defs.ts").write_text(DEFS)
    (src / "use.ts").write_text("export function a() {}\n")
    nodes = [{"id": "defs_limit", "label": "LIMIT", "source_file": str(src / "defs.ts")},
             {"id": "use_a", "label": "a()", "source_file": str(src / "use.ts")}]
    edges: list = []
    refs = [(src / "use.ts", {"owner": owner, "spec": "./defs", "imported": "LIMIT", "ctx": "read", "line": 1})
            for owner in ("use_a", "ghost")]
    _apply_symbol_resolution_facts([src / "defs.ts", src / "use.ts"], nodes, edges, tmp_path,
                                   _SymbolResolutionFacts(), rll_import_refs=refs)
    assert [(e["source"], e["target"]) for e in edges if e["relation"] == "references"] == [("use_a", "defs_limit")]


def test_a_read_does_not_crowd_out_a_namespace_member_call(tmp_path):
    """`defs.helper()` is resolved by a member-call resolver that keeps one edge
    per pair; the read of the same member, added earlier, must not occupy that pair.
    On super-rll 44 such calls vanished before the pair sets ignored reads."""
    r = _extract(tmp_path, {
        "defs.ts": DEFS,
        # the receiver matches the module's stem, which is what lets that resolver bind it
        "use.ts": "import * as defs from './defs.js';\nexport function a() { return defs.helper(); }\n",
        "tool.py": "import os\n\ndef main():\n    return os.getcwd()\n",  # a .py makes every resolver run
    })
    ids = {n["label"]: n["id"] for n in r["nodes"]}
    rels = {e["relation"] for e in r["edges"] if e["source"] == ids["a()"] and e["target"] == ids["helper()"]}
    assert {"calls", "references"} <= rels, rels


def test_a_cast_require_binding_is_an_import_not_a_declaration(tmp_path):
    """`const m = require('./defs') as any` is how TS types a CommonJS require: its
    members are read from defs, and `m` itself is no data declaration of this file."""
    r = _extract(tmp_path, {"defs.ts": DEFS, "use.ts": (
        "const m = require('./defs') as any;\nexport function a() { return m.LIMIT; }\n")})
    assert ("a", "LIMIT", "defs.ts") in _reads(r)
    assert not any(n.get("decl_name") == "m" for n in r["nodes"])


def test_a_local_require_from_create_require_is_still_read(tmp_path):
    r = _extract(tmp_path, {"defs.ts": DEFS, "tool.mjs": (
        "import { createRequire } from 'node:module';\n"
        "const require = createRequire(import.meta.url);\n"
        "const { LIMIT } = require('./defs');\n"
        "export function a() { return LIMIT; }\n")})
    node = {n["id"]: n for n in r["nodes"]}
    same = {(node[e["source"]]["label"], node[e["target"]]["label"]) for e in r["edges"]
            if e["relation"] == "references" and node[e["source"]]["source_file"] == node[e["target"]]["source_file"]}
    assert ("<module scope>", "require") in same, same  # the load-time call of the local helper
    assert ("a", "LIMIT", "defs.ts") in _reads(r)  # and the binding it made is still an import
