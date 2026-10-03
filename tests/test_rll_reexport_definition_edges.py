# Copyright 2026 RLL project contributors. Licensed under the Apache License, Version 2.0.
# Added by the RLL project (2026-10); not part of upstream Graphify.
"""Edges through a re-export barrel land on the definition, not the barrel.

Added by the RLL project (2026-10). `import { qux } from './index'` where
index.ts re-exports y's `baz as qux` (or `export * from`, or `import {x};
export {x}`) produced an `imports` edge to `index_qux` — a re-export site
with no node of its own, minted later as a phantom stub — next to the
correct edge to the definition.
"""
from __future__ import annotations

from graphify.build import build_from_json
from graphify.extract import extract

_FILES = {
    "src/lib/y.ts": (
        "export function foo() { return 1; }\n"
        "export function baz() { return 2; }\n"
        "export function deep() { return 3; }\n"
        "export class Svc { run() { return 4; } }\n"
    ),
    "src/mid/index.ts": "export { deep } from '../lib/y';\nexport * from '../lib/y';\n",
    "src/index.ts": (
        "export { foo } from './lib/y';\n"
        "export { baz as qux } from './lib/y';\n"
        "export * from './mid/index';\n"
        "import { Svc } from './lib/y';\n"
        "export { Svc };\n"
    ),
    "src/consumer.ts": (
        "import { foo, qux, deep, Svc } from './index';\n"
        "export function useIt() { return foo() + qux() + deep() + new Svc().run(); }\n"
    ),
}


def _extract(tmp_path):
    for name, body in _FILES.items():
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)
    return extract([tmp_path / n for n in _FILES], root=tmp_path,
                   cache_root=tmp_path / "graphify-out", parallel=False)


def test_consumer_edges_point_at_the_definitions(tmp_path):
    r = _extract(tmp_path)
    by_id = {n["id"]: n for n in r["nodes"]}
    consumer_file = next(n["id"] for n in r["nodes"] if n["label"] == "consumer.ts")
    imports = [e for e in r["edges"] if e["source"] == consumer_file and e["relation"] == "imports"]
    assert imports
    for e in imports:
        assert e["target"] in by_id, f"import edge to a re-export site with no node: {e['target']}"
        assert by_id[e["target"]]["source_file"] == "src/lib/y.ts", e
    assert {by_id[e["target"]]["label"] for e in imports} == {"foo()", "baz()", "deep()", "Svc"}
    calls = {by_id[e["target"]]["label"] for e in r["edges"]
             if e["relation"] == "calls" and by_id.get(e["source"], {}).get("label") == "useIt()"}
    assert {"foo()", "baz()", "deep()"} <= calls


def test_no_phantom_stub_for_a_reexported_name(tmp_path):
    r = _extract(tmp_path)
    G = build_from_json(r, root=tmp_path)
    phantoms = [nid for nid, d in G.nodes(data=True)
                if not d.get("source_file") and nid.startswith(("src_index_", "src_mid_index_"))]
    assert not phantoms, phantoms
