"""A call and an import on the same node pair both survive the build.

Added by the RLL project (2026-10). The graph keeps one edge per node pair;
with `file -imports-> foo` and `file -calls-> foo` the import won (alphabetic
order), so the call disappeared. The call now keeps the slot and the
surviving edge lists every relation seen on the pair.
"""
from __future__ import annotations

import pytest

from graphify.build import build_from_json


def _ext(order):
    nodes = [
        {"id": "a", "label": "a.ts", "file_type": "code", "source_file": "a.ts"},
        {"id": "b_foo", "label": "foo()", "file_type": "code", "source_file": "b.ts"},
    ]
    edge = {
        "imports": {"source": "a", "target": "b_foo", "relation": "imports", "confidence": "EXTRACTED",
                    "source_file": "a.ts", "source_location": "L1", "context": "import", "type_only": True},
        "calls": {"source": "a", "target": "b_foo", "relation": "calls", "confidence": "EXTRACTED",
                  "source_file": "a.ts", "source_location": "L3", "context": "call"},
    }
    return {"nodes": nodes, "edges": [edge[k] for k in order]}


@pytest.mark.parametrize("order", [("imports", "calls"), ("calls", "imports")])
@pytest.mark.parametrize("directed", [False, True])
def test_call_wins_the_pair_and_import_is_recorded(order, directed):
    G = build_from_json(_ext(order), directed=directed)
    d = G.edges["a", "b_foo"]
    assert d["relation"] == "calls"
    assert d["relations"] == ["calls", "imports"]
    assert d["source_location"] == "L3"
    assert "type_only" not in d  # no key of the displaced import lingers on the call


def test_unrelated_relation_pairs_keep_previous_behaviour():
    ext = _ext(("calls",))
    ext["edges"].append({"source": "a", "target": "b_foo", "relation": "references",
                         "confidence": "EXTRACTED", "source_file": "a.ts"})
    d = build_from_json(ext).edges["a", "b_foo"]
    assert d["relation"] == "calls"
    assert "relations" not in d
