# Copyright 2026 RLL project contributors. Licensed under the Apache License, Version 2.0.
# Added by the RLL project (2026-10); not part of upstream Graphify.
"""A call and an import on the same node pair both survive the build, each with its own facts.

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
    # One record per relation, each with its own facts: the displaced import keeps its
    # type_only and location in its record instead of losing them to the call.
    assert d["relations"] == [
        {"relation": "calls", "context": "call", "certainty": "certain", "source_location": "L3"},
        {"relation": "imports", "context": "import", "certainty": "certain", "type_only": True, "source_location": "L1"},
    ]
    assert d["source_location"] == "L3"
    assert "type_only" not in d  # no key of the displaced import lingers on the call itself


def test_a_read_on_a_called_pair_keeps_the_call_and_lists_both():
    """S3 (design §3.2): a function that both calls and reads a declaration keeps one
    edge, the call, and the read is listed beside it instead of being dropped."""
    ext = _ext(("calls",))
    ext["edges"].append({"source": "a", "target": "b_foo", "relation": "references", "context": "read",
                         "confidence": "EXTRACTED", "source_file": "a.ts", "source_location": "L3"})
    d = build_from_json(ext).edges["a", "b_foo"]
    assert d["relation"] == "calls"
    assert {(r["relation"], r.get("context")) for r in d["relations"]} == {("calls", "call"), ("references", "read")}


def test_a_read_and_a_write_of_one_declaration_both_survive():
    ext = _ext(())
    for ctx, line in (("read", "L2"), ("write", "L5")):
        ext["edges"].append({"source": "a", "target": "b_foo", "relation": "references", "context": ctx,
                             "confidence": "EXTRACTED", "source_file": "a.ts", "source_location": line})
    d = build_from_json(ext, directed=True).edges["a", "b_foo"]
    assert [(r["context"], r["source_location"]) for r in d["relations"]] == [("read", "L2"), ("write", "L5")]


def test_one_relation_seen_twice_adds_no_relations_list():
    ext = _ext(("calls", "calls"))
    d = build_from_json(ext).edges["a", "b_foo"]
    assert "relations" not in d


def test_directed_graph_keeps_both_directions_of_a_relation():
    nodes = [{"id": "a", "label": "a()", "file_type": "code", "source_file": "a.ts"},
             {"id": "b", "label": "b()", "file_type": "code", "source_file": "b.ts"}]
    edges = [{"source": "a", "target": "b", "relation": "calls", "confidence": "EXTRACTED", "source_file": "a.ts", "source_location": "L1"},
             {"source": "b", "target": "a", "relation": "calls", "confidence": "EXTRACTED", "source_file": "b.ts", "source_location": "L2"}]
    D = build_from_json({"nodes": nodes, "edges": edges}, directed=True)
    assert D.has_edge("a", "b") and D.has_edge("b", "a"), "mutual calls are two edges"
    U = build_from_json({"nodes": nodes, "edges": edges}, directed=False)
    assert U.number_of_edges() == 1, "the undirected graph collapses them (why the RLL graph is directed)"


def test_uncertainty_on_either_side_survives_the_merge():
    ext = _ext(("calls",))
    ext["edges"].append({"source": "a", "target": "b_foo", "relation": "imports", "confidence": "AMBIGUOUS",
                         "source_file": "a.ts", "source_location": "L1", "context": "import",
                         "uncertain": True, "reason": "dynamic specifier"})
    d = build_from_json(ext, directed=True).edges["a", "b_foo"]
    imp = [r for r in d["relations"] if r["relation"] == "imports"][0]
    assert imp["certainty"] == "uncertain" and imp["reason"] == "dynamic specifier"
