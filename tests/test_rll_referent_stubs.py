# Copyright 2026 RLL project contributors. Licensed under the Apache License, Version 2.0.
# Added by the RLL project (2026-10); not part of upstream Graphify.
"""A referent (a command, tier or document a file merely names) is a sourceless stub.

Upstream's sourceless stubs carry ``source_file: ""``: the key is present, so the
schema validator is satisfied, and the empty value is what every "is this a stub"
check reads (``not d.get("source_file")``). The RLL referents omitted the key
instead, and every graph build printed "106x missing required field
'source_file'". An earlier placeholder path was worse: the incremental prune
matched it as a deleted file and took every invokes edge with it.
"""
from __future__ import annotations

import json
from pathlib import Path

from networkx.readwrite import json_graph

from graphify.build import build_from_json, build_merge
from graphify.extract import extract
from graphify.validate import validate_extraction

_SKILL = "---\nname: {n}\ndescription: d\ninvokes: [status]\n---\nbody\n"


def _extract(tmp_path, names=("a", "b")):
    paths = []
    for n in names:
        p = tmp_path / "skills" / n / "SKILL.md"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(_SKILL.format(n=n))
        paths.append(p)
    return extract(paths, root=tmp_path, cache_root=tmp_path / "graphify-out", parallel=False)


def _status_nodes(nodes):
    return [n for n in nodes if n.get("id") == "rll_command_status"]


def test_a_referent_carries_an_empty_source_file_and_validates(tmp_path):
    r = _extract(tmp_path)
    stubs = _status_nodes(r["nodes"])
    assert len(stubs) == 2, "each citing file mints the same id; the build merges them"
    assert all("source_file" in s and s["source_file"] == "" for s in stubs), stubs
    missing = [e for e in validate_extraction(r) if "missing required field 'source_file'" in e]
    assert missing == []
    assert len(_status_nodes([{"id": n} for n in build_from_json(r).nodes])) == 1


def test_pruning_one_citer_keeps_the_referent_and_the_other_citers_edge(tmp_path):
    r = _extract(tmp_path)
    G = build_from_json(r)
    gp = tmp_path / "graph.json"
    gp.write_text(json.dumps(json_graph.node_link_data(G, edges="links")), encoding="utf-8")
    G2 = build_merge([{"nodes": [], "edges": [], "hyperedges": []}], graph_path=str(gp),
                     prune_sources=["skills/a/SKILL.md"], root=str(tmp_path))
    assert "rll_command_status" in G2, "a referent is never matched as a deleted source file"
    citers = {d.get("source_file") for _, _, d in G2.edges("rll_command_status", data=True)}
    assert citers == {"skills/b/SKILL.md"}
