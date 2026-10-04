# Copyright 2026 RLL project contributors. Licensed under the Apache License, Version 2.0.
# Added by the RLL project (2026-10); not part of upstream Graphify.
"""`extract --directed` builds a directed graph, and the choice persists.

Added by the RLL project (2026-10). An undirected graph keeps one edge per node pair,
so mutual calls (a→b and b→a) collapsed into one and a reverse walk lost a direction.
`extract` had no way to ask for a directed graph; and a flag that is not persisted
would be undone by the next flag-less `extract`/hook run.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

PYTHON = sys.executable


def _extract(repo: Path, *extra: str) -> dict:
    env = os.environ.copy()
    env["GRAPHIFY_OUT"] = str(repo / "graphify-out")
    r = subprocess.run([PYTHON, "-m", "graphify", "extract", ".", "--code-only", *extra],
                       cwd=repo, capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    return json.loads((repo / "graphify-out" / "graph.json").read_text(encoding="utf-8"))


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("from b import pong\n\ndef ping(n):\n    return pong(n - 1) if n else 0\n", encoding="utf-8")
    (repo / "b.py").write_text("def pong(n):\n    from a import ping\n    return ping(n - 1) if n else 1\n", encoding="utf-8")
    return repo


def _calls(g: dict) -> set:
    return {(e["source"], e["target"]) for e in g["links"] if e.get("relation") == "calls"}


def test_directed_extract_keeps_both_directions_and_persists(tmp_path):
    repo = _repo(tmp_path)
    g = _extract(repo, "--directed")
    assert g["directed"] is True
    calls = _calls(g)
    assert any((t, s) in calls for s, t in calls), f"mutual calls are both present: {calls}"
    cfg = json.loads((repo / "graphify-out" / ".graphify_build.json").read_text(encoding="utf-8"))
    assert cfg.get("directed") is True
    # A later flag-less run keeps the persisted choice.
    (repo / "a.py").write_text((repo / "a.py").read_text(encoding="utf-8") + "\n# touch\n", encoding="utf-8")
    assert _extract(repo)["directed"] is True


def test_without_the_flag_the_graph_stays_undirected(tmp_path):
    g = _extract(_repo(tmp_path))
    assert g["directed"] is False
