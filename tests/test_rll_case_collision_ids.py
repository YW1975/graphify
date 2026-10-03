"""Top-level JS/TS names that differ only in case keep separate nodes.

Added by the RLL project (2026-10). Node ids are case-folded, so `interface
WorktreeFingerprint` and `function worktreeFingerprint()` in one file minted
the same id and the function node was silently dropped.
"""
from __future__ import annotations

from graphify.extract import extract

_SRC = (
    "export interface WorktreeFingerprint { files: string[] }\n"
    "export function worktreeFingerprint(): WorktreeFingerprint { return { files: [] }; }\n"
    "export const SETTINGS = 1;\n"
    "export const settings = () => SETTINGS;\n"
    "export function local() { return worktreeFingerprint(); }\n"
)
_USER = (
    "import { worktreeFingerprint, settings } from './fp';\n"
    "export function caller() { return worktreeFingerprint() && settings(); }\n"
)


def _extract(tmp_path):
    for name, body in {"fp.ts": _SRC, "user.ts": _USER}.items():
        (tmp_path / name).write_text(body)
    return extract([tmp_path / "fp.ts", tmp_path / "user.ts"], root=tmp_path,
                   cache_root=tmp_path / "graphify-out", parallel=False)


def test_both_case_variants_survive(tmp_path):
    r = _extract(tmp_path)
    labels = [n["label"] for n in r["nodes"] if n["source_file"] == "fp.ts"]
    assert "WorktreeFingerprint" in labels and "worktreeFingerprint()" in labels
    assert "SETTINGS" in labels and "settings()" in labels
    ids = [n["id"] for n in r["nodes"]]
    assert len(ids) == len(set(ids))


def test_the_callable_keeps_the_plain_id_and_receives_the_calls(tmp_path):
    r = _extract(tmp_path)
    by_id = {n["id"]: n for n in r["nodes"]}
    fn = next(n for n in r["nodes"] if n["label"] == "worktreeFingerprint()")
    assert fn["id"] == "fp_worktreefingerprint"
    callers = {by_id[e["source"]]["label"] for e in r["edges"]
               if e["relation"] == "calls" and e["target"] == fn["id"]}
    assert {"local()", "caller()"} <= callers
    st = next(n for n in r["nodes"] if n["label"] == "settings()")
    assert any(e["relation"] == "calls" and e["target"] == st["id"]
               and by_id[e["source"]]["label"] == "caller()" for e in r["edges"])
