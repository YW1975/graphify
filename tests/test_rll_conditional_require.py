"""RLL: a module-scope `require()` under an if / switch case / try / loop is tagged
`conditional`; an unconditional module-scope one and one inside a function are not.

The danger is the false positive: a require tagged conditional that is not lets a
consumer narrow what a change reaches. So the plain cases must stay untagged.
"""
from __future__ import annotations

from pathlib import Path

from graphify.extract import _file_node_id, extract


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _require_edges(result: dict, importer: str):
    src = _file_node_id(Path(importer))
    return [e for e in result["edges"] if e["relation"] == "imports_from" and e.get("source_file") == importer
            and (e["source"] == src or not e["source"].startswith("ref_"))]


def test_module_scope_require_under_a_statement_is_conditional(tmp_path: Path):
    for name in ("a", "b", "c", "d", "e"):
        _write(tmp_path / f"src/{name}.ts", f"export function {name}() {{ return 1 }}\n")
    importer = _write(tmp_path / "src/cli.ts", "\n".join([
        'const cmd = process.argv[2];',
        'const { a } = require("./a");',                       # plain module scope: NOT conditional
        '{ const { b } = require("./b"); b(); }',               # bare block: runs on load, NOT conditional
        'switch (cmd) {',
        '  case "c": { const { c } = require("./c"); c(); break; }',
        '}',
        'if (cmd === "d") { const { d } = require("./d"); d(); }',
        'export function lazy() { const { e } = require("./e"); return e(); }',  # in a function: NOT conditional
        '',
    ]))
    result = extract([*(tmp_path / "src").glob("*.ts")], cache_root=tmp_path)
    by_line = {e["source_location"]: e for e in result["edges"] if e["relation"] == "imports_from" and e.get("source_file") == "src/cli.ts"}
    assert by_line["L2"].get("conditional") is None, "plain module-scope require"
    assert by_line["L3"].get("conditional") is None, "a bare block runs on load"
    assert by_line["L5"].get("conditional") is True, "switch case"
    assert by_line["L7"].get("conditional") is True, "if"
    assert by_line["L8"].get("conditional") is None, "inside a function it belongs to the function"
    assert by_line["L8"]["source"] != _file_node_id(Path("src/cli.ts")), "attributed to lazy(), not the file"
