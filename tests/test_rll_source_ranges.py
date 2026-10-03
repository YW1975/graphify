"""Callable / class nodes carry their full line range (``source_range``).

Added by the RLL project (2026-10). ``source_location`` only names a node's
first line, so a diff hunk in the middle of a function could not be mapped to
it. Each range below is checked against the fixture's own line numbers.
"""
from __future__ import annotations

from graphify.extract import extract

_SRC = """\
export function decl(x: number): number {
  const y = x + 1;
  return y;
}

export const arrow = (z: number) => {
  return decl(z);
};

export class Klass {
  method(a: number) {
    function nested() {
      return decl(a);
    }
    return nested();
  }

  other() { return 1; }
}

const oneLiner = () => 42;
"""


def _ranges(tmp_path, name="mod.ts", src=_SRC):
    p = tmp_path / name
    p.write_text(src)
    r = extract([p], root=tmp_path, cache_root=tmp_path / "graphify-out", parallel=False)
    return {n["label"]: n.get("source_range") for n in r["nodes"]}


def test_function_declaration_range(tmp_path):
    assert _ranges(tmp_path)["decl()"] == [1, 4]


def test_arrow_const_range(tmp_path):
    rng = _ranges(tmp_path)
    assert rng["arrow()"] == [6, 8]
    assert rng["oneLiner()"] == [21, 21]


def test_class_and_method_ranges(tmp_path):
    rng = _ranges(tmp_path)
    assert rng["Klass"] == [10, 19]
    assert rng[".method()"] == [11, 16]
    assert rng[".other()"] == [18, 18]


def test_nested_function_range(tmp_path):
    assert _ranges(tmp_path)["nested()"] == [12, 14]


def test_file_node_has_no_callable_range(tmp_path):
    assert _ranges(tmp_path)["mod.ts"] is None


def test_python_ranges_too(tmp_path):
    src = "def f():\n    x = 1\n    return x\n\n\nclass C:\n    def m(self):\n        return f()\n"
    rng = _ranges(tmp_path, "m.py", src)
    assert rng["f()"] == [1, 3]
    assert rng["C"] == [6, 8]
    assert rng[".m()"] == [7, 8]
