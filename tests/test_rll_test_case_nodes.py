# Copyright 2026 RLL project contributors. Licensed under the Apache License, Version 2.0.
# Added by the RLL project (2026-10); not part of upstream Graphify.
"""describe / it / test calls become test-case nodes that own their calls.

Added by the RLL project (2026-10). Calls written inside an anonymous
`it('…', () => {…})` callback used to be attributed to the file node, where
the file's own `imports` edge to the same symbol displaced them (one edge per
node pair) — so most test files had no call edge into product code at all.
"""
from __future__ import annotations

from graphify.extract import extract

_PROD = (
    "export function foo(x: number) { return x; }\n"
    "export function bar() { return 2; }\n"
    "export function baz() { return 3; }\n"
    "export function qux() { return 4; }\n"
)

_TEST = """\
import { describe, it, test } from 'node:test';
import { foo, bar, baz, qux } from '../prod';

function localHelper() {
  return bar();
}

describe('Outer', () => {
  const shared = 1;
  it('calls foo', () => {
    foo(shared);
  });

  describe('Inner', () => {
    it.skip('async helper', async () => {
      await Promise.resolve();
      localHelper();
    });
    test.only('with options', { timeout: 5 }, function () {
      baz();
    });
  });

  it('same title', () => { foo(1); });
  it('same title', () => { qux(); });
  it.todo('later');
});

test('node subtests', async (t) => {
  await t.test('sub one', () => { qux(); });
});

for (const n of [1, 2]) {
  it(`dynamic ${n}`, () => { foo(n); });
}
"""


def _extract(tmp_path):
    files = {"src/prod.ts": _PROD, "src/test/prod.test.ts": _TEST}
    for name, body in files.items():
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)
    r = extract([tmp_path / n for n in files], root=tmp_path,
                cache_root=tmp_path / "graphify-out", parallel=False)
    by_id = {n["id"]: n for n in r["nodes"]}
    return r, by_id


def _case(r, name):
    hits = [n for n in r["nodes"] if n.get("test_name") == name]
    assert len(hits) == 1, (name, [n.get("test_name") for n in r["nodes"] if n.get("test_kind")])
    return hits[0]


def _calls_from(r, by_id, nid):
    return {by_id[e["target"]]["label"] for e in r["edges"]
            if e["source"] == nid and e["relation"] == "calls" and e["target"] in by_id}


def test_one_node_per_suite_and_case_with_kind_and_role(tmp_path):
    r, _ = _extract(tmp_path)
    tests = {n["test_name"]: n for n in r["nodes"] if n.get("test_kind")}
    assert set(tests) == {
        "Outer", "Outer > calls foo", "Outer > Inner", "Outer > Inner > async helper",
        "Outer > Inner > with options", "Outer > same title", "Outer > later",
        "node subtests", "node subtests > sub one", "dynamic ${n}",
    }
    assert tests["Outer"]["test_kind"] == "suite"
    assert tests["Outer > Inner > async helper"]["test_kind"] == "case"
    assert all(n["role"] == "test-case" for n in tests.values())
    assert tests["Outer > Inner > async helper"]["test_modifiers"] == ["skip"]
    assert tests["Outer > Inner > with options"]["test_modifiers"] == ["only"]
    assert tests["Outer > later"]["test_modifiers"] == ["todo"]
    assert tests["dynamic ${n}"].get("test_title_dynamic") is True
    assert tests["Outer > calls foo"]["qualified_name"] == "src/test/prod.test.ts::Outer > calls foo"


def test_duplicate_titles_get_distinct_deterministic_ids(tmp_path):
    r, by_id = _extract(tmp_path)
    same = [n for n in r["nodes"] if n.get("test_name") == "Outer > same title"]
    # both duplicates exist under the same name; the second id is suffixed
    assert len(same) == 2
    ids = sorted(n["id"] for n in same)
    assert len(ids) == 2 and ids[1] == ids[0] + "_2", ids
    first, second = (by_id[i] for i in ids)
    assert _calls_from(r, by_id, first["id"]) == {"foo()"}
    assert _calls_from(r, by_id, second["id"]) == {"qux()"}
    r2, _ = _extract(tmp_path)
    assert ids == sorted(n["id"] for n in r2["nodes"] if (n.get("test_name") or "").endswith("same title"))


def test_calls_are_attributed_to_the_case(tmp_path):
    r, by_id = _extract(tmp_path)
    assert _calls_from(r, by_id, _case(r, "Outer > calls foo")["id"]) == {"foo()"}
    assert _calls_from(r, by_id, _case(r, "Outer > Inner > with options")["id"]) == {"baz()"}
    assert _calls_from(r, by_id, _case(r, "node subtests > sub one")["id"]) == {"qux()"}
    assert "foo()" in _calls_from(r, by_id, _case(r, "dynamic ${n}")["id"])
    # helper called from a case: case -> helper -> product
    helper_case = _case(r, "Outer > Inner > async helper")["id"]
    assert "localHelper()" in _calls_from(r, by_id, helper_case)
    helper = next(n for n in r["nodes"] if n["label"] == "localHelper()")
    assert helper["role"] == "test-helper"
    assert _calls_from(r, by_id, helper["id"]) == {"bar()"}
    # the file node no longer owns the cases' calls
    file_nid = next(n["id"] for n in r["nodes"] if n["label"] == "prod.test.ts")
    assert not ({"foo()", "baz()", "qux()"} & _calls_from(r, by_id, file_nid))


def test_contains_edges_follow_nesting(tmp_path):
    r, _ = _extract(tmp_path)
    contains = {(e["source"], e["target"]) for e in r["edges"] if e["relation"] == "contains"}
    file_nid = next(n["id"] for n in r["nodes"] if n["label"] == "prod.test.ts")
    outer, inner = _case(r, "Outer")["id"], _case(r, "Outer > Inner")["id"]
    assert (file_nid, outer) in contains
    assert (outer, inner) in contains
    assert (inner, _case(r, "Outer > Inner > with options")["id"]) in contains
    assert (_case(r, "node subtests")["id"], _case(r, "node subtests > sub one")["id"]) in contains


def test_case_range_is_the_callback(tmp_path):
    r, _ = _extract(tmp_path)
    assert _case(r, "Outer > calls foo")["source_range"] == [10, 12]
    assert _case(r, "Outer")["source_range"] == [8, 27]
    assert _case(r, "Outer > calls foo")["source_location"] == "L10"


def test_test_case_labels_never_resolve_as_call_targets(tmp_path):
    files = {
        "src/a.test.ts": "import { it } from 'node:test';\nit('helper', () => {});\n",
        "src/b.ts": "export function run() { return helper(); }\n",
    }
    for name, body in files.items():
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)
    r = extract([tmp_path / n for n in files], root=tmp_path,
                cache_root=tmp_path / "graphify-out", parallel=False)
    case = next(n for n in r["nodes"] if n.get("test_kind"))
    assert case["label"] == "it: helper"
    assert not [e for e in r["edges"] if e["target"] == case["id"] and e["relation"] != "contains"]


def test_non_test_file_calls_named_describe_are_left_alone(tmp_path):
    p = tmp_path / "lib.ts"
    p.write_text("function describe(x: string, f: () => void) { f(); }\n"
                 "export function main() { describe('x', () => {}); }\n")
    r = extract([p], root=tmp_path, cache_root=tmp_path / "graphify-out", parallel=False)
    assert not [n for n in r["nodes"] if n.get("test_kind")]
