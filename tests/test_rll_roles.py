"""Every node carries a ``role`` derived from its path and kind.

Added by the RLL project (2026-10). Without it, test code and product code
were indistinguishable in the graph, so "which product code does this test
reach" walked straight back into other tests' helpers.
"""
from __future__ import annotations

import json
from collections import deque

from graphify.extract import extract
from graphify.rll_granularity import load_role_rules, path_role

_FILES = {
    "src/prod.ts": (
        "export function foo(x: number) { return bar(x); }\n"
        "function bar(y: number) { return y; }\n"
        "export function unrelated() { return 0; }\n"
    ),
    "src/test-lib/h.ts": (
        "import { foo } from '../prod';\n"
        "export function helper() { return foo(1); }\n"
        "export function otherHelper() { return helper(); }\n"
    ),
    "src/test/prod.test.ts": (
        "import { foo } from '../prod';\n"
        "import { helper } from '../test-lib/h';\n"
        "function localHelper() { return helper(); }\n"
        "it('uses foo', () => { foo(2); localHelper(); });\n"
    ),
    "src/test/fixtures/sample.ts": "export function fixtureFn() { return 1; }\n",
    "scripts/build.ts": "import { foo } from '../src/prod';\nexport function main() { foo(3); }\n",
}


def _extract(tmp_path):
    for name, body in _FILES.items():
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)
    return extract([tmp_path / n for n in _FILES], root=tmp_path,
                   cache_root=tmp_path / "graphify-out", parallel=False)


def _by(r, label, sf_part):
    return next(n for n in r["nodes"] if n["label"] == label and sf_part in n["source_file"])


def test_every_node_has_a_role(tmp_path):
    r = _extract(tmp_path)
    assert all(n.get("role") for n in r["nodes"]), [n for n in r["nodes"] if not n.get("role")]


def test_roles_by_path_and_kind(tmp_path):
    r = _extract(tmp_path)
    assert _by(r, "foo()", "src/prod.ts")["role"] == "product"
    assert _by(r, "prod.ts", "src/prod.ts")["role"] == "product"
    assert _by(r, "helper()", "test-lib")["role"] == "test-helper"
    assert _by(r, "h.ts", "test-lib")["role"] == "test-helper"
    # a named function declared in a test file is a helper, the file itself a test entry
    assert _by(r, "localHelper()", "prod.test.ts")["role"] == "test-helper"
    assert _by(r, "prod.test.ts", "prod.test.ts")["role"] == "test-case"
    assert _by(r, "fixtureFn()", "fixtures")["role"] == "test-helper"
    assert _by(r, "main()", "scripts/")["role"] == "script"


def test_product_traversal_from_helper_never_enters_test_code(tmp_path):
    r = _extract(tmp_path)
    role = {n["id"]: n["role"] for n in r["nodes"]}
    sf = {n["id"]: n["source_file"] for n in r["nodes"]}
    out: dict[str, list[str]] = {}
    for e in r["edges"]:
        out.setdefault(e["source"], []).append(e["target"])
    start = _by(r, "helper()", "test-lib")["id"]
    seen = {start}
    q = deque([start])
    reached_product = set()
    while q:
        cur = q.popleft()
        for t in out.get(cur, []):
            # leave the start freely, then stay on product code only
            if t in seen or role.get(t) != "product":
                continue
            seen.add(t)
            reached_product.add(t)
            q.append(t)
    labels = {_by_id["label"] for _by_id in r["nodes"] if _by_id["id"] in reached_product}
    assert {"foo()", "bar()"} <= labels, labels
    assert not [t for t in reached_product if "/test" in sf[t] or "scripts/" in sf[t]], \
        [(t, sf[t]) for t in reached_product]
    # and nothing under a test directory is ever classified as product
    assert not [n for n in r["nodes"]
                if n["role"] == "product" and ("/test/" in n["source_file"] or "test-lib" in n["source_file"])]


def test_role_rules_are_configurable(tmp_path):
    cfg = tmp_path / "roles.json"
    cfg.write_text(json.dumps({"rules": [{"role": "test-helper", "globs": ["e2e/support/**"]}]}))
    rules = load_role_rules({"GRAPHIFY_ROLE_RULES": str(cfg)})
    assert path_role("e2e/support/login.ts", rules) == "test-helper"
    assert path_role("src/app.ts", rules) == "product"
    assert path_role("cli/src/test/a.test.ts", rules) == "test-file"
    assert path_role("cli/src/test-lib/index.ts", rules) == "test-helper"
    assert path_role("cli/templates/skills/x/scripts/run.py", rules) == "script"
    assert path_role(".github/workflows/README.md", rules) == "doc"
    replaced = load_role_rules({"GRAPHIFY_ROLE_RULES": str(cfg)} | {})
    assert replaced  # sanity: loading twice is stable
    cfg.write_text(json.dumps({"replace": True, "rules": [{"role": "script", "globs": ["**/*.ts"]}]}))
    only = load_role_rules({"GRAPHIFY_ROLE_RULES": str(cfg)})
    assert path_role("cli/src/test/a.test.ts", only) == "script"
