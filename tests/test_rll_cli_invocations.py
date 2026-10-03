"""Tests that launch the CLI as a subprocess get an edge to the command handler.

Added by the RLL project (2026-10). `spawnSync(process.execPath, [CLI,
'whose-turn'])` — directly, or through helpers that forward a parameter into
the argv — never produced a call edge, so the test looked unrelated to the
handler it exercises. Each scenario below checks the resolved `invokes_cli`
edge, or the `uncertain` edge (with its reason) when the subcommand is not
statically known.
"""
from __future__ import annotations

from graphify.extract import extract
from graphify.rll_granularity import SENTINEL_ID

_COMMANDS = """\
export function cmdWhoseTurn() { return 1; }
export function cmdStatus(a: string[]) { return a; }
export function cmdWecomFeedback(a: string[]) { return a; }
"""

_CLI = """\
import { cmdWhoseTurn, cmdStatus, cmdWecomFeedback } from './commands';
const [cmd, ...rest] = process.argv.slice(2);
switch (cmd) {
  case "whose-turn":
  case "check-turn":
    cmdWhoseTurn();
    break;
  case "status":
    cmdStatus(rest);
    break;
  case "wecom-feedback":
    cmdWecomFeedback(rest);
    break;
  default:
    console.log("help");
}
"""

_RUNNER = """\
import * as path from 'node:path';
import { execFileSync } from 'node:child_process';
const CLI_PATH = path.join(__dirname, '..', 'cli.js');
export function runCli(args: string[]) {
  return execFileSync('node', [CLI_PATH, ...args], { encoding: 'utf8' });
}
"""

_WRAP = """\
import { runCli } from './runner';
export function runSub(sub: string) {
  return runCli([sub, '--json']);
}
"""

_TEST = """\
import { describe, it } from 'node:test';
import * as path from 'node:path';
import { spawnSync, fork, execSync } from 'node:child_process';
import { runSub } from '../test-lib/wrap';

const CLI = path.join(__dirname, '..', 'cli.js');
const DIST_CLI = path.resolve(__dirname, '../../dist/cli.js');

function run(dir: string, args: string[]) {
  return spawnSync(process.execPath, [CLI, ...args], { cwd: dir });
}

function runOne(sub: string) {
  return spawnSync(process.execPath, [DIST_CLI, sub]);
}

function neverCalled(sub: string) {
  return spawnSync(process.execPath, [CLI, sub]);
}

declare function computeIt(): string;

describe('cli', () => {
  it('direct literal', () => {
    spawnSync(process.execPath, [CLI, 'status']);
  });
  it('one-level helper with spread', () => {
    run('/tmp', ['whose-turn']);
  });
  it('fallthrough alias', () => {
    run('/tmp', ['check-turn', '--json']);
  });
  it('two-level helper chain across files', () => {
    runSub('wecom-feedback');
  });
  it('positional param', () => {
    runOne('status');
  });
  it('fork and shell string', () => {
    fork(CLI, ['status']);
    execSync(`node ${CLI} whose-turn --json`);
  });
  it('computed', () => {
    const sub = computeIt();
    run('/tmp', [sub]);
  });
  it('loop', () => {
    for (const c of ['status', 'whose-turn']) {
      run('/tmp', [c]);
    }
  });
  it('map callback', () => {
    ['status'].map((c) => run('/tmp', [c]));
  });
  it('template', () => {
    const x = 'us';
    run('/tmp', [`stat${x}`]);
  });
  it('unknown subcommand', () => {
    run('/tmp', ['no-such-command']);
  });
  it('script path', () => {
    spawnSync('bash', [path.join(__dirname, 'fixtures', 'x.sh')]);
    spawnSync(process.execPath, [path.join(__dirname, '..', 'scripts', 'gen.mjs')]);
  });
  it('not the cli', () => {
    spawnSync('git', ['status']);
  });
  it('local runner closure', () => {
    const runLocal = (args: string[]) => spawnSync(process.execPath, [CLI, ...args]);
    runLocal(['status']);
  });
  it('local closure never called', () => {
    const idle = (args: string[]) => spawnSync(process.execPath, [CLI, ...args]);
    return idle;
  });
});

function fixture() {
  const go = (sub: string) => runSub(sub);
  go('whose-turn');
  return {};
}
"""


def _extract(tmp_path):
    files = {
        "src/commands.ts": _COMMANDS,
        "src/cli.ts": _CLI,
        "src/test-lib/runner.ts": _RUNNER,
        "src/test-lib/wrap.ts": _WRAP,
        "src/test/cli.test.ts": _TEST,
    }
    for name, body in files.items():
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)
    r = extract([tmp_path / n for n in files], root=tmp_path,
                cache_root=tmp_path / "graphify-out", parallel=False)
    by_id = {n["id"]: n for n in r["nodes"]}
    inv = [e for e in r["edges"] if e["relation"] == "invokes_cli"]
    return r, by_id, inv


def _from(by_id, inv, label):
    return [e for e in inv if by_id.get(e["source"], {}).get("label") == label]


def _targets(by_id, edges):
    return {by_id[e["target"]]["label"] for e in edges if not e.get("uncertain")}


def test_direct_spawn_literal(tmp_path):
    _, by_id, inv = _extract(tmp_path)
    edges = _from(by_id, inv, "it: direct literal")
    assert _targets(by_id, edges) == {"cmdStatus()"}
    assert edges[0]["via"] == "direct" and edges[0]["subcommand"] == "status"
    assert edges[0]["cli_entry"] == "src/cli.ts"


def test_one_level_helper_with_spread(tmp_path):
    _, by_id, inv = _extract(tmp_path)
    edges = _from(by_id, inv, "it: one-level helper with spread")
    assert _targets(by_id, edges) == {"cmdWhoseTurn()"}
    e = edges[0]
    assert e["via"] == "helper"
    assert [by_id[x]["label"] for x in e["helper_chain"]] == ["run()"]
    # the alias case label falls through to the same handler
    assert _targets(by_id, _from(by_id, inv, "it: fallthrough alias")) == {"cmdWhoseTurn()"}


def test_two_level_helper_chain_across_files(tmp_path):
    _, by_id, inv = _extract(tmp_path)
    edges = _from(by_id, inv, "it: two-level helper chain across files")
    assert _targets(by_id, edges) == {"cmdWecomFeedback()"}
    assert [by_id[x]["label"] for x in edges[0]["helper_chain"]] == ["runSub()", "runCli()"]
    # nothing is emitted from the helpers themselves: their argv is parameter-bound
    assert not [e for e in inv if by_id[e["source"]]["label"] in ("runSub()", "runCli()")]


def test_positional_param_and_dist_path(tmp_path):
    _, by_id, inv = _extract(tmp_path)
    assert _targets(by_id, _from(by_id, inv, "it: positional param")) == {"cmdStatus()"}


def test_fork_and_shell_string(tmp_path):
    _, by_id, inv = _extract(tmp_path)
    assert _targets(by_id, _from(by_id, inv, "it: fork and shell string")) == {
        "cmdStatus()", "cmdWhoseTurn()"}


def _uncertain(by_id, inv, label):
    return {(e["reason"], by_id[e["target"]]["label"]) for e in _from(by_id, inv, label)
            if e.get("uncertain")}


def test_non_literal_subcommands_are_uncertain_not_dropped(tmp_path):
    _, by_id, inv = _extract(tmp_path)
    disp = "<module scope>"  # the dispatcher: cli.ts module scope, which holds the switch
    assert _uncertain(by_id, inv, "it: computed") == {("computed", disp)}
    assert _uncertain(by_id, inv, "it: loop") == {("loop-variable", disp)}
    assert _uncertain(by_id, inv, "it: map callback") == {("callback-param", disp)}
    assert _uncertain(by_id, inv, "it: template") == {("template-literal", disp)}
    assert _uncertain(by_id, inv, "it: unknown subcommand") == {("unknown-subcommand", disp)}
    for label in ("it: computed", "it: loop", "it: map callback", "it: template"):
        assert not _targets(by_id, _from(by_id, inv, label)), label
        assert all(e["confidence"] == "AMBIGUOUS" for e in _from(by_id, inv, label))


def test_script_path_spawn_is_uncertain_to_the_sentinel(tmp_path):
    r, by_id, inv = _extract(tmp_path)
    assert _uncertain(by_id, inv, "it: script path") == {
        ("script-path", "<unresolved process invocation>")}
    sentinel = by_id[SENTINEL_ID]
    assert sentinel["role"] == "sentinel"
    assert not _from(by_id, inv, "it: not the cli")


def test_unbound_parameter_is_reported(tmp_path):
    _, by_id, inv = _extract(tmp_path)
    assert _uncertain(by_id, inv, "neverCalled()") == {
        ("unbound-parameter", "<unresolved process invocation>")}


def test_facts_do_not_leak_into_output(tmp_path):
    r, _, _ = _extract(tmp_path)
    assert not [n for n in r["nodes"] if "_rll_cli" in n]
    # what an incremental rebuild needs is kept: helper summaries + the dispatch table
    assert [n for n in r["nodes"] if n.get("_rll_cli_summary") and n["label"] == "runCli()"]
    assert [n for n in r["nodes"] if n.get("_rll_dispatch")]


def test_cli_path_pattern_is_configurable(tmp_path, monkeypatch):
    monkeypatch.setenv("GRAPHIFY_CLI_PATH_PATTERN", r"(?:^|/)main\.js$")
    _, by_id, inv = _extract(tmp_path)
    # cli.js no longer counts as the CLI, so the literal spawn is a script path
    assert _uncertain(by_id, inv, "it: direct literal") == {
        ("script-path", "<unresolved process invocation>")}


def test_local_runner_closure_is_bound_at_the_owners_call_site(tmp_path):
    _, by_id, inv = _extract(tmp_path)
    edges = _from(by_id, inv, "it: local runner closure")
    assert _targets(by_id, edges) == {"cmdStatus()"}
    assert _uncertain(by_id, inv, "it: local closure never called") == {
        ("callback-param", "<module scope>")}
    # a local closure forwarding to a named helper is bound too
    assert _targets(by_id, _from(by_id, inv, "fixture()")) == {"cmdWhoseTurn()"}
