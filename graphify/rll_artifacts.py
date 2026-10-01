"""Agent-instruction artifacts (skills, presets, personas, system prompts).

MODIFIED BY YW1975 — this module is not part of upstream Graphify.
Copyright 2026 YW1975. Licensed under the Apache License, Version 2.0.

An agent-driven repository carries a second body of source besides its code:
the natural-language instructions that decide what the agent does. A skill
declares a capability and the command that backs it; a preset declares which
skill and which test tiers a change type requires; a persona declares which
other persona documents it loads. Those are *declarations with referents*, and
they break the same way code breaks — a skill naming a command that no longer
exists is as real a defect as a call to a deleted function. Today they reach
no extractor at all: `.json` presets are rejected as "data JSON" by
`json_config._is_config_json` (recognition is by manifest name or a
dependencies/$schema key probe, which a preset matches neither of), and `.md`
skills/personas/prompts classify as DOCUMENT, so `--code-only` drops them and
the semantic pass would need an LLM. Both routes miss them, so an
agent-instruction defect is invisible to the graph.

This module routes them to the deterministic AST path instead, mirroring how
package manifests are handled (#1377): a path predicate promotes them to
FileType.CODE so they are extracted without an API key and refreshed by
`update` like any other source file.

Confidence is the load-bearing distinction here. A field a file *declares*
(frontmatter `invokes:`, a preset's `requiredTiers`) is EXTRACTED: the
referent is stated, and reading it is parsing. A name merely *appearing in
prose* is INFERRED and nothing more, because deciding whether
"never invent a `ralph-lisa deploy` command" is an invocation or a prohibition
is a reading of intent, not a parse. Encoding that judgement as a keyword or
polarity pattern is what this module deliberately does not do: it reports the
occurrence and leaves the judgement to the agent consuming the graph.
"""
from __future__ import annotations

import json
import re
from pathlib import Path, PurePosixPath

# Imported from graphify.ids rather than extractors.base: the latter's package
# __init__ loads every language extractor, so a classification-time import
# would pull in all tree-sitter grammars.
from graphify.ids import make_id as _make_id

#: Artifact kinds, used as the node ``label`` prefix and for edge provenance.
KIND_SKILL = "skill"
KIND_PRESET = "preset"
KIND_PERSONA = "persona"
KIND_PROMPT = "system-prompt"
KIND_MANIFEST = "gate-manifest"

#: Directory segments that never hold agent-instruction artifacts even when a
#: filename matches. `docs/` is excluded by design: design documents are prose
#: about the system, not instructions the agent executes, and indexing them is
#: what buried an earlier graph of this repository under page and heading nodes.
_EXCLUDED_SEGMENTS = ("docs", "node_modules", ".git", "site", "dist", "build")

#: Stable, citer-independent `source_file` for reference stubs. A constant
#: rather than "" because an empty string reads as a missing field to any
#: downstream consumer that filters on it, while this is self-describing.
REFERENT_SOURCE = "<rll-referent>"

_PRESET_KEYS = frozenset({"requiredTiers", "optionalTiers", "perTierConfig", "changeType"})


def _posix(path: Path) -> str:
    return PurePosixPath(path.as_posix()).as_posix()


def _segments(path: Path) -> tuple[str, ...]:
    return tuple(p.lower() for p in path.parts)


def artifact_kind(path: Path) -> str | None:
    """Classify *path* as an agent-instruction artifact, or None.

    Recognition is positional (where the file sits) plus filename, never file
    content: classification runs before extraction and must stay cheap.
    """
    segs = _segments(path)
    if any(seg in _EXCLUDED_SEGMENTS for seg in segs):
        return None
    name = path.name
    if name == "SKILL.md" and "skills" in segs:
        return KIND_SKILL
    if name == "gate-manifest.json":
        return KIND_MANIFEST
    if path.suffix == ".json" and "presets" in segs:
        return KIND_PRESET
    if path.suffix == ".md" and "roles" in segs:
        return KIND_PERSONA
    if path.suffix == ".md" and name.endswith("-prompt.md"):
        return KIND_PROMPT
    return None


def is_rll_artifact_path(path: Path) -> bool:
    """True if *path* is an agent-instruction artifact (see module docstring)."""
    return artifact_kind(path) is not None


def _split_frontmatter(text: str) -> tuple[str, int]:
    """Return (frontmatter body, line offset of the document body).

    Deliberately not a YAML parse: only scalar keys are read below, and
    depending on a YAML library here would add a dependency for the one shape
    these artifacts actually use.
    """
    if not text.startswith("---"):
        return "", 1
    end = text.find("\n---", 3)
    if end == -1:
        return "", 1
    fm = text[3:end]
    return fm, text[: end + 4].count("\n") + 1


def _scalar(frontmatter: str, key: str) -> str | None:
    m = re.search(rf"^{re.escape(key)}:\s*(.+)$", frontmatter, re.M)
    if not m:
        return None
    return m.group(1).strip().strip("\"'") or None


def _list_field(frontmatter: str, key: str) -> list[str]:
    """Read a declared list, in either inline (``[a, b]``) or block (``- a``) form."""
    inline = re.search(rf"^{re.escape(key)}:\s*\[(.*?)\]\s*$", frontmatter, re.M | re.S)
    if inline:
        return [v.strip().strip("\"'") for v in inline.group(1).split(",") if v.strip()]
    block = re.search(rf"^{re.escape(key)}:\s*$((?:\n\s*-\s*.+)+)", frontmatter, re.M)
    if block:
        return [
            line.split("-", 1)[1].strip().strip("\"'")
            for line in block.group(1).splitlines()
            if line.strip().startswith("-")
        ]
    return []


class _Collector:
    def __init__(self, path: Path) -> None:
        self.src = _posix(path)
        self.nodes: list[dict] = []
        self.edges: list[dict] = []
        self._seen: set[str] = set()
        self._edge_keys: set[tuple[str, str, str]] = set()

    def node(self, nid: str, label: str, line: int = 1, kind: str = "code") -> str:
        """A definition node: it lives in this file, so it carries its source."""
        if nid not in self._seen:
            self._seen.add(nid)
            self.nodes.append({
                "id": nid, "label": label, "file_type": kind,
                "source_file": self.src, "source_location": f"L{line}",
            })
        return nid

    def referent(self, nid: str, label: str) -> str:
        """A sourceless reference stub, for an entity this file merely names.

        A command, a tier or a loaded document is one entity repo-wide: the
        point of the graph is that every skill naming `status` points at the
        SAME node, so "who invokes this" is one traversal. Definition nodes get
        a per-file id prefix, which would mint `status` once per citing file —
        observed as 101 command nodes for 38 distinct commands, and 38 tier
        nodes for 8 tiers. Omitting `source_file` marks these as the reference
        stubs upstream already recognizes (extract.py: "sourceless reference
        stubs are not definitions"), so they stay un-prefixed and collapse by
        name. The citing file is not lost — it is on the edge.
        """
        if nid not in self._seen:
            self._seen.add(nid)
            # `concept`, not `code`: a code node is a symbol defined in this
            # file and the pipeline namespaces its id by the file, which mints
            # `status` once per citing skill (observed: 101 command nodes for
            # 38 commands). A concept is the referent itself, so the id stays
            # bare and every citer converges on one node. `source_file` is a
            # required field — it records who named it first, while each edge
            # keeps its own citing file.
            self.nodes.append({
                "id": nid, "label": label, "file_type": "concept",
                # A constant, not this file: node dedup merges only nodes that are
                # exactly equal, so recording the citer here makes a referent
                # named by N files N non-equal nodes, which the collision pass
                # then namespaces per file — observed as `e2e (tier)` splitting
                # into 5 path-prefixed nodes while single-citer tiers merged
                # fine. A referent belongs to no one file; the citing file is
                # carried by each edge. The key is present because it is
                # required, and constant so every citer mints the same node.
                "source_file": REFERENT_SOURCE, "source_location": "L1",
            })
        return nid

    def edge(self, src: str, tgt: str, relation: str, line: int,
             declared: bool, context: str | None = None) -> None:
        if not src or not tgt or src == tgt:
            return
        edge = {
            "source": src, "target": tgt, "relation": relation,
            # A declared referent is parsed; a prose occurrence is not a claim
            # about intent. See the module docstring.
            "confidence": "EXTRACTED" if declared else "INFERRED",
            "source_file": self.src, "source_location": f"L{line}", "weight": 1.0,
        }
        if context:
            edge["context"] = context
        # Repeated citations of the same declared referent are one relation: a
        # persona that names the same document twice still loads it once. An
        # undeclared prose mention is NOT deduped — each occurrence carries the
        # sentence the agent must read to judge it, so they are distinct facts.
        if declared:
            key = (src, tgt, relation)
            if key in self._edge_keys:
                return
            self._edge_keys.add(key)
        self.edges.append(edge)


def _command_node(c: _Collector, name: str) -> str:
    return c.referent(_make_id("rll-command", name), f"{name} (command)")


def _skill_node(c: _Collector, name: str) -> str:
    """A skill named by someone else (e.g. a preset) is a referent."""
    return c.referent(_make_id("rll-skill", name), f"{name} ({KIND_SKILL})")


def _extract_skill(c: _Collector, path: Path, text: str) -> None:
    fm, body_line = _split_frontmatter(text)
    name = _scalar(fm, "name") or path.parent.name
    self_id = _skill_node(c, name)

    for cmd in _list_field(fm, "invokes"):
        c.edge(self_id, _command_node(c, cmd), "invokes", 1, declared=True)
    for target in _list_field(fm, "reads"):
        c.edge(self_id, c.referent(_make_id("rll-asset", target), target), "reads", 1, declared=True)
    for target in _list_field(fm, "writes"):
        c.edge(self_id, c.referent(_make_id("rll-asset", target), target), "writes", 1, declared=True)

    # Prose occurrences of a command name. Reported, not interpreted: the same
    # surface carries invocations and prohibitions, and only the agent reading
    # the surrounding sentence can tell which this is.
    declared = {cmd for cmd in _list_field(fm, "invokes")}
    for lineno, line in enumerate(text.splitlines()[body_line - 1:], start=body_line):
        for m in re.finditer(r"ralph-lisa\s+([a-z][a-z0-9-]*)", line):
            cmd = m.group(1)
            if cmd in declared:
                continue
            c.edge(self_id, _command_node(c, cmd), "mentions_command_undeclared",
                   lineno, declared=False, context=line.strip()[:160])


def _extract_preset(c: _Collector, path: Path, text: str) -> None:
    try:
        obj = json.loads(text)
    except ValueError:
        return
    if not isinstance(obj, dict) or not (_PRESET_KEYS & obj.keys()):
        return
    name = path.stem
    self_id = c.node(_make_id("rll-preset", name), f"{name} ({KIND_PRESET})")

    for key in ("requiredTiers", "optionalTiers"):
        for tier in obj.get(key) or []:
            if isinstance(tier, str):
                c.edge(self_id, c.referent(_make_id("rll-tier", tier), f"{tier} (tier)"),
                       "requires" if key == "requiredTiers" else "optionally_requires",
                       1, declared=True)

    # A preset's `cmd` is a declared command line, so the skill it drives is
    # read from the declaration rather than guessed.
    per_tier = obj.get("perTierConfig")
    if isinstance(per_tier, dict):
        for tier, cfg in per_tier.items():
            if not isinstance(cfg, dict):
                continue
            cmd = cfg.get("cmd")
            if not isinstance(cmd, str):
                continue
            for m in re.finditer(r"ralph-lisa\s+skill\s+([\w-]+)", cmd):
                c.edge(self_id, _skill_node(c, m.group(1)), "invokes", 1,
                       declared=True, context=f"perTierConfig.{tier}.cmd")
            for m in re.finditer(r"ralph-lisa\s+([a-z][a-z0-9-]*)", cmd):
                if m.group(1) != "skill":
                    c.edge(self_id, _command_node(c, m.group(1)), "invokes", 1,
                           declared=True, context=f"perTierConfig.{tier}.cmd")


def _extract_manifest(c: _Collector, path: Path, text: str) -> None:
    try:
        obj = json.loads(text)
    except ValueError:
        return
    if not isinstance(obj, dict):
        return
    self_id = c.node(_make_id("rll-manifest", _posix(path)), f"{path.name} ({KIND_MANIFEST})")
    for tier in obj.get("canonical_tier_ids") or []:
        if isinstance(tier, str):
            c.edge(self_id, c.referent(_make_id("rll-tier", tier), f"{tier} (tier)"),
                   "whitelists", 1, declared=True)
    for tier in obj.get("default_baseline") or []:
        if isinstance(tier, str):
            c.edge(self_id, c.referent(_make_id("rll-tier", tier), f"{tier} (tier)"),
                   "baseline_includes", 1, declared=True)


def _extract_persona_or_prompt(c: _Collector, path: Path, text: str, kind: str) -> None:
    fm, _ = _split_frontmatter(text)
    name = _scalar(fm, "name") or path.stem
    prefix = "rll-persona" if kind == KIND_PERSONA else "rll-prompt"
    self_id = c.node(_make_id(prefix, _posix(path)), f"{name} ({kind})")

    # A persona document composes other documents; that link is the load order
    # an agent actually follows, so it is a declared edge.
    for m in re.finditer(r"`([\w./-]+\.md)`|\]\(([\w./-]+\.md)\)", text):
        target = m.group(1) or m.group(2)
        if not target or target.lower().startswith("docs/"):
            continue
        line = text[: m.start()].count("\n") + 1
        c.edge(self_id, c.referent(_make_id("rll-doc", target), target), "loads",
               line, declared=True)


def extract_rll_artifact(path: Path) -> dict:
    """Extract declarations and referents from an agent-instruction artifact."""
    kind = artifact_kind(path)
    if kind is None:
        return {"nodes": [], "edges": []}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return {"nodes": [], "edges": [], "error": f"unreadable: {exc}"}

    c = _Collector(path)
    if kind == KIND_SKILL:
        _extract_skill(c, path, text)
    elif kind == KIND_PRESET:
        _extract_preset(c, path, text)
    elif kind == KIND_MANIFEST:
        _extract_manifest(c, path, text)
    else:
        _extract_persona_or_prompt(c, path, text, kind)
    return {"nodes": c.nodes, "edges": c.edges}
