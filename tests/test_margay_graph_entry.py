"""The fork ships under its own name, margay-graph, beside upstream's `graphify`."""
import tomllib
from pathlib import Path


def test_margay_graph_names_the_same_entry_point_as_graphify():
    scripts = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())["project"]["scripts"]
    assert scripts["margay-graph"] == scripts["graphify"] == "graphify.__main__:main"
