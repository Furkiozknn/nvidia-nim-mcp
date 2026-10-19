"""server.json is what `mcp-publisher publish` sends to the MCP Registry.

The registry schema (2025-12-11, ServerDetail) caps `description` at 100
characters; the old 157-character text would have been rejected at the
first tagged release. These checks pin the schema limits that matter and
keep server.json in step with pyproject.toml, offline.
"""
import json
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SERVER = json.loads((ROOT / "server.json").read_text(encoding="utf-8"))
PROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]


def test_description_fits_the_registry_limit():
    assert 1 <= len(SERVER["description"]) <= 100, len(SERVER["description"])


def test_name_is_a_github_namespace_the_readme_claims():
    assert re.fullmatch(r"io\.github\.Furkiozknn/[a-zA-Z0-9._-]+", SERVER["name"])
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert f"mcp-name: {SERVER['name']}" in readme


def test_versions_and_package_match_pyproject():
    assert SERVER["version"] == PROJECT["version"]
    (package,) = SERVER["packages"]
    assert package["identifier"] == PROJECT["name"]
    assert package["version"] == PROJECT["version"]
