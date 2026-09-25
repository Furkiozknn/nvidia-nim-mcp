"""The README, server.json and the code describe the same server.

Model IDs change often on NVIDIA's free tier; when one is swapped in the
code, the README table and the health check must not keep advertising the
old one. server.json is checked against the MCP Registry's own limits so a
tagged release is not the first place a bad value shows up."""
import json
import re
import tomllib
from pathlib import Path

import pytest

import nvidia_image

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")


def _all_code_models() -> set[str]:
    return (
        {m["slug"] for m in nvidia_image.IMAGE_MODELS}
        | set(nvidia_image.TRANSLATE_MODELS)
        | set(nvidia_image.LLM_MODELS)
        | set(nvidia_image.VISION_MODELS)
        | {nvidia_image.SAFETY_MODEL, nvidia_image.EMBED_MODEL}
        | {p["model"] for p in nvidia_image.EXTRA_PROVIDERS}
        | {p["model"] for p in nvidia_image.VISION_PROVIDERS}
    )


@pytest.mark.parametrize("model", sorted(_all_code_models()))
def test_every_model_in_the_code_is_named_in_the_readme(model):
    # The README tools table uses the short name (after the vendor prefix)
    # for NVIDIA models and the full litellm id for the fallback providers.
    short = model.split("/", 1)[1] if model.count("/") == 1 else model
    assert f"`{short}`" in README or f"`{model}`" in README, f"{model} missing from README"


@pytest.mark.asyncio
async def test_health_check_lists_every_model_the_tools_use(no_nvidia_key, fake_async_client, monkeypatch):
    for p in (*nvidia_image.EXTRA_PROVIDERS, *nvidia_image.VISION_PROVIDERS):
        monkeypatch.delenv(p["env"], raising=False)
    fake_async_client()

    report = await nvidia_image.check_provider_health()

    for model in _all_code_models():
        assert model in report, f"{model} missing from check_provider_health"


def test_readme_names_all_seven_tools():
    for name in (
        "generate_image",
        "translate_text",
        "ask_llm",
        "describe_image",
        "check_content_safety",
        "create_embedding",
        "check_provider_health",
    ):
        assert f"`{name}`" in README


def test_server_json_fits_the_registry_schema_and_matches_pyproject():
    server = json.loads((ROOT / "server.json").read_text(encoding="utf-8"))
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]

    # https://static.modelcontextprotocol.io/schemas/2025-12-11/server.schema.json
    assert 1 <= len(server["description"]) <= 100
    assert re.fullmatch(r"io\.github\.Furkiozknn/[a-zA-Z0-9._-]+", server["name"])
    assert server["version"] == project["version"]
    (package,) = server["packages"]
    assert package["identifier"] == project["name"]
    assert package["version"] == project["version"]
    assert f"mcp-name: {server['name']}" in README


def test_server_reports_the_package_version():
    # serverInfo.version comes from here; it was empty before 0.1.0 shipped.
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert nvidia_image.mcp.version == pyproject["project"]["version"]
