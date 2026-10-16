"""What an MCP client sees: titles, behaviour hints, and check_provider_health's
structured report. Every HTTP and litellm call is mocked."""
from unittest.mock import AsyncMock

import pytest

import nvidia_image
from conftest import FakeResponse

TOOLS = {
    "generate_image", "translate_text", "ask_llm", "describe_image",
    "check_content_safety", "create_embedding", "check_provider_health",
}


async def _tools():
    return {t.name: t for t in await nvidia_image.mcp.list_tools()}


@pytest.mark.asyncio
async def test_the_seven_tools_keep_their_names():
    assert set(await _tools()) == TOOLS


@pytest.mark.asyncio
@pytest.mark.parametrize("name,read_only,destructive", [
    ("generate_image", False, False),
    ("create_embedding", False, False),
    ("translate_text", True, None),
    ("ask_llm", True, None),
    ("describe_image", True, None),
    ("check_content_safety", True, None),
    ("check_provider_health", True, None),
])
async def test_annotations(name, read_only, destructive):
    tool = (await _tools())[name]
    a = tool.annotations
    assert a.read_only_hint is read_only
    assert a.destructive_hint is destructive
    assert a.open_world_hint is True, "every tool sends its input to a third-party provider"
    assert a.idempotent_hint is False
    assert tool.title


@pytest.mark.asyncio
async def test_health_description_is_the_original_docstring():
    assert "single-token liveness probe" in (await _tools())["check_provider_health"].description


@pytest.mark.asyncio
async def test_health_publishes_an_output_schema():
    schema = (await _tools())["check_provider_health"].output_schema
    assert set(schema["properties"]) == {"ok", "failed", "probes"}


def _no_extra_keys(monkeypatch):
    for provider in (*nvidia_image.EXTRA_PROVIDERS, *nvidia_image.VISION_PROVIDERS):
        monkeypatch.delenv(provider["env"], raising=False)


@pytest.mark.asyncio
async def test_health_structured_rows_match_the_text(nvidia_key, fake_async_client, monkeypatch):
    _no_extra_keys(monkeypatch)

    def flaky_post(url, *args, **kwargs):
        if "riva-translate" in kwargs.get("json", {}).get("model", ""):
            return FakeResponse(503, text="model overloaded")
        return FakeResponse(200)

    fake_async_client(post_side_effect=flaky_post, get_side_effect=lambda *a, **kw: FakeResponse(200))

    result = await nvidia_image.mcp.call_tool("check_provider_health", {})

    text = result.content[0].text
    assert text == await nvidia_image.check_provider_health(), "text view unchanged"
    data = result.structured_content
    probes = data["probes"]
    assert data["ok"] + data["failed"] == len(probes)
    assert data["ok"] == sum(p["ok"] for p in probes)
    for p in probes:
        env = f" ({p['env']})" if p["tier"] == "free-tier fallback" else ""
        assert f"{'OK ' if p['ok'] else 'FAIL'} {p['model']}{env} - {p['detail']}" in text
    riva = [p for p in probes if "riva-translate" in p["model"]]
    assert riva and all(not p["ok"] and "503" in p["detail"] for p in riva)
    assert {p["tier"] for p in probes} == {"nvidia", "keyless", "free-tier fallback"}
    poll = [p for p in probes if p["model"] == "pollinations (fallback)"]
    assert poll == [{"group": "generate_image", "model": "pollinations (fallback)", "ok": True,
                     "detail": "ok", "tier": "keyless", "env": None}]


@pytest.mark.asyncio
async def test_health_groups_cover_every_tool(nvidia_key, fake_async_client, monkeypatch):
    _no_extra_keys(monkeypatch)
    fake_async_client(post_side_effect=lambda *a, **kw: FakeResponse(200))
    result = await nvidia_image.mcp.call_tool("check_provider_health", {})
    groups = {p["group"] for p in result.structured_content["probes"]}
    assert TOOLS - {"check_provider_health"} <= groups
    unconfigured = [p for p in result.structured_content["probes"] if p["tier"] == "free-tier fallback"]
    assert unconfigured and all(p["detail"] == "not configured" for p in unconfigured)


@pytest.mark.asyncio
async def test_health_without_any_key_still_reports_structured_rows(no_nvidia_key, fake_async_client, monkeypatch):
    _no_extra_keys(monkeypatch)
    fake_async_client(get_side_effect=lambda *a, **kw: FakeResponse(200))
    result = await nvidia_image.mcp.call_tool("check_provider_health", {})
    nvidia_rows = [p for p in result.structured_content["probes"] if p["tier"] == "nvidia"]
    assert nvidia_rows and all(not p["ok"] and "not configured" in p["detail"] for p in nvidia_rows)


@pytest.mark.asyncio
async def test_a_configured_fallback_provider_is_marked_ok(nvidia_key, fake_async_client, monkeypatch):
    _no_extra_keys(monkeypatch)
    monkeypatch.setenv("GROQ_API_KEY", "groq-key")
    fake_async_client(post_side_effect=lambda *a, **kw: FakeResponse(200),
                      get_side_effect=lambda *a, **kw: FakeResponse(200))
    monkeypatch.setattr(nvidia_image.litellm, "acompletion", AsyncMock(return_value=object()))
    result = await nvidia_image.mcp.call_tool("check_provider_health", {})
    groq = [p for p in result.structured_content["probes"] if p["env"] == "GROQ_API_KEY"]
    assert groq and all(p["ok"] for p in groq)
