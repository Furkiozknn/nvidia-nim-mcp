"""check_provider_health: the 7th tool. Pure liveness probing - never a real
generation - concurrent, per-model failure isolation, and a readable report."""
from unittest.mock import AsyncMock

import pytest

import nvidia_image
from conftest import FakeResponse


# --- individual probe helpers ---------------------------------------------

@pytest.mark.asyncio
async def test_probe_nvidia_chat_model_ok():
    client = AsyncMock()
    client.post = AsyncMock(return_value=FakeResponse(200))

    ok, detail = await nvidia_image._probe_nvidia_chat_model(client, "nvidia/some-model")

    assert ok is True
    assert detail == "ok"


@pytest.mark.asyncio
async def test_probe_nvidia_chat_model_http_error():
    client = AsyncMock()
    client.post = AsyncMock(return_value=FakeResponse(503))

    ok, detail = await nvidia_image._probe_nvidia_chat_model(client, "nvidia/some-model")

    assert ok is False
    assert "503" in detail


@pytest.mark.asyncio
async def test_probe_nvidia_chat_model_timeout_is_caught_not_raised():
    client = AsyncMock()
    client.post = AsyncMock(side_effect=nvidia_image.httpx2.TimeoutException("slow"))

    ok, detail = await nvidia_image._probe_nvidia_chat_model(client, "nvidia/some-model")

    assert ok is False
    assert "timed out" in detail


@pytest.mark.asyncio
async def test_probe_nvidia_chat_model_unexpected_exception_is_caught_not_raised():
    client = AsyncMock()
    client.post = AsyncMock(side_effect=RuntimeError("connection reset"))

    ok, detail = await nvidia_image._probe_nvidia_chat_model(client, "nvidia/some-model")

    assert ok is False
    assert "connection reset" in detail


@pytest.mark.asyncio
async def test_probe_nvidia_image_model_ok():
    client = AsyncMock()
    client.post = AsyncMock(return_value=FakeResponse(200))

    ok, detail = await nvidia_image._probe_nvidia_image_model(client, "black-forest-labs/flux.1-dev")

    assert ok is True


@pytest.mark.asyncio
async def test_probe_nvidia_embed_model_failure():
    client = AsyncMock()
    client.post = AsyncMock(return_value=FakeResponse(401, text="unauthorized"))

    ok, detail = await nvidia_image._probe_nvidia_embed_model(client, nvidia_image.EMBED_MODEL)

    assert ok is False
    assert "401" in detail


@pytest.mark.asyncio
async def test_probe_extra_provider_skipped_when_key_missing(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    ok, detail = await nvidia_image._probe_extra_provider(nvidia_image.EXTRA_PROVIDERS[0])

    assert ok is False
    assert detail == "not configured"


@pytest.mark.asyncio
async def test_probe_extra_provider_ok_when_configured_and_reachable(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "groq-key")
    monkeypatch.setattr(nvidia_image.litellm, "acompletion", AsyncMock(return_value=object()))

    ok, detail = await nvidia_image._probe_extra_provider(nvidia_image.EXTRA_PROVIDERS[0])

    assert ok is True
    assert detail == "ok"


@pytest.mark.asyncio
async def test_probe_extra_provider_reports_failure_without_raising(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "groq-key")
    monkeypatch.setattr(
        nvidia_image.litellm, "acompletion", AsyncMock(side_effect=RuntimeError("rate limited"))
    )

    ok, detail = await nvidia_image._probe_extra_provider(nvidia_image.EXTRA_PROVIDERS[0])

    assert ok is False
    assert "rate limited" in detail


# --- the full tool ----------------------------------------------------------

@pytest.mark.asyncio
async def test_check_provider_health_report_shape(nvidia_key, fake_async_client, monkeypatch):
    # Every NVIDIA HTTP call succeeds; no extra provider keys configured.
    for provider in nvidia_image.EXTRA_PROVIDERS:
        monkeypatch.delenv(provider["env"], raising=False)
    fake_async_client(post_side_effect=lambda *a, **kw: FakeResponse(200))

    report = await nvidia_image.check_provider_health()

    assert "NVIDIA NIM provider health check:" in report
    for tool_section in (
        "generate_image:",
        "translate_text:",
        "ask_llm:",
        "describe_image:",
        "check_content_safety:",
        "create_embedding:",
    ):
        assert tool_section in report
    assert "Cross-provider fallback" in report
    # Every model probed successfully -> no FAIL markers among the NVIDIA rows.
    assert "FAIL" not in report.split("Cross-provider fallback")[0]
    # Unconfigured providers show up as not configured, not as a hard failure.
    assert "not configured" in report


@pytest.mark.asyncio
async def test_check_provider_health_isolates_per_model_failures(nvidia_key, fake_async_client, monkeypatch):
    for provider in nvidia_image.EXTRA_PROVIDERS:
        monkeypatch.delenv(provider["env"], raising=False)

    def flaky_post(url, *args, **kwargs):
        model = kwargs.get("json", {}).get("model", "")
        if "riva-translate" in model:
            return FakeResponse(503, text="model overloaded")
        return FakeResponse(200)

    fake_async_client(post_side_effect=flaky_post)

    report = await nvidia_image.check_provider_health()

    assert "FAIL nvidia/riva-translate-4b-instruct-v2 - HTTP 503" in report
    # A single failing model doesn't blank out the rest of the report.
    assert "OK  nvidia/llama-3.3-nemotron-super-49b-v1.5" in report or "OK nvidia/llama-3.3-nemotron-super-49b-v1.5" in report


@pytest.mark.asyncio
async def test_check_provider_health_does_not_raise_on_total_network_failure(
    nvidia_key, fake_async_client, monkeypatch
):
    for provider in nvidia_image.EXTRA_PROVIDERS:
        monkeypatch.delenv(provider["env"], raising=False)
    fake_async_client(post_side_effect=RuntimeError("DNS resolution failed"))

    report = await nvidia_image.check_provider_health()  # must not raise

    assert "FAIL" in report
