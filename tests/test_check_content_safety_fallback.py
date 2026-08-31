"""check_content_safety: NVIDIA's dedicated safety model first, then a
best-effort classification prompt via the same cross-provider chat chain
translate_text/ask_llm already use, if the safety model fails."""
from unittest.mock import AsyncMock

import pytest

import nvidia_image
from conftest import FakeResponse


def _clear_all_provider_keys(monkeypatch):
    for provider in nvidia_image.EXTRA_PROVIDERS:
        monkeypatch.delenv(provider["env"], raising=False)


@pytest.mark.asyncio
async def test_succeeds_on_nvidia_safety_model_without_trying_fallback(nvidia_key, fake_async_client, monkeypatch):
    _clear_all_provider_keys(monkeypatch)
    fake_async_client(
        post_side_effect=lambda *a, **kw: FakeResponse(200, {"choices": [{"message": {"content": "SAFE"}}]})
    )
    monkeypatch.setattr(nvidia_image.litellm, "acompletion", AsyncMock(side_effect=AssertionError("should not be called")))

    result = await nvidia_image.check_content_safety(text="hello world")

    assert result == "SAFE"


@pytest.mark.asyncio
async def test_falls_back_to_cross_provider_chain_when_safety_model_fails(nvidia_key, fake_async_client, monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "mistral-key")
    for provider in nvidia_image.EXTRA_PROVIDERS:
        if provider["env"] != "MISTRAL_API_KEY":
            monkeypatch.delenv(provider["env"], raising=False)

    fake_async_client(post_side_effect=lambda *a, **kw: FakeResponse(503, text="overloaded"))

    class FakeMessage:
        content = "SAFE - no policy violations found"

    class FakeChoice:
        message = FakeMessage()

    class FakeCompletionResponse:
        choices = [FakeChoice()]
        model = "mistral/mistral-small-latest"

    captured = {}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        return FakeCompletionResponse()

    monkeypatch.setattr(nvidia_image.litellm, "acompletion", fake_acompletion)

    result = await nvidia_image.check_content_safety(text="hello world")

    assert "SAFE - no policy violations found" in result
    assert "best-effort fallback verdict from mistral/mistral-small-latest" in result
    # The fallback must actually classify - not just echo the raw input as a chat message.
    assert captured["messages"][0]["role"] == "system"
    assert "SAFE" in captured["messages"][0]["content"] or "UNSAFE" in captured["messages"][0]["content"]


@pytest.mark.asyncio
async def test_reports_failure_when_nvidia_and_fallback_both_fail(nvidia_key, fake_async_client, monkeypatch):
    _clear_all_provider_keys(monkeypatch)
    fake_async_client(post_side_effect=lambda *a, **kw: FakeResponse(503, text="overloaded"))

    result = await nvidia_image.check_content_safety(text="hello world")

    assert result == "Content safety check failed."
