"""describe_image: NVIDIA vision models first, then a vision-capable
free-tier provider (Groq/Mistral/Gemini) if both NVIDIA models fail."""
from unittest.mock import AsyncMock

import pytest

import nvidia_image
from conftest import FakeResponse


@pytest.fixture
def sample_image(tmp_path):
    path = tmp_path / "test.jpg"
    path.write_bytes(b"\xff\xd8\xff\xe0fake-jpeg-bytes")
    return str(path)


def _clear_vision_provider_keys(monkeypatch):
    for provider in nvidia_image.VISION_PROVIDERS:
        monkeypatch.delenv(provider["env"], raising=False)


@pytest.mark.asyncio
async def test_describe_image_succeeds_on_nvidia_model_without_trying_fallback(
    nvidia_key, fake_async_client, sample_image, monkeypatch
):
    _clear_vision_provider_keys(monkeypatch)
    fake_async_client(
        post_side_effect=lambda *a, **kw: FakeResponse(200, {"choices": [{"message": {"content": "a cat"}}]})
    )
    monkeypatch.setattr(nvidia_image.litellm, "acompletion", AsyncMock(side_effect=AssertionError("should not be called")))

    result = await nvidia_image.describe_image(image_path=sample_image)

    assert "a cat" in result
    assert "model: nvidia/nemotron-nano-12b-v2-vl" in result


@pytest.mark.asyncio
async def test_describe_image_falls_back_to_vision_provider_when_nvidia_fails(
    nvidia_key, fake_async_client, sample_image, monkeypatch
):
    monkeypatch.setenv("GROQ_API_KEY", "groq-key")
    for provider in nvidia_image.VISION_PROVIDERS:
        if provider["env"] != "GROQ_API_KEY":
            monkeypatch.delenv(provider["env"], raising=False)

    fake_async_client(post_side_effect=lambda *a, **kw: FakeResponse(503, text="overloaded"))

    class FakeMessage:
        content = "a described cat, from fallback"

    class FakeChoice:
        message = FakeMessage()

    class FakeCompletionResponse:
        choices = [FakeChoice()]
        model = "groq/qwen/qwen3.6-27b"

    monkeypatch.setattr(nvidia_image.litellm, "acompletion", AsyncMock(return_value=FakeCompletionResponse()))

    result = await nvidia_image.describe_image(image_path=sample_image)

    assert "a described cat, from fallback" in result
    assert "model: groq/qwen/qwen3.6-27b" in result


@pytest.mark.asyncio
async def test_describe_image_reports_failure_when_nvidia_and_fallback_both_fail(
    nvidia_key, fake_async_client, sample_image, monkeypatch
):
    _clear_vision_provider_keys(monkeypatch)
    fake_async_client(post_side_effect=lambda *a, **kw: FakeResponse(503, text="overloaded"))

    result = await nvidia_image.describe_image(image_path=sample_image)

    assert result == "All vision models failed or timed out."
