"""Every tool must refuse to call the API and return the documented message
when NVIDIA_API_KEY isn't set - no network call should even be attempted."""
import pytest

import nvidia_image

GUARD_MESSAGE = "NVIDIA_API_KEY not set in .env - can't call the API."

TOOL_CALLS = {
    "generate_image": lambda: nvidia_image.generate_image(prompt="a cat"),
    "translate_text": lambda: nvidia_image.translate_text(text="hello", target_language="Turkish"),
    "ask_llm": lambda: nvidia_image.ask_llm(question="what is 2+2?"),
    "describe_image": lambda: nvidia_image.describe_image(image_path="/nonexistent/file.jpg"),
    "check_content_safety": lambda: nvidia_image.check_content_safety(text="hello"),
    "create_embedding": lambda: nvidia_image.create_embedding(text="hello"),
    "check_provider_health": lambda: nvidia_image.check_provider_health(),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", sorted(TOOL_CALLS))
async def test_tool_returns_guard_message_without_key(tool_name, no_nvidia_key):
    result = await TOOL_CALLS[tool_name]()
    assert result == GUARD_MESSAGE


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", sorted(TOOL_CALLS))
async def test_tool_returns_guard_message_with_empty_string_key(tool_name, monkeypatch):
    # "" is falsy too - an emptied-out .env value should be treated the same
    # as an unset one.
    monkeypatch.setattr(nvidia_image, "API_KEY", "")
    result = await TOOL_CALLS[tool_name]()
    assert result == GUARD_MESSAGE


def test_all_six_original_tools_plus_health_check_are_covered():
    assert set(TOOL_CALLS) == {
        "generate_image",
        "translate_text",
        "ask_llm",
        "describe_image",
        "check_content_safety",
        "create_embedding",
        "check_provider_health",
    }
