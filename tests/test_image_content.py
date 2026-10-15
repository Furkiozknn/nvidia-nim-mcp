"""generate_image as an MCP client sees it: the unchanged sentence, the image
itself as image content, a resource_link, and structuredContent."""
import base64
from pathlib import Path

import pytest

import nvidia_image
from conftest import FakeResponse

JPEG = b"\xff\xd8\xff" + b"fake-jpeg-bytes"
PNG = b"\x89PNG\r\n\x1a\n" + b"fake-png"


@pytest.mark.asyncio
async def test_generate_image_publishes_an_output_schema():
    tools = {t.name: t for t in await nvidia_image.mcp.list_tools()}
    schema = tools["generate_image"].output_schema
    assert set(schema["properties"]) >= {"path", "uri", "model", "tier", "note", "format", "bytes", "inline"}
    assert "Tries multiple models in order" in tools["generate_image"].description


@pytest.mark.asyncio
async def test_nvidia_image_comes_back_inline_with_link_and_structure(nvidia_key, fake_async_client, tmp_path, monkeypatch):
    monkeypatch.setattr(nvidia_image, "OUTPUT_DIR", tmp_path)
    fake_async_client(post_side_effect=lambda *a, **kw: FakeResponse(
        200, json_data={"artifacts": [{"base64": base64.b64encode(JPEG).decode()}]}))

    result = await nvidia_image.mcp.call_tool("generate_image", {"prompt": "a cat"})

    assert not result.is_error
    text, image, link = result.content
    assert text.text.startswith("Image saved to ") and "black-forest-labs/flux.1-dev" in text.text
    assert image.type == "image" and image.mime_type == "image/jpeg"
    assert base64.b64decode(image.data) == JPEG
    data = result.structured_content
    assert data["tier"] == "nvidia" and data["model"] == "black-forest-labs/flux.1-dev"
    assert data["inline"] is True and data["format"] == "jpeg" and data["bytes"] == len(JPEG)
    assert data["note"] is None
    assert Path(data["path"]).read_bytes() == JPEG
    assert link.type == "resource_link" and link.uri == data["uri"] and data["uri"].startswith("file://")


@pytest.mark.asyncio
async def test_keyless_pollinations_tier_is_reported_with_its_reason(no_nvidia_key, fake_async_client, tmp_path, monkeypatch):
    monkeypatch.setattr(nvidia_image, "OUTPUT_DIR", tmp_path)
    fake_async_client(get_side_effect=lambda *a, **kw: FakeResponse(200, content=PNG,
                                                                     headers={"content-type": "image/png"}))

    result = await nvidia_image.mcp.call_tool("generate_image", {"prompt": "a cat"})

    data = result.structured_content
    assert data["tier"] == "pollinations" and data["model"] == "pollinations"
    assert data["note"] == "keyless tier, NVIDIA_API_KEY not set"
    assert data["format"] == "png", "the format comes from the bytes, not the .jpg file name"
    assert result.content[1].mime_type == "image/png"


@pytest.mark.asyncio
async def test_large_images_are_linked_but_not_inlined(nvidia_key, fake_async_client, tmp_path, monkeypatch):
    monkeypatch.setattr(nvidia_image, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(nvidia_image, "INLINE_IMAGE_MAX_BYTES", 10)
    fake_async_client(post_side_effect=lambda *a, **kw: FakeResponse(
        200, json_data={"artifacts": [{"base64": base64.b64encode(JPEG).decode()}]}))

    result = await nvidia_image.mcp.call_tool("generate_image", {"prompt": "a cat"})

    assert [c.type for c in result.content] == ["text", "resource_link"]
    assert result.structured_content["inline"] is False


@pytest.mark.asyncio
async def test_bytes_that_are_not_an_image_are_never_inlined(nvidia_key, fake_async_client, tmp_path, monkeypatch):
    monkeypatch.setattr(nvidia_image, "OUTPUT_DIR", tmp_path)
    fake_async_client(post_side_effect=lambda *a, **kw: FakeResponse(
        200, json_data={"artifacts": [{"base64": base64.b64encode(b"<html>oops</html>").decode()}]}))

    result = await nvidia_image.mcp.call_tool("generate_image", {"prompt": "a cat"})

    assert result.structured_content["format"] is None and result.structured_content["inline"] is False
    assert [c.type for c in result.content] == ["text", "resource_link"]


@pytest.mark.asyncio
async def test_total_failure_is_an_error_result(nvidia_key, fake_async_client, tmp_path, monkeypatch):
    monkeypatch.setattr(nvidia_image, "OUTPUT_DIR", tmp_path)
    fake_async_client(post_side_effect=lambda *a, **kw: FakeResponse(503, text="overloaded"),
                      get_side_effect=RuntimeError("pollinations down"))

    result = await nvidia_image.mcp.call_tool("generate_image", {"prompt": "a cat"})

    assert result.is_error
    assert result.structured_content is None
    assert result.content[0].text.startswith("All image models failed")


@pytest.mark.parametrize("sentence,model,why", [
    ("Image saved to /o/flux.1-dev_1.jpg (model: black-forest-labs/flux.1-dev)", "black-forest-labs/flux.1-dev", None),
    ("Image saved to C:\\out dir\\p_1.jpg (model: pollinations, fallback after NVIDIA models failed)",
     "pollinations", "fallback after NVIDIA models failed"),
])
def test_the_sentence_format_is_pinned(sentence, model, why):
    m = nvidia_image._SAVED_IMAGE.match(sentence)
    assert m and m["model"] == model and m["why"] == why
