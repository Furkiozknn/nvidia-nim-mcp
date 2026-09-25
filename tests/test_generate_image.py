"""generate_image: NVIDIA models first, Pollinations.ai as a last-resort,
free fallback tier when every NVIDIA model fails."""
import pytest

import nvidia_image
from conftest import FakeAsyncClient, FakeResponse


@pytest.mark.asyncio
async def test_generate_image_succeeds_on_first_nvidia_model_without_calling_pollinations(
    nvidia_key, fake_async_client, tmp_path, monkeypatch
):
    monkeypatch.setattr(nvidia_image, "OUTPUT_DIR", tmp_path)
    client = fake_async_client(
        post_side_effect=lambda *a, **kw: FakeResponse(200, json_data={"artifacts": [{"base64": "aGVsbG8="}]})
    )

    result = await nvidia_image.generate_image(prompt="a cat")

    assert "model: black-forest-labs/flux.1-dev" in result
    assert client.get.await_count == 0


@pytest.mark.asyncio
async def test_generate_image_falls_back_to_pollinations_when_all_nvidia_models_fail(
    nvidia_key, fake_async_client, tmp_path, monkeypatch
):
    monkeypatch.setattr(nvidia_image, "OUTPUT_DIR", tmp_path)
    client = fake_async_client(
        post_side_effect=lambda *a, **kw: FakeResponse(503, text="overloaded"),
        get_side_effect=lambda *a, **kw: FakeResponse(200, content=b"\xff\xd8\xfffake-jpeg-bytes"),
    )

    result = await nvidia_image.generate_image(prompt="a cat")

    assert "model: pollinations" in result
    assert client.get.await_count == 1
    saved = list(tmp_path.glob("pollinations_*.jpg"))
    assert len(saved) == 1
    assert saved[0].read_bytes() == b"\xff\xd8\xfffake-jpeg-bytes"


@pytest.mark.asyncio
async def test_generate_image_reports_pollinations_error_when_everything_fails(
    nvidia_key, fake_async_client, tmp_path, monkeypatch
):
    monkeypatch.setattr(nvidia_image, "OUTPUT_DIR", tmp_path)
    fake_async_client(
        post_side_effect=lambda *a, **kw: FakeResponse(503, text="overloaded"),
        get_side_effect=RuntimeError("pollinations down"),
    )

    result = await nvidia_image.generate_image(prompt="a cat")

    assert "All image models failed" in result
    assert "pollinations" in result
    assert "pollinations down" in result


@pytest.mark.asyncio
async def test_probe_pollinations_ok():
    client = FakeAsyncClient(get_side_effect=lambda *a, **kw: FakeResponse(200))

    ok, detail = await nvidia_image._probe_pollinations(client)

    assert ok is True
    assert detail == "ok"


@pytest.mark.asyncio
async def test_probe_pollinations_http_error():
    client = FakeAsyncClient(get_side_effect=lambda *a, **kw: FakeResponse(500))

    ok, detail = await nvidia_image._probe_pollinations(client)

    assert ok is False
    assert "500" in detail


@pytest.mark.asyncio
async def test_probe_pollinations_timeout_is_caught_not_raised():
    client = FakeAsyncClient(get_side_effect=nvidia_image.httpx2.TimeoutException("slow"))

    ok, detail = await nvidia_image._probe_pollinations(client)

    assert ok is False
    assert "timed out" in detail


# --- the bounded-and-verified Pollinations download (audit F4) ---


@pytest.mark.asyncio
async def test_pollinations_html_error_page_is_not_saved_as_a_jpg(
    nvidia_key, fake_async_client, tmp_path, monkeypatch
):
    """A 200 with an HTML body (what a free endpoint serves under load) must
    be reported as a failure, never written to disk as an image."""
    monkeypatch.setattr(nvidia_image, "OUTPUT_DIR", tmp_path)
    fake_async_client(
        post_side_effect=lambda *a, **kw: FakeResponse(503, text="overloaded"),
        get_side_effect=lambda *a, **kw: FakeResponse(
            200, content=b"<html>rate limited</html>", headers={"content-type": "text/html"}
        ),
    )

    result = await nvidia_image.generate_image(prompt="a cat")

    assert "All image models failed" in result
    assert "not an image" in result
    assert list(tmp_path.glob("*.jpg")) == []


@pytest.mark.asyncio
async def test_pollinations_image_content_type_with_non_image_bytes_is_rejected(
    nvidia_key, fake_async_client, tmp_path, monkeypatch
):
    """Content-Type is a claim; the magic bytes are checked too."""
    monkeypatch.setattr(nvidia_image, "OUTPUT_DIR", tmp_path)
    fake_async_client(
        post_side_effect=lambda *a, **kw: FakeResponse(503, text="overloaded"),
        get_side_effect=lambda *a, **kw: FakeResponse(
            200, content=b"<html>lies</html>", headers={"content-type": "image/jpeg"}
        ),
    )

    result = await nvidia_image.generate_image(prompt="a cat")

    assert "All image models failed" in result
    assert "not a recognizable image" in result
    assert list(tmp_path.glob("*.jpg")) == []


@pytest.mark.asyncio
async def test_pollinations_download_is_byte_capped(
    nvidia_key, fake_async_client, tmp_path, monkeypatch
):
    monkeypatch.setattr(nvidia_image, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(nvidia_image, "POLLINATIONS_MAX_DOWNLOAD_BYTES", 16)
    fake_async_client(
        post_side_effect=lambda *a, **kw: FakeResponse(503, text="overloaded"),
        get_side_effect=lambda *a, **kw: FakeResponse(200, content=b"\xff\xd8\xff" + b"x" * 64),
    )

    result = await nvidia_image.generate_image(prompt="a cat")

    assert "All image models failed" in result
    assert "download limit" in result
    assert list(tmp_path.glob("*.jpg")) == []


def test_sniff_image_format_recognizes_the_four_formats_and_rejects_html():
    assert nvidia_image._sniff_image_format(b"\xff\xd8\xff\xe0rest") == "jpeg"
    assert nvidia_image._sniff_image_format(b"\x89PNG\r\n\x1a\nrest") == "png"
    assert nvidia_image._sniff_image_format(b"RIFF\x00\x00\x00\x00WEBPrest") == "webp"
    assert nvidia_image._sniff_image_format(b"GIF89a...") == "gif"
    assert nvidia_image._sniff_image_format(b"<html></html>") is None
    assert nvidia_image._sniff_image_format(b"") is None


# --- output filenames and error redaction --------------------------------


def test_stamp_has_microsecond_precision():
    """Second precision let two calls in the same wall-clock second write to
    the same filename and silently overwrite each other."""
    stamp = nvidia_image._stamp()
    date_part, time_part, micro_part = stamp.split("_")
    assert len(date_part) == 8
    assert len(time_part) == 6
    assert len(micro_part) == 6


@pytest.mark.asyncio
async def test_rapid_nvidia_calls_do_not_collide_on_the_same_filename(
    nvidia_key, fake_async_client, tmp_path, monkeypatch
):
    monkeypatch.setattr(nvidia_image, "OUTPUT_DIR", tmp_path)
    fake_async_client(
        post_side_effect=lambda *a, **kw: FakeResponse(200, json_data={"artifacts": [{"base64": "aGVsbG8="}]})
    )

    first = await nvidia_image.generate_image(prompt="a cat")
    second = await nvidia_image.generate_image(prompt="a dog")

    assert first != second
    assert len(list(tmp_path.glob("flux.1-dev_*.jpg"))) == 2


@pytest.mark.asyncio
async def test_rapid_pollinations_calls_do_not_collide_on_the_same_filename(
    nvidia_key, fake_async_client, tmp_path, monkeypatch
):
    monkeypatch.setattr(nvidia_image, "OUTPUT_DIR", tmp_path)
    fake_async_client(
        post_side_effect=lambda *a, **kw: FakeResponse(503, text="overloaded"),
        get_side_effect=lambda *a, **kw: FakeResponse(200, content=b"\xff\xd8\xfffake-jpeg-bytes"),
    )

    await nvidia_image.generate_image(prompt="a cat")
    await nvidia_image.generate_image(prompt="a dog")

    assert len(list(tmp_path.glob("pollinations_*.jpg"))) == 2


def test_redact_replaces_the_secret_and_is_a_noop_otherwise():
    assert nvidia_image._redact("bad key nvapi-abc123", "nvapi-abc123") == "bad key ***"
    assert nvidia_image._redact("some error", None) == "some error"
    assert nvidia_image._redact("some error", "") == "some error"
    assert nvidia_image._redact("some error", "unrelated") == "some error"


@pytest.mark.asyncio
async def test_provider_error_body_is_returned_with_the_api_key_scrubbed(
    nvidia_key, fake_async_client, tmp_path, monkeypatch
):
    """A provider error body is attacker-influenced text this tool hands
    straight back to the caller - a 401 echoing the offending Authorization
    header must not carry the key out with it."""
    monkeypatch.setattr(nvidia_image, "OUTPUT_DIR", tmp_path)
    fake_async_client(
        post_side_effect=lambda *a, **kw: FakeResponse(
            401, text=f"invalid credentials: Bearer {nvidia_key}"
        ),
        get_side_effect=RuntimeError("pollinations down"),
    )

    result = await nvidia_image.generate_image(prompt="a cat")

    assert nvidia_key not in result
    assert "***" in result
