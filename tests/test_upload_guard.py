"""describe_image uploads a local file to a third party, so the guard must
look at the file's bytes, not only at its extension."""
import base64

import pytest

import nvidia_image
from conftest import FakeResponse

JPEG = b"\xff\xd8\xff\xe0" + b"0" * 32
PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 32


def _no_upload(monkeypatch, fake_async_client):
    def boom(*a, **kw):
        raise AssertionError("no bytes may leave the machine for a refused file")

    fake_async_client(post_side_effect=boom)

    async def boom_async(**kw):
        raise AssertionError("no bytes may leave the machine for a refused file")

    monkeypatch.setattr(nvidia_image.litellm, "acompletion", boom_async)


@pytest.mark.parametrize(
    "name, body",
    [
        ("report.png", b"%PDF-1.7\n" + b"0" * 32),
        ("key.jpg", b"-----BEGIN OPENSSH PRIVATE KEY-----\n"),
        ("anim.webp", b"GIF89a" + b"0" * 32),
        ("empty.png", b""),
    ],
)
async def test_a_renamed_non_image_is_refused_before_upload(nvidia_key, fake_async_client, tmp_path, monkeypatch, name, body):
    _no_upload(monkeypatch, fake_async_client)
    path = tmp_path / name
    path.write_bytes(body)

    result = await nvidia_image.describe_image(image_path=str(path))

    assert result.startswith(f"Not uploading {path}")
    assert "file signature" in result


async def test_the_data_url_is_named_after_the_sniffed_format(nvidia_key, fake_async_client, tmp_path):
    """A JPEG saved as .png is uploaded as image/jpeg, not image/png."""
    sent = []

    def answer(*a, **kw):
        sent.append(kw["json"])
        return FakeResponse(200, {"choices": [{"message": {"content": "a cat"}}]})

    fake_async_client(post_side_effect=answer)
    path = tmp_path / "photo.png"
    path.write_bytes(JPEG)

    result = await nvidia_image.describe_image(image_path=str(path))

    assert "a cat" in result
    url = sent[0]["messages"][0]["content"][1]["image_url"]["url"]
    assert url == "data:image/jpeg;base64," + base64.b64encode(JPEG).decode()


async def test_the_size_cap_holds_on_the_read_itself(nvidia_key, fake_async_client, tmp_path, monkeypatch):
    """A file that passes the stat() check but is larger when read (it grew in
    between) is still refused."""
    _no_upload(monkeypatch, fake_async_client)
    path = tmp_path / "growing.png"
    path.write_bytes(PNG)
    monkeypatch.setattr(nvidia_image, "DESCRIBE_IMAGE_MAX_BYTES", len(PNG))
    path.write_bytes(PNG + b"more")
    real_stat = type(path).stat
    monkeypatch.setattr(type(path), "stat", lambda self, *a, **kw: type("S", (), {"st_size": len(PNG), "st_mode": real_stat(self).st_mode})())

    result = await nvidia_image.describe_image(image_path=str(path))

    assert "grew past" in result


async def test_an_unreadable_file_is_a_message_not_a_crash(nvidia_key, fake_async_client, tmp_path, monkeypatch):
    _no_upload(monkeypatch, fake_async_client)
    path = tmp_path / "photo.png"
    path.write_bytes(PNG)

    def denied(p):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(nvidia_image, "_read_image_bytes", denied)

    result = await nvidia_image.describe_image(image_path=str(path))

    assert result == f"Cannot read {path}: Permission denied"
