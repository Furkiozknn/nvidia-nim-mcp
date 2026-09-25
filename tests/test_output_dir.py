"""OUTPUT_DIR's environment-driven default.

The bug this pins: `OUTPUT_DIR = Path(__file__).parent / "output"` wrote
generated images and embeddings inside the installed package, which for a
`pip install`ed user is site-packages. Two sibling repositories shipped the
identical defect; this is the third.
"""
import importlib
from pathlib import Path

import nvidia_image


def test_output_dir_follows_the_environment_variable(tmp_path, monkeypatch):
    """OUTPUT_DIR is resolved at import time, so this reloads the module -
    reload mutates the existing module object in place, leaving every other
    test's `import nvidia_image` reference valid."""
    custom = tmp_path / "custom-media-out"
    monkeypatch.setenv("NVIDIA_NIM_OUTPUT_DIR", str(custom))
    try:
        importlib.reload(nvidia_image)
        assert nvidia_image.OUTPUT_DIR == custom
    finally:
        monkeypatch.delenv("NVIDIA_NIM_OUTPUT_DIR", raising=False)
        importlib.reload(nvidia_image)


def test_output_dir_defaults_under_the_cwd_not_the_installed_package(tmp_path, monkeypatch):
    monkeypatch.delenv("NVIDIA_NIM_OUTPUT_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    try:
        importlib.reload(nvidia_image)
        assert nvidia_image.OUTPUT_DIR == tmp_path / "output"
        package_dir = Path(nvidia_image.__file__).resolve().parent
        assert package_dir not in nvidia_image.OUTPUT_DIR.resolve().parents
    finally:
        importlib.reload(nvidia_image)


def test_a_nested_override_is_created_not_refused(tmp_path, monkeypatch):
    """mkdir uses parents=True: an override two levels deep must work."""
    nested = tmp_path / "a" / "b" / "media"
    monkeypatch.setenv("NVIDIA_NIM_OUTPUT_DIR", str(nested))
    try:
        importlib.reload(nvidia_image)
        nvidia_image.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        assert nested.is_dir()
    finally:
        monkeypatch.delenv("NVIDIA_NIM_OUTPUT_DIR", raising=False)
        importlib.reload(nvidia_image)
