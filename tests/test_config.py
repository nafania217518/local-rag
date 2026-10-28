import sys
from pathlib import Path

import pytest

from localrag.config import app_data_dir


@pytest.mark.parametrize("platform", ["darwin", "win32", "linux"])
def test_app_data_dir_per_os(platform, monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setenv("APPDATA", str(tmp_path))
    expected = {
        "darwin": Path.home() / "Library" / "Application Support" / "LocalRAG",
        "win32": tmp_path / "LocalRAG",
        "linux": Path.home() / ".local" / "share" / "LocalRAG",
    }[platform]
    assert app_data_dir() == expected
