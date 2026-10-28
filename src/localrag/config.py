import os
import sys
from pathlib import Path


def app_data_dir() -> Path:
    """Single source of truth for the app data path. Does not create it."""
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "LocalRAG"
    if sys.platform == "win32":
        return Path(os.environ["APPDATA"]) / "LocalRAG"
    return Path.home() / ".local" / "share" / "LocalRAG"
