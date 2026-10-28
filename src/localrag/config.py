import json
import os
import sys
from pathlib import Path

EMBED_DIM = 4096

DEFAULTS = {
    "api_key": "",
    "chat_model": "deepseek-ai/deepseek-v4-flash",
    "embed_model": "nvidia/nemotron-3.5-embed-8b",
    "rerank_model": "nvidia/nemotron-3.5-rerank-8b",
    "parse_model": "nvidia/nemotron-parse-2.0",
    "rerank": True,
    "reasoning_effort": "none",
}


def app_data_dir() -> Path:
    """Single source of truth for the app data path. Does not create it."""
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "LocalRAG"
    if sys.platform == "win32":
        return Path(os.environ["APPDATA"]) / "LocalRAG"
    return Path.home() / ".local" / "share" / "LocalRAG"


def config_path() -> Path:
    return app_data_dir() / "config.json"


def load_config(path: Path | None = None) -> dict:
    path = path or config_path()
    cfg = dict(DEFAULTS)
    if path.exists():
        cfg.update(json.loads(path.read_text(encoding="utf-8")))
    return cfg


def save_config(cfg: dict, path: Path | None = None) -> None:
    path = path or config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
