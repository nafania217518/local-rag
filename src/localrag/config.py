import json
import os
import sys
from pathlib import Path

EMBED_DIM = 4096

DEFAULTS = {
    "api_key": "",
    "chat_model": "nvidia/nemotron-3.5-super-vl-120b-a12b",
    "embed_model": "nvidia/nemotron-3.5-embed-8b",
    "rerank_model": "nvidia/nemotron-3.5-rerank-8b",
    "parse_model": "nvidia/nemotron-parse",
    "rerank": True,
    "paraphrase": True,
    "paraphrases": 2,
    "reasoning_effort": "none",
    "base_url": "https://integrate.api.nvidia.com/v1",
    "rerank_url": "https://ai.api.nvidia.com/v1/retrieval/{model}/reranking",
    "rpm": 40,  # build.nvidia.com limit, shared by all models
    "timeout": 120.0,
    "system_prompt": (
        "You answer questions about the user's documents.\n"
        "- Answer only from the numbered context passages.\n"
        "- Cite passages inline as [n] after each claim, where n is the passage number.\n"
        "- If the context does not contain the answer, say so and do not guess.\n"
        "- Answer in the language of the question.\n"
        "- Be concise."
    ),
    "temperature": 0.2,
    "max_tokens": 4096,
    "history_limit": 10,
    "sources": 6,
    "candidates": 30,
    "rerank_candidates": 20,
    "chunk_size": 1500,
    "chunk_overlap": 200,
    "ocr_quality_min": 0.8,
    "min_text_chars": 20,
    "render_dpi": 150,
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
