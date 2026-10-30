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


from localrag.config import DEFAULTS, EMBED_DIM, config_path, load_config, save_config


def test_embed_dim_and_config_path():
    assert EMBED_DIM == 4096
    assert config_path() == app_data_dir() / "config.json"


def test_load_config_defaults_when_missing(tmp_path):
    assert load_config(tmp_path / "c.json") == DEFAULTS


def test_config_round_trip(tmp_path):
    p = tmp_path / "c.json"
    cfg = {"api_key": "k", "rerank": False}
    save_config(cfg, p)
    assert load_config(p) == {**DEFAULTS, **cfg}


def test_load_config_merges_partial_file(tmp_path):
    p = tmp_path / "c.json"
    p.write_text('{"api_key": "x"}')
    assert load_config(p) == {**DEFAULTS, "api_key": "x"}


def test_load_config_returns_copy(tmp_path):
    load_config(tmp_path / "c.json")["api_key"] = "mutated"
    assert DEFAULTS["api_key"] == ""


def test_save_config_creates_parent_dir(tmp_path):
    p = tmp_path / "a" / "b" / "c.json"
    save_config({"api_key": "k"}, p)
    assert p.exists()


def test_default_path_used_when_none(tmp_path, monkeypatch):
    import localrag.config as c
    monkeypatch.setattr(c, "config_path", lambda: tmp_path / "d" / "c.json")
    save_config({"api_key": "z"})
    assert load_config()["api_key"] == "z"


def test_default_chat_model_is_nemotron_super_vl():
    from localrag import config
    assert config.DEFAULTS["chat_model"] == "nvidia/nemotron-3.5-super-vl-120b-a12b"


def test_tr1_default_parse_model_is_nemotron_3_5_parse():
    from localrag import config
    assert config.DEFAULTS["parse_model"] == "nvidia/nemotron-3.5-parse"


def test_defaults_cover_every_tunable():
    expected = {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "rerank_url": "https://ai.api.nvidia.com/v1/retrieval/{model}/reranking",
        "parse_url": "https://ai.api.nvidia.com/v1/retrieval/{model}/parse",
        "rpm": 40, "timeout": 120.0, "temperature": 0.2, "max_tokens": 4096, "history_limit": 10,
        "sources": 6, "candidates": 30, "rerank_candidates": 20, "chunk_size": 1500, "chunk_overlap": 200,
        "ocr_quality_min": 0.8, "min_text_chars": 20, "render_dpi": 150,
    }
    for k, v in expected.items():
        assert DEFAULTS[k] == v and type(DEFAULTS[k]) is type(v), k
    assert DEFAULTS["system_prompt"].startswith("You answer questions about the user's documents.")
    assert "[n]" in DEFAULTS["system_prompt"]


def test_defaults_paraphrase_on_with_two_rewrites():
    assert DEFAULTS["paraphrase"] is True and DEFAULTS["paraphrases"] == 2
