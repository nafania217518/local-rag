import json
import os
import sqlite3
from pathlib import Path

import pytest

from localrag import config, db, nvidia, retrieval

pytestmark = [pytest.mark.live,
              pytest.mark.skipif(not os.environ.get("NVIDIA_API_KEY"), reason="needs NVIDIA_API_KEY"),
              pytest.mark.skipif(not os.environ.get("LOCALRAG_EVAL_DB"), reason="needs LOCALRAG_EVAL_DB")]

QUESTIONS = json.loads((Path(__file__).parent / "data" / "keyword_eval.json").read_text(encoding="utf-8"))


def copy_eval_db(tmp_path):
    # db.connect writes the schema, so the user's database is only ever opened read-only
    source = sqlite3.connect(f"file:{os.environ['LOCALRAG_EVAL_DB']}?mode=ro", uri=True)
    target = sqlite3.connect(tmp_path / "eval.db")
    source.backup(target)
    source.close()
    target.close()
    return db.connect(tmp_path / "eval.db")


def is_hit(chunks, entry):
    return any(c["document_name"] == entry["document"] and c["page"] in entry["pages"] for c in chunks)


def report(label, without, with_keywords):
    gained = [n for n in with_keywords if n not in without]
    lost = [n for n in without if n not in with_keywords]
    total = len(QUESTIONS)
    print(f"{label}: without {len(without)}/{total}, with {len(with_keywords)}/{total}, "
          f"gained {gained}, lost {lost}")


# req: 10-keyword-and-synonym-expansion-for-keyword-search/TR-10
def test_live_keyword_ranking_does_not_lower_the_fused_hit_rate(tmp_path):
    conn = copy_eval_db(tmp_path)
    cfg = {**config.DEFAULTS, "api_key": os.environ["NVIDIA_API_KEY"]}
    ws = conn.execute("SELECT workspace_id FROM documents WHERE name = ?", (QUESTIONS[0]["document"],)).fetchone()[0]
    fused_hits = {"without": [], "with": []}
    source_hits = {"without": [], "with": []}
    with_keywords_count = 0
    for number, entry in enumerate(QUESTIONS, 1):
        question = entry["question"]
        expansion = retrieval.expand(cfg, question)
        with_keywords_count += bool(expansion.keywords)
        rankings = retrieval.search_rankings(conn, cfg, ws, [question, *expansion.paraphrases], nvidia.embed)
        without = retrieval.fuse(cfg, rankings)
        with_keywords = retrieval.fuse(cfg, rankings + [retrieval.keyword_ranking(conn, cfg, ws, expansion.keywords)])
        sources_without = retrieval.pick_sources(conn, cfg, question, without, nvidia.rerank)
        sources_with = (sources_without if with_keywords == without
                        else retrieval.pick_sources(conn, cfg, question, with_keywords, nvidia.rerank))
        for variant, fused, sources in (("without", without, sources_without), ("with", with_keywords, sources_with)):
            if is_hit(db.get_chunks(conn, [i for i, _ in fused]), entry):
                fused_hits[variant].append(number)
            if is_hit(sources, entry):
                source_hits[variant].append(number)
    report(f"fused top {cfg['rerank_candidates']}", fused_hits["without"], fused_hits["with"])
    report(f"top {cfg['sources']} after rerank", source_hits["without"], source_hits["with"])
    print(f"questions with keywords: {with_keywords_count}/{len(QUESTIONS)}")
    assert len(fused_hits["with"]) >= len(fused_hits["without"])
