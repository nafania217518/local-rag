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


def first_hit_rank(conn, ids, entry):
    rows = db.get_chunks(conn, ids)
    return next((rank for rank, c in enumerate(rows, 1) if is_hit([c], entry)), None)


def mean_reciprocal_rank(ranks):
    return sum(1 / r for r in ranks if r) / len(ranks)


def hit_numbers(ranks):
    return [n for n, r in enumerate(ranks, 1) if r]


def rank_shifts(before, after):
    worse_than_none = float("inf")
    better = [n for n, (b, a) in enumerate(zip(before, after), 1) if (a or worse_than_none) < (b or worse_than_none)]
    worse = [n for n, (b, a) in enumerate(zip(before, after), 1) if (a or worse_than_none) > (b or worse_than_none)]
    return better, worse


def report(label, without, with_keywords):
    gained = [n for n in with_keywords if n not in without]
    lost = [n for n in without if n not in with_keywords]
    total = len(QUESTIONS)
    print(f"{label}: without {len(without)}/{total}, with {len(with_keywords)}/{total}, "
          f"gained {gained}, lost {lost}")


def eval_workspace(conn):
    documents = sorted({e["document"] for e in QUESTIONS})
    found = {name: ws for name, ws in conn.execute(
        f"SELECT name, workspace_id FROM documents WHERE name IN ({','.join('?' * len(documents))})", documents)}
    missing = [name for name in documents if name not in found]
    assert not missing, f"documents of the question set are missing from the database: {missing}"
    workspaces = set(found.values())
    assert len(workspaces) == 1, f"documents of the question set sit in several workspaces: {sorted(workspaces)}"
    return workspaces.pop()


# req: 10-keyword-and-synonym-expansion-for-keyword-search/TR-10, 10-keyword-and-synonym-expansion-for-keyword-search/TR-12, 10-keyword-and-synonym-expansion-for-keyword-search/TR-14
def test_live_keyword_ranking_does_not_lower_the_fused_hit_rate(tmp_path):
    conn = copy_eval_db(tmp_path)
    cfg = {**config.DEFAULTS, "api_key": os.environ["NVIDIA_API_KEY"]}
    ws = eval_workspace(conn)
    fused_hits = {"without": [], "with": []}
    source_hits = {"without": [], "with": []}
    with_keywords_count = 0
    fused_ranks = {"without": [], "with": []}
    alone_ranks = {"question": [], "keywords": []}
    for number, entry in enumerate(QUESTIONS, 1):
        question = entry["question"]
        expansion = retrieval.expand(cfg, question)
        with_keywords_count += bool(expansion.keywords)
        rankings = retrieval.search_rankings(conn, cfg, ws, [question, *expansion.paraphrases], nvidia.embed)
        without = retrieval.fuse(cfg, rankings)
        keyword_terms = retrieval.keyword_ranking(conn, cfg, ws, expansion.keywords)
        with_keywords = retrieval.fuse(cfg, rankings + [keyword_terms])
        question_ranking = [i for i, _ in db.fts_search(conn, ws, question, cfg["candidates"])]
        alone_ranks["question"].append(first_hit_rank(conn, question_ranking, entry))
        alone_ranks["keywords"].append(first_hit_rank(conn, keyword_terms, entry))
        sources_without = retrieval.pick_sources(conn, cfg, question, without, nvidia.rerank)
        sources_with = (sources_without if with_keywords == without
                        else retrieval.pick_sources(conn, cfg, question, with_keywords, nvidia.rerank))
        for variant, fused, sources in (("without", without, sources_without), ("with", with_keywords, sources_with)):
            rank = first_hit_rank(conn, [i for i, _ in fused], entry)
            fused_ranks[variant].append(rank)
            if rank:
                fused_hits[variant].append(number)
            if is_hit(sources, entry):
                source_hits[variant].append(number)
    print()  # pytest -s leaves the cursor after the test name
    report(f"fused top {cfg['rerank_candidates']}", fused_hits["without"], fused_hits["with"])
    report(f"top {cfg['sources']} after rerank", source_hits["without"], source_hits["with"])
    print(f"questions with keywords: {with_keywords_count}/{len(QUESTIONS)}")
    better, worse = rank_shifts(fused_ranks["without"], fused_ranks["with"])
    print(f"fused MRR: without {mean_reciprocal_rank(fused_ranks['without']):.3f}, "
          f"with {mean_reciprocal_rank(fused_ranks['with']):.3f}, first hit better {better}, worse {worse}")
    total = len(QUESTIONS)
    question_hits, keyword_hits = hit_numbers(alone_ranks["question"]), hit_numbers(alone_ranks["keywords"])
    for label, ranks in (("question", alone_ranks["question"]), ("keyword-term", alone_ranks["keywords"])):
        print(f"{label} ranking alone, top {cfg['candidates']}: {len(hit_numbers(ranks))}/{total}, "
              f"MRR {mean_reciprocal_rank(ranks):.3f}")
    print(f"only keyword-term ranking hits {[n for n in keyword_hits if n not in question_hits]}, "
          f"only question ranking hits {[n for n in question_hits if n not in keyword_hits]}")
    for lang in ("uk", "ru", "en"):
        numbers = [n for n, e in enumerate(QUESTIONS, 1) if e["lang"] == lang]
        print(f"keyword-term ranking alone, {lang}: {len([n for n in numbers if n in keyword_hits])}/{len(numbers)}")
    assert len(fused_hits["with"]) >= len(fused_hits["without"])
