import re

from localrag import db, nvidia

RRF_K = 60


def rrf(rankings: list[list[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, i in enumerate(ranking, 1):
            scores[i] = scores.get(i, 0.0) + 1 / (k + rank)
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))


PARAPHRASE_PROMPT = ("Write {n} different phrasings of the question below, in the same language as the question. "
                     "The documents are technical standards and specifications: use the formal technical terms a "
                     "standard uses instead of everyday words. One phrasing per line. No numbering, no quotes, "
                     "no other text. Example for a Ukrainian question: 'Як варити труби?' becomes "
                     "'Як зварювати трубопроводи?'. Write in the language of the question, not in the language "
                     "of the example.\n\nQuestion: {question}")


def paraphrase(cfg, question, chat_stream=nvidia.chat_stream) -> list[str]:
    n = cfg["paraphrases"]
    if not cfg["paraphrase"] or n < 1:
        return []
    try:
        messages = [{"role": "user", "content": PARAPHRASE_PROMPT.format(n=n, question=question)}]
        text = "".join(t for kind, t in chat_stream(cfg, messages, "none") if kind == "token")
        out = []
        for line in text.splitlines():
            line = re.sub(r"^\s*(?:[-*•]|\d+[.)])?\s*", "", line).strip().strip("\"'«»“”").strip()
            if line and line.lower() != question.strip().lower() and line not in out:
                out.append(line)
        return out[:n]
    except Exception:
        return []  # the answer runs on the original question alone


def retrieve(conn, cfg, workspace_id, question, *, embed=nvidia.embed, rerank=nvidia.rerank,
             chat_stream=nvidia.chat_stream) -> list[dict]:
    if not question.strip():
        return []
    queries = [question, *paraphrase(cfg, question, chat_stream)]
    fts = [[i for i, _ in db.fts_search(conn, workspace_id, q, cfg["candidates"])] for q in queries]
    vec = [[i for i, _ in db.vec_search(conn, workspace_id, v, cfg["candidates"])]
           for v in embed(cfg, queries, "query")]
    fused = rrf(fts + vec)[:cfg["rerank_candidates"]]
    if not fused:
        return []
    rows = db.get_chunks(conn, [i for i, _ in fused])
    if cfg["rerank"]:
        ranked = rerank(cfg, question, [r["text"] for r in rows])[:cfg["sources"]]
        picked = [(rows[i], s) for i, s in ranked]
    else:
        picked = list(zip(rows, [s for _, s in fused]))[:cfg["sources"]]
    keys = ("id", "document_id", "document_name", "page", "text")
    return [{**{k: r[k] for k in keys}, "score": s} for r, s in picked]
