from localrag import db, nvidia

RRF_K = 60


def rrf(rankings: list[list[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, i in enumerate(ranking, 1):
            scores[i] = scores.get(i, 0.0) + 1 / (k + rank)
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))


def retrieve(conn, cfg, workspace_id, question, *, embed=nvidia.embed, rerank=nvidia.rerank) -> list[dict]:
    if not question.strip():
        return []
    fts_ids = [i for i, _ in db.fts_search(conn, workspace_id, question, cfg["candidates"])]
    qvec = embed(cfg, [question], "query")[0]
    vec_ids = [i for i, _ in db.vec_search(conn, workspace_id, qvec, cfg["candidates"])]
    fused = rrf([fts_ids, vec_ids])[:cfg["rerank_candidates"]]
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
