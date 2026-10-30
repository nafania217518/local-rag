import pytest

from localrag import config, db
from localrag.retrieval import paraphrase, retrieve, rrf

E1, E2, E3, E4 = [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]


@pytest.fixture
def cfg():
    return {**config.DEFAULTS, "api_key": "k", "paraphrase": False}


@pytest.fixture
def env(tmp_path):
    conn = db.connect(tmp_path / "t.db", embed_dim=4)
    ws = db.create_workspace(conn, "w")
    d1 = db.create_document(conn, ws, "a.pdf", "a.pdf")
    d2 = db.create_document(conn, ws, "b.pdf", "b.pdf")
    return conn, ws, d1, d2


def fake_embed(vec):
    def embed(cfg, texts, input_type):
        return [vec for _ in texts]
    return embed


def no_rerank(*a):
    raise AssertionError("rerank called")


def test_rrf_sums_reciprocal_ranks():
    out = rrf([[1, 2, 3], [3, 1]], k=60)
    assert [i for i, _ in out] == [1, 3, 2]
    s = dict(out)
    assert s[1] == pytest.approx(1 / 61 + 1 / 62)
    assert s[3] == pytest.approx(1 / 63 + 1 / 61)
    assert s[2] == pytest.approx(1 / 62)


def test_rrf_tie_breaks_by_id():
    assert [i for i, _ in rrf([[5, 2], [2, 5]])] == [2, 5]


def test_retrieve_finds_exact_token_via_fts(env, cfg):
    conn, ws, d1, d2 = env
    ids = db.add_chunks(conn, d1, ws, [(0, 1, "the zyxquart valve", E1), (1, 1, "plain text", E2)])
    cfg["rerank"] = False
    out = retrieve(conn, cfg, ws, "zyxquart", embed=fake_embed(E2), rerank=no_rerank)
    assert out[0]["id"] == ids[0]


def test_retrieve_finds_paraphrase_via_vector(env, cfg):
    conn, ws, d1, d2 = env
    ids = db.add_chunks(conn, d1, ws, [(0, 1, "automobile repair", E1), (1, 1, "cooking", E2)])
    cfg["rerank"] = False
    out = retrieve(conn, cfg, ws, "fixing a car", embed=fake_embed(E1), rerank=no_rerank)
    assert out[0]["id"] == ids[0]


def test_retrieve_cross_lingual_shared_vector(env, cfg):
    conn, ws, d1, d2 = env
    ids = db.add_chunks(conn, d1, ws, [(0, 1, "Київ є столицею України", E1), (1, 1, "other", E2)])
    cfg["rerank"] = False
    assert db.fts_search(conn, ws, "capital of Ukraine", 30) == []
    out = retrieve(conn, cfg, ws, "capital of Ukraine", embed=fake_embed(E1), rerank=no_rerank)
    assert out[0]["id"] == ids[0]


def test_retrieve_merges_fts_and_vector_with_rrf(env, cfg):
    conn, ws, d1, d2 = env
    a, b, c = db.add_chunks(conn, d1, ws, [
        (0, 1, "apple only words", E1),   # fts rank 1, vec rank 2
        (1, 1, "nothing here", E2),       # vec rank 1 only
        (2, 1, "apple again pie", E3),    # fts rank 2, vec rank 3
    ])
    cfg["rerank"] = False
    q = [0.6, 0.8, 0, 0]  # nearest: b, then a
    out = retrieve(conn, cfg, ws, "apple", embed=fake_embed(q), rerank=no_rerank)
    assert out[0]["id"] == a


def test_retrieve_applies_rerank_order_and_logit_scores(env, cfg):
    conn, ws, d1, d2 = env
    db.add_chunks(conn, d1, ws, [(0, 1, "t0", E1), (1, 1, "t1", E2), (2, 1, "t2", E3)])
    seen = {}

    def rerank(cfg, query, passages):
        seen["p"] = passages
        return [(2, 5.0), (0, 1.5), (1, -2.0)]

    out = retrieve(conn, cfg, ws, "q", embed=fake_embed([0.9, 0.4, 0.1, 0]), rerank=rerank)
    assert seen["p"] == ["t0", "t1", "t2"]
    assert [r["text"] for r in out] == ["t2", "t0", "t1"]
    assert [r["score"] for r in out] == [5.0, 1.5, -2.0]


def test_retrieve_rerank_off_uses_rrf_scores(env, cfg):
    conn, ws, d1, d2 = env
    db.add_chunks(conn, d1, ws, [(0, 1, "t0", E1), (1, 1, "t1", E2)])
    cfg["rerank"] = False
    out = retrieve(conn, cfg, ws, "q", embed=fake_embed(E1), rerank=no_rerank)
    assert [r["score"] for r in out] == pytest.approx([1 / 61, 1 / 62])


def test_retrieve_limits_to_final_k(env, cfg):
    conn, ws, d1, d2 = env
    db.add_chunks(conn, d1, ws, [(i, 1, f"match {i}", E1) for i in range(10)])
    cfg["rerank"] = False
    out = retrieve(conn, cfg, ws, "match", embed=fake_embed(E1), rerank=no_rerank)
    assert len(out) == config.DEFAULTS["sources"] == 6


def test_retrieve_empty_workspace_returns_empty_without_rerank(env, cfg):
    conn, ws, d1, d2 = env
    assert retrieve(conn, cfg, ws, "q", embed=fake_embed(E1), rerank=no_rerank) == []


def test_retrieve_blank_question_returns_empty_without_embed(env, cfg):
    conn, ws, d1, d2 = env
    assert retrieve(conn, cfg, ws, "  ", embed=no_rerank, rerank=no_rerank) == []


def test_retrieve_result_shape(env, cfg):
    conn, ws, d1, d2 = env
    db.add_chunks(conn, d2, ws, [(0, 3, "hello", E1)])
    cfg["rerank"] = False
    (r,) = retrieve(conn, cfg, ws, "hello", embed=fake_embed(E1), rerank=no_rerank)
    assert set(r) == {"id", "document_id", "document_name", "page", "text", "score"}
    assert (r["document_id"], r["document_name"], r["page"]) == (d2, "b.pdf", 3)


def test_retrieve_passes_query_input_type(env, cfg):
    conn, ws, d1, d2 = env
    db.add_chunks(conn, d1, ws, [(0, 1, "x", E1)])
    cfg["rerank"] = False

    def embed(cfg, texts, input_type):
        assert input_type == "query" and texts == ["x"]
        return [E1]

    assert len(retrieve(conn, cfg, ws, "x", embed=embed, rerank=no_rerank)) == 1


def test_retrieve_uses_candidates_rerank_candidates_and_sources_from_config(env, cfg):
    conn, ws, d1, d2 = env
    db.add_chunks(conn, d1, ws, [(i, 1, f"match {i}", E1) for i in range(10)])
    seen = []

    def rerank(c, q, passages):
        seen.append(len(passages))
        return [(i, 1.0) for i in range(len(passages))]

    cfg.update(candidates=4, rerank_candidates=5, sources=2)
    out = retrieve(conn, cfg, ws, "match", embed=fake_embed(E1), rerank=rerank)
    assert len(out) == 2
    assert seen == [5]  # 4 fts + 4 vec candidates fused, then cut to 5
    cfg.update(candidates=1, rerank_candidates=5)
    seen.clear()
    retrieve(conn, cfg, ws, "match", embed=fake_embed(E1), rerank=rerank)
    assert seen[0] <= 2  # one fts + one vec candidate


def fake_chat(reply, calls=None):
    def chat_stream(cfg, messages, reasoning_effort=None):
        if calls is not None:
            calls.append((cfg, messages, reasoning_effort))
        yield "thinking", "hmm"
        yield "token", reply
    return chat_stream


def test_paraphrase_asks_chat_model_once_for_n_lines_with_no_reasoning(cfg):
    cfg.update(paraphrase=True, paraphrases=2)
    calls = []
    out = paraphrase(cfg, "what is x?", fake_chat("p1\np2", calls))
    assert out == ["p1", "p2"]
    assert len(calls) == 1
    _, messages, effort = calls[0]
    assert effort == "none"
    assert "what is x?" in messages[0]["content"] and "2" in messages[0]["content"]


def test_tr24_paraphrase_prompt_asks_for_formal_technical_terms(cfg):
    cfg.update(paraphrase=True, paraphrases=2)
    calls = []
    paraphrase(cfg, "як варити труби?", fake_chat("p1\np2", calls))
    content = calls[0][1][0]["content"]
    assert "technical terms" in content
    assert "зварювати трубопроводи" in content
    assert "same language" in content
    assert content.endswith("Question: як варити труби?")


def test_paraphrase_strips_numbering_blanks_and_duplicates_and_caps_at_n(cfg):
    cfg.update(paraphrase=True, paraphrases=2)
    assert paraphrase(cfg, "q", fake_chat('1. A\n\n- B\nA\n"C"\nq\n')) == ["A", "B"]


def test_paraphrase_failure_returns_no_rewrites(cfg):
    cfg.update(paraphrase=True)
    def bad(*a, **k):
        raise RuntimeError("down")
        yield
    assert paraphrase(cfg, "q", bad) == []


def test_paraphrase_off_does_not_call_chat_model(cfg):
    def boom(*a, **k):
        raise AssertionError("called")
    assert paraphrase(cfg, "q", boom) == []


def test_retrieve_fuses_fts_hits_found_only_by_a_paraphrase(env, cfg):
    conn, ws, d1, d2 = env
    ids = db.add_chunks(conn, d1, ws, [(0, 1, "alpha words", E1), (1, 1, "omega words", E1)])
    cfg.update(rerank=False, paraphrase=True, paraphrases=1, sources=1)
    out = retrieve(conn, cfg, ws, "nothing", embed=fake_embed(E1), rerank=no_rerank,
                   chat_stream=fake_chat("omega"))
    assert out[0]["id"] == ids[1]


def test_retrieve_embeds_original_and_paraphrases_in_one_call(env, cfg):
    conn, ws, d1, d2 = env
    db.add_chunks(conn, d1, ws, [(0, 1, "text", E1)])
    cfg.update(rerank=False, paraphrase=True, paraphrases=2)
    calls = []
    def embed(cfg, texts, input_type):
        calls.append(texts)
        return [E1 for _ in texts]
    retrieve(conn, cfg, ws, "q", embed=embed, rerank=no_rerank, chat_stream=fake_chat("p1\np2"))
    assert calls == [["q", "p1", "p2"]]


def test_retrieve_reranks_with_original_question(env, cfg):
    conn, ws, d1, d2 = env
    db.add_chunks(conn, d1, ws, [(0, 1, "text", E1)])
    cfg.update(rerank=True, paraphrase=True)
    queries = []
    def rerank(cfg, query, passages):
        queries.append(query)
        return [(i, 1.0) for i in range(len(passages))]
    retrieve(conn, cfg, ws, "q", embed=fake_embed(E1), rerank=rerank, chat_stream=fake_chat("p1"))
    assert queries == ["q"]
