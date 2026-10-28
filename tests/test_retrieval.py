import pytest

from localrag import config, db
from localrag.retrieval import FINAL_K, retrieve, rrf

E1, E2, E3, E4 = [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]


@pytest.fixture
def cfg():
    return {**config.DEFAULTS, "api_key": "k"}


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
    assert len(out) == FINAL_K == 6


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
