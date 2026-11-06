import pytest

from localrag import config, db
from localrag.retrieval import Expansion, expand, fuse, keyword_ranking, retrieve, rrf, search_rankings

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


# req: 10-keyword-and-synonym-expansion-for-keyword-search/TR-4
def test_expand_asks_chat_model_once_with_no_reasoning(cfg):
    cfg.update(paraphrase=True, paraphrases=2)
    calls = []
    out = expand(cfg, "what is x?", fake_chat("PHRASING: p1\nPHRASING: p2", calls))
    assert out.paraphrases == ("p1", "p2")
    assert len(calls) == 1
    _, messages, effort = calls[0]
    assert effort == "none"
    assert "what is x?" in messages[0]["content"] and "2 different phrasings" in messages[0]["content"]


# req: 10-keyword-and-synonym-expansion-for-keyword-search/TR-4
def test_expand_prompt_asks_for_formal_technical_terms(cfg):
    cfg.update(paraphrase=True, paraphrases=2)
    calls = []
    expand(cfg, "як варити труби?", fake_chat("", calls))
    content = calls[0][1][0]["content"]
    assert "technical terms" in content
    assert "зварювати трубопроводи" in content
    assert "same language" in content
    assert content.endswith("Question: як варити труби?")


# req: 10-keyword-and-synonym-expansion-for-keyword-search/TR-4
def test_expand_prompt_example_is_a_question_and_the_language_rule_follows_it(cfg):
    cfg.update(paraphrase=True, paraphrases=2)
    calls = []
    expand(cfg, "how to check welds?", fake_chat("", calls))
    content = calls[0][1][0]["content"]
    example = "'Як варити труби?' becomes 'Як зварювати трубопроводи?'"
    clause = "Write the intent and the phrasings in the language of the question, not in the language of the example."
    assert example in content
    assert content.index(clause) > content.index(example) > content.index("No numbering")
    assert content.index(example) > content.index("KEYWORDS:")


# req: 10-keyword-and-synonym-expansion-for-keyword-search/TR-4
def test_expand_prompt_asks_for_intent_then_phrasings_then_keywords(cfg):
    cfg.update(paraphrase=True, paraphrases=2)
    calls = []
    expand(cfg, "q", fake_chat("", calls))
    content = calls[0][1][0]["content"]
    assert content.index("INTENT:") < content.index("PHRASING:") < content.index("KEYWORDS:")
    assert "one line" in content.lower()


# req: 10-keyword-and-synonym-expansion-for-keyword-search/TR-4
def test_expand_prompt_asks_for_keywords_in_the_question_language_and_in_english(cfg):
    cfg.update(paraphrase=True)
    calls = []
    expand(cfg, "q", fake_chat("", calls))
    content = calls[0][1][0]["content"]
    assert "<English term>" in content
    assert "then the same term" in content and "in English" in content
    assert "For an English question, write English only." in content


# req: 10-keyword-and-synonym-expansion-for-keyword-search/TR-4
def test_expand_prompt_takes_the_group_and_synonym_caps_from_config(cfg):
    cfg.update(paraphrase=True, keyword_groups=7, keyword_synonyms=5)
    calls = []
    expand(cfg, "q", fake_chat("", calls))
    content = calls[0][1][0]["content"]
    assert "At most 7 lines" in content and "at most 5 synonyms" in content


# req: 10-keyword-and-synonym-expansion-for-keyword-search/TR-5
def test_expand_strips_numbering_blanks_and_duplicates_and_caps_at_n(cfg):
    cfg.update(paraphrase=True, paraphrases=2)
    reply = '1. PHRASING: A\n\n- PHRASING: B\nPHRASING: A\nPHRASING: "C"\nPHRASING: q\n'
    assert expand(cfg, "q", fake_chat(reply)).paraphrases == ("A", "B")


# req: 10-keyword-and-synonym-expansion-for-keyword-search/TR-6
def test_expand_failure_returns_an_empty_expansion(cfg):
    cfg.update(paraphrase=True)
    def bad(*a, **k):
        raise RuntimeError("down")
        yield
    assert expand(cfg, "q", bad) == Expansion()


# req: 10-keyword-and-synonym-expansion-for-keyword-search/TR-4
def test_expand_off_does_not_call_chat_model(cfg):
    def boom(*a, **k):
        raise AssertionError("called")
    assert expand(cfg, "q", boom) == Expansion()


# req: 10-keyword-and-synonym-expansion-for-keyword-search/TR-7
def test_retrieve_fuses_fts_hits_found_only_by_a_paraphrase(env, cfg):
    conn, ws, d1, d2 = env
    ids = db.add_chunks(conn, d1, ws, [(0, 1, "alpha words", E1), (1, 1, "omega words", E1)])
    cfg.update(rerank=False, paraphrase=True, paraphrases=1, sources=1)
    out = retrieve(conn, cfg, ws, "nothing", embed=fake_embed(E1), rerank=no_rerank,
                   chat_stream=fake_chat("PHRASING: omega"))
    assert out[0]["id"] == ids[1]


# req: 10-keyword-and-synonym-expansion-for-keyword-search/TR-7
def test_retrieve_embeds_original_and_paraphrases_in_one_call(env, cfg):
    conn, ws, d1, d2 = env
    db.add_chunks(conn, d1, ws, [(0, 1, "text", E1)])
    cfg.update(rerank=False, paraphrase=True, paraphrases=2)
    calls = []
    def embed(cfg, texts, input_type):
        calls.append(texts)
        return [E1 for _ in texts]
    retrieve(conn, cfg, ws, "q", embed=embed, rerank=no_rerank, chat_stream=fake_chat("PHRASING: p1\nPHRASING: p2"))
    assert calls == [["q", "p1", "p2"]]


# req: 10-keyword-and-synonym-expansion-for-keyword-search/TR-7
def test_retrieve_reranks_with_original_question(env, cfg):
    conn, ws, d1, d2 = env
    db.add_chunks(conn, d1, ws, [(0, 1, "text", E1)])
    cfg.update(rerank=True, paraphrase=True)
    queries = []
    def rerank(cfg, query, passages):
        queries.append(query)
        return [(i, 1.0) for i in range(len(passages))]
    reply = "INTENT: the intent\nPHRASING: p1\nKEYWORDS: weld; seam"
    retrieve(conn, cfg, ws, "q", embed=fake_embed(E1), rerank=rerank, chat_stream=fake_chat(reply))
    assert queries == ["q"]


# req: 10-keyword-and-synonym-expansion-for-keyword-search/TR-13
@pytest.mark.parametrize("paraphrases", [0, -1])
def test_expand_with_paraphrases_below_one_gives_keywords_only(cfg, paraphrases):
    cfg.update(paraphrase=True, paraphrases=paraphrases)
    calls = []
    out = expand(cfg, "q", fake_chat("PHRASING: p1\nPHRASING: p2\nKEYWORDS: weld; pipe", calls))
    assert out == Expansion((), ("weld", "pipe"))
    assert "- 0 lines PHRASING" in calls[0][1][0]["content"]


# req: 10-keyword-and-synonym-expansion-for-keyword-search/TR-5
def test_expand_reads_phrasings_and_keyword_terms_from_labelled_lines(cfg):
    cfg.update(paraphrase=True, paraphrases=2)
    reply = "INTENT: welding rules\nPHRASING: p1\nPHRASING: p2\nKEYWORDS: зварювання; weld\nKEYWORDS: труба; pipe"
    assert expand(cfg, "q", fake_chat(reply)) == Expansion(("p1", "p2"), ("зварювання", "weld", "труба", "pipe"))


# req: 10-keyword-and-synonym-expansion-for-keyword-search/TR-5
def test_expand_drops_the_intent_line(cfg):
    cfg.update(paraphrase=True)
    out = expand(cfg, "q", fake_chat("INTENT: secret intent\nPHRASING: p1\nKEYWORDS: weld"))
    assert out == Expansion(("p1",), ("weld",))


# req: 10-keyword-and-synonym-expansion-for-keyword-search/TR-5
def test_expand_splits_a_keyword_group_on_semicolons_and_drops_blank_terms(cfg):
    cfg.update(paraphrase=True)
    assert expand(cfg, "q", fake_chat("KEYWORDS: weld; ; seam ;;joint;")).keywords == ("weld", "seam", "joint")


# req: 10-keyword-and-synonym-expansion-for-keyword-search/TR-5
def test_expand_dedupes_keyword_terms_across_groups_ignoring_case(cfg):
    cfg.update(paraphrase=True)
    reply = "KEYWORDS: Weld; seam\nKEYWORDS: weld; pipe; SEAM"
    assert expand(cfg, "q", fake_chat(reply)).keywords == ("Weld", "seam", "pipe")


# req: 10-keyword-and-synonym-expansion-for-keyword-search/TR-5
def test_expand_caps_keyword_groups_at_keyword_groups(cfg):
    cfg.update(paraphrase=True, keyword_groups=2)
    reply = "KEYWORDS: a; b\nKEYWORDS: c\nKEYWORDS: d"
    assert expand(cfg, "q", fake_chat(reply)).keywords == ("a", "b", "c")


# req: 10-keyword-and-synonym-expansion-for-keyword-search/TR-5
def test_expand_accepts_lowercase_labels_bullets_and_bold(cfg):
    cfg.update(paraphrase=True, paraphrases=3)
    reply = "- phrasing: p1\n2. **PHRASING:** p2\n**Phrasing**: p3\n* **keywords:** weld; seam"
    assert expand(cfg, "q", fake_chat(reply)) == Expansion(("p1", "p2", "p3"), ("weld", "seam"))


# req: 10-keyword-and-synonym-expansion-for-keyword-search/TR-5
def test_expand_ignores_unlabelled_lines(cfg):
    cfg.update(paraphrase=True)
    assert expand(cfg, "q", fake_chat("Here you go:\np1\nweld; seam\nPHRASING: p2")) == Expansion(("p2",))


# req: 10-keyword-and-synonym-expansion-for-keyword-search/TR-5
def test_expand_without_keyword_lines_keeps_the_phrasings(cfg):
    cfg.update(paraphrase=True, paraphrases=2)
    assert expand(cfg, "q", fake_chat("INTENT: i\nPHRASING: p1\nPHRASING: p2")) == Expansion(("p1", "p2"))


# req: 10-keyword-and-synonym-expansion-for-keyword-search/TR-5
def test_expand_without_phrasing_lines_keeps_the_keywords(cfg):
    cfg.update(paraphrase=True)
    assert expand(cfg, "q", fake_chat("INTENT: i\nKEYWORDS: weld; seam")) == Expansion((), ("weld", "seam"))


# req: 10-keyword-and-synonym-expansion-for-keyword-search/TR-6
def test_retrieve_runs_on_the_original_question_when_expansion_fails(env, cfg):
    conn, ws, d1, d2 = env
    ids = db.add_chunks(conn, d1, ws, [(0, 1, "weld rules", E1)])
    cfg.update(rerank=False, paraphrase=True)
    embedded = []
    def embed(cfg, texts, input_type):
        embedded.append(texts)
        return [E2 for _ in texts]
    def bad(*a, **k):
        raise RuntimeError("down")
        yield
    out = retrieve(conn, cfg, ws, "weld", embed=embed, rerank=no_rerank, chat_stream=bad)
    assert embedded == [["weld"]]
    assert [r["id"] for r in out] == ids


# req: 10-keyword-and-synonym-expansion-for-keyword-search/TR-7
def test_retrieve_finds_a_chunk_matched_only_by_a_keyword_synonym(env, cfg):
    conn, ws, d1, d2 = env
    ids = db.add_chunks(conn, d1, ws, [(0, 1, "alpha words", E1), (1, 1, "трубопровід words", E2)])
    cfg.update(rerank=False, paraphrase=True, sources=1)
    out = retrieve(conn, cfg, ws, "nothing", embed=fake_embed(E1), rerank=no_rerank,
                   chat_stream=fake_chat("INTENT: pipes\nKEYWORDS: труба; трубопровід; pipe"))
    assert out[0]["id"] == ids[1]


# req: 10-keyword-and-synonym-expansion-for-keyword-search/TR-7
def test_retrieve_does_not_embed_keywords_or_the_intent(env, cfg):
    conn, ws, d1, d2 = env
    db.add_chunks(conn, d1, ws, [(0, 1, "text", E1)])
    cfg.update(rerank=False, paraphrase=True)
    calls = []
    def embed(cfg, texts, input_type):
        calls.append(texts)
        return [E1 for _ in texts]
    reply = "INTENT: the intent\nPHRASING: p1\nKEYWORDS: weld; seam"
    retrieve(conn, cfg, ws, "q", embed=embed, rerank=no_rerank, chat_stream=fake_chat(reply))
    assert calls == [["q", "p1"]]


# req: 10-keyword-and-synonym-expansion-for-keyword-search/TR-7
def test_keyword_ranking_without_keywords_is_empty(env, cfg):
    conn, ws, d1, d2 = env
    db.add_chunks(conn, d1, ws, [(0, 1, "weld", E1)])
    assert keyword_ranking(conn, cfg, ws, ()) == []


# req: 10-keyword-and-synonym-expansion-for-keyword-search/TR-7
def test_fuse_with_and_without_the_keyword_ranking_from_one_expansion(env, cfg):
    conn, ws, d1, d2 = env
    ids = db.add_chunks(conn, d1, ws, [(0, 1, "alpha", E1), (1, 1, "seam", E2)])
    expansion = Expansion(keywords=("seam",))
    base = search_rankings(conn, cfg, ws, ["alpha"], fake_embed(E1))
    without = fuse(cfg, base)
    with_keywords = fuse(cfg, base + [keyword_ranking(conn, cfg, ws, expansion.keywords)])
    assert [i for i, _ in without] == [ids[0], ids[1]]
    assert dict(with_keywords)[ids[1]] == pytest.approx(dict(without)[ids[1]] + 1 / 61)


# req: 10-keyword-and-synonym-expansion-for-keyword-search/TR-9
def test_keyword_eval_set_has_about_30_questions_in_three_languages():
    import json
    from collections import Counter
    from pathlib import Path

    entries = json.loads((Path(__file__).parent / "data" / "keyword_eval.json").read_text(encoding="utf-8"))
    langs = Counter(e["lang"] for e in entries)
    assert 27 <= len(entries) <= 33
    assert set(langs) == {"uk", "ru", "en"}
    assert all(langs[lang] >= 8 for lang in ("uk", "ru", "en"))
    assert all(set(e) == {"lang", "question", "document", "pages"} and e["question"] and e["document"]
               and e["pages"] and all(isinstance(p, int) for p in e["pages"]) for e in entries)
