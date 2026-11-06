# Plan: Keyword and synonym expansion for keyword search

Parent: [Keyword and synonym expansion for keyword search](https://github.com/nafania217518/local-rag/issues/10) (no milestone).

## Goal

Keyword search finds passages that use another term than the question. The chat model
writes the keywords of a question with their synonyms in the paraphrase request, and
keyword search runs them as one more ranking.

## Requirements

The `BR` items live in the parent item. Each new test carries `# req: 10-keyword-and-synonym-expansion-for-keyword-search/TR-n`.

- TR-1 (BR-4): `db.fts_search_terms(conn, workspace_id, terms, k)` MUST OR each term as
  one quoted FTS5 phrase (inner quotes doubled), filter by workspace, order by rank and
  take the top k. `fts_search` MUST delegate to it and keep its behaviour.
  - tests: `tests/test_db.py`: `test_fts_search_terms_matches_a_multi_word_term_only_as_a_phrase`,
    `test_fts_search_terms_ranks_a_chunk_with_more_terms_first`,
    `test_fts_search_terms_does_not_match_another_word_form`,
    `test_fts_search_terms_escapes_double_quotes_in_a_term`,
    `test_fts_search_terms_with_no_terms_returns_empty`,
    `test_fts_search_terms_returns_only_chunks_of_the_workspace`
- TR-2 (BR-8): `config.DEFAULTS` MUST hold `keyword_groups` 4 and `keyword_synonyms` 3 (int).
  - tests: `tests/test_config.py::test_defaults_cover_every_tunable` (rework)
- TR-3 (BR-8): the Settings dialog MUST show the inputs `keyword_groups` (min 1, max 10)
  and `keyword_synonyms` (min 0, max 10) in the Retrieval group and save them as integers.
  - tests: `tests/test_ui.py::test_settings_saves_keyword_limits` (e2e)
- TR-4 (BR-1, BR-2, BR-3, BR-5): `retrieval.expand` MUST replace `paraphrase()`. With
  `paraphrase` on, it MUST send one chat request with reasoning "none". The prompt MUST
  ask, in this order, for one INTENT line, `paraphrases` PHRASING lines and at most
  `keyword_groups` KEYWORDS lines; each KEYWORDS line holds one domain term with at most
  `keyword_synonyms` synonyms in the question's language, then the same in English, or
  English only for an English question. The prompt MUST keep the formal-terms rule, the
  Ukrainian example question after the rules and the language clause. With `paraphrase`
  off, `expand` MUST send no request.
  - tests: `tests/test_retrieval.py`: `test_expand_asks_chat_model_once_with_no_reasoning` (rework),
    `test_expand_prompt_asks_for_formal_technical_terms` (rework),
    `test_expand_prompt_example_is_a_question_and_the_language_rule_follows_it` (rework),
    `test_expand_prompt_asks_for_intent_then_phrasings_then_keywords`,
    `test_expand_prompt_asks_for_keywords_in_the_question_language_and_in_english`,
    `test_expand_prompt_takes_the_group_and_synonym_caps_from_config`,
    `test_expand_off_does_not_call_chat_model` (rework)
- TR-5 (BR-1, BR-2, BR-6): `expand` MUST parse the reply into `Expansion`: PHRASING lines
  become paraphrases (quotes stripped, copies of the question and duplicates dropped,
  capped at `paraphrases`); the first `keyword_groups` KEYWORDS lines are split on `;`,
  blank terms dropped, duplicates removed across groups ignoring case; INTENT lines are
  dropped; labels match case-insensitively after a bullet or number and inside markdown
  bold; unlabelled lines are ignored; a reply with only one kind of line keeps that kind.
  - tests: `tests/test_retrieval.py`: `test_expand_reads_phrasings_and_keyword_terms_from_labelled_lines`,
    `test_expand_drops_the_intent_line`,
    `test_expand_splits_a_keyword_group_on_semicolons_and_drops_blank_terms`,
    `test_expand_dedupes_keyword_terms_across_groups_ignoring_case`,
    `test_expand_caps_keyword_groups_at_keyword_groups`,
    `test_expand_accepts_lowercase_labels_bullets_and_bold`,
    `test_expand_strips_numbering_blanks_and_duplicates_and_caps_at_n` (rework),
    `test_expand_ignores_unlabelled_lines`,
    `test_expand_without_keyword_lines_keeps_the_phrasings`,
    `test_expand_without_phrasing_lines_keeps_the_keywords`
- TR-6 (BR-6): when the chat request raises, `expand` MUST return an empty `Expansion`,
  and `retrieve` MUST search and embed the original question alone.
  - tests: `tests/test_retrieval.py`: `test_expand_failure_returns_an_empty_expansion` (rework),
    `test_retrieve_runs_on_the_original_question_when_expansion_fails`
- TR-7 (BR-4): `retrieve` MUST add exactly one FTS ranking of `Expansion.keywords`
  (`fts_search_terms`, top `candidates`) to the RRF input. It MUST NOT embed the keywords,
  MUST NOT send them to rerank, and MUST NOT let the intent text reach FTS, embed or
  rerank. `retrieve` MUST be composed of the public steps `search_rankings`,
  `keyword_ranking`, `fuse` and `pick_sources`.
  - tests: `tests/test_retrieval.py`: `test_retrieve_finds_a_chunk_matched_only_by_a_keyword_synonym`,
    `test_retrieve_does_not_embed_keywords_or_the_intent`,
    `test_retrieve_reranks_with_original_question` (rework),
    `test_keyword_ranking_without_keywords_is_empty`,
    `test_fuse_with_and_without_the_keyword_ranking_from_one_expansion`,
    `test_retrieve_fuses_fts_hits_found_only_by_a_paraphrase` (rework),
    `test_retrieve_embeds_original_and_paraphrases_in_one_call` (rework)
- TR-8 (BR-2, BR-3): with the real chat model, a Ukrainian or Russian question MUST get
  exactly one INTENT line before the first KEYWORDS line, and keyword terms in Cyrillic
  and in Latin script. An English question MUST get keyword terms with no Cyrillic.
  - tests: `tests/test_live_languages.py`: `test_live_expansion_states_intent_first_and_keywords_in_both_languages`
    (uk, ru), `test_live_expansion_of_an_english_question_has_english_keywords_only`
- TR-9 (BR-7): `tests/data/keyword_eval.json` MUST hold 27 to 33 entries
  `{lang, question, document, pages}` with at least 8 each of `uk`, `ru` and `en`. The
  main session drafts it from the three НАЕК standards of the app database (read-only),
  with everyday wording where the standard uses a formal term. The user MUST approve the
  list before the first measurement.
  - tests: `tests/test_retrieval.py::test_keyword_eval_set_has_about_30_questions_in_three_languages`
- TR-10 (BR-7): on the question set, the hit rate with the keyword ranking MUST NOT be
  lower than without it. A hit is an expected (document, page) among the fused top
  `rerank_candidates`, before rerank. Both variants MUST come from one expansion per
  question. The test MUST print both rates, the top-`sources` hit rate after rerank for
  both variants, and the count of questions that got keywords.
  - tests: `tests/test_live_retrieval.py::test_live_keyword_ranking_does_not_lower_the_fused_hit_rate`
    (live; skipped without `NVIDIA_API_KEY` or `LOCALRAG_EVAL_DB`)
- TR-11 (BR-1 to BR-8): `docs/DESIGN.md` MUST describe the new behaviour: section 4.3
  (expansion step, keyword-ranking step, the cost line moved below the list), section 4.7
  (the two settings), section 8 (caveats) and the dated measurement result.
  - tests: none, documentation; TR-1 to TR-10 prove the behaviour it describes

Gates and verification:

- unit: `uv run pytest -m "not ui and not live" --no-cov`
- e2e: `uv run pytest -m ui --no-cov` (TR-3)
- coverage: `uv run pytest -m "not live"`
- integration: `uv run pytest -m live --no-cov` with `NVIDIA_API_KEY` (TR-8)
- measurement: `LOCALRAG_EVAL_DB="$HOME/Library/Application Support/LocalRAG/localrag.db" uv run pytest -m live tests/test_live_retrieval.py --no-cov -s`
  with `NVIDIA_API_KEY` set (TR-10); the printed rates go into `docs/DESIGN.md`.
- Manual: `uv run localrag`, ask a question with an everyday term, check that the
  sources hold the passage with the formal term.

## Test scope

- `tests/test_retrieval.py::test_paraphrase_asks_chat_model_once_for_n_lines_with_no_reasoning` : rework : `paraphrase` becomes `expand`; the reply needs PHRASING labels.
- `tests/test_retrieval.py::test_tr24_paraphrase_prompt_asks_for_formal_technical_terms` : rework : calls `expand`; the name drops the `tr24_` prefix.
- `tests/test_retrieval.py::test_tr25_paraphrase_prompt_example_is_a_question_and_the_language_rule_follows_it` : rework : calls `expand`; the name drops the `tr25_` prefix.
- `tests/test_retrieval.py::test_paraphrase_strips_numbering_blanks_and_duplicates_and_caps_at_n` : rework : the reply lines get PHRASING labels.
- `tests/test_retrieval.py::test_paraphrase_failure_returns_no_rewrites` : rework : expects an empty `Expansion`.
- `tests/test_retrieval.py::test_paraphrase_off_does_not_call_chat_model` : rework : calls `expand`.
- `tests/test_retrieval.py::test_retrieve_fuses_fts_hits_found_only_by_a_paraphrase` : rework : the unlabelled reply would parse to nothing.
- `tests/test_retrieval.py::test_retrieve_embeds_original_and_paraphrases_in_one_call` : rework : the reply gets PHRASING labels.
- `tests/test_retrieval.py::test_retrieve_reranks_with_original_question` : rework : the reply gets PHRASING and KEYWORDS lines, else the test passes with no evidence.
- `tests/test_config.py::test_defaults_cover_every_tunable` : rework : two new keys.

## Risks

- The measured hit rate drops, or shows no gain. On a drop the code run stops and asks
  the user (a requirement change). No gain is reported with the numbers; BR-7 asks only
  for "not lower".
- The model skips the labels: strict parsing drops the whole reply, paraphrases too. The
  measurement prints how many questions got keywords.
- The reply is 2 to 3 times longer, so the answer starts later.
- Exact tokens miss inflected Ukrainian and Russian forms; the gain stays small until the
  word-forms issue ships. English keywords match only Latin abbreviations in a Ukrainian
  corpus.
- 30 questions is a small set: one question is 3.3 percentage points. The paired design
  removes the expansion noise from the comparison, not the noise between runs.
- The measurement is valid only while the default embed model is the one that indexed
  the database.
- The question set names three document files and pages in a public repo; it holds no
  document text.

## Stack choices

No new dependency. Python stdlib `sqlite3` FTS5, the existing chat endpoint.

Design choices:

- Data shape, `src/localrag/retrieval.py`: frozen dataclass
  `Expansion(paraphrases: tuple[str, ...] = (), keywords: tuple[str, ...] = ())`. The
  groups are flattened to distinct terms. There is no intent field, so the intent cannot
  be searched, shown or stored.
- Reply format, labelled lines with fixed English labels:
  ```
  INTENT: <what the user wants to find, one line>
  PHRASING: <phrasing>                                   ({n} lines)
  KEYWORDS: <term>; <synonym>; ...; <English term>; <English synonym>   (at most {groups} lines)
  ```
  The rules come first, then the Ukrainian example question, then the clause "Write the
  intent and the phrasings in the language of the question, not in the language of the
  example".
- Functions, each with one job:
  - `db.fts_search_terms(conn, ws, terms, k)`: the only builder of the MATCH string.
    `db.fts_search` ([db.py:153](../../src/localrag/db.py:153)) delegates to it with `query.split()`.
  - `retrieval.expand(cfg, question, chat_stream) -> Expansion` replaces `paraphrase()`
    ([retrieval.py:24](../../src/localrag/retrieval.py:24)); it calls the private pure parser.
  - `search_rankings`, `keyword_ranking`, `fuse`, `pick_sources`: the public steps of
    `retrieve()`, which keeps its signature ([chat.py:25](../../src/localrag/chat.py:25) does not change).
    The measurement fuses with and without the keyword ranking from one expansion.
- Keyword query: a flat OR of every distinct term, each a quoted phrase
  (`"t1" OR "a b" OR "t2"`). AND of per-group OR groups is rejected: with exact tokens a
  chunk rarely holds a form from every group, so the ranking would be empty.
- Settings: two number inputs in the Retrieval group after "Paraphrases per question":
  `keyword_groups` (1 to 10) and `keyword_synonyms` (0 to 10). `SettingsIn`
  ([api.py:33](../../src/localrag/api.py:33)) is built from `DEFAULTS`, so the API needs no change.
- Measurement: a `live` test, gated on `NVIDIA_API_KEY` and `LOCALRAG_EVAL_DB`, like the
  `LOCALRAG_SCAN_PDF` test. It copies the database with `Connection.backup` into
  `tmp_path`, uses `config.DEFAULTS`, and costs about 120 requests (about 3 minutes).

## Out of scope

- Word forms: stemming, prefix queries, LLM stems (the separate issue).
- A static thesaurus; a custom FTS5 tokenizer (apsw).
- Intent routing: skip retrieval, search modes.
- An own toggle for keyword expansion; stop-word removal from the question query.
- Keywords in the vector search or in the reranker input.

## Accepted tech debt

None.

## Overrides

None.

## Caveats

- `db.connect` writes the schema on open, so the measurement runs on a backup copy of the app database : tests/test_live_retrieval.py
- The expansion parser ignores unlabelled lines, so a reply in the old one-phrasing-per-line format gives no paraphrases : src/localrag/retrieval.py
- Keyword terms match exact tokens only; inflected Ukrainian and Russian forms miss until word forms ship : src/localrag/retrieval.py
- The expansion prompt's language clause covers only the intent and the phrasings, because the keywords must also be in English : src/localrag/retrieval.py
- FTS5 bm25 scores each OR'd phrase on its own and counts a repeated phrase twice, so keyword terms are deduplicated and run as a separate ranking : src/localrag/db.py
