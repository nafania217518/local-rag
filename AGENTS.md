# LocalRAG

At session start, load the armory `core` skill when its rules are not in your context.

## What this is

LocalRAG is a desktop app that answers questions about the user's own documents (PDF,
DOCX, TXT). Documents and the search index live in one local SQLite file (FTS5 plus
sqlite-vec, merged with reciprocal rank fusion, then reranked). The language models run
on build.nvidia.com through its OpenAI-compatible API: embeddings, rerank, OCR for
scanned pages and the streamed, cited answer. The backend is FastAPI; the UI is a static
page shown in pywebview; `localrag` starts both. It works in Ukrainian, Russian and
English. A Windows x64 one-file exe is built with PyInstaller. The design is in
`docs/DESIGN.md`.

## Commands

Format: `<check>: <command>`; `implement` section Gates per commit says how to run
them.

- unit: `uv run pytest -m "not ui and not live" --no-cov`
- integration: `uv run pytest -m live --no-cov`
- e2e: `uv run pytest -m ui --no-cov`
- coverage: `uv run pytest -m "not live"`
- run: `uv run localrag`

## Gates

Format: `<gate>: on`, or `<gate>: off <reason>`.

- lint: off (no linter configured)
- owasp: off (no scanner configured)
- unit: on
- integration: on
- coverage: on
- contract: off (not chosen)
- functional: off (not chosen)
- e2e: on

## Test levels

Only what this project adds to or changes in the armory `test` skill, section Levels.

Unit style: sociable

- integration: the `live` tests call the real build.nvidia.com service and need
  `NVIDIA_API_KEY`.
- e2e: the `ui` tests drive the local app page in headless Chromium through Playwright;
  there is no deployed stack. It is the test level of the frontend surface
  (`src/localrag/static/`).

## Tracker

Format: `<verb>: <binding>`, one line per bound verb of the armory `core` skill,
section Host terms, term `tracker`, then `target: <the project's own tracker>`, then
`closes by PR merge: <no | the closing reference form, e.g. Closes #<id>>`, then,
with the label verbs bound, `label <role>: <label>`, one per role of the armory
`triage` skill.

- read an item: `gh issue view <id> --json title,body,comments,labels,state,assignees`
- list an item's children: `gh issue view <id> --json subIssues`, then
  `gh issue view <child> --json state,assignees,blockedBy` per child
- create an item: `gh issue create --title <title> --body-file <file>`
- edit an item body: `gh issue edit <id> --body-file <file>`
- link a child to its parent: `gh issue edit <child> --parent <parent>`
- add a blocked-by edge: `gh issue edit <id> --add-blocked-by <blocker>`
- claim: `gh issue edit <id> --add-assignee @me`
- comment: `gh issue comment <id> --body-file <file>`
- set the status: open `gh issue reopen <id>`; done
  `gh issue close <id> --reason completed`; dropped
  `gh issue close <id> --reason "not planned"`
- list items: `gh issue list --state <open|closed|all> --label <label> --json number,title,labels,state`
- list labels: `gh label list`
- set labels: `gh issue edit <id> --add-label <label> --remove-label <label>`
- create label: `gh label create <name>`
- target: GitHub Issues of `nafania217518/local-rag`
- closes by PR merge: `Closes #<id>`
- label needs-triage: `needs-triage`
- label needs-info: `question`
- label accepted: `accepted`
- label bug: `bug`
- label enhancement: `enhancement`

## Rules

Project-only rules. The armory skills hold the rest.

- Documentation lives in `docs/`
- All new and changed code MUST follow SOLID:
  - Single responsibility: every class has only one responsibility.
  - Open-closed: entities are open for extension, but closed for modification.
  - Liskov substitution: a subtype works wherever its base type is used.
  - Interface segregation: clients do not depend on interface methods they do not use.
  - Dependency inversion: high-level modules do not depend on low-level modules; both
    depend on abstractions, and abstractions do not depend on details.
- Research MUST check at least 4 sources for each claim.
- A coding spawn MUST run on Opus with low effort.

## Glossary

Format: `Term : one-line meaning`, plus `. Avoid: <synonyms>` when a synonym was rejected.
A line MUST hold only a term of the project's domain, with no implementation detail.

- Expansion : the intent, the paraphrases and the keyword groups that the chat model writes for one question
- Intent : one line that says what the user wants to find; it guides the keywords and is never shown or stored
- Keyword group : one domain term of the question with its synonyms
- Question set : the fixed list of questions, each with its expected document and pages, that measures search quality
- Hit rate : the share of the question set whose expected page is among the top candidates

## Caveats

Format: the caveat line of the armory `core` skill, section Reports

- Scanned pages go to nemotron-3.5-parse through a multipart parse endpoint on ai.api.nvidia.com (not chat completions); options accept only decoding.max_output_tokens up to 9000; the response is {content, finish_reason, usage} : docs/DESIGN.md
- Embeddings need input_type passage/query or retrieval quality drops : docs/DESIGN.md
- Rerank endpoint is on ai.api.nvidia.com, not the OpenAI base URL : docs/DESIGN.md
- build.nvidia.com free tier is 40 requests/min across all models; every call goes through the throttle in nvidia.py : docs/DESIGN.md
- Hosted model ids churn (deepseek-v4-flash went 410); default chat model is nemotron-3.5-super-vl-120b-a12b, live-tested on Ukrainian and Russian : docs/DESIGN.md
- vec0 table fixes embedding dims; changing the embed model needs a full re-index : docs/DESIGN.md
- Cross-lingual retrieval needs no translation; the chat model language list is the limit : docs/DESIGN.md
- PyMuPDF is AGPL; acceptable because the project is non-commercial : docs/DESIGN.md
- macOS system Python is 3.9 and blocks SQLite extensions (sqlite-vec needs them); use uv Python 3.12+ : docs/DESIGN.md
- Windows needs WebView2 Runtime for pywebview : docs/DESIGN.md
- PyInstaller: webview is imported via importlib (hiddenimports), sqlite-vec has no hook (collect_dynamic_libs), the PyMuPDF 2.0 layout stack (170 MB) is excluded : docs/DESIGN.md
- Windows bundle has no console; the launcher must not print : docs/DESIGN.md
- Text-layer quality heuristic flags about 2 % of pages on real scans (figures, forms, covers); it cannot catch well-formed wrong words : docs/DESIGN.md
- nemotron-3.5-parse returned the full large table that v1 dropped; the redo-mode guard stays (OCR replaces the layer only when at least half as long and scoring higher) : docs/DESIGN.md
- nemotron-3.5-parse returns tables as LaTeX tabular with \thead and \mbox, puts box and class tokens inline, and still mixes Latin look-alikes into Cyrillic words : docs/DESIGN.md
- A saved parse_model of nemotron-parse or nemotron-parse-2.0 is migrated to the default at load, because those names 404 on the parse URL : docs/DESIGN.md
- Only a Windows x64 one-file exe is published (zip on a GitHub release, built by hand; CI runs tests only, on pull requests); one-file mode unpacks to a temp folder at every start : docs/DESIGN.md
- All tunables live in config.DEFAULTS (no module constants); chunk and OCR settings apply at indexing time only, a changed document must be deleted and added again : docs/DESIGN.md
- Hosted nemotron-3.5-embed-8b returns 500 `service inference failed` for some multi-input batches (2 to 12 inputs failed every time, 16 passed, 64 flaky) and at random for long single inputs (about 40 % for a 900-token chunk, never for 350 tokens, never on embed-1b; probe 2026-10-30); the client splits a failed batch in halves down to single texts and gives a single text `embed_attempts` attempts (setting, default 50) : docs/DESIGN.md
- Logs go to stderr only (no log file), so the Windows exe, which has no console, shows none : docs/DESIGN.md
- The chat model copies the language of a bare example in the paraphrase prompt (an English question got Ukrainian paraphrases); the example is a full question pair after the rules, with an explicit "language of the question, not of the example" clause : docs/DESIGN.md
- Question paraphrasing costs one extra chat request per question (four requests per question with rerank and the answer); a failed paraphrase call falls back to the original question : docs/DESIGN.md
- A Rosetta crash (rosetta error: invalid gdt selector index 5) can kill Wine's pip during the Docker build at random (2 of 3 runs on 2026-10-30, same image and settings as a passing run the day before); a rerun passes; scripts/build-windows.sh detects it from the build log : docs/DESIGN.md
- `db.connect` writes the schema on open, so the hit-rate measurement runs on a backup copy of the app database : tests/test_live_retrieval.py
- The expansion parser ignores unlabelled lines, so a reply in the old one-phrasing-per-line format gives no paraphrases : src/localrag/retrieval.py
- Keyword terms match exact tokens only; inflected Ukrainian and Russian forms miss until word forms ship : src/localrag/retrieval.py
- The expansion prompt has two language sentences: the clause after the example covers the intent and the phrasings, and the next sentence covers the keywords, which must also be in English : src/localrag/retrieval.py
- FTS5 bm25 scores each OR'd phrase on its own and counts a repeated phrase twice, so keyword terms are deduplicated and run as a separate ranking : src/localrag/db.py
- `chunk_overlap` applies only inside one page; chunks of adjacent pages share no text, so an answer across a page break is split between two pages : src/localrag/index.py
- OCR text has spacing noise inside numbers ("2 0 мм" for "20 мм"), so exact-token keyword search can miss numeric answers : —
- The real chat model copied the Ukrainian example into the keywords of an English question; one more sentence after the language clause of the expansion prompt fixes it : src/localrag/retrieval.py
- `expand` sends the chat request also when `paraphrases` is below 1; only the `paraphrase` toggle stops it : src/localrag/retrieval.py
- The hit-rate measurement stops before any API request when a document of the question set is missing or the documents sit in more than one workspace : tests/test_live_retrieval.py
- On the 757-chunk НАЕК corpus the question set hits 30/30 in the fused top 20 with and without the keyword ranking (2026-11-06), so that cutoff cannot show a gain or a loss : tests/test_live_retrieval.py
- On the question set the keyword-term ranking alone hits 5 to 6 of 30 against 14 of 30 for the question ranking (two runs, 2026-11-06) : docs/DESIGN.md
- The hit-rate measurement runs on `config.DEFAULTS` (paraphrases 2, sources 6), not on the user's saved settings, so its numbers describe a new install : tests/test_live_retrieval.py
- In the second measurement run of 2026-11-06 the keyword ranking lowered the fused mean reciprocal rank from 0.795 to 0.740 with every hit rate unchanged; the first run gave 0.760 against 0.761 : docs/DESIGN.md
- The autouse fixture `no_configure_logging` lets the real `configure_logging` run only for the tests it lists by name, so a rename of either `test_configure_logging_*` test makes it fail : tests/test_app.py
