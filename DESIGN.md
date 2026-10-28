# Local RAG Desktop — Design

Status: v1 implemented (phases 0 to 8), 2026-10-28. Decisions come from the design quiz and the sources in [Sources](#sources). Live-probe results are from the same day.

## 1. Goal

A desktop app for macOS and Windows. The user creates a workspace, uploads 100+ documents into it, and chats with a model that answers from those documents. All storage is local. All models come from build.nvidia.com through the OpenAI-compatible API.

## 2. Decisions (from the quiz)

| Topic | Decision |
|---|---|
| Use | Study project, non-commercial. AGPL and trial-tier terms are acceptable |
| Stack | One Python process: FastAPI backend + plain HTML/JS UI inside a pywebview window |
| Formats | PDF (text and scanned), DOCX, TXT/MD (free with the same pipeline) |
| OCR | NVIDIA hosted model, no local binaries |
| Workspaces | A workspace owns documents and one index. A workspace has many chat sessions |
| Answer features | Citations, streaming, reranking, hybrid search |
| OS | macOS + Windows |
| Languages | Documents and questions in Ukrainian, Russian and English |
| Chat model | Chosen after research, see 3. Model name is a setting |
| API key | Plain config file in the app data folder |

## 3. Models and endpoints

All calls use one API key (`NVIDIA_API_KEY`) as a Bearer token.

| Role | Model | Endpoint | Notes |
|---|---|---|---|
| Chat | `nvidia/nemotron-3.5-super-vl-120b-a12b` | `POST https://integrate.api.nvidia.com/v1/chat/completions` | 1M context, 120B MoE with 12B active, free hosted endpoint (the backend serves the text Super 3.5 model). Live-tested 2026-10-28: correct Ukrainian and Russian answers with [n] citations in about 1 s. `reasoning_effort: "none"` accepted and disables thinking. `stream: true` for SSE |
| Embeddings | `nvidia/nemotron-3.5-embed-8b` | `POST https://integrate.api.nvidia.com/v1/embeddings` | Max 10240 tokens per input, 4096 dims, L2-normalized. Evaluated on 34 text languages (Ukrainian and Russian included). `input_type` is required: `passage` for indexing, `query` for questions. `truncate: END` |
| Rerank | `nvidia/nemotron-3.5-rerank-8b` | `POST https://ai.api.nvidia.com/v1/retrieval/nvidia/nemotron-3.5-rerank-8b/reranking` | Same 34 languages as the embed model. Not OpenAI-shaped: body is `{model, query:{text}, passages:[{text}], truncate:"END"}`, max 512 passages |
| OCR / page parse | `nvidia/nemotron-parse` | `POST https://integrate.api.nvidia.com/v1/chat/completions` | One page image per request as a base64 `image_url` content part. Returns the page as a `markdown_bbox` tool call: a JSON list of elements with `bbox`, `text`, `type` in reading order; the client joins the texts. `max_tokens` 3500. Live-tested 2026-10-28. `nemotron-parse-2.0` is the successor, but its hosted endpoint returned 502 upstream all day; the client also accepts plain `content`, so switching is a setting |

Why parse instead of the OCR NIM: the hosted `nemoretriever-ocr-v1` endpoint is deprecated, and `nemotron-ocr-v2` returns word boxes, not reading-order text. `nemotron-parse` runs through the same chat endpoint and returns page Markdown. One client, one auth path.

Rate limit: build.nvidia.com allows 40 requests per minute in total, across all models. `nvidia.py` enforces one global sliding-window throttle (`RPM = 40`, thread-safe) that every outgoing request passes through, and retries 429 and 5xx responses with exponential backoff (1, 2, 4, 8 s, or the `Retry-After` header), at most 5 attempts. The SDK's own retries are off. Consequence for ingestion: a scanned 30-page PDF costs about 31 requests (30 parse calls plus one embed batch), so about one such document per minute. The UI shows per-document status so this is visible, not silent.

The OpenAI Python SDK handles chat, embeddings and parse (`base_url="https://integrate.api.nvidia.com/v1"`, `extra_body={"input_type": ...}` for embeddings). Rerank is one `httpx.post`.

Why 8b and not 1b for embeddings: the 1b model is a distillation of the 8b model. On ViDoRe V3 text retrieval the 8b scores 60.65 nDCG@10 against 57.80 for the 1b. Both run on the same hosted endpoint, so the only local cost is vector size (4096 vs 2048 floats). For 100 documents that is about 160 MB on disk. Accuracy wins.

Chat model comparison for Ukrainian, Russian and English (all hosted on the catalog):

| Model | Ukrainian / Russian in official language list | Context | Notes |
|---|---|---|---|
| `nvidia/nemotron-3.5-super-vl-120b-a12b` | No. Post-training: FR, DE, IT, JA, ES, ZH | 1M | **Default.** Live probe 2026-10-28: correct Ukrainian and Russian RAG answers with citations in 1 s. Backend id `nemotron-3.5-super-120b-a12b` |
| `nvidia/nemotron-3.5-lightning-30b-a3b` | No. EN, ES, FR, DE, IT, JA | 1M | Fallback. Live probe: correct Ukrainian answer with citation in 1 s |
| `deepseek-ai/deepseek-v4-flash` | No official list; top-3 on a community Ukrainian leaderboard | 1M | Gone: catalog returns 410, end of life 2026-08-07. `deepseek-ai/deepseek-v4.1-flash` is listed but timed out twice (504, then no answer in 150 s) |
| `qwen/qwen3-next-80b-a3b-instruct` | Yes, both (Qwen3 report, 119 languages; Russian benchmarked) | 256K | Best language evidence, but no free hosted endpoint (user check, 2026-10-28) |
| `nvidia/nemotron-3-super-120b-a12b` | No. EN, FR, DE, IT, JA, ES, ZH | 1M | Live probe: 404 for this account |
| `nvidia/nemotron-3-ultra-550b-a55b` | No. EN, FR, ES, IT, DE, JA, KO, HI, PT, ZH | 1M | Largest, slowest, no gain for these languages |
| `z-ai/glm-5.3-flash` | No list published. Model card tags English and Chinese only | 1M | Live probe: no answer in 150 s |

All model names are settings, not code. Switching needs no re-index.

## 4. Architecture

```
pywebview window  ──HTTP/SSE──▶  FastAPI (uvicorn, background thread, 127.0.0.1:random port)
                                     │
                                     ├── api.py        routes
                                     ├── chat.py       prompt build, stream, citations
                                     ├── retrieval.py  FTS5 + vector + RRF + rerank
                                     ├── index.py      parse → chunk → embed → store
                                     ├── parsers.py    pdf / docx / txt
                                     ├── nvidia.py     chat, embed, rerank, parse clients
                                     ├── db.py         SQLite schema + queries
                                     └── config.py     app data dir, config.json
```

Package: `src/localrag/`. UI: `src/localrag/static/index.html` (vanilla HTML/CSS/JS, no build step). Launcher: `src/localrag/app.py` picks a free localhost port, starts uvicorn in a daemon thread, waits for `/health`, then `webview.create_window(url)` and `webview.start()`; closing the window stops the server. Every blocking collaborator is injectable so the launcher has unit tests without a window.

Packaging: `localrag.spec` at the repo root, one-folder PyInstaller build (`uv run pyinstaller localrag.spec`). Output: `dist/LocalRAG/` on both OS plus `dist/LocalRAG.app` on macOS (about 95 MB). The spec collects `static/`, the sqlite-vec `vec0` library (no community hook), names `webview` as a hidden import (app.py loads it through importlib), and excludes the optional PyMuPDF layout stack (`pymupdf.layout`, onnxruntime, numpy, networkx, psutil; about 170 MB and its ONNX resources are not collected). `LocalRAG --serve <port>` runs the server without a window; the CI `build` job uses it to probe `/health` on the real bundle on macOS and Windows and uploads `dist/` as an artifact. The bundle is not code-signed; macOS Gatekeeper and Windows SmartScreen will warn.

### 4.1 Storage (local)

One SQLite file: `<app data>/localrag.db`. Uploaded files are copied to `<app data>/files/<workspace_id>/<document_id>.<ext>`.

App data dir: `~/Library/Application Support/LocalRAG` (macOS), `%APPDATA%\LocalRAG` (Windows). Resolved in `config.py` only (SSOT).

```sql
workspaces(id, name, created_at)
documents(id, workspace_id, name, path, status, pages, error, created_at)      -- status: pending|indexed|failed
chunks(id, document_id, workspace_id, ord, page, text)
chunks_fts  FTS5 virtual table(text, content=chunks, content_rowid=id)         -- keyword index
chunks_vec  vec0 virtual table(chunk_id integer primary key, workspace_id integer partition key,
                               embedding float[4096] distance_metric=cosine)   -- sqlite-vec
sessions(id, workspace_id, title, created_at)
messages(id, session_id, role, content, citations JSON, created_at)
```

Vector search: `sqlite-vec` (`pip install sqlite-vec`, loaded with `sqlite_vec.load(conn)`). KNN query: `SELECT chunk_id, distance FROM chunks_vec WHERE embedding MATCH ? AND workspace_id = ? AND k = 30`. Vectors are stored as float32 blobs (`serialize_float32`). The embedding dimension is fixed in the table; changing the embed model means re-indexing.

Verified on this machine: sqlite-vec v0.1.9 loads under uv-managed Python 3.12 on macOS arm64 and answers a KNN query. PyPI ships a `win_amd64` wheel. The macOS system Python blocks `enable_load_extension`, so the app always runs on uv-managed Python.

FTS5 is compiled into the Python 3.12+ `sqlite3` module on macOS and Windows. Startup tests assert both FTS5 and sqlite-vec.

### 4.2 Ingestion (`index.py`)

1. Copy file into the files folder, insert `documents` row with `status=pending`.
2. Parse → list of `(page, text)`:
   - PDF: PyMuPDF `page.get_text()`. If a page has fewer than 20 characters of text, render it to PNG at 150 DPI and send it to the parse model (`parse_model` setting). Per-page decision, so mixed PDFs work.
   - DOCX: `python-docx` paragraphs and table cells, page = 1.
   - TXT/MD: whole file, page = 1.
   - Parsers are a dict `{".pdf": parse_pdf, ...}`; a new format is one new entry (open-closed).
3. Chunk: split each page's text into ~1500-character chunks on paragraph boundaries, 200-character overlap, keep `page`.
4. Embed in batches of 64 with `input_type=passage`.
5. Insert chunks, FTS rows and vec rows in one transaction. Set `status=indexed` or `failed` + `error`.

Runs in a background thread per document. The UI polls document status.

### 4.3 Retrieval (`retrieval.py`)

1. FTS5 `MATCH` on the question, top 30 (BM25 order).
2. Embed question with `input_type=query`, sqlite-vec KNN top 30 (cosine).
3. Reciprocal Rank Fusion (`k=60`), top 20.
4. Rerank the 20 with `nemotron-3.5-rerank-8b`, keep top 6.
5. Return chunks with `document name`, `page`, `text`, `score`.

Reranking is a setting (`rerank: true`). Off → RRF top 6.

### 4.4 Chat (`chat.py`)

Prompt: system message with rules ("answer only from the context, cite as [n], say when the context does not contain the answer") + numbered context blocks + last 10 messages of the session + the question.

Request parameters: `temperature: 0.2`, `max_tokens: 4096`, `reasoning_effort` from settings (default `none`). When reasoning is on, the stream carries `reasoning_content` deltas before `content`; the server forwards them as SSE `thinking` events and the UI shows them collapsed. `# ponytail: thinking events are forwarded raw, no token budget; add one if users turn reasoning on and answers get slow.`

Streaming: `stream=True` → FastAPI `StreamingResponse` with SSE events `token`, then one `citations` event with the chunk list, then `done`. The assistant message and its citations are saved to `messages` when the stream ends.

### 4.5 API (`api.py`)

```
GET/POST        /workspaces                  list, create
DELETE          /workspaces/{id}
GET/POST        /workspaces/{id}/documents   list, upload (multipart, many files)
DELETE          /workspaces/{id}/documents/{doc_id}
GET/POST        /workspaces/{id}/sessions    list, create
GET             /sessions/{id}/messages
POST            /sessions/{id}/chat          body {question} → SSE stream
GET/PUT         /settings                    config.json keys, see 4.7
GET             /health                      checks FTS5 and sqlite-vec
```

### 4.6 UI (`static/index.html`)

Three columns: workspaces + documents list (left), chat with a session selector (centre), sources panel for the selected answer (right). Settings dialog for the API key and model names. File picker is a plain `<input type="file" multiple>` (works inside pywebview and is testable with Playwright; the native dialog is not needed). Streaming through `fetch` + `ReadableStream`. Thinking text, when present, sits in a collapsed `<details>`. Browser tests run with pytest-playwright against the FastAPI app with fake model clients.

### 4.7 Settings (`config.py`)

`<app data>/config.json`: `{"api_key": "", "chat_model": "...", "embed_model": "...", "rerank_model": "...", "parse_model": "...", "rerank": true, "reasoning_effort": "none"}`. Defaults live in `config.py` only. The key is stored in plain text by the user's decision.

### 4.8 Languages

No translation layer. The embed and rerank models map text from different languages into one vector space, so a question in one language finds passages in another (cross-lingual retrieval). Both were evaluated on 34 text languages, Ukrainian and Russian included. The chat model answers in the language of the question when the system prompt says so.

Limits to know:
- FTS5 uses the `unicode61` tokenizer with `remove_diacritics 2`. It matches exact tokens in any script and does no stemming (the `porter` stemmer is English only). Inflected languages lose some keyword recall; the vector path and the reranker cover that.
- The chat model is the only language-limited component. Every Nemotron chat model lists Western European languages, Japanese and Chinese, none lists Ukrainian. The default is `nvidia/nemotron-3.5-super-vl-120b-a12b`: despite its official list it answered Ukrainian and Russian RAG questions correctly in the live probe (see the table in section 3). The system prompt tells the model to answer in the language of the question. Lightning is the fast fallback; Qwen3-Next if it gets a free endpoint.
- `tests/test_live_languages.py` (opt-in, needs `NVIDIA_API_KEY`): one Ukrainian question over a Ukrainian passage, one Russian, one cross-lingual (English question, Ukrainian passage). All three passed on 2026-10-28.
- Scanned pages: `nemotron-parse` is multilingual by model card. Not yet verified on a Ukrainian or Russian scanned page.

### 4.9 SOLID mapping

- Single responsibility: one module per step above.
- Open-closed: parsers dict; model names are config.
- Liskov / interface segregation / dependency inversion: `nvidia.py` exposes four small functions (`chat_stream`, `embed`, `rerank`, `parse_page`). `index.py`, `retrieval.py` and `chat.py` take them as parameters with those defaults. Tests pass fakes. No class hierarchy, no interface with one implementation.

## 5. Testing strategy (TDD, ≥90% per core module)

| Module | Test type | Fakes |
|---|---|---|
| db.py | unit, temp SQLite file, FTS5 match, vec0 KNN order | none |
| parsers.py | unit with fixture files (text PDF, scanned PDF, DOCX, TXT) | `parse_page` fake |
| chunker (in index.py) | unit | none |
| nvidia.py | unit with `respx` mocked HTTP, asserts URLs, headers, `input_type` | respx |
| index.py / retrieval.py / chat.py | unit, fake embed/rerank/chat functions | in-memory fakes |
| api.py | integration with FastAPI `TestClient`, SSE parsed | fakes injected |
| UI | browser tests with Playwright against the FastAPI server + fake NVIDIA | fakes |
| packaging | CI job builds the bundle on macOS and Windows and runs `/health` | real build |

Tools: `pytest`, `pytest-cov`, `respx`, `pytest-playwright`. Live NVIDIA calls are a separate opt-in test marked `live`, skipped without `NVIDIA_API_KEY`.

## 6. Implementation phases

Each phase: requirements → tests → code, by an Opus sub-agent (low effort). A phase is done when its tests pass and coverage holds. Each phase is one branch (`phase-N/<name>`) and one commit.

| # | Phase | Deliverable | Key requirements (= test names) |
|---|---|---|---|
| 0 | Scaffold | `pyproject.toml` (uv, Python 3.12), `src/localrag`, pytest, GitHub Actions matrix macOS + Windows | `test_fts5_available`, `test_sqlite_vec_loads`, `test_app_data_dir_per_os` |
| 1 | Storage | `db.py`, `config.py` | CRUD per table, FTS sync, config defaults, config round-trip |
| 2 | Clients | `nvidia.py` | each client sends the right URL/body/headers, `input_type` set, streaming yields tokens, HTTP errors raise |
| 3 | Ingestion | `parsers.py`, `index.py` | text PDF extracted per page, scanned page routed to parse, DOCX tables included, chunks keep page, overlap, status transitions, failure recorded |
| 4 | Retrieval | `retrieval.py` | FTS finds exact term, vector finds paraphrase, cross-lingual query finds passage (fake embed with shared vectors), RRF merges, rerank order applied, rerank off path |
| 5 | Chat | `chat.py` | prompt contains numbered context, history limited to 10, SSE order token→citations→done, `reasoning_content` deltas become `thinking` events, `reasoning_effort` passed from settings, message persisted with citations |
| 6 | API | `api.py` | one test per route, upload of many files, SSE end-to-end with fakes |
| 7 | UI + launcher | `static/index.html`, `app.py` | Playwright: create workspace, upload, see status, ask, see streamed answer and sources, open settings |
| 8 | Packaging | `localrag.spec`, `--serve` flag, CI `build` job with artifacts | `test_spec_*`, `test_main_serve_flag_*`, CI smoke `/health` on the bundle on both OS |

All phases are implemented; `uv run pytest` is the proof.

## 7. Out of scope (v1)

Local models, translation layer, images in answers, document editing, multi-user, cloud sync, Linux, code signing/notarization.

## 8. Caveats found during research

- `nemoretriever-ocr-v1` hosted endpoint is deprecated. Scanned pages go through the chat endpoint with `nemotron-parse` (v1), which answers as a `markdown_bbox` tool call. `nemotron-parse-2.0` is the successor, but its hosted context is 4096 tokens (`max_tokens` above that is a 400) and its endpoint returned 502 upstream on 2026-10-28.
- `nemotron-3.5-embed-8b` requires `input_type`; wrong mode silently degrades retrieval.
- The vec0 table fixes the embedding dimension (4096). Changing the embed model requires a full re-index.
- Rerank is not on the OpenAI base URL; it lives on `ai.api.nvidia.com/v1/retrieval/...`.
- PyMuPDF is AGPL-3.0. Fine for this non-commercial study project.
- macOS system Python (`/usr/bin/python3`, 3.9) blocks SQLite extensions and is too old. sqlite-vec needs `enable_load_extension`; use uv-managed Python 3.12+ everywhere, including PyInstaller builds.
- pywebview on Windows needs the WebView2 Runtime (present on current Windows 10/11; the installer must check).
- pywebview docs suggest py2app on macOS; PyInstaller also works and is the single tool used here.
- No Nemotron chat model on the catalog lists Ukrainian in its hosted model card, yet Super VL and Lightning answered Ukrainian and Russian correctly in the live probe. Qwen3-Next lists Ukrainian but has no free hosted endpoint. DeepSeek V4 Flash is end of life (410).
- Hosted model ids churn: DeepSeek V4 Flash disappeared within months. The chat model is a setting; the Settings dialog should show the error text from the API so a 410 is understood.
- Provider limit is 40 requests per minute for all models together. The throttle is global per process; two app instances would share nothing and could exceed it.
- PyMuPDF 2.0 depends on `pymupdf-layout` and `onnxruntime` and activates the layout model at import. The bundle excludes them; `import pymupdf.layout` then raises ImportError, which PyMuPDF catches. Verified: text PDF parses in the frozen app.
- sqlite-vec has no PyInstaller hook; `collect_dynamic_libs("sqlite_vec")` ships `vec0` next to the package so `sqlite_vec.load` finds it.
- PyInstaller only sees static imports; `importlib.import_module("webview")` needs `hiddenimports`.
- The Windows bundle is windowed (no console); `sys.stdout` is None there, so the launcher must not print.
- Windows bundle smoke test is only verified in CI; the local build check ran on macOS.
- Remote default branch is `master`. Each phase branch was cut from the previous phase branch, because `master` holds only the initial commit until the phases are merged.

## 9. Feature: OCR pass for scanned documents

Status: planned 2026-10-28, branch `feature/ocr-pass`. Quiz answers: OCR runs for pages without a text layer (as now), for pages whose text layer is broken, and on demand for a whole document; engine stays the hosted `nemotron-parse`; the UI shows page progress; OCR text goes to the index only.

### 9.1 What exists

`parsers.parse_pdf` sends a page to `parse_page` when its text layer has fewer than 20 characters. There is no quality check, no progress, no way to force OCR, and the parse model has not been verified on a Ukrainian scanned page.

### 9.2 Design

Three OCR modes, named after ocrmypdf's: `skip` (pages with a usable text layer keep it), `redo` (pages with a broken layer are re-read), `force` (every page is re-read). Automatic indexing runs skip+redo; the manual action runs force.

Page routing, in `parsers.py`:
1. `is_scanned(page)`: a page is a scan when one image covers at least half of the page area (`page.get_image_info()` bboxes). Digital pages are never sent to OCR unless forced; a logo-only cover page with little text is still treated as text.
2. `text_quality(text) -> float` in `[0, 1]`: share of good tokens among whitespace tokens of two or more characters, with these rmgarbage-style rules marking a token bad: longer than 40 characters; four or more identical characters in a row; fewer than half of its characters alphanumeric; Cyrillic and Latin letters mixed in one token. If more than 30 % of the tokens are single letters (character-separated text), the score is capped at 0.5. Pages with fewer than 5 tokens score 1.0 and are decided by the 20-character rule as now.
3. A scanned page goes to OCR when its text has fewer than 20 characters or `text_quality < OCR_QUALITY_MIN` (constant, 0.8 to start). `# ponytail: fixed threshold, calibrate on real scans; a setting only if users ask`.
4. `parse_pdf(path, cfg, parse_page, *, force_ocr=False, on_page=None)`; `on_page(done, total, via_ocr)` is called after every page.

Limits of the heuristic, stated up front: it catches mojibake, symbol soup, mixed scripts and character-separated text. It cannot catch a wrong but well-formed word (`ДОКУМЕНТІГ`) without a dictionary; that is what the manual force mode is for.

Progress and record, in `db.py` and `index.py`:
- `documents` gets `pages_done INTEGER` and `ocr_pages INTEGER` (added by `ALTER TABLE` on existing databases, same pattern as `sha256`). `pages` is set to the total at the start of parsing, not at the end.
- `index_document(..., force_ocr=False)` passes `on_page` that writes `pages_done` and counts OCR'd pages; the existing cancel check runs before every OCR call. At the end `ocr_pages` is stored.
- Manual run: `db.delete_chunks(conn, document_id)` (chunks, FTS rows and vec rows), status back to `pending`, `pages_done` 0, then `index_document(force_ocr=True)` in the background.

API, in `api.py`: `POST /workspaces/{ws}/documents/{id}/ocr` → 202 with the document row; 404 for unknown; 409 while the document is `pending`. `GET .../documents` already returns the new columns.

UI, in `index.html`:
- Pending row: the amber dot keeps pulsing; a small pencil-colored `34 / 170` appears after the name while `pages` is known; the tooltip says `Indexing, page 34 of 170`.
- Indexed row: tooltip `Indexed` or `Indexed, 12 pages via OCR`.
- Row action: a scan icon button (aria-label `Run OCR`) next to the × on hover. It opens a confirm dialog: `Run OCR on <name>? All 170 pages are sent to the parse model, about 5 minutes.` (minutes = pages / 40, rounded up) with Run OCR and Cancel. The row returns to pending and polling resumes.

Rate limit consequence: force mode on a 170-page document is 170 parse requests plus about 6 embed requests, about 4.5 minutes at 40 rpm, and it blocks nothing else except the shared throttle.

### 9.3 Tests (= requirements)

| Module | Tests |
|---|---|
| parsers.py | `test_text_quality_clean_page_scores_high`, `test_text_quality_mixed_script_and_symbol_soup_scores_low`, `test_text_quality_char_separated_text_capped`, `test_is_scanned_requires_image_over_half_page`, `test_scanned_page_with_bad_text_layer_goes_to_ocr`, `test_digital_page_with_bad_text_is_not_ocred`, `test_force_ocr_reads_every_page`, `test_on_page_reports_progress_and_ocr_flag` |
| db.py | `test_schema_adds_progress_columns_to_existing_db`, `test_set_document_progress`, `test_delete_chunks_removes_fts_and_vec_rows` |
| index.py | `test_index_sets_total_pages_before_parsing`, `test_index_records_pages_done_and_ocr_pages`, `test_force_ocr_reindex_replaces_chunks`, cancel tests keep passing |
| api.py | `test_run_ocr_route_resets_and_reindexes`, `test_run_ocr_conflict_while_pending`, `test_run_ocr_unknown_document_404` |
| UI | `test_pending_row_shows_page_progress`, `test_indexed_row_tooltip_counts_ocr_pages`, `test_run_ocr_button_confirms_then_row_goes_pending` |
| live | `test_live_parse_reads_ukrainian_scanned_page`: render page 1 of a Ukrainian scan to PNG, call `parse_page`, assert the expected heading words; records the DESIGN caveat as verified |

### 9.4 Phases

| # | Step | Deliverable |
|---|---|---|
| 1 | Quality heuristic and routing | `text_quality`, `is_scanned`, `parse_pdf` modes and `on_page` |
| 2 | Progress and record | schema columns, `set_document_progress`, `delete_chunks`, `index_document` wiring |
| 3 | Manual run | route, row action, confirm dialog, progress in the row |
| 4 | Calibration and live check | run the heuristic over real scans (the three НАЕК standards, 433 pages, once they are back in `test_pdf_docs/`), adjust `OCR_QUALITY_MIN`, run the live test, update caveats |

Calibration needs real scans in `test_pdf_docs/`; the folder is untracked and was not present on 2026-10-28 21:40 when the plan was written.

## Sources

- https://build.nvidia.com/llms.txt — base URL, auth, OpenAI compatibility
- https://docs.api.nvidia.com/nim/reference/llm-apis — chat endpoint and model list
- https://build.nvidia.com/nvidia/nemotron-3.5-lightning-30b-a3b/modelcard — 1M context, commercial use, RAG use case
- https://research.nvidia.com/labs/nemotron/Nemotron-3.5-Super-and-Lightning/ — Lightning long-context phase
- https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-3_5-embed-8b-infer — embeddings params (4096 dims, 10240 tokens)
- https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-3_5-embed-8b — 8b model card: 34 languages, ViDoRe V3 scores
- https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-3_5-embed-1b — 1b model card: distilled from 8b, scores
- https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-3_5-rerank-8b — rerank model card: 34 languages
- https://huggingface.co/nvidia/Nemotron-3.5-Rerank-8B-BF16 — rerank languages (second source)
- https://catalog.ngc.nvidia.com/orgs/nim/nvidia/containers/nemotron-embed/ — Nemotron Embed: 34 text languages (third source)
- https://developer.nvidia.com/blog/develop-multilingual-and-cross-lingual-information-retrieval-systems-with-efficient-data-storage/ — cross-lingual retrieval without translation
- https://beluga-ai.org/docs/guides/capabilities/rag/cross-lingual-retrieval/ — cross-lingual similarity search without translation
- https://arxiv.org/html/2504.03616v1 — multilingual RAG strategies, translation vs direct multilingual retrieval
- https://pypi.org/project/sqlite-vec/ — 0.1.9 wheels incl. win_amd64
- https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct — Qwen3-Next model card (candidate; no free endpoint)
- https://arxiv.org/html/2505.09388v1 — Qwen3 technical report: 119 languages incl. ukr_Cyrl and rus_Cyrl, Russian benchmark table
- https://qwenlm.github.io/blog/qwen3/ — Qwen3 language table lists Ukrainian and Russian
- https://arxiv.org/html/2605.10296 — UNLP 2026 Ukrainian document RAG: Qwen3 embed/rerank/generator pipeline
- https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-3-super-120b-a12b — Super languages
- https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-3-ultra-550b-a55b — Ultra languages
- https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-3_5-super-vl-120b-a12b — 3.5 Super VL post-training languages
- https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-Base-BF16 — Super base card (adds Russian)
- https://huggingface.co/zai-org/GLM-5.3-Flash — GLM-5.3-Flash card (no language list)
- https://ai-llm-leaderboard.com/leaderboards/ukrainian — community Ukrainian ranking (DeepSeek V4 top)
- https://docs.api.nvidia.com/nim/reference/deepseek-ai-deepseek-v4-flash-infer — `reasoning_effort` none/high/max, max_tokens 16384, SSE (model is 410 since 2026-08-07)
- https://developer.nvidia.com/blog/build-with-deepseek-v4-using-nvidia-blackwell-and-gpu-accelerated-endpoints/ — DeepSeek V4 hosted on build.nvidia.com
- https://docs.api.nvidia.com/nim/reference/nvidia-llama-3_2-nv-embedqa-1b-v2-infer — `input_type` warning
- https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-3_5-rerank-8b-infer — rerank endpoint and body
- https://docs.api.nvidia.com/nim/reference/nvidia-llama-nemotron-rerank-1b-v2-infer — rerank body shape (second source)
- https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-parse-2_0-infer — parse via chat completions, one image per request
- https://build.nvidia.com/nvidia/nemoretriever-ocr — deprecation notice
- https://docs.nvidia.com/nim/ingestion/image-ocr/latest/use-the-api.html — OCR NIM output shape (word boxes)
- https://alexgarcia.xyz/sqlite-vec/python.html — macOS extension block, SQLite version notes
- https://simonwillison.net/2024/Oct/4/hybrid-full-text-search-and-vector-search-with-sqlite/ — hybrid search + RRF
- https://blog.sqlite.ai/building-a-rag-on-sqlite — FTS5 + vector + RRF
- https://programmer.ie/solutions/sqlite-vec-hybrid-rag/ — RRF for small/medium RAG
- https://keenhenry.github.io/posts/python-sqlite-fts5/ — FTS5 bundled in Python 3.13 sqlite3
- https://pywebview.flowrl.com/guide/installation.html — WebView2 and pyobjc deps
- https://pywebview.flowrl.com/guide/freezing.html — PyInstaller / py2app
- https://parsebridge.com/blog/pymupdf-license/ — PyMuPDF AGPL
- https://www.nutrient.io/blog/extract-text-from-pdf-pymupdf/ — scanned PDF detection with PyMuPDF
- https://pyinstaller.org/en/stable/spec-files.html — spec file: datas, binaries, hiddenimports, excludes, macOS BUNDLE
- https://github.com/pyinstaller/pyinstaller-hooks-contrib — community hooks exist for pymupdf, uvicorn, webview, pydantic; none for sqlite_vec
- https://deepwiki.com/git-liu835/codex-cn-bridge/4.1-backend-build-and-pyinstaller-packaging — FastAPI + uvicorn PyInstaller build example
- https://github.com/mothercoremasterchief-prog/ocr-bench — OCR quality without ground truth: gibberish ratio, single-letter ratio, char-separated text
- https://github.com/erl-ang/interactive-ocr — rmgarbage heuristics (Taghva et al.) for OCR quality without ground truth
- https://ocrmypdf.readthedocs.io/en/latest/cookbook.html — skip / redo / force OCR modes; redo replaces only OCR text, force rasterizes all pages
- https://getsearchablepdf.com/blog/force-ocr-fix-bad-text-layer.html — symptoms of a bad text layer and when to force OCR
