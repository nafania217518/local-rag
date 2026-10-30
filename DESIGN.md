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
| OS | Windows x64 (published build); macOS only for development |
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
| OCR / page parse | `nvidia/nemotron-3.5-parse` | `POST https://ai.api.nvidia.com/v1/retrieval/nvidia/nemotron-3.5-parse/parse` | Not OpenAI-shaped and not a chat model. Request: `multipart/form-data` with a `document` part (one PNG or JPEG page image, `page.png`; the whole request stays under 100 MiB) and an `options` part (JSON `{"decoding": {"max_output_tokens": 9000}}`). `decoding.max_output_tokens` is the only option; its range is 1 to 9000 (400 above). No `model` field in the body. Response: `{content, finish_reason, usage}`. `content` is one string of page elements in reading order, each as `<x_0.11><y_0.08>TEXT<x_0.93><y_0.12><class_Text>` (normalized box corners and a class token). `finish_reason` is `length` when the output hits the cap. The client strips the box and class tokens. Token cap 9000. Live-tested 2026-10-30 |

Why parse instead of the OCR NIM: the hosted `nemoretriever-ocr-v1` endpoint is deprecated, and `nemotron-ocr-v2` returns word boxes, not reading-order text. The OCR path uses the dedicated parse endpoint on `ai.api.nvidia.com`, the same host as rerank. On 2026-10-30 it replaced the chat-endpoint `nemotron-parse` (v1) with `nemotron-3.5-parse`. The old client shape (chat completions with an image part) no longer applies. One auth path: the same Bearer key.

Rate limit: build.nvidia.com allows 40 requests per minute in total, across all models. `nvidia.py` enforces one global sliding-window throttle (`rpm` setting, default 40, thread-safe) that every outgoing request passes through, and retries 429 and 5xx responses with exponential backoff (1, 2, 4, 8 s, or the `Retry-After` header), at most 5 attempts. The SDK's own retries are off. The embedding endpoint also rejects some multi-text batches with a 5xx at once (see 9.7), and a retry cannot fix that. `nvidia.embed` still sends 64 texts per request. When a batch of more than one text gets a 5xx, the client splits it in halves, embeds each half and repeats down to single texts, in input order. A single text keeps the backoff retries. A 429 keeps the backoff at every batch size and does not split. The worst case for one bad 64-text batch is about 35 requests instead of 1, about one minute at 40 rpm. Consequence for ingestion: a scanned 30-page PDF costs about 31 requests (30 parse calls plus one embed batch), so about one such document per minute. The UI shows per-document status so this is visible, not silent.

The OpenAI Python SDK handles chat and embeddings (`base_url="https://integrate.api.nvidia.com/v1"`, `extra_body={"input_type": ...}` for embeddings). Rerank and parse are each one `httpx.post`.

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

Logging: `app.main` calls `configure_logging()` first. It sets the Python `logging` root to INFO with the format `%(asctime)s %(levelname)s %(name)s: %(message)s`, writes to stderr and sets the `httpx` logger to WARNING. It installs the handler only when `sys.stderr` exists. The Windows one-file exe has no console, so it logs nothing, and the launcher still prints nothing. The log holds:
- one INFO line per NVIDIA request on success: `<kind> <model> x<size>: ok in <s>s` (kinds chat, embed, rerank, parse);
- a WARNING per retry (status, server message, attempt, wait);
- a WARNING per embedding batch split (`splitting into 32 + 32`);
- INFO when a document starts and finishes indexing (pages, OCR pages);
- ERROR with traceback when indexing fails;
- INFO when indexing stops because the document was deleted.

A 125-page scan writes about 130 lines. There is no log file; the console is the only sink (decided 2026-10-30).

Packaging: `localrag.spec` at the repo root, one-file PyInstaller build (`uv run pyinstaller localrag.spec`). Output: `dist/LocalRAG.exe` on Windows (one fat executable; on macOS the same spec gives `dist/LocalRAG`, used only for local checks, never published). The spec collects `static/`, the sqlite-vec `vec0` library (no community hook), names `webview` as a hidden import (app.py loads it through importlib), and excludes the optional PyMuPDF layout stack (`pymupdf.layout`, onnxruntime, numpy, networkx, psutil; about 170 MB and its ONNX resources are not collected). `LocalRAG --serve <port>` runs the server without a window; it serves as the smoke check of a build. Releases are built by hand on Windows and the zip `LocalRAG-windows-x64.zip` is attached to a GitHub release; CI runs only the test suite, on pull requests to `master`. The exe is not code-signed; Windows SmartScreen will warn.

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
   - PDF: PyMuPDF `page.get_text()`. If a page has fewer than `min_text_chars` (20) characters of text, render it to PNG at `render_dpi` (150) and send it to the parse model (`parse_model` and `parse_url` settings). Per-page decision, so mixed PDFs work.
   - DOCX: `python-docx` paragraphs and table cells, page = 1.
   - TXT/MD: whole file, page = 1.
   - Parsers are a dict `{".pdf": parse_pdf, ...}`; a new format is one new entry (open-closed).
3. Chunk: split each page's text into ~1500-character chunks on paragraph boundaries, 200-character overlap, keep `page`.
4. Embed in batches of 64 with `input_type=passage`.
5. Insert chunks, FTS rows and vec rows in one transaction. Set `status=indexed` or `failed` + `error`.

Runs in a background thread per document. The UI polls document status.

### 4.3 Retrieval (`retrieval.py`)

Sizes are settings (section 4.7); defaults in brackets.

0. Paraphrase: when `paraphrase` is on, one chat call with reasoning off writes `paraphrases` (2) alternative phrasings of the question in its own language, one per line. Numbering, quotes, blanks and duplicates are stripped. If the call fails, the question runs alone. The queries are the original plus the paraphrases.
1. FTS5 `MATCH` per query, top `candidates` (30, BM25 order).
2. Embed all queries in one call with `input_type=query`, sqlite-vec KNN per vector, top `candidates` (30, cosine).
3. Reciprocal Rank Fusion (`k=60`) over all FTS and vector rankings, top `rerank_candidates` (20).
4. Rerank them against the original question with `nemotron-3.5-rerank-8b`, keep top `sources` (6).

Cost per question: one chat request for the paraphrases, one embed, one rerank, then the answer stream. The paraphrases are not shown or stored; a follow-up question is paraphrased without the chat history.
5. Return chunks with `document name`, `page`, `text`, `score`.

Reranking is a setting (`rerank: true`). Off → RRF top 6.

### 4.4 Chat (`chat.py`)

Prompt: system message with rules ("answer only from the context, cite as [n], say when the context does not contain the answer") + numbered context blocks + last 10 messages of the session + the question.

Request parameters `temperature`, `max_tokens` and `reasoning_effort` come from settings (defaults 0.2, 4096, `none`); the system prompt and the history limit (10 messages) too. When reasoning is on, the stream carries `reasoning_content` deltas before `content`; the server forwards them as SSE `thinking` events and the UI shows them collapsed. `# ponytail: thinking events are forwarded raw, no token budget; add one if users turn reasoning on and answers get slow.`

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

`<app data>/config.json` holds every tunable. `config.DEFAULTS` in `config.py` is the single source of truth: the code reads `cfg[key]`, there are no module constants for these values, and `PUT /settings` is validated by a pydantic model generated from `DEFAULTS` (wrong type is a 422, unset and null fields are ignored). `GET /settings/defaults` feeds the "Reset to defaults" button, which fills the form except the API key and does not save. The key is stored in plain text by the user's decision.

| Section | Key | Default |
|---|---|---|
| Connection | `api_key` | `""` |
| Connection | `base_url` | `https://integrate.api.nvidia.com/v1` |
| Connection | `rerank_url` | `https://ai.api.nvidia.com/v1/retrieval/{model}/reranking` |
| Connection | `parse_url` | `https://ai.api.nvidia.com/v1/retrieval/{model}/parse` |
| Connection | `rpm` | 40 |
| Connection | `timeout` | 120 s |
| Models | `chat_model`, `embed_model`, `rerank_model`, `parse_model` | see section 3 |
| Chat | `system_prompt` | the prompt in section 4.4 |
| Chat | `temperature` | 0.2 |
| Chat | `max_tokens` | 4096 |
| Chat | `history_limit` | 10 |
| Chat | `reasoning_effort` | `none` |
| Retrieval | `rerank` | true |
| Retrieval | `paraphrase` | true |
| Retrieval | `paraphrases` | 2 |
| Retrieval | `sources` | 6 |
| Retrieval | `candidates` | 30 |
| Retrieval | `rerank_candidates` | 20 |
| Indexing | `chunk_size` | 1500 |
| Indexing | `chunk_overlap` | 200 |
| Indexing | `ocr_quality_min` | 0.8 |
| Indexing | `min_text_chars` | 20 |
| Indexing | `render_dpi` | 150 |

Kept hardcoded: the RRF constant, retries and backoff, the embed batch size, the OCR token cap (9000), the embedding dimension. `load_config` replaces a saved `parse_model` of `nvidia/nemotron-parse` or `nvidia/nemotron-parse-2.0` with the default, because those names return 404 on the parse URL. Chunk and OCR settings apply at indexing time only: a document indexed before the change keeps its chunks until it is deleted and added again (the scan button re-indexes too, but forces OCR).

### 4.8 Languages

No translation layer. The embed and rerank models map text from different languages into one vector space, so a question in one language finds passages in another (cross-lingual retrieval). Both were evaluated on 34 text languages, Ukrainian and Russian included. The chat model answers in the language of the question when the system prompt says so.

Limits to know:
- FTS5 uses the `unicode61` tokenizer with `remove_diacritics 2`. It matches exact tokens in any script and does no stemming (the `porter` stemmer is English only). Inflected languages lose some keyword recall; the vector path and the reranker cover that.
- The chat model is the only language-limited component. Every Nemotron chat model lists Western European languages, Japanese and Chinese, none lists Ukrainian. The default is `nvidia/nemotron-3.5-super-vl-120b-a12b`: despite its official list it answered Ukrainian and Russian RAG questions correctly in the live probe (see the table in section 3). The system prompt tells the model to answer in the language of the question. Lightning is the fast fallback; Qwen3-Next if it gets a free endpoint.
- `tests/test_live_languages.py` (opt-in, needs `NVIDIA_API_KEY`): one Ukrainian question over a Ukrainian passage, one Russian, one cross-lingual (English question, Ukrainian passage). All three passed on 2026-10-28.
- Scanned pages: `nemotron-parse` (v1) verified live on 2026-10-28 on Ukrainian scans (see 9.5); `nemotron-3.5-parse` verified live on 2026-10-30 (see 9.6).

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
| packaging | text asserts on the spec and the workflow; the exe is checked by hand with `--serve` and `/health` | real build |

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
| 8 | Packaging | `localrag.spec`, `--serve` flag, CI `build` job with artifacts (later cut to one Windows zip on the release, see caveats) | `test_spec_*`, `test_ci_*`, `test_main_serve_flag_*`, CI smoke `/health` on the exe |

All phases are implemented; `uv run pytest` is the proof.

## 7. Out of scope (v1)

Local models, translation layer, images in answers, document editing, multi-user, cloud sync, Linux, code signing/notarization.

## 8. Caveats found during research

- `nemoretriever-ocr-v1` hosted endpoint is deprecated. Scanned pages go to `nemotron-3.5-parse` through the multipart parse endpoint on `ai.api.nvidia.com` (not chat completions). `nemotron-parse` (v1) on the chat endpoint was replaced on 2026-10-30; `nemotron-parse-2.0` returned 502 upstream on 2026-10-28 and is not used. A saved config with an old parse model name is migrated to the default at load.
- `nemotron-3.5-embed-8b` requires `input_type`; wrong mode silently degrades retrieval.
- The vec0 table fixes the embedding dimension (4096). Changing the embed model requires a full re-index.
- Rerank is not on the OpenAI base URL; it lives on `ai.api.nvidia.com/v1/retrieval/...`.
- PyMuPDF is AGPL-3.0. Fine for this non-commercial study project.
- macOS system Python (`/usr/bin/python3`, 3.9) blocks SQLite extensions and is too old. sqlite-vec needs `enable_load_extension`; use uv-managed Python 3.12+ everywhere, including PyInstaller builds.
- pywebview on Windows needs the WebView2 Runtime (present on current Windows 10/11; the installer must check).
- pywebview docs suggest py2app on macOS; PyInstaller also works and is the single tool used here.
- No Nemotron chat model on the catalog lists Ukrainian in its hosted model card, yet Super VL and Lightning answered Ukrainian and Russian correctly in the live probe. Qwen3-Next lists Ukrainian but has no free hosted endpoint. DeepSeek V4 Flash is end of life (410).
- Hosted model ids churn: DeepSeek V4 Flash disappeared within months. The chat model is a setting; the Settings dialog should show the error text from the API so a 410 is understood.
- `nemotron-3.5-embed-8b` rejects some multi-text batches at once with `500 service inference failed` (2026-10-30 probe: 2 to 12 texts fail, 1 and 16 pass, 64 are flaky). Backoff retries cannot help; the client splits a failed batch in halves down to single texts (see 9.7).
- Provider limit is 40 requests per minute for all models together. The throttle is global per process; two app instances would share nothing and could exceed it.
- PyMuPDF 2.0 depends on `pymupdf-layout` and `onnxruntime` and activates the layout model at import. The bundle excludes them; `import pymupdf.layout` then raises ImportError, which PyMuPDF catches. Verified: text PDF parses in the frozen app.
- sqlite-vec has no PyInstaller hook; `collect_dynamic_libs("sqlite_vec")` ships `vec0` next to the package so `sqlite_vec.load` finds it.
- PyInstaller only sees static imports; `importlib.import_module("webview")` needs `hiddenimports`.
- The Windows bundle is windowed (no console); `sys.stdout` is None there, so the launcher must not print.
- Windows bundle smoke test is only verified in CI; the local build check ran on macOS.
- The macOS build was dropped, then the CI build too: CI runs the tests on Windows on pull requests; the release zip with the one-file exe is built by hand. A one-file exe unpacks itself into a temp folder on every start, so launch is slower than the one-folder build and antivirus may scan it; acceptable for one user per machine.
- Remote default branch is `master`. Each phase branch was cut from the previous phase branch, because `master` holds only the initial commit until the phases are merged.

## 9. Feature: OCR pass for scanned documents

Status: planned 2026-10-28, branch `feature/ocr-pass`. Quiz answers: OCR runs for pages without a text layer (as now), for pages whose text layer is broken, and on demand for a whole document; engine stays the hosted `nemotron-parse` (moved to `nemotron-3.5-parse` on 2026-10-30, see 9.6); the UI shows page progress; OCR text goes to the index only.

### 9.1 What exists

`parsers.parse_pdf` sends a page to `parse_page` when its text layer has fewer than 20 characters. There is no quality check, no progress, no way to force OCR, and the parse model has not been verified on a Ukrainian scanned page.

### 9.2 Design

Three OCR modes, named after ocrmypdf's: `skip` (pages with a usable text layer keep it), `redo` (pages with a broken layer are re-read), `force` (every page is re-read). Automatic indexing runs skip+redo; the manual action runs force.

Page routing, in `parsers.py`:
1. `is_scanned(page)`: a page is a scan when one image covers at least half of the page area (`page.get_image_info()` bboxes). Digital pages are never sent to OCR unless forced; a logo-only cover page with little text is still treated as text.
2. `text_quality(text) -> float` in `[0, 1]`: runs of three or more dots are removed first (table-of-contents leaders are not garbage), then the share of good tokens among whitespace tokens of two or more characters, with these rmgarbage-style rules marking a token bad: longer than 40 characters; four or more identical characters in a row; fewer than half of its characters alphanumeric; Cyrillic and Latin letters mixed in one token. If more than 30 % of the tokens are single letters (character-separated text), the score is capped at 0.5. Pages with fewer than 5 tokens score 1.0 and are decided by the 20-character rule as now.
3. A page goes to OCR when its text has fewer than 20 characters (any page, as now), or when it is scanned and `text_quality < ocr_quality_min` (setting, default 0.8).
4. `parse_pdf(path, cfg, parse_page, *, force_ocr=False, on_page=None)`; `on_page(done, total, via_ocr)` is called once with `done = 0` before the first page (so the total is known before a slow first OCR call) and then after every page.

Limits of the heuristic, stated up front: it catches mojibake, symbol soup, mixed scripts and character-separated text. It cannot catch a wrong but well-formed word (`ДОКУМЕНТІГ`) without a dictionary; that is what the manual force mode is for.

Progress and record, in `db.py` and `index.py`:
- `documents` gets `pages_done INTEGER` and `ocr_pages INTEGER` (added by `ALTER TABLE` on existing databases, same pattern as `sha256`). `pages` is set to the total at the start of parsing, not at the end.
- `index_document(..., force_ocr=False)` passes `on_page` that writes `pages_done` and counts OCR'd pages; the existing cancel check runs before every OCR call. At the end `ocr_pages` is stored. The final `pages` is the total page count for PDFs (DOCX and TXT have one page), so `pages_done` never exceeds `pages`.
- `index.reindex(conn, cfg, document_id, force_ocr=True, ...)`: `delete_chunks`, status `pending`, `pages_done` 0, then `index_document`.
- Manual run: `db.delete_chunks(conn, document_id)` (chunks, FTS rows and vec rows), status back to `pending`, `pages_done` 0, then `index_document(force_ocr=True)` in the background.

API, in `api.py`: `POST /workspaces/{ws}/documents/{id}/ocr` → 202 with the document row; 404 for unknown; 409 while the document is `pending`. `GET .../documents` already returns the new columns.

UI, in `index.html`:
- Pending row: the amber dot keeps pulsing; a small pencil-colored `34 / 170` appears after the name while `pages` is known; the tooltip says `Indexing, page 34 of 170`.
- Indexed row: tooltip `Indexed` or `Indexed, 12 pages via OCR`.
- Failed row: the red dot is a button (aria-label `Show error`). It opens a dialog with the document name, the full error text and one Close button. The pending and indexed dots stay plain markers. The row tooltip keeps the error. When indexing fails on an HTTP error with a response body, the stored `error` holds the exception text plus the first 500 characters of the body, so the dialog and the log show the server message.
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
| 4 | Calibration and live check | heuristic measured over the three НАЕК standards (433 pages), live test, keep-better rule, LaTeX cleanup, caveats |

Calibration on 2026-10-28 over the three НАЕК standards (433 pages, kept outside the repo in `../test_pdf_docs/`): every page passes the image gate; median score 1.0, 10th percentile 0.93 to 0.99; 9 pages (2 %) score below 0.8: figure pages full of symbol soup and a cover page with spaced letters, which is where OCR helps. Table-of-contents pages scored 0.8 before the dot-leader rule and 1.0 after it. Threshold stays 0.8.

### 9.5 Live results (2026-10-28)

`test_live_parse_reads_ukrainian_scanned_page` passed on page 10 of СОУ НАЕК 160 (body text): the parse model's output scores 0.99 against the text layer's 1.00, same length. The eleven probe pages (nine flagged plus two references):

| Page | Text layer | Parse model | Note |
|---|---|---|---|
| 160 p10 (body) | 1.00, 2874 ch | 0.99, 2837 ch | match |
| 158 p4 (contents) | 1.00 | 0.96 | not flagged after the dot-leader rule |
| 158 p97, p98, p108; 159 p93, p100, p122 (tables, forms) | 0.50 | 0.71 to 0.85 | tables returned as LaTeX `tabular`; cell text correct, some Latin look-alikes (`COV HAEK` for `СОУ НАЕК`) |
| 158 p122 (large table) | 0.50, 4984 ch | 1.00, 17 ch | the model returned only the running header: blind replacement would lose the page |
| 160 p1 (cover) | 0.50 (spaced letters) | 0.67 | mixed-script look-alikes in the title |

Consequences built into phase 4: in redo mode the OCR text replaces the layer only when it is at least half as long and scores higher; the no-text and force cases replace unconditionally. LaTeX table markup from the model is flattened to `cell | cell` rows in `nvidia._clean`. Known limit: the model sometimes writes Latin look-alike letters inside Ukrainian words; the quality score sees those as mixed script, which is why OCR output scores below 1.0.

### 9.6 Live results (2026-10-30), nemotron-3.5-parse

The OCR path moved to `nvidia/nemotron-3.5-parse` on the parse endpoint. The probe used the same pages as 9.5 (standards in `../test_pdf_docs/`), rendered at 150 dpi, one page per request.

| Page | Text layer (score, chars) | nemotron-3.5-parse | Note |
|---|---|---|---|
| 160 p10 (body) | 1.00, 2874 ch | 200 in 12 s, 2489 tokens, 3692 ch with tokens; `test_live_parse_reads_ukrainian_scanned_page` passed through the new client | all body text present, Ukrainian correct; elements Page-header, Text, Section-header, Picture (the Picture held the page number) |
| 158 p122 (large table) | 0.50, 4984 ch | 200 in 9 s, 953 tokens; cleaned 1.00, 860 ch | full table as LaTeX `tabular`, caption `Таблиця К.4` and running header; v1 returned 17 ch here, this limit is gone |
| 158 p97 (form tables) | n/a | 200 in 12 s | four `tabular` blocks and a Footnote element; cell text correct |
| 160 p1 (cover) | 0.50 (spaced letters) | 200 in 4 s; cleaned 0.98, 487 ch | Title and Text elements, correct Ukrainian (v1 scored 0.67); the logo text came back as a Picture element |

Tables still come as LaTeX `tabular`, now with `\thead{...}` around header cells and `\mbox{}` for empty cells. `nvidia._clean` strips the box and class tokens, unwraps `\thead` and nested `\multirow`/`\multicolumn` cells, drops `\mbox{}`, renders `<sup>N</sup>` as `^N` (so `кгс/см<sup>2</sup>` stays `кгс/см^2`, not `кгс/см2`) and flattens the tabular to `cell | cell` rows. Latin look-alike letters still appear in some table cells (`МІЛА` for `МПа`, `Нознака` for `Познака`); the quality score still sees those as mixed script. The redo-mode guard stays: the OCR text replaces the layer only when it is at least half as long and scores higher.

### 9.7 Embedding failure and logs (2026-10-30)

Finding: indexing a 125-page scan failed with `Error code: 500 - service inference failed` after all pages were parsed. The 500 came from the hosted embedding endpoint `nvidia/nemotron-3.5-embed-8b` (`POST https://integrate.api.nvidia.com/v1/embeddings`), not from the parse endpoint. A rejection is not a timeout: it returns in 0.3 s with the body `{"object":"error","message":"service inference failed","type":"server_error"}`. The NVIDIA status page showed no incident. English and Cyrillic text and the `passage` and `query` input types behave the same. The same pair of texts passes on `nvidia/nemotron-3.5-embed-1b`. Five backoff retries cannot help a deterministic rejection.

| Inputs per request | Result on the chunks of this document |
|---|---|
| 1 | passes every time |
| 2 to 12 | fails every time, within 0.3 s |
| 16 | passes |
| 64 (the app batch size) | depends on the batch: the first batch of this document failed 4 of 5 times, a later one passed 2 of 3 |

Fix (TR-10): the client splits a failed batch in halves, as described in section 3. `EMBED_BATCH` stays 64.

Logging (TR-11 to TR-13): see section 4. Before this change the app logged nothing, and the only trace of the failure was the `error` column.

Stored error (TR-15) and UI dialog (TR-14): see section 9.2. The failed row now shows the server message on request.

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
- https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-parse-2_0-infer — parse-2.0 via chat completions, one image per request (history)
- https://build.nvidia.com/nvidia/nemotron-3.5-parse — hosted endpoint, Python snippet with the multipart request
- https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-3_5-parse-infer — API reference: `document` and `options` parts, status codes 400/413/415/422/503/504
- https://huggingface.co/nvidia/Nemotron-3.5-Parse — model card: architecture, control tokens, recommended max input 1664x2048, released 2026-10-17
- https://docs.nvidia.com/nim/vision-language-models/1.7.0/examples/nemotron-parse/api.html — the older v1.2 NIM used chat completions with control-token prompts; shows why the old client shape no longer applies
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
- https://pyinstaller.org/en/stable/spec-files.html — spec file: datas, binaries, hiddenimports, excludes
- https://pyinstaller.org/en/stable/operating-mode.html — one-file mode unpacks to a temp folder at start
- https://github.com/pyinstaller/pyinstaller-hooks-contrib — community hooks exist for pymupdf, uvicorn, webview, pydantic; none for sqlite_vec
- https://deepwiki.com/git-liu835/codex-cn-bridge/4.1-backend-build-and-pyinstaller-packaging — FastAPI + uvicorn PyInstaller build example
- https://github.com/mothercoremasterchief-prog/ocr-bench — OCR quality without ground truth: gibberish ratio, single-letter ratio, char-separated text
- https://github.com/erl-ang/interactive-ocr — rmgarbage heuristics (Taghva et al.) for OCR quality without ground truth
- https://ocrmypdf.readthedocs.io/en/latest/cookbook.html — skip / redo / force OCR modes; redo replaces only OCR text, force rasterizes all pages
- https://getsearchablepdf.com/blog/force-ocr-fix-bad-text-layer.html — symptoms of a bad text layer and when to force OCR
