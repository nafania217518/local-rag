# Plan: embed fallback and console logs (branch `feature/nemotron-3.5-parse`, mode `min`)

## Context

Indexing of a 125-page scan failed today with `Error code: 500 - service inference failed` after every page was parsed. Probes showed the cause: the hosted `nvidia/nemotron-3.5-embed-8b` endpoint rejects some multi-input batches at once (2 to 12 of these chunks fail every time; 16 pass; 64 pass or fail per batch), while a single input always passes. The app sends 64 chunks per request and retries five times with backoff, which cannot help a deterministic rejection. Nothing in the app logs what happened, so the only trace was the `error` column in the database.

Two changes, both in this branch:

1. When an embedding batch of more than one text gets a 5xx, split it in halves and retry each half, down to single texts. Single texts keep the existing backoff retries. 429 keeps the backoff at every size (it is a rate limit, not a rejection).
2. Console logs (stderr, INFO, timestamped) for every NVIDIA call, every retry and split, and every indexing start, end and failure. Quiz answers: console only; every NVIDIA call logged.

The existing code: `nvidia._retry` (status checks and backoff), `nvidia._throttled` (throttle + retry wrapper used by `chat_stream`, `embed`, `rerank`, `parse_page`), `index.index_document` (records `failed` with `str(exc)`), `app.main` (entry point; uvicorn runs at `log_level="warning"`). No `logging` or `print` exists in `src/localrag` today. The Windows one-file exe has no console: `sys.stderr` is `None` there, so the handler must be installed only when stderr exists.

## Requirements (continue the ids of this branch; the tests start with the id)

```
TR-10 [opus]: embed MUST split a batch of more than one text in halves and embed each half when the request gets a 5xx, recursively down to single texts; the returned vectors MUST keep the input order.
  tests: test_tr10_embed_splits_batch_in_halves_on_5xx (64 texts: first request 500, halves 200; vectors in order; 3 requests),
         test_tr10_embed_splits_down_to_single_texts (2 texts: pair 500, each single 200),
         test_tr10_embed_single_text_5xx_still_retries_with_backoff (1 text: 503, 503, 200; sleeps [1, 2]),
         test_tr10_embed_429_on_a_batch_backs_off_instead_of_splitting (2 texts: 429 then 200; 2 requests, one sleep, no split),
         test_tr10_embed_single_text_5xx_gives_up_after_max_attempts (raises after MAX_ATTEMPTS),
         test_tr10_embed_other_4xx_on_a_batch_raises_without_split (401 on 2 texts: 1 request, raises)
TR-11 [opus]: every outgoing NVIDIA request MUST log one INFO line with its kind, model, size and seconds on success; every retry MUST log a WARNING with the HTTP status, the server message (response body, first 200 chars), the attempt and the wait; every split MUST log a WARNING with the batch size and status.
  tests: test_tr11_throttled_call_logs_kind_model_size_and_seconds (caplog, INFO, embed of 2 texts),
         test_tr11_retry_logs_status_body_attempt_and_wait (caplog, WARNING on 429 with Retry-After),
         test_tr11_split_logs_batch_size_and_status (caplog, WARNING),
         test_tr11_parse_and_rerank_calls_are_logged_too (one INFO line each)
TR-12 [opus]: index_document MUST log INFO at start (document id and name) and at the end (pages, OCR pages), ERROR with the exception and its traceback when indexing fails, and INFO (not ERROR) when the document was deleted during indexing (Cancelled).
  tests: test_tr12_index_logs_start_and_result (caplog),
         test_tr12_failed_index_logs_error_with_exception_message (embed raises; ERROR record carries the message and exc_info),
         test_tr12_cancelled_index_logs_info_not_error
TR-13 [opus]: app.main MUST configure console logging once before the server starts: level INFO, format `%(asctime)s %(levelname)s %(name)s: %(message)s`, stream sys.stderr, the `httpx` logger at WARNING (it logs every request at INFO otherwise); when sys.stderr is None (Windows windowed exe) no handler is installed and main still works.
  tests: test_tr13_configure_logging_installs_stderr_handler_at_info (isolated root logger via monkeypatch or a fresh logger; httpx logger level WARNING),
         test_tr13_configure_logging_skips_when_stderr_is_none,
         test_tr13_main_configures_logging (main calls configure_logging; use the existing run_main helper with a fake)
```

```
TR-14 [opus]: in the documents list, the status dot of a failed row MUST be a button (aria-label "Show error") that opens a dialog with the document name and the full error text; the dot of a pending or indexed row stays a plain marker; the dialog has one Close button. The row tooltip keeps the error as today.
  tests (ui): test_tr14_failed_row_dot_opens_error_dialog_with_message (stubbed docs with a long error; click the dot; dialog visible with name and full text; Close hides it),
              test_tr14_indexed_and_pending_dots_are_not_buttons
TR-15 [opus]: when indexing fails on an HTTP error whose response has a body, the stored error MUST include that body (first 500 chars) so the dialog and the log show the server message (today an httpx error from parse_page stores only "Server error '500 ...' for url ...").
  tests: test_tr15_failed_index_stores_server_message_from_response_body (parse_page raises httpx.HTTPStatusError with a JSON body; the document error contains the body text),
         test_tr15_failed_index_without_response_keeps_exception_text
```

## Design, by file

- `src/localrag/nvidia.py`
  - `log = logging.getLogger(__name__)`.
  - `_retry(fn, label, *, retry_5xx=True)`: keep the status rules; when `retry_5xx` is False a 5xx raises at once (429 still backs off). Each retry logs `WARNING "%s: HTTP %d %s, retry %d/%d in %ds"` with `e.response.text[:200]` (works for both `openai.APIStatusError` and `httpx.HTTPStatusError`, both carry an httpx response).
  - `_throttled(cfg, fn, label, *, retry_5xx=True)`: times the call and logs `INFO "%s: ok in %.1fs"` on success. Callers pass a label such as `f"embed {model} x{len(batch)}"`, `f"chat {model}"`, `f"rerank {model} x{len(passages)}"`, `f"parse {model}"`.
  - `embed` keeps the 64-slice loop and calls `_embed_batch(cfg, client, texts, input_type)`, which does: `retry_5xx = len(texts) == 1`; on `openai.APIStatusError` with 5xx and more than one text: `WARNING "%s: HTTP %d, splitting into %d + %d"`, then return `_embed_batch(first half) + _embed_batch(second half)`; otherwise re-raise. The result stays `[d.embedding for d in sorted(resp.data, key=lambda d: d.index)]`. `EMBED_BATCH` stays 64.
- `src/localrag/index.py`: `log = logging.getLogger(__name__)`; in `index_document`: INFO at start, INFO at success with pages and OCR pages; `except Cancelled: log.info(...)` then re-raise (status update as today); `except Exception: log.exception("indexing %d %s failed: %s", ...)`, status `failed` as today, re-raise. (`Cancelled` is defined in this module, so the order of the except clauses is the only change in structure.)
- `src/localrag/app.py`: `configure_logging()` guarded by `sys.stderr is not None`; `logging.basicConfig(level=logging.INFO, format=..., stream=sys.stderr)`; `logging.getLogger("httpx").setLevel(logging.WARNING)`. Called first thing in `main()` for both the `--serve` path and the window path. uvicorn stays at `log_level="warning"`.
- `src/localrag/index.py` (TR-15): one helper `_error_text(exc)` that returns `str(exc)` plus `exc.response.text[:500]` when the exception has a `response` with text not already contained in `str(exc)`; used for the stored `error` and the ERROR log line.
- `src/localrag/static/index.html` (TR-14): in `loadDocuments`, when `d.status === "failed"` the status element is `el("button", "status", d.status)` with `aria-label="Show error"` and `onclick` that fills and opens `#error-dialog`; otherwise the `span` as today. Add `<dialog id="error-dialog">` next to `#ocr-dialog` with `<h2 id="error-title">`, `<pre id="error-text">` (wrap long text: `white-space: pre-wrap`) and a Close button (`method="dialog"` form). CSS: `button.status` inherits the dot look (same `.status` rules apply to the class; add `cursor: pointer; border: 0; background: none; padding: 0`). Keep `.status` text hidden (font-size 0) as today so `to_have_text("failed")` keeps working.
- No change to `parsers.py`, `api.py`, config or dependencies.

## Test scope (existing tests that change)

- `tests/test_ui.py::test_failed_document_shows_error` (line 128) and the status-dot test around line 333-344 assert `.status` text `failed` and the row `title`; both keep passing because the button keeps the class, the text and the row title. If a selector there assumes `span.status`, that one assertion is in scope to change to `.status`.
- `tests/test_index.py` tests that assert the stored error equals `str(exc)` exactly (search for `["error"]`): if any uses an exception with a response, it is in scope for TR-15; plain exceptions keep the exact text.
- Otherwise none need to change by design: the retry tests use a single text or `rerank`, `test_embed_batches_by_64` and `test_embed_sends_input_type_and_truncate_and_returns_vectors_in_order` keep their request counts, `test_every_client_call_passes_through_throttle` keeps 4 throttle calls (a split only happens on a 5xx). `test_app.run_main` passes fakes into `main`; `configure_logging` runs for real there and must not break those tests (basicConfig is a no-op when the root logger already has handlers under pytest). If any of these fails, the coder stops and reports; the failure is a bug, not a scope gap.

## Gates

unit, e2e (UI unchanged; run once at the end), coverage (90 %, expect 100 % on `nvidia.py`, `index.py`, `app.py` except the existing 3 lines). Then one live check: `uv run pytest -m live` is not enough here; instead re-index the failed document in the running app (or run the embed step script) and read the console: expect `WARNING ... HTTP 500 ..., splitting into 32 + 32` lines and a final `indexed` line.

## Risks

- A deterministic bad 64-batch costs up to about 35 requests instead of 1 (≈1 minute at 40 rpm) before every chunk is embedded one by one. Acceptable; indexing succeeds instead of failing.
- INFO per request is about 130 lines for a 125-page scan. Chosen by the user.
- `log.exception` prints a traceback for API failures too (about 10 lines per failed document). Chosen for diagnosability.

## Docs after implementation (same branch, by the writer agent)

- DESIGN.md: rate-limit paragraph gains the split rule and the logging paragraph (what is logged, where, the Windows note); section 8 caveat on the embed endpoint rejecting small batches (2026-10-30 probe numbers).
- CLAUDE.md caveats: one line on the embed-8b batch rejection and the split fallback; one line that logs go to stderr only (the Windows exe shows none).
- `docs/requirements/feature/nemotron-3.5-parse.md`: append TR-10 to TR-13 with BR-4 "indexing MUST survive a server rejection of one embedding batch" and BR-5 "the console MUST show the cause of an indexing failure".

## Verification

1. `uv run pytest -m "not live"` passes with coverage ≥ 90 %.
2. `uv run localrag`, re-add (or scan) the failed СОУ НАЕК 160 document; the console shows the INFO request lines, WARNING split lines for the first batch, and the document ends as `indexed`.
3. In the app, the still-failed row from today: click its red dot; a dialog shows the name and `Error code: 500 - service inference failed`; Close hides it.

On "implement", run the `code` skill on Opus (coder agent), items TR-10 to TR-15 all `[opus]`. Docs afterwards also cover TR-14 (README: the OCR section gets one sentence on the red dot and the error dialog) and TR-15.
