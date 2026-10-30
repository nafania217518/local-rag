# Requirements: switch OCR to nemotron-3.5-parse

Branch `feature/nemotron-3.5-parse`, 2026-10-30. Mode `none`: the user asked for one change, so the requirements are written here without a quiz. Every test name starts with its id.

## Business requirements

- BR-1: Scanned pages MUST be read by the hosted `nvidia/nemotron-3.5-parse` model instead of the chat-endpoint `nvidia/nemotron-parse` (v1).
- BR-2: An install upgraded from 0.1.5 MUST keep working without a manual settings change.
- BR-3: The OCR text in the index MUST keep every cell of a table and every number as written on the page.
- BR-4: Indexing MUST survive a server rejection of one embedding batch.
- BR-5: The console MUST show the cause of an indexing failure.
- BR-6: A failed document MUST show its error in the UI on request.
- BR-7: Question paraphrases MUST use the formal technical terms of standards so that a question in everyday words finds the passage written in standard terms.

## Technical requirements

| Id | Executor | Requirement | From |
|---|---|---|---|
| TR-1 | opus | `config.DEFAULTS["parse_model"]` is `nvidia/nemotron-3.5-parse`; `config.DEFAULTS["parse_url"]` is `https://ai.api.nvidia.com/v1/retrieval/{model}/parse` | BR-1 |
| TR-2 | opus | `nvidia.parse_page` posts multipart/form-data to `parse_url` with the Bearer key, a `document` PNG part and an `options` JSON part `{"decoding": {"max_output_tokens": 9000}}`, through the shared throttle; a 4xx raises `httpx.HTTPStatusError` | BR-1 |
| TR-3 | opus | `parse_page` returns the cleaned `content`: box and class tokens stripped, old markers stripped, `\thead{X}` unwrapped, `\mbox{}` dropped, LaTeX tabular flattened to `cell | cell` rows | BR-1, BR-3 |
| TR-4 | opus | an empty or null `content` returns `""` | BR-1 |
| TR-5 | opus | the Settings dialog shows a `Parse URL` input (`name="parse_url"`) in the Connection group after Rerank URL, and the value round-trips through Save | BR-1 |
| TR-6 | opus | `load_config` replaces a saved `parse_model` of `nvidia/nemotron-parse` or `nvidia/nemotron-parse-2.0` with the default | BR-2 |
| TR-7 | opus | nested merged cells (`\multicolumn{..}{..}{\multirow{..}{..}{X}}`) unwrap to `X` | BR-3 (review finding) |
| TR-8 | opus | `<sup>N</sup>` renders as `^N`, so `10<sup>6</sup>` stays `10^6` and not `106` | BR-3 (review finding) |
| TR-9 | opus | two redundant tests (empty-string content, plain returns-content) are removed | review finding |
| TR-10 | opus | `embed` MUST split a batch of more than one text in halves and embed each half when the request gets a 5xx, recursively down to single texts; the returned vectors MUST keep the input order. A single text keeps the backoff retries on 5xx; a 429 keeps the backoff at every size and does not split | BR-4 |
| TR-11 | opus | every outgoing NVIDIA request MUST log one INFO line with its kind, model, size and seconds on success; every retry MUST log a WARNING with the HTTP status, the server message (response body, first 200 chars), the attempt and the wait; every split MUST log a WARNING with the batch size and status | BR-5 |
| TR-12 | opus | `index_document` MUST log INFO at start (document id and name) and at the end (pages, OCR pages), ERROR with the exception and its traceback when indexing fails, and INFO (not ERROR) when the document was deleted during indexing | BR-5 |
| TR-13 | opus | `app.main` MUST configure console logging once before the server starts: level INFO, format `%(asctime)s %(levelname)s %(name)s: %(message)s`, stream `sys.stderr`, the `httpx` logger at WARNING; when `sys.stderr` is None (Windows windowed exe) no handler is installed and `main` still works | BR-5 |
| TR-14 | opus | in the documents list, the status dot of a failed row MUST be a button (aria-label "Show error") that opens a dialog with the document name and the full error text; the dot of a pending or indexed row stays a plain marker; the dialog has one Close button; the row tooltip keeps the error | BR-6 |
| TR-15 | opus | when indexing fails on an HTTP error whose response has a body, the stored error MUST include that body (first 500 chars), so the dialog and the log show the server message | BR-5, BR-6 |
| TR-16 | opus | `_error_text` appends the response body only for `httpx.HTTPStatusError`; an `openai.APIStatusError` already carries the server message in its text, so it is not repeated | BR-5, BR-6 (review finding) |
| TR-17 | opus | the `Cancelled` branch of `index_document` no longer calls `set_document_status` (dead code: the row is gone) | review finding |
| TR-18 | opus | the `test_app.py` tests stub `configure_logging` so the `httpx` logger level does not leak between tests | review finding |
| TR-19 | opus | the odd split (3 texts → 1 + 2 → 1 + 1 + 1) is covered by a test that checks the vector order | review finding |
| TR-20 | opus | (replaced by TR-21) a single-text embed request gets 10 attempts with the backoff wait capped at 8 s; other calls keep 5 attempts | BR-4 (second finding: random single-input failures at about 40 % for long chunks) |
| TR-21 | opus | `embed_attempts` setting in `config.DEFAULTS`, default 50, sets the attempts for a single-text embed request; the backoff wait stays capped at 8 s; other calls keep 5 attempts | BR-4 |
| TR-22 | opus | the Settings dialog Connection group has an "Embed retries" number input (`name="embed_attempts"`, min 1) that round-trips | BR-4 |
| TR-23 | opus | the Settings input "Paraphrases per question" (`name="paraphrases"`) accepts up to 10 (`max="10"`); the default stays 2 | BR-7 |
| TR-24 | opus | (replaced by TR-25) the paraphrase prompt tells the chat model that the documents are technical standards and specifications and asks for the formal technical terms a standard uses instead of everyday words, with the example `варити труби` → `зварювати трубопроводи`; the same-language, one-per-line and no-numbering rules stay | BR-7 |
| TR-25 | opus | the paraphrase prompt gives the example as a full question pair after the rules (`Як варити труби?` → `Як зварювати трубопроводи?`) and says to write in the language of the question, not of the example; a bare example made the model answer English questions in Ukrainian (live probe) | BR-7 (own finding) |

## Out of scope

- A warning when `finish_reason` is `length` (output cut at 9000 tokens); no page in the probe corpus came near the cap.
- A log file: logs go to the console only (chosen 2026-10-30).
- A change of `EMBED_BATCH` (stays 64).
- `&` outside tables and `\\` inside `\thead{}`; not seen in the model output.
