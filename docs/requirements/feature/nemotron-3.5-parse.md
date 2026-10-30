# Requirements: switch OCR to nemotron-3.5-parse

Branch `feature/nemotron-3.5-parse`, 2026-10-30. Mode `none`: the user asked for one change, so the requirements are written here without a quiz. Every test name starts with its id.

## Business requirements

- BR-1: Scanned pages MUST be read by the hosted `nvidia/nemotron-3.5-parse` model instead of the chat-endpoint `nvidia/nemotron-parse` (v1).
- BR-2: An install upgraded from 0.1.5 MUST keep working without a manual settings change.
- BR-3: The OCR text in the index MUST keep every cell of a table and every number as written on the page.

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

## Out of scope

- A warning when `finish_reason` is `length` (output cut at 9000 tokens); no page in the probe corpus came near the cap.
- `&` outside tables and `\\` inside `\thead{}`; not seen in the model output.
