# CLAUDE.md

## Rules

1. KISS: keep it simple.
2. TDD approach for coding: Requirement -> Test -> Code. The requirement is the test name. New requirement = new test. The task is done when all requirements have tests and all tests pass.
3. Use only Opus sub-agents with low effort for coding tasks (Agent tool: `model: opus`, `effort: low`). Instruct them to follow the TDD approach and give them the necessary set of requirements and instructions.
4. Tests must pass. Coverage must be no less than 90% per core functionality module.
5. Tests for every feature, no exceptions. Every feature ships with its test(s) unless the user explicitly states otherwise for that feature.
6. TDD on bug reports. When the user reports a bug, first add test(s). Only after those tests reproduce the bug, fix it.
7. TDD: tests are the only proof code works. All code must have test coverage via unit tests and/or functional, integration, browser, or end-to-end tests. Frontend: unit and/or browser tests. Backend: unit and/or integration tests. Both are acceptable per surface.
8. You may have outdated information. Do research and check at least 4 sources for each claim you make. Put anything worth noticing under the Caveats section below in the format `SHORT_DESCRIPTION : file_reference.md`.
9. TDD on own findings. Every issue, problem, bug or other finding you find yourself MUST be fixed through the TDD process when you are asked to fix it or decide to fix it: first add test(s) that reproduce it, then fix it.
10. Existing tests change only within a defined scope. Before a change, define its scope and list the existing tests it will break or that need rework or removal (for example: removing a module removes its tests; renaming a label changes the tests that check that label). Give this list to the sub-agent: it may change or remove only the listed tests. A test that fails outside the list most likely points to a newly introduced bug: the sub-agent stops, investigates the root cause, and reports the test with proof that the failure is not its bug. Then decide: if the change is expected and the test was only missing from the scope, add it to the scope; if the change is not expected, escalate to the user before any test changes.
11. Start every new task in a new branch. Before you start, fetch origin and create the branch from the latest `master` (`git fetch origin` and `git switch -c <task-branch> origin/master`). Never commit directly to `master`.
12. SOLID. Apply the five principles to all new and changed code:
    - Single responsibility: every class has only one responsibility.
    - Open-closed: entities are open for extension, but closed for modification.
    - Liskov substitution: a subtype must work wherever its base type is used; this enables polymorphic behavior and makes code more flexible and reusable.
    - Interface segregation: clients must not depend on interface methods that they do not use.
    - Dependency inversion: high-level modules must not depend on low-level modules; both depend on abstractions, and abstractions do not depend on details.

## Caveats

- Scanned pages go to nemotron-3.5-parse through a multipart parse endpoint on ai.api.nvidia.com (not chat completions); options accept only decoding.max_output_tokens up to 9000; the response is {content, finish_reason, usage} : DESIGN.md
- Embeddings need input_type passage/query or retrieval quality drops : DESIGN.md
- Rerank endpoint is on ai.api.nvidia.com, not the OpenAI base URL : DESIGN.md
- build.nvidia.com free tier is 40 requests/min across all models; every call goes through the throttle in nvidia.py : DESIGN.md
- Hosted model ids churn (deepseek-v4-flash went 410); default chat model is nemotron-3.5-super-vl-120b-a12b, live-tested on Ukrainian and Russian : DESIGN.md
- vec0 table fixes embedding dims; changing the embed model needs a full re-index : DESIGN.md
- Cross-lingual retrieval needs no translation; the chat model language list is the limit : DESIGN.md
- PyMuPDF is AGPL; acceptable because the project is non-commercial : DESIGN.md
- macOS system Python is 3.9 and blocks SQLite extensions (sqlite-vec needs them); use uv Python 3.12+ : DESIGN.md
- Windows needs WebView2 Runtime for pywebview : DESIGN.md
- PyInstaller: webview is imported via importlib (hiddenimports), sqlite-vec has no hook (collect_dynamic_libs), the PyMuPDF 2.0 layout stack (170 MB) is excluded : DESIGN.md
- Windows bundle has no console; the launcher must not print : DESIGN.md
- Text-layer quality heuristic flags about 2 % of pages on real scans (figures, forms, covers); it cannot catch well-formed wrong words : DESIGN.md
- nemotron-3.5-parse returned the full large table that v1 dropped; the redo-mode guard stays (OCR replaces the layer only when at least half as long and scoring higher) : DESIGN.md
- nemotron-3.5-parse returns tables as LaTeX tabular with \thead and \mbox, puts box and class tokens inline, and still mixes Latin look-alikes into Cyrillic words : DESIGN.md
- A saved parse_model of nemotron-parse or nemotron-parse-2.0 is migrated to the default at load, because those names 404 on the parse URL : DESIGN.md
- Only a Windows x64 one-file exe is published (zip on a GitHub release, built by hand; CI runs tests only, on pull requests); one-file mode unpacks to a temp folder at every start : DESIGN.md
- All tunables live in config.DEFAULTS (no module constants); chunk and OCR settings apply at indexing time only, a changed document must be deleted and added again : DESIGN.md
- Hosted nemotron-3.5-embed-8b returns 500 `service inference failed` for some multi-input batches (2 to 12 inputs failed every time, 16 passed, 64 flaky) and at random for long single inputs (about 40 % for a 900-token chunk, never for 350 tokens, never on embed-1b; probe 2026-10-30); the client splits a failed batch in halves down to single texts and gives a single text `embed_attempts` attempts (setting, default 50) : DESIGN.md
- Logs go to stderr only (no log file), so the Windows exe, which has no console, shows none : DESIGN.md
- The chat model copies the language of a bare example in the paraphrase prompt (an English question got Ukrainian paraphrases); the example is a full question pair after the rules, with an explicit "language of the question, not of the example" clause : DESIGN.md
- Question paraphrasing costs one extra chat request per question (four requests per question with rerank and the answer); a failed paraphrase call falls back to the original question : DESIGN.md
- A Rosetta crash (rosetta error: invalid gdt selector index 5) can kill Wine's pip during the Docker build at random (2 of 3 runs on 2026-10-30, same image and settings as a passing run the day before); a rerun passes; scripts/build-windows.sh detects it from the build log : DESIGN.md
