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
11. Start every new task in a new branch. Before you start, fetch origin and create the branch from the latest `main` (`git fetch origin` and `git switch -c <task-branch> origin/main`). Never commit directly to `main`.
12. SOLID. Apply the five principles to all new and changed code:
    - Single responsibility: every class has only one responsibility.
    - Open-closed: entities are open for extension, but closed for modification.
    - Liskov substitution: a subtype must work wherever its base type is used; this enables polymorphic behavior and makes code more flexible and reusable.
    - Interface segregation: clients must not depend on interface methods that they do not use.
    - Dependency inversion: high-level modules must not depend on low-level modules; both depend on abstractions, and abstractions do not depend on details.

## Caveats

- Hosted nemoretriever-ocr-v1 is deprecated; scanned pages go through nemotron-parse-2.0 on the chat endpoint : DESIGN.md
- Embeddings need input_type passage/query or retrieval quality drops : DESIGN.md
- Rerank endpoint is on ai.api.nvidia.com, not the OpenAI base URL : DESIGN.md
- PyMuPDF is AGPL; acceptable because the project is non-commercial : DESIGN.md
- vec0 table fixes embedding dims; changing the embed model needs a full re-index : DESIGN.md
- Cross-lingual retrieval needs no translation; chat model language list is the limit : DESIGN.md
- deepseek-v4-flash is end of life (410); default chat model is nemotron-3.5-super-vl-120b-a12b, live-tested on Ukrainian and Russian : DESIGN.md
- nemotron-parse-2.0 hosted endpoint is 502 upstream; default parse model is nemotron-parse (v1), which returns a markdown_bbox tool call; max_tokens 3500 : DESIGN.md
- macOS system Python is 3.9 and blocks SQLite extensions (sqlite-vec needs them); use uv Python 3.12+ : DESIGN.md
- Windows needs WebView2 Runtime for pywebview : DESIGN.md
- Remote default branch is master, not main : DESIGN.md
- build.nvidia.com free tier is 40 requests/min across all models; every call goes through the throttle in nvidia.py : DESIGN.md
- PyMuPDF 2.0 pulls pymupdf-layout + onnxruntime (170 MB); the PyInstaller spec excludes them : DESIGN.md
- sqlite-vec has no PyInstaller hook; the spec collects vec0 with collect_dynamic_libs : DESIGN.md
- webview is imported via importlib; the spec needs hiddenimports=["webview"] : DESIGN.md
- Windows bundle has no console; the launcher must not print : DESIGN.md
