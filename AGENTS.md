# LocalRAG

The key words MUST, MUST NOT, SHOULD, SHOULD NOT and MAY in this file and in the armory
skills follow RFC 2119. MUST has no exception. A SHOULD rule is on by default; `## Gates`
turns it off for this project, with a reason. MAY is the agent's choice.

## What this is

LocalRAG is a desktop app that answers questions about the user's own documents (PDF,
DOCX, TXT). Documents and the search index live in one local SQLite file (FTS5 plus
sqlite-vec, merged with reciprocal rank fusion, then reranked). The language models run
on build.nvidia.com through its OpenAI-compatible API: embeddings, rerank, OCR for
scanned pages and the streamed, cited answer. The backend is FastAPI; the UI is a static
page shown in pywebview; `localrag` starts both. It works in Ukrainian, Russian and
English. A Windows x64 one-file exe is built with PyInstaller. The design is in
`DESIGN.md`.

## Commands

One line per check. A gate in `## Gates` has the command of the same name here; a gate
with no command is `off` there. `run` is not a gate. `worktree-per-agent` and `delegate`
have no command.

- unit: `uv run pytest -m "not ui and not live" --no-cov`
- integration: `uv run pytest -m live --no-cov` (calls build.nvidia.com, needs `NVIDIA_API_KEY`)
- e2e: `uv run pytest -m ui --no-cov` (headless Chromium through Playwright)
- coverage: `uv run pytest -m "not live"` (`addopts` in `pyproject.toml` carries `--cov-fail-under=90`)
- run: `uv run localrag`

The `coverage` command MUST fail below the threshold set in the build tools (default 90%
per core module).

## Test levels

What each level covers in this project. The armory `test` skill holds the rules.

- unit: one unit (a function, a class, or a small cluster), fast, runs on every change.
  Frontend unit tests with the backend stubbed are `unit`.
- integration: narrow, one integration point at a time (a real database, the file
  system, one separate service or its double), at the boundary of the service.
- contract: consumer-driven contract tests for each interface with a separate service.
- functional: acceptance from the user's view, black box, given/when/then, through the
  API or the UI, externals doubled.
- e2e: the deployed stack through its UI or API, the main user journeys only.

Unit style: sociable (`sociable`: real collaborators, doubles only for remote,
non-deterministic, slow or side-effecting ones; `solitary`: every collaborator doubled).

Surface rule: each surface MUST have at least one test level. Frontend: `unit` or `e2e`
(browser). Backend: `unit` or `integration`. CLI and library: `unit`.

Surfaces here: frontend (`src/localrag/static/`, tested by `e2e`), backend (FastAPI,
SQLite, NVIDIA client, tested by `unit` and `integration`), CLI (the `localrag` entry
point in `app.py`, tested by `unit`).

## Gates

`on`, or `off <reason>`. `contract`, `functional` and `e2e` are the flexible levels: off
until the project turns them on. Besides these gates, any SHOULD rule of a skill is off
when a line names it: `<skill> section <n>: off <reason>`.

- worktree-per-agent: on
- delegate: on
- lint: off (no linter configured)
- owasp: off (no scanner configured)
- unit: on
- integration: on
- contract: off (not chosen)
- functional: off (not chosen)
- e2e: on
- coverage: on

## Models

| Task kind | Model | Effort | Agent | Why |
|---|---|---|---|---|
| Pre-work: research, requirement quizzes, the plan; tasks Opus cannot carry: architecture, data risk, non-trivial root-cause analysis, core conflicts | fable | high | researcher, tag `[fable]` | heavy reasoning only; a trivial task MAY run pre-work on opus |
| Coding (TDD implementation) | opus | low | coder | follows a brief |
| Deep research | opus | medium | researcher | sources and judgement |
| Code review, audit | opus | medium | reviewer | needs judgement, no writing |
| Mechanics: docs, test runs, i18n strings, formatting, routine edits per a ready spec or ready tests | sonnet | low | writer; coder, tag `[sonnet]` or `[haiku]` | no reasoning needed |
| Routine lookup, grep, file reads | sonnet | low | none (built-in Explore) | no reasoning needed |

Fable MUST be used only for the first row: the plan, and the items the plan tags
`[fable]`. The table binds the session model as well as the agents. A plan tag
(`[opus]`, `[fable]`, `[sonnet]`, `[haiku]`) names the model for one item; on a spawn
it overrides the model in the agent file. Agent files hold the model of their untagged
row.

## Rules

Workflows live in the armory skills. This section holds the
base rules when the global `~/.claude/CLAUDE.md` does not, then project-only rules.

1. Questions to the user MUST go through the `AskUserQuestion` tool: at least 2 options,
   the recommended one first. The message text holds only the context, not the question.

Project rules:

- Documentation MUST live in `docs/`. Only the README stays at the root. Requirements
  live in `docs/requirements/<task-branch>.md`, plans in `docs/plans/<task-branch>.md`.
- Every feature MUST ship with its tests. The only exception is a feature for which the
  user states so explicitly.
- All new and changed code MUST follow SOLID:
  - Single responsibility: every class has only one responsibility.
  - Open-closed: entities are open for extension, but closed for modification.
  - Liskov substitution: a subtype works wherever its base type is used.
  - Interface segregation: clients do not depend on interface methods they do not use.
  - Dependency inversion: high-level modules do not depend on low-level modules; both
    depend on abstractions, and abstractions do not depend on details.

## Glossary

Format: `Term : one-line meaning`

## Caveats

Format: `SHORT_DESCRIPTION : file_reference.md`
