# LocalRAG

LocalRAG is a desktop app that answers questions about your own documents. It keeps the documents and the search index in one local SQLite file. The language models run on build.nvidia.com through its OpenAI-compatible API, so the app needs an NVIDIA API key and a network connection, nothing else.

## What it does

- Workspaces: each workspace owns its documents and its chats.
- Documents: PDF, DOCX and TXT. A file that is already in the workspace (same checksum) is not added twice.
- Search: full-text search (FTS5) and vector search (sqlite-vec) merged with reciprocal rank fusion, then reranked.
- Answers: streamed, with numbered citations that open the source passage and page in the Sources column.
- Languages: Ukrainian, Russian and English. A question in one language finds passages in the others.
- Theme: follows the OS by default, with a manual light or dark switch.

## OCR for scanned documents

Indexing runs page by page. A page goes to the OCR model (`nvidia/nemotron-parse`) automatically when:

- it has almost no text layer (fewer than 20 characters), or
- it is a scan (an image covers at least half of the page) and its text layer fails a quality check (garbage tokens, repeated characters, mixed Cyrillic and Latin letters).

For a scanned page that already had a text layer, the OCR text replaces it only when it is at least half as long and scores higher on the same check. Otherwise the original text stays.

Each document row has a scan button to force OCR on every page of that document. It asks for confirmation, then re-indexes the document and shows the page progress, for example `34 / 170`. The tooltip on an indexed row tells how many pages went through OCR.

## Install

### Windows x64

Download `LocalRAG-windows-x64.zip` from a [release](https://github.com/nafania217518/local-rag/releases), unzip it and start `LocalRAG.exe`. The executable is not code-signed, so Windows SmartScreen warns on first start. It needs the WebView2 Runtime, which current Windows 10 and 11 already have.

### From source (Windows or macOS)

Needs Python 3.12 or newer and a clone of this repository.

With [uv](https://docs.astral.sh/uv/) (recommended; it installs its own Python if needed):

```
pip install uv
uv sync
uv run localrag
```

With plain pip in a virtual environment:

```
python -m venv .venv
.venv\Scripts\activate      # macOS: source .venv/bin/activate
pip install .
localrag
```

The app needs a `sqlite3` module that can load extensions (sqlite-vec). The python.org installer on Windows and uv-managed Python have it. The Microsoft Store Python and the macOS system Python do not; if the app fails on start with a message about `enable_load_extension`, use uv.

## Setup

Open Settings (gear icon, bottom left) and paste your API key from [build.nvidia.com](https://build.nvidia.com). The chat, embedding, rerank and parse models can be changed there too, and so can the endpoint URLs, the rate limit, the system prompt, the retrieval sizes and the chunking and OCR thresholds. "Reset to defaults" restores everything except the API key. The free tier allows 40 requests per minute across all models; the app throttles itself to stay under it. Chunk settings apply to documents indexed after the change.

App data lives in `~/Library/Application Support/LocalRAG` on macOS and `%APPDATA%\LocalRAG` on Windows.

## Tests

```
uv sync --all-groups
uv run playwright install chromium
uv run pytest
```

Live tests against the real NVIDIA endpoints run only when `NVIDIA_API_KEY` is set.

## Build the Windows exe

On any machine with Docker (macOS, Linux or Windows; an amd64 image runs under emulation on Apple silicon):

```
docker build -f Dockerfile.windows --output dist/windows-x64 .
```

The result is `dist/windows-x64/LocalRAG.exe`, a one-file build made with PyInstaller inside Wine with Windows Python 3.12. The first build pulls the `tobix/pywine` image and takes a few minutes. Zip the exe as `LocalRAG-windows-x64.zip` to publish it.

On Windows itself, without Docker:

```
uv sync --all-groups
uv run pyinstaller localrag.spec --noconfirm
```

## Release

Releases are built by hand with one of the commands above and attached to a GitHub release. CI only runs the test suite on pull requests.

See [DESIGN.md](DESIGN.md) for the design and the known caveats.
