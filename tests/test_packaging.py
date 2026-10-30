from pathlib import Path

ROOT = Path(__file__).parent.parent
SPEC = (ROOT / "localrag.spec").read_text()
CI = (ROOT / ".github/workflows/ci.yml").read_text()
DOCKERFILE = (ROOT / "Dockerfile.windows").read_text()


def test_spec_collects_static_files_and_sqlite_vec_binary():
    assert '"localrag/static"' in SPEC and 'collect_dynamic_libs("sqlite_vec")' in SPEC


def test_spec_entry_is_the_launcher():
    assert "src/localrag/app.py" in SPEC


def test_spec_excludes_pymupdf_layout_and_onnx():
    assert "excludes=" in SPEC and '"pymupdf.layout"' in SPEC and '"onnxruntime"' in SPEC


def test_spec_declares_webview_hidden_import():
    assert 'hiddenimports=["webview"]' in SPEC


def test_spec_builds_one_file_windowed_exe():
    exe = SPEC[SPEC.index("EXE("):]
    assert "COLLECT(" not in SPEC and "BUNDLE(" not in SPEC
    assert "a.binaries" in exe and "a.datas" in exe and "console=False" in exe


def test_ci_runs_only_on_windows():
    assert "macos" not in CI and "runs-on: windows-latest" in CI


def test_ci_runs_tests_only_on_pull_requests():
    assert "pull_request:" in CI and "uv run pytest" in CI
    assert not any(s in CI for s in ("tags:", "pyinstaller", "gh release", "contents: write"))


def test_windows_dockerfile_builds_the_spec_one_file_exe():
    assert all(s in DOCKERFILE for s in (
        "tobix/pywine:3.12", "pyinstaller localrag.spec", "FROM scratch", "/src/dist/LocalRAG.exe"))


def test_tr1_dockerfile_wine_steps_fail_on_error():
    wine_runs = [l for l in DOCKERFILE.splitlines() if l.startswith("RUN") and "wine " in l]
    assert wine_runs and all("; rc=$?; wineserver -w; exit $rc'" in l for l in wine_runs)
