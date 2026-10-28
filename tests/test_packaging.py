from pathlib import Path

SPEC = (Path(__file__).parent.parent / "localrag.spec").read_text()


def test_spec_collects_static_files_and_sqlite_vec_binary():
    assert '"localrag/static"' in SPEC and 'collect_dynamic_libs("sqlite_vec")' in SPEC


def test_spec_entry_is_the_launcher():
    assert "src/localrag/app.py" in SPEC


def test_spec_excludes_pymupdf_layout_and_onnx():
    assert "excludes=" in SPEC and '"pymupdf.layout"' in SPEC and '"onnxruntime"' in SPEC


def test_spec_declares_webview_hidden_import():
    assert 'hiddenimports=["webview"]' in SPEC
