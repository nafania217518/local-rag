from pathlib import Path

import pytest

from localrag import config, db, index


@pytest.fixture
def conn(tmp_path):
    return db.connect(tmp_path / "t.db", embed_dim=4)


def _count(conn, table):
    return conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


def test_chunk_text_respects_size_and_overlap():
    text = "\n\n".join(f"para {i} " + "x" * 80 for i in range(30))
    chunks = index.chunk_text(text, size=300, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 350 for c in chunks)
    for a, b in zip(chunks, chunks[1:]):
        assert b.startswith(a[-50:].strip())


def test_chunk_text_prefers_paragraph_boundaries():
    paras = ["a" * 40, "b" * 40, "c" * 40]
    chunks = index.chunk_text("\n\n".join(paras), size=90, overlap=0)
    assert chunks == ["a" * 40 + "\n\n" + "b" * 40, "c" * 40]


def test_chunk_text_hard_splits_long_paragraph():
    chunks = index.chunk_text("z" * 250, size=100, overlap=0)
    assert chunks == ["z" * 100, "z" * 100, "z" * 50]


def test_chunk_text_empty_returns_empty():
    assert index.chunk_text("") == []
    assert index.chunk_text("  \n\n ") == []


def test_chunk_pages_assigns_global_ord_and_keeps_page():
    pages = [(1, "a" * 2000), (3, "short")]
    out = index.chunk_pages(pages)
    assert [o for o, _, _ in out] == list(range(len(out)))
    assert [p for _, p, _ in out] == [1, 1, 3]
    assert out[-1][2] == "short"


def test_files_dir_is_under_app_data_dir():
    assert index.files_dir() == config.app_data_dir() / "files"


def test_store_file_copies_into_workspace_folder_and_sets_path(conn, tmp_path):
    ws = db.create_workspace(conn, "w")
    src = tmp_path / "My Doc.TXT"
    src.write_text("hello", encoding="utf-8")
    doc_id = index.store_file(conn, ws, src, dest_root=tmp_path / "files")
    row = db.get_document(conn, doc_id)
    expected = tmp_path / "files" / str(ws) / f"{doc_id}.TXT"
    assert row["name"] == "My Doc.TXT"
    assert row["path"] == str(expected)
    assert expected.read_text(encoding="utf-8") == "hello"


def _stored(conn, tmp_path, text):
    ws = db.create_workspace(conn, "w")
    src = tmp_path / "a.txt"
    src.write_text(text, encoding="utf-8")
    return index.store_file(conn, ws, src, dest_root=tmp_path / "files")


def test_index_document_parses_chunks_embeds_and_marks_indexed(conn, tmp_path):
    doc_id = _stored(conn, tmp_path, "hello world\n\nsecond paragraph")
    calls = []

    def fake_embed(cfg, texts, input_type):
        calls.append((texts, input_type))
        return [[0.0] * 4 for _ in texts]

    index.index_document(conn, {}, doc_id, embed=fake_embed, parse_page=None)
    assert _count(conn, "chunks") == 1
    assert _count(conn, "chunks_vec") == 1
    row = db.get_document(conn, doc_id)
    assert (row["status"], row["pages"]) == ("indexed", 1)
    assert calls[0][1] == "passage"


def test_index_document_marks_failed_on_error_and_reraises(conn, tmp_path):
    doc_id = _stored(conn, tmp_path, "hello")

    def boom(cfg, texts, input_type):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        index.index_document(conn, {}, doc_id, embed=boom)
    row = db.get_document(conn, doc_id)
    assert row["status"] == "failed"
    assert "boom" in row["error"]


def test_index_document_zero_chunks_marks_indexed_without_embed(conn, tmp_path):
    doc_id = _stored(conn, tmp_path, "")

    def no_embed(*a):
        raise AssertionError("embed called")

    index.index_document(conn, {}, doc_id, embed=no_embed)
    row = db.get_document(conn, doc_id)
    assert (row["status"], row["pages"]) == ("indexed", 0)
    assert _count(conn, "chunks") == 0


def test_store_file_uses_given_name(conn, tmp_path):
    ws = db.create_workspace(conn, "w")
    src = tmp_path / "tmp123.txt"
    src.write_text("x")
    doc_id = index.store_file(conn, ws, src, dest_root=tmp_path / "files", name="Real.txt")
    assert db.get_document(conn, doc_id)["name"] == "Real.txt"


def test_store_file_records_sha256(conn, tmp_path):
    import hashlib
    doc_id = _stored(conn, tmp_path, "some bytes")
    assert db.get_document(conn, doc_id)["sha256"] == hashlib.sha256(b"some bytes").hexdigest()


def test_index_stops_when_document_is_deleted_during_indexing(conn, tmp_path):
    from localrag import nvidia
    doc_id = _stored(conn, tmp_path, "\n\n".join(f"{i:03d}" + "x" * 400 for i in range(150)))
    calls = []

    def fake_embed(cfg, texts, input_type):
        calls.append(texts)
        db.delete_document(conn, doc_id)
        return [[0.0] * 4 for _ in texts]

    with pytest.raises(index.Cancelled):
        index.index_document(conn, {}, doc_id, embed=fake_embed, parse_page=None)
    assert len(calls) == 1 and len(calls[0]) <= nvidia.EMBED_BATCH
    assert _count(conn, "chunks") == 0


def test_index_cancelled_before_parse_of_each_scanned_page(conn, tmp_path):
    import pymupdf
    pdf = pymupdf.open()
    for _ in range(2):
        pdf.new_page().draw_rect(pymupdf.Rect(50, 50, 200, 200), color=(0, 0, 0), fill=(0, 0, 0))
    src = tmp_path / "s.pdf"
    pdf.save(src)
    doc_id = index.store_file(conn, db.create_workspace(conn, "w"), src, dest_root=tmp_path / "files")
    calls = []

    def fake_parse(cfg, png):
        calls.append(1)
        db.delete_document(conn, doc_id)
        return "OCR"

    with pytest.raises(index.Cancelled):
        index.index_document(conn, {}, doc_id, embed=None, parse_page=fake_parse)
    assert len(calls) == 1


def _pdf_doc(conn, tmp_path, pages):
    import pymupdf
    pdf = pymupdf.open()
    for text in pages:
        page = pdf.new_page()
        if text:
            page.insert_text((72, 72), text)
    src = tmp_path / "p.pdf"
    pdf.save(src)
    return index.store_file(conn, db.create_workspace(conn, "w"), src, dest_root=tmp_path / "files")


def _embed(cfg, texts, input_type):
    return [[1.0, 0, 0, 0] for _ in texts]


def test_index_sets_total_pages_before_parsing(conn, tmp_path):
    doc_id = _pdf_doc(conn, tmp_path, ["Hello text page with enough chars", None])
    seen = []

    def fake_parse(cfg, png):
        seen.append(db.get_document(conn, doc_id)["pages"])
        return "OCR TEXT"

    index.index_document(conn, {}, doc_id, embed=_embed, parse_page=fake_parse)
    assert seen == [2]


def test_index_records_pages_done_and_ocr_pages(conn, tmp_path):
    doc_id = _pdf_doc(conn, tmp_path, ["Hello text page with enough chars", None, "Third page with enough chars"])
    index.index_document(conn, {}, doc_id, embed=_embed, parse_page=lambda c, png: "OCR TEXT")
    row = db.get_document(conn, doc_id)
    assert (row["status"], row["pages"], row["pages_done"], row["ocr_pages"]) == ("indexed", 3, 3, 1)


def test_force_ocr_reindex_replaces_chunks(conn, tmp_path):
    doc_id = _pdf_doc(conn, tmp_path, ["Hello text page with enough chars", "Second page with enough chars"])
    index.index_document(conn, {}, doc_id, embed=_embed, parse_page=lambda c, png: "OCR TEXT")
    assert db.get_document(conn, doc_id)["ocr_pages"] == 0 and _count(conn, "chunks") > 0
    index.reindex(conn, {}, doc_id, force_ocr=True, embed=_embed, parse_page=lambda c, png: "FAKE OCR")
    row = db.get_document(conn, doc_id)
    texts = [r[0] for r in conn.execute("SELECT text FROM chunks")]
    assert texts and all(t == "FAKE OCR" for t in texts)
    assert (row["status"], row["ocr_pages"]) == ("indexed", 2)


def test_final_pages_is_total_page_count_even_with_blank_pages(conn, tmp_path):
    doc_id = _pdf_doc(conn, tmp_path, ["Hello text page with enough chars", None, "Third page with enough chars"])
    index.index_document(conn, {}, doc_id, embed=_embed, parse_page=None)
    row = db.get_document(conn, doc_id)
    assert (row["status"], row["pages"], row["pages_done"], row["ocr_pages"]) == ("indexed", 3, 3, 0)


def test_index_sets_total_pages_before_first_page_ocr(conn, tmp_path):
    doc_id = _pdf_doc(conn, tmp_path, [None, "Second page with enough chars"])
    seen = []

    def fake_parse(cfg, png):
        row = db.get_document(conn, doc_id)
        seen.append((row["pages"], row["pages_done"]))
        return "OCR TEXT"

    index.index_document(conn, {}, doc_id, embed=_embed, parse_page=fake_parse)
    assert seen == [(2, 0)]


def test_reset_document_clears_chunks_and_sets_pending(conn, tmp_path):
    doc_id = _pdf_doc(conn, tmp_path, ["Hello text page with enough chars"])
    index.index_document(conn, {}, doc_id, embed=_embed, parse_page=None)
    index.reset_document(conn, doc_id)
    row = db.get_document(conn, doc_id)
    assert (row["status"], row["pages_done"], _count(conn, "chunks")) == ("pending", 0, 0)
