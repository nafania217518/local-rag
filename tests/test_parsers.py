import docx
import pymupdf
import pytest

from localrag import parsers


def _pdf(path, pages):
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text)
        else:
            page.draw_rect(pymupdf.Rect(50, 50, 200, 200), color=(0, 0, 0), fill=(0, 0, 0))
    doc.save(path)
    return path


class FakeParse:
    def __init__(self):
        self.calls = []

    def __call__(self, cfg, png):
        self.calls.append(png)
        return "OCR TEXT"


def test_parse_pdf_extracts_text_per_page(tmp_path):
    p = _pdf(tmp_path / "a.pdf", ["Hello text page", "Second page sentence here"])
    assert [(n, t.strip()) for n, t in parsers.parse_pdf(p)] == [
        (1, "Hello text page"), (2, "Second page sentence here")]


def test_parse_pdf_routes_pages_without_text_to_parse_page(tmp_path):
    fake = FakeParse()
    p = _pdf(tmp_path / "s.pdf", [None])
    assert parsers.parse_pdf(p, {}, fake) == [(1, "OCR TEXT")]
    assert fake.calls[0].startswith(b"\x89PNG")


def test_parse_pdf_mixed_only_parses_image_pages(tmp_path):
    fake = FakeParse()
    p = _pdf(tmp_path / "m.pdf", ["Hello text page with enough chars", None])
    result = parsers.parse_pdf(p, {}, fake)
    assert len(fake.calls) == 1
    assert result[1] == (2, "OCR TEXT")
    assert "Hello text page" in result[0][1]


def test_parse_pdf_without_parse_page_skips_empty_pages(tmp_path):
    p = _pdf(tmp_path / "m.pdf", ["Hello text page", None])
    assert [n for n, _ in parsers.parse_pdf(p)] == [1]


def test_parse_docx_includes_paragraphs_and_table_cells(tmp_path):
    d = docx.Document()
    d.add_paragraph("First para")
    d.add_paragraph("Second para")
    t = d.add_table(rows=2, cols=2)
    for r in range(2):
        for c in range(2):
            t.cell(r, c).text = f"c{r}{c}"
    d.save(tmp_path / "a.docx")
    [(page, text)] = parsers.parse_docx(tmp_path / "a.docx")
    assert page == 1
    assert text == "First para\n\nSecond para\n\nc00 | c01\n\nc10 | c11"


def test_parse_txt_and_md(tmp_path):
    (tmp_path / "a.txt").write_text("plain", encoding="utf-8")
    (tmp_path / "b.md").write_text("# md", encoding="utf-8")
    assert parsers.parse_txt(tmp_path / "a.txt") == [(1, "plain")]
    assert parsers.parse_file(tmp_path / "b.md") == [(1, "# md")]


def test_parse_file_dispatches_by_suffix_case_insensitive(tmp_path):
    p = _pdf(tmp_path / "A.PDF", ["Hello text page"])
    assert parsers.parse_file(p)[0][1].strip() == "Hello text page"


def test_parse_file_rejects_unknown_suffix(tmp_path):
    with pytest.raises(ValueError, match="unsupported file type: .xyz"):
        parsers.parse_file(tmp_path / "a.xyz")
