import re
from pathlib import Path

import docx
import pymupdf

MIN_TEXT_CHARS = 20
RENDER_DPI = 150
OCR_QUALITY_MIN = 0.8
MAX_TOKEN_LEN = 40


def _bad(tok):
    cyr = any("\u0400" <= c <= "\u04ff" for c in tok)
    lat = any(c.isascii() and c.isalpha() for c in tok)
    return (
        len(tok) > MAX_TOKEN_LEN
        or re.search(r"(.)\1{3}", tok) is not None
        or sum(c.isalnum() for c in tok) * 2 < len(tok)
        or (cyr and lat)
    )


def text_quality(text):
    tokens = re.sub(r"\.{3,}", " ", text).split()
    words = [t for t in tokens if len(t) >= 2]
    if len(words) < 5:
        return 1.0
    score = 1 - sum(map(_bad, words)) / len(words)
    singles = sum(len(t) == 1 and t.isalpha() for t in tokens)
    return min(score, 0.5) if singles > 0.3 * len(words) else score


def is_scanned(page):
    half = page.rect.get_area() / 2
    return any(pymupdf.Rect(i["bbox"]).get_area() >= half for i in page.get_image_info())


def _nonempty(pages):
    return [(n, t) for n, t in pages if t.strip()]


def _needs_ocr(page, text, force_ocr):
    if force_ocr or len(text.strip()) < MIN_TEXT_CHARS:
        return True
    return is_scanned(page) and text_quality(text) < OCR_QUALITY_MIN


def parse_pdf(path, cfg=None, parse_page=None, *, force_ocr=False, on_page=None):
    pages = []
    with pymupdf.open(path) as doc:
        if on_page:
            on_page(0, len(doc), False)
        for n, page in enumerate(doc, 1):
            text = page.get_text()
            via_ocr = bool(parse_page) and _needs_ocr(page, text, force_ocr)
            if via_ocr:
                text = parse_page(cfg, page.get_pixmap(dpi=RENDER_DPI).tobytes("png"))
            pages.append((n, text))
            if on_page:
                on_page(n, len(doc), via_ocr)
    return _nonempty(pages)


def parse_docx(path, cfg=None, parse_page=None, **kw):
    d = docx.Document(path)
    parts = [p.text for p in d.paragraphs]
    parts += [" | ".join(c.text for c in row.cells) for t in d.tables for row in t.rows]
    return _nonempty([(1, "\n\n".join(parts))])


def parse_txt(path, cfg=None, parse_page=None, **kw):
    return _nonempty([(1, Path(path).read_text(encoding="utf-8", errors="replace"))])


PARSERS = {".pdf": parse_pdf, ".docx": parse_docx, ".txt": parse_txt, ".md": parse_txt}


def parse_file(path, cfg=None, parse_page=None, **kw):
    suffix = Path(path).suffix.lower()
    if suffix not in PARSERS:
        raise ValueError(f"unsupported file type: {suffix}")
    return PARSERS[suffix](path, cfg, parse_page, **kw)
