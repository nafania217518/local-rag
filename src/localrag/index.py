import hashlib
import logging
import re
import shutil
from pathlib import Path

from localrag import config, db, nvidia
from localrag.parsers import parse_file

log = logging.getLogger(__name__)

def _bodies(text, size):
    bodies, cur = [], ""
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        pieces = [para[i:i + size] for i in range(0, len(para), size)]
        for piece in pieces:
            joined = f"{cur}\n\n{piece}" if cur else piece
            if len(joined) <= size:
                cur = joined
            else:
                bodies.append(cur)
                cur = piece
    if cur:
        bodies.append(cur)
    return bodies


def chunk_text(text, size=config.DEFAULTS["chunk_size"], overlap=config.DEFAULTS["chunk_overlap"]):
    chunks = []
    for body in _bodies(text, size):
        prefix = chunks[-1][-overlap:] if chunks and overlap else ""
        chunk = (prefix + body).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def chunk_pages(pages, cfg):
    out = []
    for page, text in pages:
        for chunk in chunk_text(text, cfg["chunk_size"], cfg["chunk_overlap"]):
            out.append((len(out), page, chunk))
    return out


def files_dir() -> Path:
    return config.app_data_dir() / "files"


class Cancelled(Exception):
    """The document was deleted while it was being indexed."""


def _check(conn, document_id):
    if db.get_document(conn, document_id) is None:
        raise Cancelled(document_id)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def store_file(conn, workspace_id, src_path, dest_root=None, name=None):
    src = Path(src_path)
    doc_id = db.create_document(conn, workspace_id, name=name or src.name, path="")
    dest = Path(dest_root or files_dir()) / str(workspace_id) / f"{doc_id}{src.suffix}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dest)
    with conn:
        conn.execute("UPDATE documents SET path = ?, sha256 = ? WHERE id = ?",
                     [str(dest), file_sha256(dest), doc_id])
    return doc_id


def _error_text(exc):
    text = str(exc)
    body = getattr(getattr(exc, "response", None), "text", "")  # httpx errors keep the server message only here
    return f"{text} - {body[:500]}" if body and body not in text else text


def index_document(conn, cfg, document_id, *, embed=nvidia.embed, parse_page=nvidia.parse_page,
                   force_ocr=False):
    name = None
    try:
        doc = db.get_document(conn, document_id)
        name = doc["name"]
        log.info("indexing %d %s", document_id, name)
        checked_parse = parse_page and (lambda c, png: (_check(conn, document_id), parse_page(c, png))[1])
        ocr, totals = [], []

        def on_page(done, total, via_ocr):
            ocr.append(via_ocr)
            totals.append(total)
            db.set_document_progress(conn, document_id, done, pages=total)

        pages = parse_file(doc["path"], cfg, checked_parse, force_ocr=force_ocr, on_page=on_page)
        chunks = chunk_pages(pages, {**config.DEFAULTS, **cfg})
        if chunks:
            texts = [t for _, _, t in chunks]
            vectors = []
            for i in range(0, len(texts), nvidia.EMBED_BATCH):
                _check(conn, document_id)
                vectors += embed(cfg, texts[i:i + nvidia.EMBED_BATCH], "passage")
            _check(conn, document_id)
            db.add_chunks(conn, document_id, doc["workspace_id"],
                          [(o, p, t, v) for (o, p, t), v in zip(chunks, vectors)])
        pages_total = totals[-1] if totals else len(pages)
        db.set_document_status(conn, document_id, "indexed", pages=pages_total, ocr_pages=sum(ocr))
        log.info("indexed %d %s: %d pages, %d via OCR", document_id, name, pages_total, sum(ocr))
    except Cancelled as exc:
        log.info("indexing %d cancelled", document_id)
        db.set_document_status(conn, document_id, "failed", error=str(exc))  # as before: a no-op, the row is gone
        raise
    except Exception as exc:
        error = _error_text(exc)
        log.exception("indexing %d %s failed: %s", document_id, name, error)
        db.set_document_status(conn, document_id, "failed", error=error)
        raise


def reset_document(conn, document_id):
    db.delete_chunks(conn, document_id)
    db.set_document_status(conn, document_id, "pending")
    db.set_document_progress(conn, document_id, 0)


def reindex(conn, cfg, document_id, *, force_ocr=False, **kw):
    reset_document(conn, document_id)
    index_document(conn, cfg, document_id, force_ocr=force_ocr, **kw)
