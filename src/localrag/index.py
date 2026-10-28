import hashlib
import re
import shutil
from pathlib import Path

from localrag import config, db, nvidia
from localrag.parsers import parse_file

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200


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


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    for body in _bodies(text, size):
        prefix = chunks[-1][-overlap:] if chunks and overlap else ""
        chunk = (prefix + body).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def chunk_pages(pages):
    out = []
    for page, text in pages:
        for chunk in chunk_text(text):
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


def index_document(conn, cfg, document_id, *, embed=nvidia.embed, parse_page=nvidia.parse_page,
                   force_ocr=False):
    try:
        doc = db.get_document(conn, document_id)
        checked_parse = parse_page and (lambda c, png: (_check(conn, document_id), parse_page(c, png))[1])
        ocr, totals = [], []

        def on_page(done, total, via_ocr):
            ocr.append(via_ocr)
            totals.append(total)
            db.set_document_progress(conn, document_id, done, pages=total)

        pages = parse_file(doc["path"], cfg, checked_parse, force_ocr=force_ocr, on_page=on_page)
        chunks = chunk_pages(pages)
        if chunks:
            texts = [t for _, _, t in chunks]
            vectors = []
            for i in range(0, len(texts), nvidia.EMBED_BATCH):
                _check(conn, document_id)
                vectors += embed(cfg, texts[i:i + nvidia.EMBED_BATCH], "passage")
            _check(conn, document_id)
            db.add_chunks(conn, document_id, doc["workspace_id"],
                          [(o, p, t, v) for (o, p, t), v in zip(chunks, vectors)])
        db.set_document_status(conn, document_id, "indexed", pages=totals[-1] if totals else len(pages), ocr_pages=sum(ocr))
    except Exception as exc:
        db.set_document_status(conn, document_id, "failed", error=str(exc))
        raise


def reset_document(conn, document_id):
    db.delete_chunks(conn, document_id)
    db.set_document_status(conn, document_id, "pending")
    db.set_document_progress(conn, document_id, 0)


def reindex(conn, cfg, document_id, *, force_ocr=False, **kw):
    reset_document(conn, document_id)
    index_document(conn, cfg, document_id, force_ocr=force_ocr, **kw)
