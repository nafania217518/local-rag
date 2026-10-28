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


def store_file(conn, workspace_id, src_path, dest_root=None, name=None):
    src = Path(src_path)
    doc_id = db.create_document(conn, workspace_id, name=name or src.name, path="")
    dest = Path(dest_root or files_dir()) / str(workspace_id) / f"{doc_id}{src.suffix}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dest)
    db.set_document_path(conn, doc_id, str(dest))
    return doc_id


def index_document(conn, cfg, document_id, *, embed=nvidia.embed, parse_page=nvidia.parse_page):
    try:
        doc = db.get_document(conn, document_id)
        pages = parse_file(doc["path"], cfg, parse_page)
        chunks = chunk_pages(pages)
        if chunks:
            vectors = embed(cfg, [t for _, _, t in chunks], "passage")
            db.add_chunks(conn, document_id, doc["workspace_id"],
                          [(o, p, t, v) for (o, p, t), v in zip(chunks, vectors)])
        db.set_document_status(conn, document_id, "indexed", pages=len(pages))
    except Exception as exc:
        db.set_document_status(conn, document_id, "failed", error=str(exc))
        raise
