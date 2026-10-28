import json
import os
import sqlite3

import sqlite_vec

from localrag import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS workspaces(id INTEGER PRIMARY KEY, name TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS documents(id INTEGER PRIMARY KEY,
    workspace_id INTEGER NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    name TEXT NOT NULL, path TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
    pages INTEGER, error TEXT, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS chunks(id INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    workspace_id INTEGER NOT NULL, ord INTEGER NOT NULL, page INTEGER NOT NULL, text TEXT NOT NULL);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text, content='chunks',
    content_rowid='id', tokenize = 'unicode61 remove_diacritics 2');
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(chunk_id INTEGER PRIMARY KEY,
    workspace_id INTEGER PARTITION KEY, embedding float[{dim}] distance_metric=cosine);
CREATE TABLE IF NOT EXISTS sessions(id INTEGER PRIMARY KEY,
    workspace_id INTEGER NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    title TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS messages(id INTEGER PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL, content TEXT NOT NULL, citations TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
"""


def connect(path: str | os.PathLike, embed_dim: int = config.EMBED_DIM) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    init_schema(conn, embed_dim)
    return conn


def init_schema(conn, embed_dim):
    conn.executescript(SCHEMA.replace("{dim}", str(int(embed_dim))))


def _insert(conn, sql, params):
    with conn:
        return conn.execute(sql, params).lastrowid


def create_workspace(conn, name):
    return _insert(conn, "INSERT INTO workspaces(name) VALUES (?)", [name])


def list_workspaces(conn):
    return conn.execute("SELECT * FROM workspaces ORDER BY id").fetchall()


def delete_workspace(conn, workspace_id):
    with conn:
        conn.execute("DELETE FROM chunks_vec WHERE workspace_id = ?", [workspace_id])
        conn.execute("DELETE FROM workspaces WHERE id = ?", [workspace_id])


def create_document(conn, workspace_id, name, path):
    return _insert(conn, "INSERT INTO documents(workspace_id, name, path) VALUES (?, ?, ?)",
                   [workspace_id, name, path])


def list_documents(conn, workspace_id):
    return conn.execute("SELECT * FROM documents WHERE workspace_id = ? ORDER BY id",
                        [workspace_id]).fetchall()


def get_document(conn, document_id):
    return conn.execute("SELECT * FROM documents WHERE id = ?", [document_id]).fetchone()


def set_document_path(conn, document_id, path):
    with conn:
        conn.execute("UPDATE documents SET path = ? WHERE id = ?", [path, document_id])


def set_document_status(conn, document_id, status, pages=None, error=None):
    with conn:
        conn.execute("UPDATE documents SET status = ?, pages = ?, error = ? WHERE id = ?",
                     [status, pages, error, document_id])


def delete_document(conn, document_id):
    with conn:
        conn.execute("DELETE FROM chunks_vec WHERE chunk_id IN "
                     "(SELECT id FROM chunks WHERE document_id = ?)", [document_id])
        conn.execute("DELETE FROM documents WHERE id = ?", [document_id])


def add_chunks(conn, document_id, workspace_id, chunks):
    ids = []
    with conn:
        for ord_, page, text, embedding in chunks:
            cid = conn.execute(
                "INSERT INTO chunks(document_id, workspace_id, ord, page, text) VALUES (?, ?, ?, ?, ?)",
                [document_id, workspace_id, ord_, page, text]).lastrowid
            conn.execute("INSERT INTO chunks_vec(chunk_id, workspace_id, embedding) VALUES (?, ?, ?)",
                         [cid, workspace_id, sqlite_vec.serialize_float32(embedding)])
            ids.append(cid)
    return ids


def get_chunks(conn, ids):
    ids = list(ids)
    rows = conn.execute(
        "SELECT c.id, c.document_id, c.page, c.text, d.name AS document_name FROM chunks c "
        f"JOIN documents d ON d.id = c.document_id WHERE c.id IN ({','.join('?' * len(ids))})",
        ids).fetchall()
    by_id = {r["id"]: r for r in rows}
    return [by_id[i] for i in ids if i in by_id]


def fts_search(conn, workspace_id, query, k):
    expr = " OR ".join('"' + t.replace('"', '""') + '"' for t in query.split())
    if not expr:
        return []
    rows = conn.execute(
        "SELECT c.id, chunks_fts.rank FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid "
        "WHERE chunks_fts MATCH ? AND c.workspace_id = ? ORDER BY chunks_fts.rank LIMIT ?",
        [expr, workspace_id, k]).fetchall()
    return [(r[0], r[1]) for r in rows]


def vec_search(conn, workspace_id, embedding, k):
    rows = conn.execute(
        "SELECT chunk_id, distance FROM chunks_vec "
        "WHERE embedding MATCH ? AND workspace_id = ? AND k = ? ORDER BY distance",
        [sqlite_vec.serialize_float32(embedding), workspace_id, k]).fetchall()
    return [(r[0], r[1]) for r in rows]


def create_session(conn, workspace_id, title):
    return _insert(conn, "INSERT INTO sessions(workspace_id, title) VALUES (?, ?)",
                   [workspace_id, title])


def list_sessions(conn, workspace_id):
    return conn.execute("SELECT * FROM sessions WHERE workspace_id = ? ORDER BY id",
                        [workspace_id]).fetchall()


def add_message(conn, session_id, role, content, citations=None):
    cit = None if citations is None else json.dumps(citations)
    return _insert(conn, "INSERT INTO messages(session_id, role, content, citations) VALUES (?, ?, ?, ?)",
                   [session_id, role, content, cit])


def list_messages(conn, session_id):
    rows = conn.execute("SELECT * FROM messages WHERE session_id = ? ORDER BY id",
                        [session_id]).fetchall()
    return [{**dict(r), "citations": r["citations"] and json.loads(r["citations"])} for r in rows]


def get_session(conn, session_id):
    return conn.execute("SELECT * FROM sessions WHERE id = ?", [session_id]).fetchone()


def get_workspace(conn, workspace_id):
    return conn.execute("SELECT * FROM workspaces WHERE id = ?", [workspace_id]).fetchone()
