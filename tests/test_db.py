import sqlite3

import sqlite_vec

from localrag.db import connect


def test_fts5_available(tmp_path):
    conn = connect(tmp_path / "t.db")
    conn.execute("CREATE VIRTUAL TABLE f USING fts5(t)")
    conn.execute("INSERT INTO f(t) VALUES ('hello world')")
    rows = conn.execute("SELECT t FROM f WHERE f MATCH 'hello'").fetchall()
    assert [r["t"] for r in rows] == ["hello world"]


def test_sqlite_vec_loads(tmp_path):
    conn = connect(tmp_path / "t.db")
    (version,) = conn.execute("SELECT vec_version()").fetchone()
    assert version.startswith("v")


def test_sqlite_vec_knn(tmp_path):
    conn = connect(tmp_path / "t.db")
    conn.execute("CREATE VIRTUAL TABLE v USING vec0(id integer primary key, e float[4])")
    conn.execute("INSERT INTO v(id, e) VALUES (1, ?)", [sqlite_vec.serialize_float32([1, 0, 0, 0])])
    conn.execute("INSERT INTO v(id, e) VALUES (2, ?)", [sqlite_vec.serialize_float32([0, 1, 0, 0])])
    row = conn.execute(
        "SELECT id FROM v WHERE e MATCH ? AND k = 1",
        [sqlite_vec.serialize_float32([0.1, 0.9, 0, 0])],
    ).fetchone()
    assert row["id"] == 2


def test_connect_returns_row_factory(tmp_path):
    conn = connect(tmp_path / "t.db")
    assert isinstance(conn.execute("SELECT 1 AS x").fetchone(), sqlite3.Row)


def test_rename_workspace(tmp_path):
    from localrag import db
    conn = connect(tmp_path / "t.db")
    w = db.create_workspace(conn, "a")
    db.rename_workspace(conn, w, "b")
    assert db.get_workspace(conn, w)["name"] == "b"


def test_delete_session_removes_messages(tmp_path):
    from localrag import db
    conn = connect(tmp_path / "t.db")
    s = db.create_session(conn, db.create_workspace(conn, "a"), "t")
    db.add_message(conn, s, "user", "hi")
    db.delete_session(conn, s)
    assert db.get_session(conn, s) is None and db.list_messages(conn, s) == []


def test_connection_usable_from_another_thread(tmp_path):
    import threading
    conn = connect(tmp_path / "t.db")
    out = {}
    def run():
        try:
            out["r"] = tuple(conn.execute("SELECT 1").fetchone())
        except Exception as e:
            out["e"] = e
    t = threading.Thread(target=run)
    t.start()
    t.join()
    assert "e" not in out and out["r"] == (1,)


def test_schema_adds_sha256_column_to_existing_db(tmp_path):
    p = tmp_path / "old.db"
    old = sqlite3.connect(p)
    old.execute("CREATE TABLE documents(id INTEGER PRIMARY KEY, workspace_id INTEGER NOT NULL, "
                "name TEXT NOT NULL, path TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending', "
                "pages INTEGER, error TEXT, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)")
    old.commit()
    old.close()
    conn = connect(p)
    assert "sha256" in [r[1] for r in conn.execute("PRAGMA table_info(documents)")]


def test_find_document_by_sha256(tmp_path):
    from localrag import db
    conn = connect(tmp_path / "t.db")
    w1, w2 = db.create_workspace(conn, "a"), db.create_workspace(conn, "b")
    d = db.create_document(conn, w1, "a.txt", "", sha256="abc")
    assert db.find_document(conn, w1, "abc")["id"] == d
    assert db.find_document(conn, w2, "abc") is None
    assert db.find_document(conn, w1, "zzz") is None


def test_schema_adds_progress_columns_to_existing_db(tmp_path):
    p = tmp_path / "old.db"
    old = sqlite3.connect(p)
    old.execute("CREATE TABLE documents(id INTEGER PRIMARY KEY, workspace_id INTEGER, name TEXT, "
                "path TEXT, status TEXT, pages INTEGER, error TEXT, sha256 TEXT, created_at TEXT)")
    old.commit()
    old.close()
    conn = connect(p, embed_dim=4)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(documents)")]
    assert "pages_done" in cols and "ocr_pages" in cols


def test_set_document_progress(tmp_path):
    from localrag import db
    conn = connect(tmp_path / "t.db", embed_dim=4)
    d = db.create_document(conn, db.create_workspace(conn, "w"), "a", "p")
    db.set_document_progress(conn, d, 1, pages=3)
    assert (db.get_document(conn, d)["pages_done"], db.get_document(conn, d)["pages"]) == (1, 3)
    db.set_document_progress(conn, d, 2)
    assert (db.get_document(conn, d)["pages_done"], db.get_document(conn, d)["pages"]) == (2, 3)


def test_delete_chunks_removes_fts_and_vec_rows(tmp_path):
    from localrag import db
    conn = connect(tmp_path / "t.db", embed_dim=4)
    ws = db.create_workspace(conn, "w")
    d = db.create_document(conn, ws, "a", "p")
    db.add_chunks(conn, d, ws, [(0, 1, "uniqueword here", [1, 0, 0, 0])])
    db.delete_chunks(conn, d)
    assert conn.execute("SELECT count(*) FROM chunks").fetchone()[0] == 0
    assert conn.execute("SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH ?", ["uniqueword"]).fetchone()[0] == 0
    assert conn.execute("SELECT count(*) FROM chunks_vec").fetchone()[0] == 0
    assert db.get_document(conn, d) is not None
