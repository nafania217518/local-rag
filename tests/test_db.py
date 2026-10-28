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
