import pytest

from localrag import db


@pytest.fixture
def conn(tmp_path):
    return db.connect(tmp_path / "t.db", embed_dim=4)


def _count(conn, table):
    return conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


def _doc(conn, name="d.txt"):
    ws = db.create_workspace(conn, "w")
    return ws, db.create_document(conn, ws, name, "/p/" + name)


def test_schema_is_idempotent(tmp_path):
    db.connect(tmp_path / "t.db", embed_dim=4).close()
    conn = db.connect(tmp_path / "t.db", embed_dim=4)
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_create_and_list_workspaces(conn):
    a = db.create_workspace(conn, "a")
    b = db.create_workspace(conn, "b")
    assert [(r["id"], r["name"]) for r in db.list_workspaces(conn)] == [(a, "a"), (b, "b")]


def test_delete_workspace_cascades_documents_chunks_sessions_messages_and_vec(conn):
    ws, d = _doc(conn)
    db.add_chunks(conn, d, ws, [(0, 1, "hello", [1, 0, 0, 0])])
    s = db.create_session(conn, ws, "t")
    db.add_message(conn, s, "user", "hi")
    db.delete_workspace(conn, ws)
    for t in ["workspaces", "documents", "chunks", "chunks_vec", "sessions", "messages"]:
        assert _count(conn, t) == 0, t


def test_create_document_defaults_to_pending(conn):
    ws, d = _doc(conn)
    (row,) = db.list_documents(conn, ws)
    assert (row["id"], row["status"], row["name"], row["path"]) == (d, "pending", "d.txt", "/p/d.txt")


def test_set_document_status_with_pages_and_error(conn):
    ws, d = _doc(conn)
    db.set_document_status(conn, d, "failed", pages=3, error="boom")
    (row,) = db.list_documents(conn, ws)
    assert (row["status"], row["pages"], row["error"]) == ("failed", 3, "boom")


def test_delete_document_removes_chunks_fts_and_vec(conn):
    ws, d = _doc(conn)
    db.add_chunks(conn, d, ws, [(0, 1, "zebra", [1, 0, 0, 0])])
    db.delete_document(conn, d)
    assert _count(conn, "chunks") == 0
    assert _count(conn, "chunks_vec") == 0
    assert db.fts_search(conn, ws, "zebra", 5) == []


def test_add_chunks_returns_ids_in_order(conn):
    ws, d = _doc(conn)
    ids = db.add_chunks(conn, d, ws, [(i, 1, f"t{i}", [1, 0, 0, 0]) for i in range(3)])
    rows = conn.execute("SELECT id, ord FROM chunks ORDER BY id").fetchall()
    assert ids == [r["id"] for r in rows] and [r["ord"] for r in rows] == [0, 1, 2]


def test_add_chunks_is_atomic(conn):
    ws, d = _doc(conn)
    bad = [(0, 1, "a", [1, 0, 0, 0]), (1, 1, "b", [1, 0]), (2, 1, "c", [1, 0, 0, 0])]
    with pytest.raises(Exception):
        db.add_chunks(conn, d, ws, bad)
    assert _count(conn, "chunks") == 0 and _count(conn, "chunks_vec") == 0


def test_get_chunks_preserves_order_and_joins_document_name(conn):
    ws, d = _doc(conn, "doc.pdf")
    ids = db.add_chunks(conn, d, ws, [(0, 1, "a", [1, 0, 0, 0]), (1, 2, "b", [0, 1, 0, 0])])
    rows = db.get_chunks(conn, ids[::-1])
    assert [r["id"] for r in rows] == ids[::-1]
    assert [(r["text"], r["page"], r["document_id"], r["document_name"]) for r in rows] == [
        ("b", 2, d, "doc.pdf"), ("a", 1, d, "doc.pdf")]


def test_fts_search_finds_exact_token_in_workspace_only(conn):
    ws1, d1 = _doc(conn)
    ws2, d2 = _doc(conn)
    (c1,) = db.add_chunks(conn, d1, ws1, [(0, 1, "the quick fox", [1, 0, 0, 0])])
    db.add_chunks(conn, d2, ws2, [(0, 1, "the quick fox", [1, 0, 0, 0])])
    res = db.fts_search(conn, ws1, "fox", 10)
    assert [r[0] for r in res] == [c1] and isinstance(res[0][1], float)


def test_fts_search_tolerates_quotes_and_operators(conn):
    ws, d = _doc(conn)
    db.add_chunks(conn, d, ws, [(0, 1, "what is this", [1, 0, 0, 0])])
    assert isinstance(db.fts_search(conn, ws, 'what\'s "this" (AND) NOT', 5), list)


def test_fts_search_matches_cyrillic(conn):
    ws, d = _doc(conn)
    (c,) = db.add_chunks(conn, d, ws, [(0, 1, "Київ є столицею України", [1, 0, 0, 0])])
    assert [r[0] for r in db.fts_search(conn, ws, "столицею", 5)] == [c]


def test_fts_search_empty_query_returns_empty(conn):
    ws, _ = _doc(conn)
    assert db.fts_search(conn, ws, "   ", 5) == []


def test_vec_search_orders_by_cosine_distance(conn):
    ws, d = _doc(conn)
    a, b, c = db.add_chunks(conn, d, ws, [
        (0, 1, "a", [1, 0, 0, 0]), (1, 1, "b", [0, 1, 0, 0]), (2, 1, "c", [1, 1, 0, 0])])
    res = db.vec_search(conn, ws, [0.9, 0.1, 0, 0], 3)
    assert [r[0] for r in res] == [a, c, b]
    assert res[0][1] < res[1][1] < res[2][1]


def test_vec_search_respects_workspace_partition(conn):
    ws1, d1 = _doc(conn)
    ws2, d2 = _doc(conn)
    (c1,) = db.add_chunks(conn, d1, ws1, [(0, 1, "a", [0, 1, 0, 0])])
    db.add_chunks(conn, d2, ws2, [(0, 1, "a", [1, 0, 0, 0])])
    assert [r[0] for r in db.vec_search(conn, ws1, [1, 0, 0, 0], 5)] == [c1]


def test_sessions_crud(conn):
    ws = db.create_workspace(conn, "w")
    s1 = db.create_session(conn, ws, "one")
    s2 = db.create_session(conn, ws, "two")
    assert [(r["id"], r["title"]) for r in db.list_sessions(conn, ws)] == [(s1, "one"), (s2, "two")]


def test_messages_round_trip_citations_json(conn):
    s = db.create_session(conn, db.create_workspace(conn, "w"), "t")
    m1 = db.add_message(conn, s, "user", "q")
    m2 = db.add_message(conn, s, "assistant", "a", citations=[{"chunk_id": 1, "page": 2}])
    msgs = db.list_messages(conn, s)
    assert [m["id"] for m in msgs] == [m1, m2]
    assert msgs[0]["citations"] is None
    assert msgs[1]["citations"] == [{"chunk_id": 1, "page": 2}]
    assert msgs[1]["role"] == "assistant" and msgs[1]["content"] == "a"


def test_get_document_returns_row(conn):
    ws, d = _doc(conn)
    row = db.get_document(conn, d)
    assert (row["id"], row["workspace_id"], row["name"]) == (d, ws, "d.txt")


def test_set_document_path_updates_path(conn):
    _, d = _doc(conn)
    db.set_document_path(conn, d, "/new/path")
    assert db.get_document(conn, d)["path"] == "/new/path"
