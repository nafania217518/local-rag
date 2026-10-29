import json
import threading

import pytest
from fastapi.testclient import TestClient

from localrag import api, config


def fake_embed(cfg, texts, input_type):
    return [[0.0] * config.EMBED_DIM for _ in texts]


def fake_rerank(cfg, query, passages):
    return [(i, 1.0) for i in range(len(passages))]


def fake_chat_stream(cfg, messages, *a, **k):
    yield from [("thinking", "hmm"), ("token", "Kyiv"), ("token", " [1]")]


def fake_parse(cfg, png):
    return ""


def make(tmp_path, **kw):
    args = dict(embed=fake_embed, rerank=fake_rerank, chat_stream=fake_chat_stream,
                parse_page=fake_parse, run_in_background=lambda fn: fn())
    args.update(kw)
    return TestClient(api.create_app(tmp_path / "t.db", tmp_path / "config.json", tmp_path / "files", **args))


@pytest.fixture
def client(tmp_path):
    return make(tmp_path)


def sse(body):
    out = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.split("\n"))
        out.append((lines["event"], json.loads(lines["data"])))
    return out


def ws(client, name="w"):
    return client.post("/workspaces", json={"name": name}).json()["id"]


def upload(client, w, *files):
    return client.post(f"/workspaces/{w}/documents",
                       files=[("files", (n, c.encode(), "text/plain")) for n, c in files])


def test_health_reports_sqlite_vec(client):
    r = client.get("/health").json()
    assert r["ok"] is True and r["sqlite_vec"].startswith("v")


def test_create_and_list_workspaces(client):
    r = client.post("/workspaces", json={"name": "a"})
    assert r.status_code == 201 and r.json()["name"] == "a"
    assert [w["name"] for w in client.get("/workspaces").json()] == ["a"]


def test_create_workspace_rejects_blank_name(client):
    assert client.post("/workspaces", json={"name": "  "}).status_code == 422
    assert client.post("/workspaces", json={"name": ""}).status_code == 422


def test_delete_workspace_204_and_404(client):
    w = ws(client)
    assert client.delete(f"/workspaces/{w}").status_code == 204
    assert client.delete(f"/workspaces/{w}").status_code == 404


def test_upload_txt_creates_document_and_indexes_it(client, tmp_path):
    w = ws(client)
    r = upload(client, w, ("a.txt", "Kyiv is the capital"))
    assert r.status_code == 202 and len(r.json()) == 1
    doc = client.get(f"/workspaces/{w}/documents").json()[0]
    assert doc["status"] == "indexed" and doc["pages"] == 1 and doc["name"] == "a.txt"
    assert (tmp_path / "files" / str(w) / f"{doc['id']}.txt").exists()
    assert list((tmp_path / "files" / "incoming").iterdir()) == []


def test_upload_many_files_in_one_request(client):
    w = ws(client)
    assert len(upload(client, w, ("a.txt", "one"), ("b.txt", "two")).json()) == 2
    assert [d["status"] for d in client.get(f"/workspaces/{w}/documents").json()] == ["indexed"] * 2


def test_upload_unknown_workspace_404(client):
    assert upload(client, 99, ("a.txt", "x")).status_code == 404
    assert client.get("/workspaces/99/documents").status_code == 404


def test_upload_unsupported_type_marks_failed(client):
    w = ws(client)
    upload(client, w, ("a.xyz", "x"))
    doc = client.get(f"/workspaces/{w}/documents").json()[0]
    assert doc["status"] == "failed" and "unsupported" in doc["error"].lower()


def test_upload_indexing_error_marks_failed_and_does_not_crash_request(tmp_path):
    def boom(*a):
        raise RuntimeError("down")
    client = make(tmp_path, embed=boom)
    w = ws(client)
    assert upload(client, w, ("a.txt", "text")).status_code == 202
    assert client.get(f"/workspaces/{w}/documents").json()[0]["status"] == "failed"


def test_delete_document_removes_file_and_rows(client, tmp_path):
    w = ws(client)
    d = upload(client, w, ("a.txt", "text")).json()[0]["id"]
    path = tmp_path / "files" / str(w) / f"{d}.txt"
    assert client.delete(f"/workspaces/{w + 1}/documents/{d}").status_code == 404
    assert client.delete(f"/workspaces/{w}/documents/{d}").status_code == 204
    assert not path.exists() and client.get(f"/workspaces/{w}/documents").json() == []
    assert client.delete(f"/workspaces/{w}/documents/{d}").status_code == 404


def test_sessions_create_default_title_and_list(client):
    w = ws(client)
    r = client.post(f"/workspaces/{w}/sessions", json={})
    assert r.status_code == 201 and r.json()["title"] == "New chat"
    client.post(f"/workspaces/{w}/sessions", json={"title": " "})
    client.post(f"/workspaces/{w}/sessions", json={"title": "x"})
    assert [s["title"] for s in client.get(f"/workspaces/{w}/sessions").json()] == ["New chat", "New chat", "x"]


def test_messages_404_for_unknown_session(client):
    assert client.get("/sessions/99/messages").status_code == 404


def _session(client):
    w = ws(client)
    upload(client, w, ("a.txt", "Kyiv is the capital"))
    return client.post(f"/workspaces/{w}/sessions", json={}).json()["id"]


def test_chat_streams_sse_events_in_order(client):
    s = _session(client)
    r = client.post(f"/sessions/{s}/chat", json={"question": "capital?"})
    assert r.headers["content-type"].startswith("text/event-stream")
    events = sse(r.text)
    assert [e for e, _ in events] == ["thinking", "token", "token", "citations", "done"]
    assert events[1][1] == "Kyiv" and events[3][1] and events[4][1] is None


def test_chat_persists_messages(client):
    s = _session(client)
    client.post(f"/sessions/{s}/chat", json={"question": "capital?"})
    msgs = client.get(f"/sessions/{s}/messages").json()
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[1]["content"] == "Kyiv [1]" and msgs[1]["citations"]


def test_chat_blank_question_422(client):
    s = _session(client)
    assert client.post(f"/sessions/{s}/chat", json={"question": " "}).status_code == 422


def test_chat_unknown_session_404(client):
    assert client.post("/sessions/99/chat", json={"question": "q"}).status_code == 404


def test_chat_error_mid_stream_emits_error_event(tmp_path):
    def bad(*a, **k):
        yield "token", "Ky"
        raise RuntimeError("model died")
    client = make(tmp_path, chat_stream=bad)
    s = _session(client)
    events = sse(client.post(f"/sessions/{s}/chat", json={"question": "q"}).text)
    assert events[0] == ("token", "Ky")
    assert events[-1][0] == "error" and "model died" in events[-1][1]["message"]


def test_settings_get_defaults_and_put_merges(client, tmp_path):
    assert client.get("/settings").json() == config.DEFAULTS
    r = client.put("/settings", json={"api_key": "k", "rerank": False})
    assert r.json() == {**config.DEFAULTS, "api_key": "k", "rerank": False}
    assert client.get("/settings").json() == r.json()
    assert (tmp_path / "config.json").exists()


def test_background_runner_default_uses_thread(tmp_path, monkeypatch):
    calls = []

    class Rec:
        def __init__(self, target, daemon):
            calls.append(daemon)
            self.target = target

        def start(self):
            calls.append("start")
            self.target()

    monkeypatch.setattr(threading, "Thread", Rec)
    client = TestClient(api.create_app(tmp_path / "t.db", tmp_path / "c.json", tmp_path / "f",
                                       embed=fake_embed, parse_page=fake_parse))
    upload(client, ws(client), ("a.txt", "x"))
    assert calls == [True, "start"]


def test_root_serves_index_html(client):
    r = client.get("/")
    assert r.status_code == 200 and "<title>LocalRAG</title>" in r.text


def test_put_workspace_renames(client):
    w = ws(client, "a")
    r = client.put(f"/workspaces/{w}", json={"name": "b"})
    assert r.status_code == 200 and r.json()["name"] == "b" and r.json()["id"] == w
    assert client.put("/workspaces/999", json={"name": "b"}).status_code == 404
    assert client.put(f"/workspaces/{w}", json={"name": "  "}).status_code == 422


def test_delete_session_route(client):
    s = client.post(f"/workspaces/{ws(client)}/sessions", json={}).json()["id"]
    assert client.delete(f"/sessions/{s}").status_code == 204
    assert client.get(f"/sessions/{s}/messages").status_code == 404
    assert client.delete(f"/sessions/{s}").status_code == 404


def test_upload_same_file_twice_is_skipped(tmp_path):
    runs = []
    client = make(tmp_path, run_in_background=lambda fn: (runs.append(fn), fn()))
    w = ws(client)
    first = upload(client, w, ("a.txt", "same")).json()[0]
    second = upload(client, w, ("a.txt", "same")).json()[0]
    assert second["id"] == first["id"]
    assert len(client.get(f"/workspaces/{w}/documents").json()) == 1
    assert len(runs) == 1
    assert list((tmp_path / "files" / "incoming").iterdir()) == []
    w2 = ws(client, "other")
    assert upload(client, w2, ("a.txt", "same")).json()[0]["id"] != first["id"]
    assert len(runs) == 2


def _pdf_bytes(*texts):
    import pymupdf
    doc = pymupdf.open()
    for t in texts:
        doc.new_page().insert_text((72, 72), t)
    return doc.tobytes()


def _upload_pdf(client, w):
    files = [("files", ("s.pdf", _pdf_bytes("Hello text page with enough chars"), "application/pdf"))]
    return client.post(f"/workspaces/{w}/documents", files=files).json()[0]


def test_run_ocr_route_resets_and_reindexes(tmp_path):
    from localrag import db
    client = make(tmp_path, parse_page=lambda cfg, png: "OCR TEXT")
    w = ws(client)
    d = _upload_pdf(client, w)
    r = client.post(f"/workspaces/{w}/documents/{d['id']}/ocr")
    assert r.status_code == 202 and r.json()["id"] == d["id"]
    row = next(x for x in client.get(f"/workspaces/{w}/documents").json() if x["id"] == d["id"])
    assert (row["status"], row["ocr_pages"], row["pages"]) == ("indexed", 1, 1)
    conn = db.connect(tmp_path / "t.db")
    assert [r[0] for r in conn.execute("SELECT text FROM chunks")] == ["OCR TEXT"]


def test_run_ocr_conflict_while_pending(client, tmp_path):
    from localrag import db
    w = ws(client)
    d = upload(client, w, ("a.txt", "text")).json()[0]["id"]
    db.set_document_status(db.connect(tmp_path / "t.db"), d, "pending")
    assert client.post(f"/workspaces/{w}/documents/{d}/ocr").status_code == 409


def test_run_ocr_unknown_document_404(client):
    w = ws(client)
    d = upload(client, w, ("a.txt", "text")).json()[0]["id"]
    assert client.post(f"/workspaces/{w}/documents/{d + 99}/ocr").status_code == 404
    assert client.post(f"/workspaces/{w + 1}/documents/{d}/ocr").status_code == 404


def test_run_ocr_route_returns_pending_row_before_background_job(tmp_path):
    from localrag import db
    jobs = []
    client = make(tmp_path, parse_page=lambda cfg, png: "OCR TEXT", run_in_background=jobs.append)
    w = ws(client)
    d = _upload_pdf(client, w)["id"]
    jobs.pop()()
    body = client.post(f"/workspaces/{w}/documents/{d}/ocr").json()
    assert (body["status"], body["pages_done"]) == ("pending", 0)
    conn = db.connect(tmp_path / "t.db")
    assert conn.execute("SELECT count(*) FROM chunks").fetchone()[0] == 0
    assert client.get(f"/workspaces/{w}/documents").json()[0]["status"] == "pending"
    jobs.pop()()
    row = client.get(f"/workspaces/{w}/documents").json()[0]
    assert (row["status"], row["ocr_pages"]) == ("indexed", 1)


def test_settings_put_rejects_wrong_types(client):
    assert client.put("/settings", json={"rpm": "abc"}).status_code == 422
    assert client.put("/settings", json={"rerank": "maybe"}).status_code == 422
    assert client.get("/settings").json() == config.DEFAULTS


def test_settings_defaults_route_returns_defaults(client):
    client.put("/settings", json={"sources": 3})
    assert client.get("/settings/defaults").json() == config.DEFAULTS
