import functools
import json
import shutil
import threading
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, field_validator

from localrag import chat, config, db, index, nvidia, retrieval


def _thread(fn):
    threading.Thread(target=fn, daemon=True).start()


def _not_blank(v):
    if not v.strip():
        raise ValueError("must not be blank")
    return v


class WorkspaceIn(BaseModel):
    name: str
    _check = field_validator("name")(_not_blank)


class SessionIn(BaseModel):
    title: str | None = None


class ChatIn(BaseModel):
    question: str
    _check = field_validator("question")(_not_blank)


def _found(row):
    if row is None:
        raise HTTPException(404)
    return row


def create_app(db_path, cfg_path, files_root, *, embed=nvidia.embed, rerank=nvidia.rerank,
               chat_stream=nvidia.chat_stream, parse_page=nvidia.parse_page, run_in_background=None):
    run_in_background = run_in_background or _thread
    files_root = Path(files_root)
    cfg_path = Path(cfg_path)
    app = FastAPI()

    def get_conn():
        conn = db.connect(db_path)
        try:
            yield conn
        finally:
            conn.close()

    Conn = Depends(get_conn)

    @app.get("/health")
    def health(conn=Conn):
        conn.execute("SELECT 1 FROM chunks_fts LIMIT 0")
        return {"ok": True, "sqlite_vec": conn.execute("SELECT vec_version()").fetchone()[0]}

    @app.get("/workspaces")
    def list_workspaces(conn=Conn):
        return [dict(r) for r in db.list_workspaces(conn)]

    @app.post("/workspaces", status_code=201)
    def create_workspace(body: WorkspaceIn, conn=Conn):
        return dict(db.get_workspace(conn, db.create_workspace(conn, body.name)))

    @app.delete("/workspaces/{ws}", status_code=204)
    def delete_workspace(ws: int, conn=Conn):
        _found(db.get_workspace(conn, ws))
        db.delete_workspace(conn, ws)
        return Response(status_code=204)

    @app.get("/workspaces/{ws}/documents")
    def list_documents(ws: int, conn=Conn):
        _found(db.get_workspace(conn, ws))
        return [dict(r) for r in db.list_documents(conn, ws)]

    def index_in_background(doc_id):
        def run():
            conn = db.connect(db_path)
            try:
                index.index_document(conn, config.load_config(cfg_path), doc_id,
                                     embed=embed, parse_page=parse_page)
            except Exception:
                pass  # status already recorded as failed by index_document
            finally:
                conn.close()
        run_in_background(run)

    @app.post("/workspaces/{ws}/documents", status_code=202)
    def upload(ws: int, files: list[UploadFile], conn=Conn):
        _found(db.get_workspace(conn, ws))
        incoming = files_root / "incoming"
        incoming.mkdir(parents=True, exist_ok=True)
        ids = []
        for f in files:
            tmp = incoming / f"{uuid.uuid4().hex}{Path(f.filename).suffix}"
            with tmp.open("wb") as out:
                shutil.copyfileobj(f.file, out)
            doc_id = index.store_file(conn, ws, tmp, files_root, name=f.filename)
            tmp.unlink()
            ids.append(doc_id)
        rows = [dict(db.get_document(conn, i)) for i in ids]
        for i in ids:
            index_in_background(i)
        return rows

    @app.delete("/workspaces/{ws}/documents/{doc_id}", status_code=204)
    def delete_document(ws: int, doc_id: int, conn=Conn):
        doc = _found(db.get_document(conn, doc_id))
        if doc["workspace_id"] != ws:
            raise HTTPException(404)
        Path(doc["path"]).unlink(missing_ok=True)
        db.delete_document(conn, doc_id)
        return Response(status_code=204)

    @app.get("/workspaces/{ws}/sessions")
    def list_sessions(ws: int, conn=Conn):
        return [dict(r) for r in db.list_sessions(conn, ws)]

    @app.post("/workspaces/{ws}/sessions", status_code=201)
    def create_session(ws: int, body: SessionIn, conn=Conn):
        title = (body.title or "").strip() or "New chat"
        return dict(db.get_session(conn, db.create_session(conn, ws, title)))

    @app.get("/sessions/{sid}/messages")
    def list_messages(sid: int, conn=Conn):
        _found(db.get_session(conn, sid))
        return db.list_messages(conn, sid)

    retrieve = functools.partial(retrieval.retrieve, embed=embed, rerank=rerank)

    @app.post("/sessions/{sid}/chat")
    def chat_route(sid: int, body: ChatIn, conn=Conn):
        _found(db.get_session(conn, sid))
        cfg = config.load_config(cfg_path)

        def events():
            # Own connection: the request dependency may close before streaming ends.
            sconn = db.connect(db_path)
            try:
                for kind, payload in chat.answer(sconn, cfg, sid, body.question,
                                                 retrieve=retrieve, chat_stream=chat_stream):
                    yield f"event: {kind}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
            except Exception as exc:
                yield f"event: error\ndata: {json.dumps({'message': str(exc)}, ensure_ascii=False)}\n\n"
            finally:
                sconn.close()

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.get("/settings")
    def get_settings():
        return config.load_config(cfg_path)

    @app.put("/settings")
    def put_settings(body: dict):
        cfg = {**config.load_config(cfg_path), **body}
        config.save_config(cfg, cfg_path)
        return cfg

    return app
