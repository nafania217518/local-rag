from collections.abc import Iterator

from localrag import db, nvidia, retrieval


def format_context(chunks: list[dict]) -> str:
    return "\n\n".join(f"[{i}] {c['document_name']}, p.{c['page']}\n{c['text']}"
                       for i, c in enumerate(chunks, 1))


def build_messages(cfg: dict, history: list[dict], chunks: list[dict], question: str) -> list[dict]:
    context = format_context(chunks) or "(no passages found)"
    return ([{"role": "system", "content": cfg["system_prompt"]}]
            + [{"role": m["role"], "content": m["content"]} for m in history[-cfg["history_limit"]:]]
            + [{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}])


def answer(conn, cfg, session_id, question, *, retrieve=retrieval.retrieve,
           chat_stream=nvidia.chat_stream) -> Iterator[tuple[str, object]]:
    session = db.get_session(conn, session_id)
    history = db.list_messages(conn, session_id)
    if not history:
        db.set_session_title(conn, session_id, question[:60])
    db.add_message(conn, session_id, "user", question)
    chunks = retrieve(conn, cfg, session["workspace_id"], question)
    parts = []
    for kind, text in chat_stream(cfg, build_messages(cfg, history, chunks, question)):
        if kind == "token":
            parts.append(text)
        yield kind, text
    citations = [{"n": i, **{k: c[k] for k in ("id", "document_id", "document_name", "page", "text")}}
                 for i, c in enumerate(chunks, 1)]
    db.add_message(conn, session_id, "assistant", "".join(parts), citations)
    yield "citations", citations
    yield "done", None
