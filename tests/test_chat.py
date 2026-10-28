import pytest

from localrag import chat, config, db

CHUNKS = [
    {"id": 11, "document_id": 1, "document_name": "a.pdf", "page": 3, "text": "Kyiv is the capital.", "score": 0.9},
    {"id": 12, "document_id": 2, "document_name": "b.docx", "page": 1, "text": "Dnipro river.", "score": 0.5},
]


@pytest.fixture
def env(tmp_path):
    conn = db.connect(tmp_path / "t.db", embed_dim=4)
    ws = db.create_workspace(conn, "w")
    s = db.create_session(conn, ws, "t")
    cfg = {**config.DEFAULTS, "api_key": "k"}
    return conn, cfg, ws, s


def fake_retrieve(chunks=CHUNKS, seen=None):
    def f(conn, cfg, workspace_id, question):
        if seen is not None:
            seen.append(workspace_id)
        return chunks
    return f


def fake_stream(seen=None, fail=False):
    def f(cfg, messages):
        if seen is not None:
            seen.append(messages)
        yield ("thinking", "hmm")
        yield ("token", "Kyiv")
        if fail:
            raise RuntimeError("boom")
        yield ("token", " [1]")
    return f


def run(env, question="Capital?", **kw):
    conn, cfg, _, s = env
    kw.setdefault("retrieve", fake_retrieve())
    kw.setdefault("chat_stream", fake_stream())
    return list(chat.answer(conn, cfg, s, question, **kw))


def test_format_context_numbers_blocks_with_document_and_page():
    assert chat.format_context(CHUNKS) == "[1] a.pdf, p.3\nKyiv is the capital.\n\n[2] b.docx, p.1\nDnipro river."


def test_format_context_empty():
    assert chat.format_context([]) == ""


def test_build_messages_starts_with_system_prompt_rules():
    first = chat.build_messages([], CHUNKS, "q")[0]
    assert first["role"] == "system"
    for phrase in ["only", "[n]", "language of the question"]:
        assert phrase in first["content"]


def test_build_messages_limits_history_to_10():
    history = [{"role": "user", "content": str(i)} for i in range(15)]
    msgs = chat.build_messages(history, CHUNKS, "q")
    assert [m["content"] for m in msgs[1:-1]] == [str(i) for i in range(5, 15)]


def test_build_messages_strips_extra_history_keys():
    history = [{"id": 1, "role": "assistant", "content": "x", "citations": [], "session_id": 2}]
    assert chat.build_messages(history, CHUNKS, "q")[1] == {"role": "assistant", "content": "x"}


def test_build_messages_ends_with_context_and_question():
    last = chat.build_messages([], CHUNKS, "Capital?")[-1]
    assert last == {"role": "user", "content": f"Context:\n{chat.format_context(CHUNKS)}\n\nQuestion: Capital?"}


def test_build_messages_no_chunks_says_no_passages():
    last = chat.build_messages([], [], "q")[-1]
    assert last["content"] == "Context:\n(no passages found)\n\nQuestion: q"


def test_answer_streams_thinking_tokens_citations_done_in_order(env):
    assert [k for k, _ in run(env)] == ["thinking", "token", "token", "citations", "done"]


def test_answer_stores_user_then_assistant_with_citations(env):
    conn, _, _, s = env
    run(env)
    msgs = db.list_messages(conn, s)
    assert [(m["role"], m["content"]) for m in msgs] == [("user", "Capital?"), ("assistant", "Kyiv [1]")]
    cits = msgs[1]["citations"]
    assert [c["n"] for c in cits] == [1, 2]
    assert [c["document_name"] for c in cits] == ["a.pdf", "b.docx"]


def test_answer_passes_prior_history_but_not_current_question_twice(env):
    conn, _, _, s = env
    db.add_message(conn, s, "user", "Earlier?")
    db.add_message(conn, s, "assistant", "Earlier answer")
    seen = []
    run(env, question="Unique question?", chat_stream=fake_stream(seen))
    msgs = seen[0]
    assert msgs[1:-1] == [{"role": "user", "content": "Earlier?"}, {"role": "assistant", "content": "Earlier answer"}]
    assert sum(m["content"].count("Unique question?") for m in msgs) == 1


def test_answer_uses_session_workspace_for_retrieval(env):
    seen = []
    run(env, retrieve=fake_retrieve(seen=seen))
    assert seen == [env[2]]


def test_answer_does_not_store_assistant_message_when_stream_fails(env):
    conn, _, _, s = env
    with pytest.raises(RuntimeError):
        run(env, chat_stream=fake_stream(fail=True))
    assert [m["role"] for m in db.list_messages(conn, s)] == ["user"]


def test_answer_with_no_chunks_yields_empty_citations(env):
    events = run(env, retrieve=fake_retrieve(chunks=[]))
    assert ("citations", []) in events
