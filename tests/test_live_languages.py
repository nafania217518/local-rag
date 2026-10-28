import os

import pytest

from localrag import chat, config, db, nvidia

pytestmark = [pytest.mark.live,
              pytest.mark.skipif(not os.environ.get("NVIDIA_API_KEY"), reason="needs NVIDIA_API_KEY")]

PASSAGES = [
    "Київ є столицею України. Місто стоїть на річці Дніпро.",
    "Байкал — самое глубокое озеро в мире, его глубина 1642 метра.",
    "The Danube flows through ten countries and ends in the Black Sea.",
]


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    cfg = {**config.DEFAULTS, "api_key": os.environ["NVIDIA_API_KEY"]}
    conn = db.connect(tmp_path_factory.mktemp("live") / "t.db")
    ws = db.create_workspace(conn, "w")
    doc = db.create_document(conn, ws, "facts.txt", "/p/facts.txt")
    vecs = nvidia.embed(cfg, PASSAGES, "passage")
    db.add_chunks(conn, doc, ws, [(i, 1, t, v) for i, (t, v) in enumerate(zip(PASSAGES, vecs))])
    return conn, cfg, ws


def ask(env, question):
    conn, cfg, ws = env
    s = db.create_session(conn, ws, question)
    events = list(chat.answer(conn, cfg, s, question))
    text = "".join(t for k, t in events if k == "token")
    citations = next(v for k, v in events if k == "citations")
    return text, citations


def test_live_ukrainian_question_cites_ukrainian_passage(env):
    text, cits = ask(env, "Яка столиця України?")
    assert "Київ" in text
    assert any(f"[{c['n']}]" in text for c in cits)
    assert cits[0]["text"] == PASSAGES[0]


def test_live_russian_question(env):
    text, _ = ask(env, "Какое самое глубокое озеро в мире?")
    assert "Байкал" in text


def test_live_cross_lingual_english_question_over_ukrainian_passage(env):
    text, cits = ask(env, "What is the capital of Ukraine?")
    assert "Kyiv" in text
    assert cits[0]["text"] == PASSAGES[0]
