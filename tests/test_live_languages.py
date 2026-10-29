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


@pytest.mark.skipif(not os.environ.get("LOCALRAG_SCAN_PDF"), reason="needs LOCALRAG_SCAN_PDF")
def test_live_parse_reads_ukrainian_scanned_page():
    import re

    import pymupdf

    from localrag import parsers
    cfg = {**config.DEFAULTS, "api_key": os.environ["NVIDIA_API_KEY"]}
    with pymupdf.open(os.environ["LOCALRAG_SCAN_PDF"]) as doc:
        page = doc[int(os.environ.get("LOCALRAG_SCAN_PAGE", "1")) - 1]
        layer = page.get_text()
        png = page.get_pixmap(dpi=config.DEFAULTS["render_dpi"]).tobytes("png")
    words = dict.fromkeys(w.casefold() for w in re.findall(r"[Ѐ-ӿ]{6,}", layer))
    expected = sorted(words, key=len, reverse=True)[:5]
    ocr = nvidia.parse_page(cfg, png)
    print("expected:", expected)
    print("quality layer:", parsers.text_quality(layer), "ocr:", parsers.text_quality(ocr))
    print("ocr:", ocr[:300])
    assert sum(w in ocr.casefold() for w in expected) >= 3
    assert parsers.text_quality(ocr) >= config.DEFAULTS["ocr_quality_min"]
