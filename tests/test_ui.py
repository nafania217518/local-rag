import uuid

import httpx
import pytest
from playwright.sync_api import expect

from localrag import api, app
from test_api import fake_chat_stream, fake_embed, fake_parse, fake_rerank

pytestmark = pytest.mark.ui


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    d = tmp_path_factory.mktemp("ui")
    a = api.create_app(d / "t.db", d / "config.json", d / "files",
                       embed=fake_embed, rerank=fake_rerank, chat_stream=fake_chat_stream,
                       parse_page=fake_parse, run_in_background=lambda fn: fn())
    port = app.free_port()
    srv = app.serve(a, port)
    url = f"http://127.0.0.1:{port}"
    assert app.wait_ready(url)
    yield url
    srv.should_exit = True


def new_ws(page, server):
    name = f"ws-{uuid.uuid4().hex[:8]}"
    page.goto(server)
    page.fill("#ws-name", name)
    page.click("#ws-create")
    expect(page.locator("#workspaces li.selected")).to_have_text(name)
    return name


def upload(page, name, data):
    page.set_input_files("#upload", {"name": name, "mimeType": "text/plain", "buffer": data})
    page.click("#upload-btn")


def ask(page, server):
    new_ws(page, server)
    upload(page, "a.txt", b"Kyiv is the capital")
    expect(page.locator("#documents li .status")).to_have_text("indexed")
    page.click("#session-new")
    expect(page.locator("#sessions option")).to_have_count(1)
    page.fill("#question", "What is the capital?")
    page.press("#question", "Enter")
    expect(page.locator(".msg.assistant .content")).to_have_text("Kyiv [1]")


def test_create_workspace_appears_in_list_and_is_selected(page, server):
    name = new_ws(page, server)
    expect(page.locator("#workspaces li", has_text=name)).to_have_class("selected")


def test_upload_txt_shows_indexed_status(page, server):
    new_ws(page, server)
    upload(page, "a.txt", b"Kyiv is the capital")
    expect(page.locator("#documents li", has_text="a.txt").locator(".status")).to_have_text("indexed")


def test_ask_streams_answer_with_citation_and_sources(page, server):
    ask(page, server)
    expect(page.locator(".msg.user")).to_have_text("What is the capital?")
    expect(page.locator(".msg.assistant .cite")).to_have_text("[1]")
    expect(page.locator("#sources .source")).to_contain_text("a.txt")


def test_thinking_is_collapsed_by_default(page, server):
    ask(page, server)
    thinking = page.locator(".msg.assistant details.thinking")
    expect(thinking).to_have_count(1)
    expect(thinking).not_to_have_attribute("open", "")
    expect(thinking.locator("pre")).to_have_text("hmm", use_inner_text=False)


def test_click_citation_highlights_source(page, server):
    ask(page, server)
    page.click(".msg.assistant .cite")
    expect(page.locator('#sources .source[data-n="1"]')).to_have_class("source highlight")


def test_messages_persist_across_reload(page, server):
    ask(page, server)
    page.reload()
    expect(page.locator(".msg.assistant .content")).to_have_text("Kyiv [1]")
    expect(page.locator("#sources .source")).to_contain_text("a.txt")


def test_settings_dialog_saves_api_key(page, server):
    page.goto(server)
    page.click("#settings-btn")
    page.fill('#settings [name="api_key"]', "nvapi-test")
    page.click("#settings-save")
    expect(page.locator("#settings")).not_to_be_visible()
    page.reload()
    page.click("#settings-btn")
    expect(page.locator('#settings [name="api_key"]')).to_have_value("nvapi-test")
    assert httpx.get(f"{server}/settings").json()["api_key"] == "nvapi-test"


def test_failed_document_shows_error(page, server):
    new_ws(page, server)
    upload(page, "b.xyz", b"junk")
    row = page.locator("#documents li", has_text="b.xyz")
    expect(row.locator(".status")).to_have_text("failed")
    assert page.locator("#documents li", has_text="b.xyz").get_attribute("title")


def test_theme_follows_os_by_default_and_manual_switch_persists(page, server):
    page.emulate_media(color_scheme="dark")
    page.goto(server)
    paper = lambda: page.evaluate("getComputedStyle(document.documentElement).getPropertyValue('--paper').trim()")
    assert paper() == "#242320"          # dark token from the OS
    page.click('#theme [value="light"]')
    assert paper() == "#F8F7F3"          # manual override wins
    page.reload()
    assert paper() == "#F8F7F3"          # persisted
    page.click('#theme [value="system"]')
    assert paper() == "#242320"          # back to the OS


def test_session_select_hidden_until_a_chat_exists(page, server):
    new_ws(page, server)
    expect(page.locator("#sessions")).to_be_hidden()
    page.click("#session-new")
    expect(page.locator("#sessions")).to_be_visible()
