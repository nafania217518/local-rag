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
    page.click("#ws-add")
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


def test_source_passage_joins_single_line_breaks(page, server):
    new_ws(page, server)
    upload(page, "a.txt", b"Kyiv is\nthe capital\n\nSecond paragraph")
    expect(page.locator("#documents li .status")).to_have_text("indexed")
    page.fill("#question", "What is the capital?")
    page.press("#question", "Enter")
    expect(page.locator(".msg.assistant .content")).to_have_text("Kyiv [1]")
    # to_have_text normalizes whitespace, so compare the raw textContent
    pre = page.locator("#sources .source pre")
    expect(pre).to_have_count(1)
    assert pre.evaluate("e => e.textContent") == "Kyiv is the capital\n\nSecond paragraph"


def test_long_document_name_is_ellipsized_with_full_name_in_tooltip(page, server):
    new_ws(page, server)
    name = "x" * 80 + ".txt"
    upload(page, name, b"Kyiv is the capital")
    el = page.locator("#documents li .name")
    expect(el).to_have_attribute("title", name)
    assert el.evaluate("e => getComputedStyle(e).textOverflow") == "ellipsis"
    assert el.evaluate("e => e.scrollWidth > e.clientWidth")


def ws_id(page):
    return page.locator("#workspaces li.selected").get_attribute("data-id")


def sessions(server, page):
    return httpx.get(f"{server}/workspaces/{ws_id(page)}/sessions").json()


def test_add_workspace_button_opens_dialog_and_ok_creates(page, server):
    page.goto(server)
    expect(page.locator("#ws-add")).to_have_attribute("aria-label", "New workspace")
    page.click("#ws-add")
    expect(page.locator("#ws-dialog")).to_be_visible()
    expect(page.locator("#ws-dialog-title")).to_have_text("New workspace")
    expect(page.locator("#ws-create")).to_have_text("OK")
    expect(page.locator("#ws-cancel")).to_have_text("Cancel")
    page.fill("#ws-name", "dlg-ok")
    page.click("#ws-create")
    expect(page.locator("#ws-dialog")).not_to_be_visible()
    expect(page.locator("#workspaces li.selected")).to_have_text("dlg-ok")


def test_add_workspace_dialog_cancel_creates_nothing(page, server):
    page.goto(server)
    page.wait_for_load_state("networkidle")
    n = page.locator("#workspaces li").count()
    page.click("#ws-add")
    page.fill("#ws-name", "dlg-cancel")
    page.click("#ws-cancel")
    expect(page.locator("#ws-dialog")).not_to_be_visible()
    expect(page.locator("#workspaces li")).to_have_count(n)


@pytest.mark.parametrize("scheme", ["light", "dark"])
def test_pending_document_dot_is_amber(page, server, scheme):
    page.emulate_media(color_scheme=scheme)
    page.goto(server)
    got, want = page.evaluate("""() => {
        const li = document.createElement('li'); li.dataset.status = 'pending';
        const s = document.createElement('span'); s.className = 'status'; li.append(s);
        document.querySelector('#documents').append(li);
        const probe = document.createElement('div');
        probe.style.backgroundColor = 'var(--amber)'; document.body.append(probe);
        return [getComputedStyle(s, '::before').backgroundColor, getComputedStyle(probe).backgroundColor];
    }""")
    assert got == want == {"light": "rgb(184, 134, 11)", "dark": "rgb(224, 176, 74)"}[scheme]


def test_settings_and_theme_buttons_are_icons_with_labels(page, server):
    page.goto(server)
    labels = {"#settings-btn": "Settings", '#theme [value="system"]': "System theme",
              '#theme [value="light"]': "Light theme", '#theme [value="dark"]': "Dark theme"}
    for sel, label in labels.items():
        b = page.locator(sel)
        expect(b).to_have_attribute("aria-label", label)
        expect(b.locator("svg")).to_have_count(1)
        assert b.inner_text().strip() == ""


def test_new_workspace_opens_an_empty_chat_by_default(page, server):
    new_ws(page, server)
    tab = page.locator("#sessions .tab")
    expect(tab).to_have_count(1)
    expect(tab.locator(".tab-title")).to_have_text("New chat")
    expect(tab).to_have_class("tab active")
    expect(page.locator("#question")).to_be_enabled()


def test_chat_tab_takes_title_from_first_question(page, server):
    ask(page, server)
    expect(page.locator("#sessions .tab.active .tab-title")).to_have_text("What is the capital?")


def test_new_chat_pen_button_adds_tab(page, server):
    new_ws(page, server)
    expect(page.locator("#sessions .tab")).to_have_count(1)
    b = page.locator("#session-new")
    expect(b).to_have_attribute("aria-label", "New chat")
    expect(b.locator("svg")).to_have_count(1)
    b.click()
    expect(page.locator("#sessions .tab")).to_have_count(2)
    expect(page.locator("#sessions .tab").nth(1)).to_have_class("tab active")
    expect(page.locator("#sessions .tab.active .tab-title")).to_have_text("New chat")


def test_close_chat_tab_deletes_session_and_last_close_opens_new_chat(page, server):
    new_ws(page, server)
    expect(page.locator("#sessions .tab")).to_have_count(1)
    page.click("#session-new")
    expect(page.locator("#sessions .tab")).to_have_count(2)
    close = page.locator("#sessions .tab.active button.tab-close")
    expect(close).to_have_attribute("aria-label", "Close chat")
    expect(close).to_have_text("×")
    close.click()
    expect(page.locator("#sessions .tab")).to_have_count(1)
    expect(page.locator("#sessions .tab")).to_have_class("tab active")
    assert len(sessions(server, page)) == 1
    with page.expect_response(lambda r: r.request.method == "POST" and r.url.endswith("/sessions")):
        page.locator("#sessions .tab.active button.tab-close").click()
    expect(page.locator("#sessions .tab")).to_have_count(1)
    expect(page.locator("#sessions .tab.active .tab-title")).to_have_text("New chat")
    assert len(sessions(server, page)) == 1


def test_workspace_menu_rename(page, server):
    name = new_ws(page, server)
    li = page.locator("#workspaces li.selected")
    menu = li.locator("button.ws-menu")
    expect(menu).to_have_attribute("aria-label", "Workspace menu")
    expect(menu.locator("svg")).to_have_count(1)
    menu.click()
    pop = page.locator("#ws-popover")
    expect(pop).to_be_visible()
    expect(pop.locator("[data-action=rename]")).to_have_text("Rename")
    expect(pop.locator("[data-action=remove]")).to_have_text("Remove")
    pop.locator("[data-action=rename]").click()
    expect(page.locator("#ws-dialog-title")).to_have_text("Rename workspace")
    expect(page.locator("#ws-name")).to_have_value(name)
    page.fill("#ws-name", name + "-r")
    page.click("#ws-create")
    expect(page.locator("#workspaces li.selected .name")).to_have_text(name + "-r")
    assert httpx.get(f"{server}/workspaces").json()[-1]["name"] == name + "-r"


def test_workspace_menu_does_not_select_and_popover_closes(page, server):
    first = new_ws(page, server)
    second = new_ws(page, server)
    page.locator("#workspaces li", has_text=first).locator("button.ws-menu").click()
    expect(page.locator("#ws-popover")).to_be_visible()
    expect(page.locator("#workspaces li.selected .name")).to_have_text(second)
    page.keyboard.press("Escape")
    expect(page.locator("#ws-popover")).to_be_hidden()
    page.locator("#workspaces li", has_text=first).locator("button.ws-menu").click()
    page.click("#messages")
    expect(page.locator("#ws-popover")).to_be_hidden()


def test_workspace_menu_remove_confirms_then_deletes(page, server):
    for w in httpx.get(f"{server}/workspaces").json():
        httpx.delete(f"{server}/workspaces/{w['id']}")
    name = new_ws(page, server)
    li = page.locator("#workspaces li", has_text=name)
    def open_remove():
        li.locator("button.ws-menu").click()
        page.click("#ws-popover [data-action=remove]")
        expect(page.locator("#ws-remove-dialog")).to_be_visible()
        expect(page.locator("#ws-remove-dialog")).to_contain_text(name)
    open_remove()
    expect(page.locator("#ws-remove-ok")).to_have_text("Remove")
    page.click("#ws-remove-cancel")
    expect(page.locator("#ws-remove-dialog")).not_to_be_visible()
    expect(li).to_have_count(1)
    open_remove()
    page.click("#ws-remove-ok")
    expect(li).to_have_count(0)
    expect(page.locator("#ws-empty")).to_be_visible()
    assert httpx.get(f"{server}/workspaces").json() == []
