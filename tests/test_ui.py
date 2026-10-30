import uuid

import httpx
import pytest
from playwright.sync_api import expect

from localrag import api, app, config
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
    expect(page.locator("#sources .source")).to_have_count(1)
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


def test_tr5_settings_dialog_saves_parse_url_after_rerank_url(page, server):
    page.goto(server)
    page.click("#settings-btn")
    expect(page.locator('#settings fieldset:has(legend:text("Connection")) label:has([name="rerank_url"]) + label'
                        ' [name="parse_url"]')).to_be_visible()
    page.fill('#settings [name="parse_url"]', "https://example.test/{model}/parse")
    page.click("#settings-save")
    expect(page.locator("#settings")).not_to_be_visible()
    page.reload()
    page.click("#settings-btn")
    expect(page.locator('#settings [name="parse_url"]')).to_have_value("https://example.test/{model}/parse")


def test_settings_saves_paraphrase_toggle_and_count(page, server):
    page.goto(server)
    page.click("#settings-btn")
    expect(page.locator("#settings")).to_be_visible()  # the form is filled from config right before it opens
    page.uncheck('#settings [name="paraphrase"]')
    page.fill('#settings [name="paraphrases"]', "3")
    page.click("#settings-save")
    expect(page.locator("#settings")).not_to_be_visible()
    got = httpx.get(f"{server}/settings").json()
    assert got["paraphrase"] is False and got["paraphrases"] == 3 and type(got["paraphrases"]) is int


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


def test_new_chat_plus_button_adds_tab(page, server):
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
    expect(pop.locator("[data-action=delete]")).to_have_text("Delete")
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


def test_workspace_menu_delete_confirms_then_deletes(page, server):
    for w in httpx.get(f"{server}/workspaces").json():
        httpx.delete(f"{server}/workspaces/{w['id']}")
    name = new_ws(page, server)
    li = page.locator("#workspaces li", has_text=name)
    def open_delete():
        li.locator("button.ws-menu").click()
        page.click("#ws-popover [data-action=delete]")
        expect(page.locator("#ws-delete-dialog")).to_be_visible()
        expect(page.locator("#ws-delete-dialog")).to_contain_text(name)
    open_delete()
    expect(page.locator("#ws-delete-ok")).to_have_text("Delete")
    page.click("#ws-delete-cancel")
    expect(page.locator("#ws-delete-dialog")).not_to_be_visible()
    expect(li).to_have_count(1)
    open_delete()
    page.click("#ws-delete-ok")
    expect(li).to_have_count(0)
    expect(page.locator("#ws-empty")).to_be_visible()
    assert httpx.get(f"{server}/workspaces").json() == []


def test_status_dot_has_tooltip_and_pulses_while_indexing(page, server):
    import json
    docs = [{"id": 1, "name": "p.txt", "status": "pending", "error": None},
            {"id": 2, "name": "i.txt", "status": "indexed", "error": None},
            {"id": 3, "name": "f.txt", "status": "failed", "error": "boom"}]
    new_ws(page, server)
    page.route("**/workspaces/*/documents", lambda r: r.fulfill(
        status=200, content_type="application/json", body=json.dumps(docs)))
    page.reload()
    page.locator("#workspaces li").last.click()
    rows = {d["status"]: page.locator(f"#documents li[data-status={d['status']}]") for d in docs}
    expect(rows["pending"]).to_have_attribute("title", "Indexing")
    expect(rows["indexed"]).to_have_attribute("title", "Indexed")
    expect(rows["failed"]).to_have_attribute("title", "boom")
    # one evaluate: the pending poll re-renders the list between separate calls
    got = page.evaluate("""() => ['pending', 'indexed'].map(k => {
        const e = document.querySelector(`#documents li[data-status=${k}] .status`);
        return [getComputedStyle(e, '::before').animationName, e.getBoundingClientRect().width]; })""")
    assert got[0][0] == "pulse" and got[1][0] == "none"
    assert got[0][1] <= 1 and got[1][1] <= 1


def test_document_count_shown_in_heading(page, server):
    new_ws(page, server)
    page.set_input_files("#upload", [{"name": n, "mimeType": "text/plain", "buffer": n.encode()}
                                     for n in ("a.txt", "b.txt")])
    page.click("#upload-btn")
    expect(page.locator("#doc-count")).to_have_text("2")


def test_document_filter_hides_non_matching_rows(page, server):
    new_ws(page, server)
    page.set_input_files("#upload", [{"name": n, "mimeType": "text/plain", "buffer": n.encode()}
                                     for n in ("Kyiv-notes.txt", "lviv.txt")])
    page.click("#upload-btn")
    expect(page.locator("#documents li")).to_have_count(2)
    b = page.locator("#doc-filter-btn")
    expect(b).to_have_attribute("aria-label", "Filter documents")
    expect(b.locator("svg")).to_have_count(1)
    b.click()
    f = page.locator("#doc-filter")
    expect(f).to_be_visible()
    expect(f).to_be_focused()
    expect(f).to_have_attribute("placeholder", "Filter documents")
    f.fill("kyiv")
    expect(page.locator("#documents li", has_text="Kyiv-notes.txt")).to_be_visible()
    lviv = page.locator("#documents li", has_text="lviv.txt")
    expect(lviv).to_have_class("hidden")
    expect(lviv).to_be_hidden()
    f.fill("")
    expect(page.locator("#documents li:visible")).to_have_count(2)


def test_document_list_scrolls_and_footer_stays_visible_with_many_documents(page, server):
    new_ws(page, server)
    w = ws_id(page)
    for i in range(1, 61):
        httpx.post(f"{server}/workspaces/{w}/documents",
                   files=[("files", (f"doc-{i:02d}.txt", f"text {i}".encode(), "text/plain"))])
    page.reload()
    page.locator(f"#workspaces li[data-id='{w}']").click()
    expect(page.locator("#documents li")).to_have_count(60)
    assert page.locator("#documents").evaluate("e => e.scrollHeight > e.clientHeight")
    expect(page.locator("#settings-btn")).to_be_in_viewport()


def test_workspace_menu_items_have_pen_and_trash_icons(page, server):
    new_ws(page, server)
    page.locator("#workspaces li.selected button.ws-menu").click()
    for action, text in (("rename", "Rename"), ("delete", "Delete")):
        b = page.locator(f"#ws-popover button[data-action={action}]")
        expect(b.locator("svg")).to_have_count(1)
        expect(b).to_have_text(text)


def test_settings_button_is_a_gear(page, server):
    page.goto(server)
    expect(page.locator("#settings-btn svg")).to_have_attribute("data-icon", "gear")
    expect(page.locator("#session-new svg")).to_have_attribute("data-icon", "plus")
    expect(page.locator("#ws-add svg")).to_have_attribute("data-icon", "plus")


def test_enter_in_workspace_dialog_creates_workspace(page, server):
    name = f"enter-{uuid.uuid4().hex[:8]}"
    page.goto(server)
    page.click("#ws-add")
    page.fill("#ws-name", name)
    page.press("#ws-name", "Enter")
    expect(page.locator("#workspaces li.selected")).to_have_text(name)
    expect(page.locator("#ws-dialog")).to_be_hidden()


def test_enter_in_rename_dialog_renames_workspace(page, server):
    name = new_ws(page, server)
    page.locator("#workspaces li.selected button.ws-menu").click()
    page.click("#ws-popover [data-action=rename]")
    page.fill("#ws-name", name + "-e")
    page.press("#ws-name", "Enter")
    expect(page.locator("#workspaces li.selected .name")).to_have_text(name + "-e")
    expect(page.locator("#ws-dialog")).to_be_hidden()


def _stub_docs(page, server, docs):
    import json
    new_ws(page, server)
    page.route("**/workspaces/*/documents", lambda r: r.fulfill(
        status=200, content_type="application/json", body=json.dumps(docs)))
    page.reload()
    page.locator("#workspaces li").last.click()


_TR14_ERROR = "Error code: 500 - service inference failed " * 5
_TR14_DOCS = [
    {"id": 1, "name": "p.pdf", "status": "pending", "error": None, "pages": None, "pages_done": None},
    {"id": 2, "name": "i.pdf", "status": "indexed", "error": None, "pages": 1, "ocr_pages": 0},
    {"id": 3, "name": "f.pdf", "status": "failed", "error": _TR14_ERROR, "pages": None}]


def test_tr14_failed_row_dot_opens_error_dialog_with_message(page, server):
    _stub_docs(page, server, _TR14_DOCS)
    page.click("#documents li[data-id='3'] button.status")
    dialog = page.locator("#error-dialog")
    expect(dialog).to_be_visible()
    expect(page.locator("#error-title")).to_have_text("f.pdf")
    expect(page.locator("#error-text")).to_have_text(_TR14_ERROR)
    dialog.get_by_role("button", name="Close").click()
    expect(dialog).to_be_hidden()


def test_tr14_indexed_and_pending_dots_are_not_buttons(page, server):
    _stub_docs(page, server, _TR14_DOCS)
    for i in (1, 2):
        row = page.locator(f"#documents li[data-id='{i}']")
        expect(row.locator("span.status")).to_have_count(1)
        expect(row.locator("button.status")).to_have_count(0)
    failed = page.locator("#documents li[data-id='3'] button.status")
    expect(failed).to_have_attribute("aria-label", "Show error")


def test_pending_row_shows_page_progress(page, server):
    _stub_docs(page, server, [
        {"id": 1, "name": "a.pdf", "status": "pending", "error": None, "pages": 170, "pages_done": 34},
        {"id": 2, "name": "b.pdf", "status": "pending", "error": None, "pages": None, "pages_done": None}])
    a, b = page.locator("#documents li[data-id='1']"), page.locator("#documents li[data-id='2']")
    expect(a.locator("span.progress")).to_have_text("34 / 170")
    expect(a).to_have_attribute("title", "Indexing, page 34 of 170")
    expect(b.locator("span.progress")).to_have_count(0)
    expect(b).to_have_attribute("title", "Indexing")


def test_indexed_row_tooltip_counts_ocr_pages(page, server):
    _stub_docs(page, server, [
        {"id": i, "name": f"{i}.pdf", "status": "indexed", "error": None, "pages": 20, "ocr_pages": n}
        for i, n in [(1, 12), (2, 1), (3, 0)]])
    for i, title in [(1, "Indexed, 12 pages via OCR"), (2, "Indexed, 1 page via OCR"), (3, "Indexed")]:
        expect(page.locator(f"#documents li[data-id='{i}']")).to_have_attribute("title", title)


def _indexed_pdf_row(page, server, name="scan.pdf"):
    from test_api import _pdf_bytes
    new_ws(page, server)
    page.set_input_files("#upload", {"name": name, "mimeType": "application/pdf",
                                     "buffer": _pdf_bytes("Hello text page with enough chars")})
    page.click("#upload-btn")
    row = page.locator("#documents li", has_text=name)
    expect(row).to_have_attribute("data-status", "indexed")
    return row


def test_run_ocr_button_confirms_then_row_goes_pending(page, server):
    row = _indexed_pdf_row(page, server)
    btn = row.locator("button.doc-ocr")
    expect(btn).to_have_attribute("aria-label", "Run OCR")
    expect(btn.locator("svg[data-icon=scan]")).to_have_count(1)
    row.hover()
    btn.click()
    dlg = page.locator("#ocr-dialog")
    expect(dlg).to_be_visible()
    for text in ["scan.pdf", "1 page", "about 1 minute"]:
        expect(dlg).to_contain_text(text)
    expect(page.locator("#ocr-ok")).to_have_text("Run OCR")
    page.click("#ocr-ok")
    expect(row).to_have_attribute("title", "Indexed, 1 page via OCR", timeout=10000)


def test_run_ocr_dialog_wraps_long_document_name(page, server):
    name = "СОУ_НАЕК_158-2025_Зварювання_та_контроль_зварних_з'єднань_обладнання_та_трубопроводів_АЕС.pdf"
    row = _indexed_pdf_row(page, server, name)
    row.hover()
    row.locator("button.doc-ocr").click()
    expect(page.locator("#ocr-dialog")).to_be_visible()
    fits = page.evaluate("""() => {
        const p = document.querySelector('#ocr-text'), r = document.querySelector('#ocr-dialog').getBoundingClientRect();
        return {text: p.scrollWidth <= p.clientWidth,
                rect: r.left >= 0 && r.top >= 0 && r.right <= innerWidth && r.bottom <= innerHeight};
    }""")
    assert fits == {"text": True, "rect": True}
    page.click("#ocr-cancel")


def test_run_ocr_cancel_sends_no_request(page, server):
    row = _indexed_pdf_row(page, server)
    posts = []
    page.on("request", lambda r: r.method == "POST" and r.url.endswith("/ocr") and posts.append(r))
    row.hover()
    row.locator("button.doc-ocr").click()
    page.click("#ocr-cancel")
    expect(page.locator("#ocr-dialog")).not_to_be_visible()
    expect(row).to_have_attribute("title", "Indexed")
    assert posts == []


def test_settings_dialog_groups_fields_and_keeps_save_in_viewport(page, server):
    page.goto(server)
    page.click("#settings-btn")
    expect(page.locator("#settings legend")).to_have_text(["Connection", "Models", "Chat", "Retrieval", "Indexing"])
    expect(page.locator("#settings-save")).to_be_in_viewport()
    page.click('#settings button[value="cancel"]')


def test_settings_saves_numbers_as_numbers(page, server):
    page.goto(server)
    page.click("#settings-btn")
    page.fill('#settings [name="sources"]', "4")
    page.click("#settings-save")
    expect(page.locator("#settings")).not_to_be_visible()
    cfg = httpx.get(f"{server}/settings").json()
    assert cfg["sources"] == 4 and type(cfg["sources"]) is int
    assert cfg["temperature"] == 0.2


def test_settings_reset_restores_defaults_but_keeps_api_key(page, server):
    page.goto(server)
    page.click("#settings-btn")
    page.fill('#settings [name="api_key"]', "nvapi-keep")
    page.fill('#settings [name="sources"]', "9")
    page.fill('#settings [name="system_prompt"]', "custom")
    page.click("#settings-reset")
    expect(page.locator('#settings [name="sources"]')).to_have_value("6")
    expect(page.locator('#settings [name="system_prompt"]')).to_have_value(config.DEFAULTS["system_prompt"])
    expect(page.locator('#settings [name="api_key"]')).to_have_value("nvapi-keep")
    expect(page.locator("#settings")).to_be_visible()
