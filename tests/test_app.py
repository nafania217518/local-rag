import socket
import types

import httpx

from localrag import api, app
from test_api import fake_chat_stream, fake_embed, fake_parse, fake_rerank


def make_real_app(tmp_path):
    return api.create_app(tmp_path / "t.db", tmp_path / "config.json", tmp_path / "files",
                          embed=fake_embed, rerank=fake_rerank, chat_stream=fake_chat_stream,
                          parse_page=fake_parse, run_in_background=lambda fn: fn())


def test_free_port_is_bindable():
    port = app.free_port()
    with socket.socket() as s:
        s.bind(("127.0.0.1", port))


def test_serve_and_wait_ready_real_server(tmp_path):
    port = app.free_port()
    server = app.serve(make_real_app(tmp_path), port)
    url = f"http://127.0.0.1:{port}"
    assert app.wait_ready(url)
    assert httpx.get(f"{url}/health").json()["ok"] is True
    server.should_exit = True
    server.thread.join(5)
    assert not server.thread.is_alive()


def test_wait_ready_times_out():
    assert app.wait_ready("http://x", timeout=0.3, probe=lambda url: False) is False


class FakeWebview:
    def __init__(self):
        self.window, self.started = None, False

    def create_window(self, *a, **k):
        self.window = (a, k)

    def start(self):
        self.started = True


def run_main(ready, argv=None):
    server, wv = types.SimpleNamespace(should_exit=False), FakeWebview()
    code = app.main(argv, make_app=lambda: "app", serve=lambda a, p: server,
                    wait_ready=lambda url: ready, webview=wv)
    return code, server, wv


def test_main_creates_window_and_stops_server():
    code, server, wv = run_main(True)
    assert code == 0 and server.should_exit is True and wv.started
    (title, url), kw = wv.window
    assert title == "LocalRAG" and url.startswith("http://127.0.0.1:")
    assert kw == {"width": 1280, "height": 800}


def test_main_returns_1_when_server_not_ready():
    code, server, wv = run_main(False)
    assert code == 1 and server.should_exit is True and wv.window is None


def test_make_app_creates_data_dir_and_serves_health(tmp_path, monkeypatch):
    monkeypatch.setattr(app.config, "app_data_dir", lambda: tmp_path / "data")
    from fastapi.testclient import TestClient
    assert TestClient(app.make_app()).get("/health").status_code == 200
    assert (tmp_path / "data" / "localrag.db").exists()


def test_main_serve_flag_runs_server_in_foreground_without_window():
    calls, wv = [], FakeWebview()
    code = app.main(["--serve", "8123"], make_app=lambda: "app", webview=wv,
                    run=lambda a, p: calls.append((a, p)))
    assert code == 0 and calls == [("app", 8123)] and wv.window is None


def test_main_without_args_reads_sys_argv(monkeypatch):
    monkeypatch.setattr("sys.argv", ["localrag"])
    code, server, wv = run_main(True)
    assert code == 0 and wv.started


def test_run_server_serves_health(tmp_path):
    import threading
    port = app.free_port()
    threading.Thread(target=app.run_server, args=(make_real_app(tmp_path), port), daemon=True).start()
    url = f"http://127.0.0.1:{port}"
    assert app.wait_ready(url)
    assert httpx.get(f"{url}/health").status_code == 200
