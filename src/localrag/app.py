import importlib
import logging
import socket
import sys
import threading
import time

import httpx
import uvicorn

from localrag import api, config, index


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def make_app():
    config.app_data_dir().mkdir(parents=True, exist_ok=True)
    return api.create_app(config.app_data_dir() / "localrag.db", config.config_path(), index.files_dir())


def serve(app, port) -> uvicorn.Server:
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    server.thread = threading.Thread(target=server.run, daemon=True)
    server.thread.start()
    return server


def _probe(url):
    try:
        return httpx.get(f"{url}/health").status_code == 200
    except httpx.HTTPError:
        return False


def wait_ready(url, timeout=10.0, probe=None) -> bool:
    probe = probe or _probe
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if probe(url):
            return True
        time.sleep(0.1)
    return False


def run_server(app, port):
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


def configure_logging():
    if sys.stderr is None:  # the windowed Windows exe has no console
        return
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    logging.getLogger("httpx").setLevel(logging.WARNING)  # it logs every request at INFO


def main(argv=None, *, make_app=make_app, serve=serve, wait_ready=wait_ready, webview=None, run=None) -> int:
    configure_logging()
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) == 2 and argv[0] == "--serve":
        (run or run_server)(make_app(), int(argv[1]))
        return 0
    webview = webview or importlib.import_module("webview")
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    server = serve(make_app(), port)
    if not wait_ready(url):
        server.should_exit = True
        return 1
    webview.create_window("LocalRAG", url, width=1280, height=800)
    webview.start()
    server.should_exit = True
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
