import base64
import threading
import time
from collections import deque
from collections.abc import Iterator

import httpx
import openai

BASE_URL = "https://integrate.api.nvidia.com/v1"
RERANK_URL = "https://ai.api.nvidia.com/v1/retrieval/{model}/reranking"
EMBED_BATCH = 64
TIMEOUT = 120.0
RPM = 40  # build.nvidia.com limit, shared by all models
MAX_ATTEMPTS = 5
BACKOFF_BASE = 1.0
WINDOW = 60.0

_sleep = time.sleep
_now = time.monotonic
_window: deque[float] = deque()
_lock = threading.Lock()


def _reset_throttle() -> None:
    with _lock:
        _window.clear()


def _throttle() -> None:
    # ponytail: per-process window; a second process would double the rate.
    while True:
        with _lock:
            now = _now()
            while _window and now - _window[0] >= WINDOW:
                _window.popleft()
            if len(_window) < RPM:
                _window.append(now)
                return
            wait = _window[0] + WINDOW - now
        _sleep(wait)


def _retry(fn):
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return fn()
        except (openai.APIStatusError, httpx.HTTPStatusError) as e:
            status = e.response.status_code
            if not (status == 429 or 500 <= status <= 599) or attempt == MAX_ATTEMPTS:
                raise
            retry_after = e.response.headers.get("Retry-After", "")
            _sleep(int(retry_after) if retry_after.isdigit() else BACKOFF_BASE * 2 ** (attempt - 1))


def _throttled(fn):
    def call():
        _throttle()
        return fn()
    return _retry(call)


def _client(cfg: dict) -> openai.OpenAI:
    # openai 3.x defaults to httpx2; pass a plain httpx client so the module uses one HTTP stack (and respx sees it).
    return openai.OpenAI(api_key=cfg["api_key"], base_url=BASE_URL, timeout=TIMEOUT, max_retries=0,
                         http_client=httpx.Client(timeout=TIMEOUT))


def chat_stream(cfg: dict, messages: list[dict], reasoning_effort: str | None = None) -> Iterator[tuple[str, str]]:
    client = _client(cfg)
    stream = _throttled(lambda: client.chat.completions.create(
        model=cfg["chat_model"], messages=messages, stream=True, temperature=0.2, max_tokens=4096,
        extra_body={"reasoning_effort": reasoning_effort or cfg["reasoning_effort"]},
    ))
    for chunk in stream:
        if not chunk.choices:  # final usage-only chunk
            continue
        delta = chunk.choices[0].delta
        if thinking := getattr(delta, "reasoning_content", None):
            yield ("thinking", thinking)
        if delta.content:
            yield ("token", delta.content)


def embed(cfg: dict, texts: list[str], input_type: str) -> list[list[float]]:
    if input_type not in ("passage", "query"):
        raise ValueError(f"input_type must be 'passage' or 'query', got {input_type!r}")
    if not texts:
        return []
    client = _client(cfg)
    vectors = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i:i + EMBED_BATCH]
        resp = _throttled(lambda: client.embeddings.create(
            model=cfg["embed_model"], input=batch, encoding_format="float",
            extra_body={"input_type": input_type, "truncate": "END"},
        ))
        vectors += [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]
    return vectors


def rerank(cfg: dict, query: str, passages: list[str]) -> list[tuple[int, float]]:
    if not passages:
        return []
    model = cfg["rerank_model"]

    def post():
        resp = httpx.post(
            RERANK_URL.format(model=model),
            headers={"Authorization": f"Bearer {cfg['api_key']}", "Accept": "application/json"},
            json={"model": model, "query": {"text": query}, "passages": [{"text": p} for p in passages],
                  "truncate": "END"},
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        return resp

    resp = _throttled(post)
    ranked = [(r["index"], r["logit"]) for r in resp.json()["rankings"]]
    return sorted(ranked, key=lambda r: r[1], reverse=True)


def parse_page(cfg: dict, image_png: bytes) -> str:
    url = "data:image/png;base64," + base64.b64encode(image_png).decode()
    client = _client(cfg)
    resp = _throttled(lambda: client.chat.completions.create(
        model=cfg["parse_model"],
        messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}]}],
        max_tokens=8192, temperature=0,
    ))
    return resp.choices[0].message.content or ""
