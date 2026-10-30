import json
import logging
import re
import threading
import time
from collections import deque
from collections.abc import Iterator

import httpx
import openai

EMBED_BATCH = 64
PARSE_MAX_TOKENS = 9000  # hosted nemotron-3.5-parse caps max_output_tokens at 9000
MAX_ATTEMPTS = 5
# The 8b embed endpoint fails long single inputs at random (DESIGN.md 9.7).
EMBED_SINGLE_ATTEMPTS = 10
# Caps the backoff so the extra single-text attempts do not wait minutes.
BACKOFF_MAX = 8.0
BACKOFF_BASE = 1.0
WINDOW = 60.0

log = logging.getLogger(__name__)
_sleep = time.sleep
_now = time.monotonic
_window: deque[float] = deque()
_lock = threading.Lock()


def _reset_throttle() -> None:
    with _lock:
        _window.clear()


def _throttle(rpm: int) -> None:
    # ponytail: per-process window; a second process would double the rate.
    while True:
        with _lock:
            now = _now()
            while _window and now - _window[0] >= WINDOW:
                _window.popleft()
            if len(_window) < rpm:
                _window.append(now)
                return
            wait = _window[0] + WINDOW - now
        _sleep(wait)


def _retry(fn, label, *, retry_5xx=True, attempts=MAX_ATTEMPTS):
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except (openai.APIStatusError, httpx.HTTPStatusError) as e:
            status = e.response.status_code
            if not (status == 429 or (retry_5xx and 500 <= status <= 599)) or attempt == attempts:
                raise
            retry_after = e.response.headers.get("Retry-After", "")
            wait = int(retry_after) if retry_after.isdigit() else min(BACKOFF_BASE * 2 ** (attempt - 1), BACKOFF_MAX)
            log.warning("%s: HTTP %d %s, retry %d/%d in %ds", label, status, e.response.text[:200], attempt,
                        attempts, wait)
            _sleep(wait)


def _throttled(cfg, fn, label, *, retry_5xx=True, attempts=MAX_ATTEMPTS):
    def call():
        _throttle(cfg["rpm"])
        return fn()
    start = _now()
    result = _retry(call, label, retry_5xx=retry_5xx, attempts=attempts)
    log.info("%s: ok in %.1fs", label, _now() - start)
    return result


def _client(cfg: dict) -> openai.OpenAI:
    # openai 3.x defaults to httpx2; pass a plain httpx client so the module uses one HTTP stack (and respx sees it).
    return openai.OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"], timeout=cfg["timeout"], max_retries=0,
                         http_client=httpx.Client(timeout=cfg["timeout"]))


def chat_stream(cfg: dict, messages: list[dict], reasoning_effort: str | None = None) -> Iterator[tuple[str, str]]:
    client = _client(cfg)
    stream = _throttled(cfg, lambda: client.chat.completions.create(
        model=cfg["chat_model"], messages=messages, stream=True, temperature=cfg["temperature"], max_tokens=cfg["max_tokens"],
        extra_body={"reasoning_effort": reasoning_effort or cfg["reasoning_effort"]},
    ), f"chat {cfg['chat_model']}")
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
        vectors += _embed_batch(cfg, client, texts[i:i + EMBED_BATCH], input_type)
    return vectors


def _embed_batch(cfg, client, texts, input_type):
    label = f"embed {cfg['embed_model']} x{len(texts)}"
    try:
        # A 5xx on a batch is a deterministic rejection: split instead of backing off.
        resp = _throttled(cfg, lambda: client.embeddings.create(
            model=cfg["embed_model"], input=texts, encoding_format="float",
            extra_body={"input_type": input_type, "truncate": "END"},
        ), label, retry_5xx=len(texts) == 1,
           attempts=EMBED_SINGLE_ATTEMPTS if len(texts) == 1 else MAX_ATTEMPTS)
    except openai.APIStatusError as e:
        if len(texts) == 1 or not 500 <= e.response.status_code <= 599:
            raise
        half = len(texts) // 2
        log.warning("%s: HTTP %d, splitting into %d + %d", label, e.response.status_code, half, len(texts) - half)
        return (_embed_batch(cfg, client, texts[:half], input_type)
                + _embed_batch(cfg, client, texts[half:], input_type))
    return [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]


def rerank(cfg: dict, query: str, passages: list[str]) -> list[tuple[int, float]]:
    if not passages:
        return []
    model = cfg["rerank_model"]

    def post():
        resp = httpx.post(
            cfg["rerank_url"].format(model=model),
            headers={"Authorization": f"Bearer {cfg['api_key']}", "Accept": "application/json"},
            json={"model": model, "query": {"text": query}, "passages": [{"text": p} for p in passages],
                  "truncate": "END"},
            timeout=cfg["timeout"],
        )
        resp.raise_for_status()
        return resp

    resp = _throttled(cfg, post, f"rerank {model} x{len(passages)}")
    ranked = [(r["index"], r["logit"]) for r in resp.json()["rankings"]]
    return sorted(ranked, key=lambda r: r[1], reverse=True)


_MARKER = re.compile(r"</?[a-z]{2,4}>|<[xy]_[\d.]+>|<class_[\w-]+>")  # box corners, element class, <sup>, <tbc>
_SUP = re.compile(r"<sup>(.*?)</sup>")  # before _MARKER: keep the exponent apart from its base


_LATEX = [
    (re.compile(r"\\thead\{([^}]*)\}"), r"\1"),  # before multirow: it nests inside its braces
    (re.compile(r"\\mbox\{\}"), ""),
    (re.compile(r"\\(?:begin|end)\{tabular\}(?:\{[^}]*\})?"), ""),
    (re.compile(r"\\multi(?:row|column)\{[^}]*\}\{[^}]*\}\{([^{}]*)\}"), r"\1"),
    (re.compile(r"\s*\\\\[ \t]*\n?"), "\n"),  # eat one newline only: keep the blank line between elements
    (re.compile(r"[ \t]*&[ \t]*"), " | "),  # spaces only: keep the row break before an empty first cell
    (re.compile(r"[ \t]{2,}"), " "),
]


def _clean(text: str) -> str:
    text = _MARKER.sub("", _SUP.sub(r"^\1", text))
    for pattern, repl in _LATEX:
        n = 1
        while n:  # nested merged cells unwrap one level per pass
            text, n = pattern.subn(repl, text)
    return text.strip()


def parse_page(cfg: dict, image_png: bytes) -> str:
    def post():
        resp = httpx.post(
            cfg["parse_url"].format(model=cfg["parse_model"]),
            headers={"Authorization": f"Bearer {cfg['api_key']}", "Accept": "application/json"},
            files={"document": ("page.png", image_png, "image/png"),
                   "options": (None, json.dumps({"decoding": {"max_output_tokens": PARSE_MAX_TOKENS}}),
                               "application/json")},
            timeout=cfg["timeout"],
        )
        resp.raise_for_status()
        return resp

    return _clean(_throttled(cfg, post, f"parse {cfg['parse_model']}").json()["content"] or "")
