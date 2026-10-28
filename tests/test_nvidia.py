import base64
import json

import httpx
import openai
import pytest
import respx

from localrag import config, nvidia

cfg = {**config.DEFAULTS, "api_key": "test-key"}
CHAT_URL = f"{nvidia.BASE_URL}/chat/completions"
EMBED_URL = f"{nvidia.BASE_URL}/embeddings"
RERANK = nvidia.RERANK_URL.format(model="nvidia/nemotron-3.5-rerank-8b")


def sse(*deltas):
    lines = [
        "data: " + json.dumps({"id": "x", "object": "chat.completion.chunk", "created": 0, "model": "m",
                               "choices": [{"index": 0, "delta": d, "finish_reason": None}]})
        for d in deltas
    ]
    body = "\n\n".join(lines + ["data: [DONE]"]) + "\n\n"
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=body)


def embed_resp(vectors):
    return {"object": "list", "model": "m", "usage": {"prompt_tokens": 1, "total_tokens": 1},
            "data": [{"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(vectors)]}


@respx.mock
def test_chat_stream_posts_to_chat_completions_with_model_and_reasoning_effort():
    route = respx.post(CHAT_URL).mock(return_value=sse(
        {"reasoning_content": "hmm"}, {"content": "Hi"}, {"content": " there"}))
    out = list(nvidia.chat_stream(cfg, [{"role": "user", "content": "q"}]))
    assert out == [("thinking", "hmm"), ("token", "Hi"), ("token", " there")]
    req = route.calls.last.request
    body = json.loads(req.content)
    assert body["model"] == cfg["chat_model"]
    assert body["stream"] is True
    assert body["reasoning_effort"] == "none"
    assert req.headers["Authorization"] == "Bearer test-key"


@respx.mock
def test_chat_stream_reasoning_effort_override():
    route = respx.post(CHAT_URL).mock(return_value=sse({"content": "x"}))
    list(nvidia.chat_stream(cfg, [], reasoning_effort="high"))
    assert json.loads(route.calls.last.request.content)["reasoning_effort"] == "high"


@respx.mock
def test_chat_stream_skips_empty_deltas():
    respx.post(CHAT_URL).mock(return_value=sse({"content": None}, {"content": ""}))
    assert list(nvidia.chat_stream(cfg, [])) == []


@respx.mock
def test_embed_sends_input_type_and_truncate_and_returns_vectors_in_order():
    route = respx.post(EMBED_URL).mock(return_value=httpx.Response(200, json=embed_resp([[0.1, 0.2], [0.3, 0.4]])))
    assert nvidia.embed(cfg, ["a", "b"], "passage") == [[0.1, 0.2], [0.3, 0.4]]
    body = json.loads(route.calls.last.request.content)
    assert body["input_type"] == "passage"
    assert body["truncate"] == "END"
    assert body["model"] == cfg["embed_model"]
    assert body["encoding_format"] == "float"


@respx.mock
def test_embed_batches_by_64():
    def echo(request):
        inputs = json.loads(request.content)["input"]
        return httpx.Response(200, json=embed_resp([[float(t)] for t in inputs]))

    route = respx.post(EMBED_URL).mock(side_effect=echo)
    texts = [str(i) for i in range(130)]
    out = nvidia.embed(cfg, texts, "query")
    assert [len(json.loads(c.request.content)["input"]) for c in route.calls] == [64, 64, 2]
    assert out == [[float(i)] for i in range(130)]


def test_embed_rejects_bad_input_type():
    with pytest.raises(ValueError):
        nvidia.embed(cfg, ["a"], "doc")


@respx.mock
def test_embed_empty_returns_empty_without_request():
    route = respx.post(EMBED_URL)
    assert nvidia.embed(cfg, [], "query") == []
    assert route.call_count == 0


@respx.mock
def test_rerank_posts_correct_body_and_sorts_by_logit():
    route = respx.post(RERANK).mock(return_value=httpx.Response(200, json={"rankings": [
        {"index": 0, "logit": -1.2}, {"index": 2, "logit": 3.1}, {"index": 1, "logit": 0.5}]}))
    assert nvidia.rerank(cfg, "q", ["a", "b", "c"]) == [(2, 3.1), (1, 0.5), (0, -1.2)]
    req = route.calls.last.request
    assert json.loads(req.content) == {"model": "nvidia/nemotron-3.5-rerank-8b", "query": {"text": "q"},
                                       "passages": [{"text": "a"}, {"text": "b"}, {"text": "c"}],
                                       "truncate": "END"}
    assert req.headers["Authorization"] == "Bearer test-key"


@respx.mock
def test_rerank_empty_passages_returns_empty_without_request():
    route = respx.post(RERANK)
    assert nvidia.rerank(cfg, "q", []) == []
    assert route.call_count == 0


@respx.mock
def test_rerank_raises_on_http_error():
    respx.post(RERANK).mock(return_value=httpx.Response(402, json={}))
    with pytest.raises(httpx.HTTPStatusError):
        nvidia.rerank(cfg, "q", ["a"])


@respx.mock
def test_parse_page_sends_base64_png_and_returns_markdown():
    route = respx.post(CHAT_URL).mock(return_value=httpx.Response(200, json={
        "id": "x", "object": "chat.completion", "created": 0, "model": "m",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "# Page"}, "finish_reason": "stop"}]}))
    png = b"\x89PNG fake"
    assert nvidia.parse_page(cfg, png) == "# Page"
    body = json.loads(route.calls.last.request.content)
    assert body["model"] == cfg["parse_model"]
    parts = body["messages"][0]["content"]
    assert len(parts) == 1 and parts[0]["type"] == "image_url"
    url = parts[0]["image_url"]["url"]
    prefix = "data:image/png;base64,"
    assert url.startswith(prefix)
    assert base64.b64decode(url[len(prefix):]) == png


@respx.mock
def test_openai_errors_propagate():
    respx.post(EMBED_URL).mock(return_value=httpx.Response(401, json={"error": "bad key"}))
    with pytest.raises(openai.APIStatusError):
        nvidia.embed(cfg, ["a"], "query")


@respx.mock
def test_chat_stream_skips_chunks_without_choices():
    chunks = [
        {"id": "x", "object": "chat.completion.chunk", "created": 0, "model": "m",
         "choices": [{"index": 0, "delta": {"content": "Hi"}, "finish_reason": None}]},
        {"id": "x", "object": "chat.completion.chunk", "created": 0, "model": "m", "choices": [],
         "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}},
    ]
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
    respx.post(CHAT_URL).mock(return_value=httpx.Response(
        200, headers={"content-type": "text/event-stream"}, text=body))
    assert list(nvidia.chat_stream(cfg, [])) == [("token", "Hi")]


# --- rate limiting -----------------------------------------------------------

import threading


@pytest.fixture(autouse=True)
def fake_time(monkeypatch):
    nvidia._reset_throttle()
    clock = {"t": 0.0}
    sleeps = []

    def sleep(s):
        sleeps.append(s)
        clock["t"] += s

    monkeypatch.setattr(nvidia, "_sleep", sleep)
    monkeypatch.setattr(nvidia, "_now", lambda: clock["t"])
    return clock, sleeps


def test_throttle_allows_rpm_requests_without_sleeping(fake_time):
    _, sleeps = fake_time
    for _ in range(nvidia.RPM):
        nvidia._throttle()
    assert sleeps == []


def test_throttle_sleeps_until_window_frees(fake_time):
    clock, sleeps = fake_time
    for _ in range(nvidia.RPM):
        nvidia._throttle()
    clock["t"] = 10.0
    nvidia._throttle()
    assert sleeps == [pytest.approx(50)]
    clock["t"] = 200.0
    nvidia._throttle()
    assert len(sleeps) == 1


def test_throttle_is_thread_safe(fake_time):
    _, sleeps = fake_time
    threads = [threading.Thread(target=lambda: [nvidia._throttle() for _ in range(4)]) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(nvidia._window) == 40
    assert sleeps == []


@respx.mock
def test_retry_on_429_honors_retry_after(fake_time):
    _, sleeps = fake_time
    limited = httpx.Response(429, headers={"Retry-After": "3"}, json={})
    route = respx.post(EMBED_URL).mock(side_effect=[
        limited, limited, httpx.Response(200, json=embed_resp([[1.0]]))])
    assert nvidia.embed(cfg, ["a"], "query") == [[1.0]]
    assert route.call_count == 3
    assert sleeps == [3, 3]


@respx.mock
def test_retry_exponential_backoff_on_5xx(fake_time):
    _, sleeps = fake_time
    down = httpx.Response(503, json={})
    respx.post(RERANK).mock(side_effect=[down, down, down, httpx.Response(200, json={"rankings": [
        {"index": 0, "logit": 1.0}]})])
    assert nvidia.rerank(cfg, "q", ["a"]) == [(0, 1.0)]
    assert sleeps == [1, 2, 4]


@respx.mock
def test_retry_gives_up_after_max_attempts(fake_time):
    route = respx.post(EMBED_URL).mock(return_value=httpx.Response(429, json={}))
    with pytest.raises(openai.APIStatusError):
        nvidia.embed(cfg, ["a"], "query")
    assert route.call_count == nvidia.MAX_ATTEMPTS


@respx.mock
def test_no_retry_on_other_4xx(fake_time):
    _, sleeps = fake_time
    route = respx.post(EMBED_URL).mock(return_value=httpx.Response(401, json={}))
    with pytest.raises(openai.APIStatusError):
        nvidia.embed(cfg, ["a"], "query")
    assert route.call_count == 1
    assert sleeps == []


@respx.mock
def test_chat_stream_retries_on_429_before_streaming(fake_time):
    route = respx.post(CHAT_URL).mock(side_effect=[httpx.Response(429, json={}), sse({"content": "Hi"})])
    assert list(nvidia.chat_stream(cfg, [])) == [("token", "Hi")]
    assert route.call_count == 2


@respx.mock
def test_every_client_call_passes_through_throttle(monkeypatch):
    calls = []
    monkeypatch.setattr(nvidia, "_throttle", lambda: calls.append(1))
    respx.post(EMBED_URL).mock(return_value=httpx.Response(200, json=embed_resp([[1.0]])))
    respx.post(RERANK).mock(return_value=httpx.Response(200, json={"rankings": []}))
    respx.post(CHAT_URL).mock(side_effect=[
        httpx.Response(200, json={"id": "x", "object": "chat.completion", "created": 0, "model": "m",
                                  "choices": [{"index": 0, "message": {"role": "assistant", "content": "p"},
                                               "finish_reason": "stop"}]}),
        sse({"content": "x"})])
    nvidia.embed(cfg, ["a"], "query")
    nvidia.rerank(cfg, "q", ["a"])
    nvidia.parse_page(cfg, b"png")
    list(nvidia.chat_stream(cfg, []))
    assert len(calls) == 4
