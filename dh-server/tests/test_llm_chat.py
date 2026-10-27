"""``POST /api/admin/llm/chat`` + ``POST /api/admin/llm/chat/stream``.

The chat endpoint pair drives the console's chat panel. The non-streaming
endpoint is a thin wrapper over :func:`server.llm.openai.chat`; the
streaming one wraps :func:`server.llm.openai.stream_chat` and frames each
delta as an SSE event.

All tests here drive the real parse path through ``httpx.MockTransport``
so a typo in the request body shape, an unexpected upstream status, or
a malformed SSE frame surfaces as a test failure rather than a silent
network call.

LLM config lives entirely in the SQLite ``settings`` table (the operator
wires it from the Models tab) -- there are no env defaults, so the tests
write the config rows directly via :func:`server.llm.config.write_config`.
"""
from __future__ import annotations

import json

import httpx
import pytest

from server.core.errors import AppError


ADMIN = {"X-Admin-Token": "dev-admin-token"}


def _mock_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _write_llm_config(db, *, api_key="test-key", base_url="https://api.example.test/v1", model="gpt-4o-mini"):
    """Drop the three LLM settings rows directly so the chat path sees
    a wired config without going through the HTTP layer.

    Bypasses :func:`server.llm.config.write_config` so tests can set
    deliberately-invalid values (empty base_url etc.) without tripping
    the form-level validation -- the chat path is what we're testing.
    """
    import json as _json
    from server.db.models import Setting

    rows = {
        "llm_api_key": api_key,
        "llm_base_url": base_url,
        "llm_model": model,
    }
    for key, value in rows.items():
        existing = db.get(Setting, key)
        if existing is None:
            db.add(Setting(key=key, value=_json.dumps(value)))
        else:
            existing.value = _json.dumps(value)
    db.commit()


# ----- the non-streaming chat surface ---------------------------------------


@pytest.mark.asyncio
async def test_chat_sends_messages_and_returns_content(_fresh_db):
    """``chat()`` POSTs the messages (with a system prompt prefix) to
    ``{base_url}/chat/completions`` and returns the assistant content.

    The system prompt rides in every request as turn 0 -- the chat panel
    itself doesn't have to know about it. ``api_key`` rides in the
    ``Authorization: Bearer ...`` header.
    """
    from server.llm.openai import chat

    _write_llm_config(_fresh_db)
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "你好世界"}}],
            },
        )

    client = _mock_client(handler)
    try:
        content = await chat(
            [{"role": "user", "content": "你好"}],
            client=client,
            db=_fresh_db,
        )
    finally:
        await client.aclose()

    assert content == "你好世界"
    assert seen["url"] == "https://api.example.test/v1/chat/completions"
    assert seen["auth"] == "Bearer test-key"
    # System prompt landed as turn 0.
    assert seen["body"]["model"] == "gpt-4o-mini"
    assert seen["body"]["messages"][0]["role"] == "system"
    assert "数字人" in seen["body"]["messages"][0]["content"]
    # User message lands after the system turn.
    assert seen["body"]["messages"][1] == {"role": "user", "content": "你好"}
    # Non-streaming by default.
    assert "stream" not in seen["body"]


@pytest.mark.asyncio
async def test_chat_rejects_empty_messages():
    """An empty message list is a 400 ``empty_messages`` -- the chat
    panel must always send at least one user turn.
    """
    from server.llm.openai import chat

    with pytest.raises(AppError) as excinfo:
        await chat([], client=_mock_client(lambda r: httpx.Response(200, json={})))
    assert excinfo.value.status == 400
    assert excinfo.value.code == "empty_messages"


@pytest.mark.asyncio
async def test_chat_surfaces_not_configured_when_api_key_missing(_fresh_db):
    """Empty api_key means "not configured": ``chat()`` raises
    503 ``llm_not_configured`` rather than calling out.
    """
    from server.llm.openai import chat

    # Wire ONLY base_url -- api_key stays at the empty default.
    _write_llm_config(_fresh_db, api_key="")
    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, json={})

    client = _mock_client(handler)
    try:
        with pytest.raises(AppError) as excinfo:
            await chat(
                [{"role": "user", "content": "hi"}],
                client=client,
                db=_fresh_db,
            )
        assert excinfo.value.status == 503
        assert excinfo.value.code == "llm_not_configured"
    finally:
        await client.aclose()

    assert called is False  # never hit the wire


@pytest.mark.asyncio
async def test_chat_surfaces_not_configured_when_base_url_missing(_fresh_db):
    """Empty base_url is also a 503 -- the chat path appends
    ``/chat/completions`` so a missing base would 404 on the wire.
    """
    from server.llm.openai import chat

    _write_llm_config(_fresh_db, base_url="")
    with pytest.raises(AppError) as excinfo:
        await chat(
            [{"role": "user", "content": "hi"}],
            client=_mock_client(lambda r: httpx.Response(200, json={})),
            db=_fresh_db,
        )
    assert excinfo.value.status == 503
    assert excinfo.value.code == "llm_not_configured"


@pytest.mark.asyncio
async def test_chat_surfaces_upstream_error(_fresh_db):
    """Non-200 upstream becomes 502 ``llm_failed`` with a slice of the
    upstream body so the operator can diagnose without flipping to a
    network debugger.
    """
    from server.llm.openai import chat

    _write_llm_config(_fresh_db)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="unauthorized")

    client = _mock_client(handler)
    try:
        with pytest.raises(AppError) as excinfo:
            await chat(
                [{"role": "user", "content": "hi"}],
                client=client,
                db=_fresh_db,
            )
        assert excinfo.value.status == 502
        assert excinfo.value.code == "llm_failed"
        assert "401" in excinfo.value.message
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_chat_surfaces_upstream_error_block(_fresh_db):
    """Some providers echo an ``error`` block on HTTP 200; the chat path
    treats it as a 502 with the upstream message verbatim.
    """
    from server.llm.openai import chat

    _write_llm_config(_fresh_db)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"error": {"message": "context length exceeded", "type": "x"}},
        )

    client = _mock_client(handler)
    try:
        with pytest.raises(AppError) as excinfo:
            await chat(
                [{"role": "user", "content": "hi"}],
                client=client,
                db=_fresh_db,
            )
        assert excinfo.value.status == 502
        assert excinfo.value.code == "llm_failed"
        assert "context length" in excinfo.value.message
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_chat_surfaces_transport_error(_fresh_db):
    """Connection refused (or any httpx transport-level failure) maps to
    502 ``llm_failed`` rather than letting the underlying exception 500
    the request.
    """
    from server.llm.openai import chat

    _write_llm_config(_fresh_db)

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    client = _mock_client(handler)
    try:
        with pytest.raises(AppError) as excinfo:
            await chat(
                [{"role": "user", "content": "hi"}],
                client=client,
                db=_fresh_db,
            )
        assert excinfo.value.status == 502
        assert excinfo.value.code == "llm_failed"
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_chat_rejects_empty_choices(_fresh_db):
    """An HTTP 200 with no ``choices`` array is a malformed reply --
    surface as 502 ``llm_failed`` rather than crashing on the shape.
    """
    from server.llm.openai import chat

    _write_llm_config(_fresh_db)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    client = _mock_client(handler)
    try:
        with pytest.raises(AppError) as excinfo:
            await chat(
                [{"role": "user", "content": "hi"}],
                client=client,
                db=_fresh_db,
            )
        assert excinfo.value.status == 502
        assert excinfo.value.code == "llm_failed"
    finally:
        await client.aclose()


# ----- the streaming chat surface -------------------------------------------


@pytest.mark.asyncio
async def test_stream_chat_yields_deltas_and_terminates_on_done(_fresh_db):
    """``stream_chat`` reads the SSE stream line-by-line and yields each
    ``delta.content`` until ``[DONE]`` closes the upstream.
    """
    from server.llm.openai import stream_chat

    _write_llm_config(_fresh_db)

    chunks = [
        "data: " + json.dumps({"choices": [{"delta": {"role": "assistant"}}]}, ensure_ascii=False) + "\n\n",
        "data: " + json.dumps({"choices": [{"delta": {"content": "你"}}]}, ensure_ascii=False) + "\n\n",
        "data: " + json.dumps({"choices": [{"delta": {"content": "好"}}]}, ensure_ascii=False) + "\n\n",
        "data: " + json.dumps({"choices": [{"delta": {"content": "世界"}}]}, ensure_ascii=False) + "\n\n",
        "data: [DONE]\n\n",
    ]
    body = "".join(chunks)

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=body.encode("utf-8"),
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        deltas: list[str] = []
        async for delta in stream_chat(
            [{"role": "user", "content": "你好"}],
            client=client,
            db=_fresh_db,
        ):
            deltas.append(delta)
    finally:
        await client.aclose()

    # Role-only first frame doesn't contribute; the three content frames do.
    assert deltas == ["你", "好", "世界"]


@pytest.mark.asyncio
async def test_stream_chat_surfaces_not_configured(_fresh_db):
    """Empty api_key means "not configured": ``stream_chat`` raises 503
    before opening the upstream connection.
    """
    from server.llm.openai import stream_chat

    _write_llm_config(_fresh_db, api_key="")
    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, text="")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(AppError) as excinfo:
            async for _ in stream_chat(
                [{"role": "user", "content": "hi"}],
                client=client,
                db=_fresh_db,
            ):
                pass
        assert excinfo.value.status == 503
        assert excinfo.value.code == "llm_not_configured"
    finally:
        await client.aclose()

    assert called is False


@pytest.mark.asyncio
async def test_stream_chat_surfaces_upstream_http_error(_fresh_db):
    """Non-200 upstream closes the stream with 502 ``llm_failed`` -- the
    chat panel surfaces this via its single error frame.
    """
    from server.llm.openai import stream_chat

    _write_llm_config(_fresh_db)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(AppError) as excinfo:
            async for _ in stream_chat(
                [{"role": "user", "content": "hi"}],
                client=client,
                db=_fresh_db,
            ):
                pass
        assert excinfo.value.status == 502
        assert excinfo.value.code == "llm_failed"
        assert "500" in excinfo.value.message
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_stream_chat_constructs_valid_timeout_when_client_is_none(
    _fresh_db, monkeypatch
):
    """Regression: ``stream_chat`` without an injected client must not
    crash building the ``httpx.Timeout`` instance.

    httpx.Timeout requires either a default or all four of connect / read /
    write / pool to be set explicitly. The historic bug at
    ``httpx.Timeout(connect=10.0, read=None)`` raised
    ``ValueError: httpx.Timeout must either include a default, or set all
    four parameters explicitly.`` on the first request -- which is exactly
    the path the admin endpoint hits, since it never injects a client.
    All other stream tests inject a client and bypass the
    ``client = client or httpx.AsyncClient(timeout=...)`` line entirely.
    """
    from server.llm import openai as openai_mod

    _write_llm_config(_fresh_db)

    captured: dict = {}

    class FakeAsyncClient:
        def __init__(self, *, timeout=None, **kwargs):
            captured["timeout"] = timeout

        def build_request(self, *args, **kwargs):
            raise RuntimeError("captured-only stub: build_request")

        async def send(self, *args, **kwargs):
            raise RuntimeError("captured-only stub: send")

        async def aclose(self):
            pass

    monkeypatch.setattr(openai_mod.httpx, "AsyncClient", FakeAsyncClient)

    with pytest.raises(RuntimeError, match="captured-only stub"):
        async for _ in openai_mod.stream_chat(
            [{"role": "user", "content": "hi"}],
            db=_fresh_db,
        ):
            pass

    # If we reached this line the Timeout construction succeeded -- otherwise
    # the ValueError from httpx would have escaped before FakeAsyncClient
    # was even instantiated.
    assert "timeout" in captured


# ----- the admin endpoint ---------------------------------------------------


def test_admin_chat_endpoint_returns_content(client, monkeypatch):
    """The non-streaming admin endpoint round-trips through the chat
    function with a stubbed response.
    """
    async def fake_chat(messages, **kwargs):
        return "fake reply"

    monkeypatch.setattr("server.llm.openai.chat", fake_chat)

    resp = client.post(
        "/api/admin/llm/chat",
        headers=ADMIN,
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    assert resp.json() == {"content": "fake reply"}


def test_admin_chat_endpoint_rejects_empty_messages(client):
    """An empty messages list is a 400 from Pydantic; ``messages`` is
    required by the schema.
    """
    resp = client.post(
        "/api/admin/llm/chat",
        headers=ADMIN,
        json={"messages": []},
    )
    # Pydantic-side validation accepts an empty list (it matches the
    # shape), so the empty check lives inside ``chat()`` and surfaces
    # as 400.
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "empty_messages"


def test_admin_chat_stream_endpoint_emits_sse_frames(client, monkeypatch):
    """The streaming endpoint serialises deltas as ``data:`` lines and
    terminates with a ``done`` frame. The chat panel parses these on the
    client side.
    """
    from typing import AsyncIterator

    async def fake_stream_chat(messages, **kwargs) -> AsyncIterator[str]:
        for chunk in ["你", "好", "世界"]:
            yield chunk

    monkeypatch.setattr("server.llm.openai.stream_chat", fake_stream_chat)

    resp = client.post(
        "/api/admin/llm/chat/stream",
        headers=ADMIN,
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    body = resp.text

    # Three delta frames + one done frame.
    assert 'data: {"delta": "你"}' in body
    assert 'data: {"delta": "好"}' in body
    assert 'data: {"delta": "世界"}' in body
    assert '"done": true' in body


def test_admin_chat_stream_endpoint_surfaces_error_frame(client, monkeypatch):
    """An :class:`AppError` from ``stream_chat`` becomes one SSE frame
    so the client can render it without breaking the connection.
    """
    from typing import AsyncIterator

    async def fake_stream_chat(messages, **kwargs) -> AsyncIterator[str]:
        # Raise before yielding any tokens: the endpoint's generator
        # catches it and writes one error frame.
        raise AppError("llm_not_configured", "set LLM config via Models tab", 503)
        yield  # pragma: no cover -- makes this a generator

    monkeypatch.setattr("server.llm.openai.stream_chat", fake_stream_chat)

    resp = client.post(
        "/api/admin/llm/chat/stream",
        headers=ADMIN,
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200  # SSE envelope is always 200; the error rides inside
    body = resp.text
    assert "llm_not_configured" in body
    assert "set LLM config via Models tab" in body


# ----- the shared _fresh_db fixture -----------------------------------------


@pytest.fixture()
def _fresh_db(tmp_path, monkeypatch):
    """Standard fresh DB with the standard test env. Mirrors the
    fixture in :mod:`tests.test_tts_config` so the LLM tests can write
    directly to the settings table.
    """
    from tests.env import apply_env
    from server.config import get_settings
    from server.db.session import init_db, get_engine, reset_engine
    from sqlalchemy.orm import sessionmaker

    apply_env(monkeypatch, tmp_path)
    reset_engine()
    init_db(get_engine())
    get_settings.cache_clear()
    Session = sessionmaker(bind=get_engine())
    db = Session()
    try:
        yield db
    finally:
        db.close()
        reset_engine()
        get_settings.cache_clear()
