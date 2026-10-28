"""``POST /api/admin/llm/chat/stream``.

The chat endpoint drives the console's chat panel. It wraps
:func:`dh_server.llm.openai.stream_chat` and frames each delta as an SSE
event.

All tests here drive the real parse path through ``httpx.MockTransport``
so a typo in the request body shape, an unexpected upstream status, or
a malformed SSE frame surfaces as a test failure rather than a silent
network call.

LLM config lives entirely in the SQLite ``settings`` table (the operator
wires it from the Models tab) -- there are no env defaults, so the tests
write the config rows directly via :func:`dh_server.llm.config.write_config`.
"""
from __future__ import annotations

import json

import httpx
import pytest

from dh_server.core.errors import AppError


ADMIN = {"X-Admin-Token": "dev-admin-token"}


def _mock_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _write_llm_config(db, *, api_key="test-key", base_url="https://api.example.test/v1", model="gpt-4o-mini"):
    """Drop the three LLM settings rows directly so the chat path sees
    a wired config without going through the HTTP layer.

    Bypasses :func:`dh_server.llm.config.write_config` so tests can set
    deliberately-invalid values (empty base_url etc.) without tripping
    the form-level validation -- the chat path is what we're testing.
    """
    import json as _json
    from dh_server.db.models import Setting

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


# ----- the streaming chat surface -------------------------------------------


@pytest.mark.asyncio
async def test_stream_chat_yields_deltas_and_terminates_on_done(_fresh_db):
    """``stream_chat`` reads the SSE stream line-by-line and yields each
    ``delta.content`` until ``[DONE]`` closes the upstream.
    """
    from dh_server.llm.openai import stream_chat

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
    from dh_server.llm.openai import stream_chat

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
    from dh_server.llm.openai import stream_chat

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
    from dh_server.llm import openai as openai_mod

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


def test_admin_chat_stream_endpoint_emits_sse_frames(client, monkeypatch):
    """The streaming endpoint serialises deltas as ``data:`` lines and
    terminates with a ``done`` frame. The chat panel parses these on the
    client side.
    """
    from typing import AsyncIterator

    async def fake_stream_chat(messages, **kwargs) -> AsyncIterator[str]:
        for chunk in ["你", "好", "世界"]:
            yield chunk

    monkeypatch.setattr("dh_server.llm.openai.stream_chat", fake_stream_chat)

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

    monkeypatch.setattr("dh_server.llm.openai.stream_chat", fake_stream_chat)

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
    from dh_server.config import get_settings
    from dh_server.db.session import init_db, get_engine, reset_engine
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
