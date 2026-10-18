"""OpenAI-compatible chat client used by the console's chat panel.

Speaks the ``/v1/chat/completions`` shape every major provider supports:

* OpenAI: ``https://api.openai.com/v1``
* Ollama: ``http://localhost:11434/v1``
* 阿里云 DashScope: ``https://dashscope.aliyuncs.com/compatible-mode/v1``

Two surfaces:

* :func:`chat` -- one POST, full reply string back. Used by the
  non-streaming chat endpoint (kept around as a fallback when SSE isn't
  desired or fails on the wire).
* :func:`stream_chat` -- async iterator yielding reply deltas. Used by
  the streaming endpoint. The caller can break out of the loop on the
  first ``StopAsyncIteration`` or by closing the underlying stream.

A short system prompt is bolted on to keep replies voice-friendly:
"你是一个数字人，正在与用户语音对话。请用简洁、口语化的中文回复
（30-80 字），不要使用表情符号、列表或 Markdown。" The chat panel
itself doesn't need to know about this prompt -- it's part of the
"basic LLM" the user asked for. A future revision can move the prompt
into the UI for the operator to tweak; for now the brief is fixed.

Both surfaces share the same error mapping:

* ``llm_not_configured`` (503) -- api_key or base_url missing.
* ``empty_messages`` (400) -- caller passed an empty list.
* ``llm_failed`` (502) -- upstream network / parse / non-200 error,
  or upstream returned a non-zero ``error`` block.
"""
from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx

from server.core.errors import AppError

#: Voice-friendly persona bolted on to every chat request. Kept short on
#: purpose: TTS round-trip + driving the avatar already adds latency; the
#: reply needs to be brief to feel like a real conversation.
SYSTEM_PROMPT = (
    "你是一个数字人，正在与用户语音对话。"
    "请用简洁、口语化的中文回复（30-80 字），"
    "不要使用表情符号、列表或 Markdown。"
)


def _read_cfg(db) -> dict[str, Any]:
    """Pull the effective LLM config (DB rows over env defaults).

    Falls back to ``Settings``-only defaults when ``db`` is None (the
    test path) so unit tests can stub the env without spinning up a DB.
    """
    from server.llm.config import read_config

    if db is not None:
        return read_config(db)
    return _settings_defaults_as_public()


def _settings_defaults_as_public() -> dict[str, Any]:
    """Mirror of ``read_config`` for callers without a DB session.

    LLM has no env-var defaults -- the operator wires it via the Models
    tab. The defaults returned here are the same empty placeholders
    ``read_config`` would surface for a freshly-installed box.
    """
    return {
        "api_key": "",
        "base_url": "",
        "model": "gpt-3.5-turbo",
    }


def _build_request(messages: list[dict[str, str]], cfg: dict[str, Any], *, stream: bool) -> dict[str, Any]:
    """Compose the OpenAI-shape body. The system prompt rides in every call.

    The caller-supplied ``messages`` is appended after the system prompt
    so a turn 0 user message lands at the right depth. ``stream`` only
    flips the boolean on the wire; the rest of the body is identical.
    """
    payload: dict[str, Any] = {
        "model": cfg["model"],
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, *messages],
    }
    if stream:
        payload["stream"] = True
    return payload


def _require_configured(cfg: dict[str, Any]) -> None:
    if not cfg.get("api_key"):
        raise AppError(
            "llm_not_configured",
            "LLM is not configured: set DH_LLM_API_KEY in .env or save it via the Models tab",
            503,
        )
    if not cfg.get("base_url"):
        raise AppError(
            "llm_not_configured",
            "LLM is not configured: set DH_LLM_BASE_URL in .env or save it via the Models tab",
            503,
        )


def _extract_content(data: dict[str, Any]) -> str:
    """Pull ``choices[0].message.content`` from a non-streaming reply."""
    try:
        choices = data.get("choices") or []
        if not choices:
            raise AppError("llm_failed", "LLM returned no choices", 502)
        message = choices[0].get("message") or {}
        content = message.get("content") or ""
    except (AttributeError, IndexError, KeyError, TypeError) as exc:
        raise AppError("llm_failed", f"LLM reply shape not recognised: {exc}", 502) from exc
    if not content:
        raise AppError("llm_failed", "LLM returned empty content", 502)
    return content


async def chat(
    messages: list[dict[str, str]],
    *,
    client: httpx.AsyncClient | None = None,
    db=None,
) -> str:
    """Send ``messages`` to the configured upstream and return the reply text.

    ``client`` is injectable so tests can drive the real parse path through
    an ``httpx.MockTransport`` instead of the network. ``db`` is the FastAPI
    request session; passed through so the effective LLM config (DB rows
    layered over env defaults) is read fresh on every call.
    """
    if not messages:
        raise AppError("empty_messages", "messages must not be empty", 400)

    cfg = _read_cfg(db)
    _require_configured(cfg)
    payload = _build_request(messages, cfg, stream=False)
    url = f"{cfg['base_url'].rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {cfg['api_key']}",
        "Content-Type": "application/json",
    }

    owns_client = client is None
    client = client or httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0))
    try:
        try:
            resp = await client.post(url, headers=headers, json=payload)
        except httpx.HTTPError as exc:
            raise AppError("llm_failed", f"LLM request failed: {exc}", 502) from exc
        if resp.status_code != 200:
            # Surface upstream HTTP errors verbatim. Some providers put a
            # JSON body with an ``error`` block; we don't try to be clever
            # about it -- just include the text the operator can read.
            raise AppError(
                "llm_failed",
                f"LLM returned HTTP {resp.status_code}: {resp.text[:200]!r}",
                502,
            )
        try:
            data = resp.json()
        except ValueError as exc:
            raise AppError(
                "llm_failed",
                f"LLM returned a non-JSON response (HTTP {resp.status_code})",
                502,
            ) from exc
    finally:
        if owns_client:
            await client.aclose()

    # Some providers echo an ``error`` block on HTTP 200; treat it like
    # any other upstream failure so the UI sees one consistent error shape.
    if isinstance(data, dict) and data.get("error"):
        raise AppError("llm_failed", f"LLM error: {data['error']}", 502)
    return _extract_content(data)


def _parse_sse_delta(line: str) -> str | None:
    """Yield the text delta from a single SSE ``data:`` line.

    Returns ``None`` for ``[DONE]`` sentinel and for blank lines (the SSE
    protocol's separator). Anything that isn't a recognised ``data:``
    line is treated as no-op so an unusual provider doesn't 502 the chat
    panel over a comment line.
    """
    if not line:
        return None
    if line.startswith(":"):
        # SSE comment; ignore.
        return None
    if not line.startswith("data:"):
        return None
    payload = line[len("data:"):].strip()
    if payload == "[DONE]":
        return None
    try:
        obj = json.loads(payload)
    except ValueError:
        return None
    choices = obj.get("choices") or []
    if not choices:
        return None
    delta = choices[0].get("delta") or {}
    content = delta.get("content")
    if not content:
        return None
    return content


async def stream_chat(
    messages: list[dict[str, str]],
    *,
    client: httpx.AsyncClient | None = None,
    db=None,
) -> AsyncIterator[str]:
    """Yield reply deltas as they arrive from the upstream SSE stream.

    Same semantics as :func:`chat` (error mapping, system prompt) but
    yields strings rather than returning one final string. The caller is
    expected to consume the iterator; ``stream=True`` is sent on the wire
    so the upstream keeps the connection open and pushes deltas as the
    model produces them.

    The iterator raises :class:`StopAsyncIteration` cleanly when the
    upstream emits ``[DONE]``. Upstream errors raise :class:`AppError`
    (502 ``llm_failed``) at the point of failure -- the chat panel
    surfaces the message to the operator via its existing error path.
    """
    if not messages:
        raise AppError("empty_messages", "messages must not be empty", 400)

    cfg = _read_cfg(db)
    _require_configured(cfg)
    payload = _build_request(messages, cfg, stream=True)
    url = f"{cfg['base_url'].rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {cfg['api_key']}",
        "Content-Type": "application/json",
        # Ask the upstream to keep the connection open and stream. Most
        # providers honour this when ``stream=true`` is in the body; the
        # header is harmless on top.
        "Accept": "text/event-stream",
    }

    owns_client = client is None
    # No read timeout on streaming -- the model can take a while between
    # tokens, and httpx's default 60 s would chop a long reply in half.
    # httpx.Timeout requires either a default OR all four of
    # connect/read/write/pool to be set explicitly; the historic bug
    # passed only ``connect`` and ``read``, which raised ``ValueError``
    # on the first request through the admin endpoint.
    client = client or httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10.0, read=None, write=10.0, pool=10.0)
    )
    try:
        try:
            req = client.build_request("POST", url, headers=headers, json=payload)
            resp = await client.send(req, stream=True)
        except httpx.HTTPError as exc:
            raise AppError("llm_failed", f"LLM request failed: {exc}", 502) from exc

        if resp.status_code != 200:
            # Drain the body so the connection releases, then surface.
            try:
                await resp.aread()
            finally:
                await resp.aclose()
            raise AppError(
                "llm_failed",
                f"LLM returned HTTP {resp.status_code}: {resp.text[:200]!r}",
                502,
            )

        try:
            # ``aiter_lines`` splits on newlines; SSE frames are separated
            # by blank lines, so we hand each line to ``_parse_sse_delta``
            # and let it filter.
            seen_done = False
            async for raw_line in resp.aiter_lines():
                if seen_done:
                    break
                # ``aiter_lines`` strips the trailing newline; lines come
                # in one at a time as the upstream emits them.
                content = _parse_sse_delta(raw_line)
                if content is None:
                    # Could be blank (frame separator), a comment, [DONE],
                    # or a non-content delta (e.g. role-only first chunk).
                    if raw_line.strip() == "data: [DONE]":
                        seen_done = True
                    continue
                yield content
        finally:
            await resp.aclose()
    finally:
        if owns_client:
            await client.aclose()
