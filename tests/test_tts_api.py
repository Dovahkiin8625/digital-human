import json

import httpx
import pytest

from server.core.errors import AppError

ADMIN = {"X-Admin-Token": "dev-admin-token"}


@pytest.fixture()
def tts_env(monkeypatch):
    """Configure MiniMax the way a real .env would, and drop the settings cache.

    get_settings() is lru_cached, so a bare setenv would be ignored until the
    cache is cleared; the yield/cleanup keeps the cache from leaking a fake key
    into the next test.
    """
    monkeypatch.setenv("DH_MINIMAX_API_KEY", "test-key")
    monkeypatch.setenv("DH_MINIMAX_GROUP_ID", "test-group")
    monkeypatch.setenv("DH_MINIMAX_BASE_URL", "https://api.example.test")
    monkeypatch.setenv("DH_MINIMAX_VOICE_ID", "male-qn-qingse")
    monkeypatch.setenv("DH_MINIMAX_MODEL", "speech-02-hd")
    monkeypatch.setenv("DH_MINIMAX_SAMPLE_RATE", "32000")
    from server.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _mock_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# --- the PCM -> WAV wrapper -------------------------------------------------


def test_wav_from_pcm_writes_a_16bit_mono_header():
    from server.tts.minimax import wav_from_pcm

    pcm = b"\x01\x02\x03\x04"
    wav = wav_from_pcm(pcm, 32000)

    assert wav[:4] == b"RIFF"
    assert wav[8:12] == b"WAVE"
    assert wav[12:16] == b"fmt "
    assert wav[36:40] == b"data"
    # RIFF chunk size covers everything after the first 8 bytes; data size is the payload.
    assert int.from_bytes(wav[4:8], "little") == len(wav) - 8
    assert int.from_bytes(wav[40:44], "little") == len(pcm)
    # fmt chunk: PCM(1), 1 channel, 32000 Hz, 16-bit -> 2 bytes/frame, 64000 B/s.
    assert int.from_bytes(wav[20:22], "little") == 1  # audio format
    assert int.from_bytes(wav[22:24], "little") == 1  # channels
    assert int.from_bytes(wav[24:28], "little") == 32000  # sample rate
    assert int.from_bytes(wav[28:32], "little") == 64000  # byte rate
    assert int.from_bytes(wav[32:34], "little") == 2  # block align
    assert int.from_bytes(wav[34:36], "little") == 16  # bits per sample
    assert wav[44:] == pcm


# --- the MiniMax client -----------------------------------------------------


@pytest.mark.asyncio
async def test_synthesize_returns_wav_and_sends_the_voice_settings(tts_env):
    from server.tts.minimax import synthesize

    pcm = b"\x10\x20\x30\x40"
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "base_resp": {"status_code": 0, "status_msg": "success"},
                "data": {"audio": pcm.hex()},
            },
        )

    client = _mock_client(handler)
    try:
        wav = await synthesize("你好世界", "female-shaonv", client=client)
    finally:
        await client.aclose()

    assert wav[:4] == b"RIFF"
    assert wav[44:] == pcm
    assert seen["url"].startswith("https://api.example.test/v1/t2a_v2")
    assert seen["auth"] == "Bearer test-key"
    assert seen["body"]["model"] == "speech-02-hd"
    assert seen["body"]["text"] == "你好世界"
    assert seen["body"]["voice_setting"]["voice_id"] == "female-shaonv"
    assert seen["body"]["audio_setting"] == {"format": "pcm", "sample_rate": 32000, "channel": 1}
    # Non-streaming, hex-encoded: the two options the WAV wrapper depends on.
    assert seen["body"]["stream"] is False
    assert seen["body"]["output_format"] == "hex"


@pytest.mark.asyncio
async def test_synthesize_rejects_empty_voice_id(tts_env):
    """``voice_id`` is required -- the operator picks it per session in the
    debug page, so there is no server-wide default. An empty string
    surfaces as 400 ``missing_voice_id`` so the UI can prompt the user
    rather than silently picking one.
    """
    from server.tts.minimax import synthesize

    with pytest.raises(AppError) as excinfo:
        await synthesize("hi", "")
    assert excinfo.value.status == 400
    assert excinfo.value.code == "missing_voice_id"

    # Whitespace-only counts as missing too (UI often ends up with that).
    with pytest.raises(AppError) as excinfo:
        await synthesize("hi", "   ")
    assert excinfo.value.status == 400
    assert excinfo.value.code == "missing_voice_id"


@pytest.mark.asyncio
async def test_synthesize_without_api_key_raises_not_configured(monkeypatch):
    monkeypatch.setenv("DH_MINIMAX_API_KEY", "")
    from server.config import get_settings
    from server.tts.minimax import synthesize

    get_settings.cache_clear()
    try:
        with pytest.raises(AppError) as excinfo:
            await synthesize("hi", "female-shaonv")
        assert excinfo.value.status == 503
        assert excinfo.value.code == "tts_not_configured"
    finally:
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_synthesize_surfaces_a_minimax_error(tts_env):
    from server.tts.minimax import synthesize

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "base_resp": {"status_code": 1004, "status_msg": "invalid api key"},
                "data": {"audio": ""},
            },
        )

    client = _mock_client(handler)
    try:
        with pytest.raises(AppError) as excinfo:
            await synthesize("hi", "female-shaonv", client=client)
    finally:
        await client.aclose()

    assert excinfo.value.status == 502
    assert excinfo.value.code == "tts_failed"
    assert "invalid api key" in excinfo.value.message


@pytest.mark.asyncio
async def test_synthesize_rejects_a_non_hex_payload(tts_env):
    # output_format=hex is pinned, but if the reply ever carries a URL instead we
    # want the 502 that names the problem, not a bare ValueError turned 500.
    from server.tts.minimax import synthesize

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "base_resp": {"status_code": 0},
                "data": {"audio": "https://cdn.example.test/a.mp3"},
            },
        )

    client = _mock_client(handler)
    try:
        with pytest.raises(AppError) as excinfo:
            await synthesize("hi", "female-shaonv", client=client)
    finally:
        await client.aclose()

    assert excinfo.value.status == 502
    assert excinfo.value.code == "tts_failed"


@pytest.mark.asyncio
async def test_synthesize_surfaces_a_transport_error(tts_env):
    from server.tts.minimax import synthesize

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    client = _mock_client(handler)
    try:
        with pytest.raises(AppError) as excinfo:
            await synthesize("hi", "female-shaonv", client=client)
    finally:
        await client.aclose()

    assert excinfo.value.status == 502
    assert excinfo.value.code == "tts_failed"


# --- the admin endpoint -----------------------------------------------------


def test_tts_endpoint_returns_wav(client, monkeypatch):
    captured: dict = {}

    async def fake_synthesize(text, voice_id=None, **kwargs):
        captured["text"] = text
        captured["voice_id"] = voice_id
        return b"RIFF\x00\x00\x00\x00WAVEfake"

    monkeypatch.setattr("server.tts.minimax.synthesize", fake_synthesize)

    resp = client.post(
        "/api/admin/tts", headers=ADMIN, json={"text": "你好", "voice_id": "female-shaonv"}
    )

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"
    assert resp.content == b"RIFF\x00\x00\x00\x00WAVEfake"
    assert captured == {"text": "你好", "voice_id": "female-shaonv"}


def test_tts_endpoint_rejects_blank_text(client):
    resp = client.post("/api/admin/tts", headers=ADMIN, json={"text": "   ", "voice_id": "female-shaonv"})
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "empty_text"


def test_tts_endpoint_reports_missing_configuration(client, monkeypatch):
    # No synthesize stub here: the real client must refuse to call out without a key.
    monkeypatch.setenv("DH_MINIMAX_API_KEY", "")
    from server.config import get_settings

    get_settings.cache_clear()

    resp = client.post(
        "/api/admin/tts", headers=ADMIN, json={"text": "hi", "voice_id": "female-shaonv"}
    )

    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "tts_not_configured"


def test_tts_endpoint_reports_missing_voice_id(client):
    """The synthesise endpoint requires a ``voice_id``; Pydantic rejects the
    missing field at the request boundary as 422 with the field name in the
    error blob. The synthesise path itself also raises 400 ``missing_voice_id``
    when called directly (e.g. the chat worker) -- that branch is covered by
    ``test_synthesize_rejects_empty_voice_id`` above.
    """
    resp = client.post("/api/admin/tts", headers=ADMIN, json={"text": "hi"})
    assert resp.status_code == 422
    assert "voice_id" in str(resp.json())
