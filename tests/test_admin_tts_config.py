"""``GET /api/admin/tts/config`` + ``PUT /api/admin/tts/config``.

The endpoint pair surfaces the MiniMax configuration to the Models tab
and persists operator edits to the SQLite ``settings`` table. The
synthesise endpoint reads through the same helper, so a successful PUT
takes effect on the next TTS request without a server restart.
"""
from __future__ import annotations


def test_get_returns_masked_api_key(client):
    """The GET response masks ``api_key`` as ``****<last4>`` so the UI
    never has to display the raw bearer token.

    The test DB carries no MiniMax rows, so the effective api_key is
    empty and the masked value is the "(未设置)" sentinel; the field's
    *presence* is what matters here.
    """
    body = client.get("/api/admin/tts/config").json()
    assert "api_key_masked" in body
    assert body["api_key_masked"] == "(未设置)"  # nothing configured yet
    # The unmasked ``api_key`` is *not* in the GET response.
    assert "api_key" not in body


def test_get_includes_public_config_fields(client):
    """The GET response carries every operator-facing field. The
    ``model`` / ``sample_rate`` / ``group_id`` / ``base_url`` values come
    from the built-in defaults when the DB has no rows. ``voice_id``
    used to live here -- the operator now picks it per session in the
    debug page, so it isn't a server-wide setting anymore.
    """
    body = client.get("/api/admin/tts/config").json()
    for field in ("model", "sample_rate", "group_id", "base_url"):
        assert field in body
    # ``voice_id`` is gone from the GET shape.
    assert "voice_id" not in body
    # Built-in defaults.
    assert body["model"] == "speech-02-hd"
    assert body["sample_rate"] == 32000


def test_put_updates_db_and_subsequent_get_reflects(client):
    """A PUT writes to the ``settings`` table; the next GET returns
    exactly what was sent (so the operator's "保存" feedback loop is
    honest)."""
    payload = {
        "api_key": "test-key-1234567890",
        "model": "speech-02-turbo",
        "sample_rate": 24000,
        "group_id": "g-test",
        "base_url": "https://api.example.com",
    }
    resp = client.put("/api/admin/tts/config", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["model"] == "speech-02-turbo"
    # api_key_masked reflects the new value: last 4 of "test-key-1234567890".
    assert body["api_key_masked"] == "****7890"
    # Raw api_key stays out of the response -- only the masked form leaks.
    assert "api_key" not in body
    # ``voice_id`` is gone from the response shape.
    assert "voice_id" not in body

    # And the next GET agrees.
    body2 = client.get("/api/admin/tts/config").json()
    assert body2["model"] == "speech-02-turbo"
    assert body2["sample_rate"] == 24000


def test_put_rejects_voice_id_field(client):
    """``voice_id`` used to be a configurable field -- it isn't anymore.
    Pydantic's ``extra='forbid'`` rejects it at the request boundary so a
    stale client sending voice_id gets a 422 with the field name in the
    error rather than a silently dropped key.
    """
    resp = client.put(
        "/api/admin/tts/config",
        json={"voice_id": "female-shaonv"},
    )
    assert resp.status_code == 422
    assert "voice_id" in str(resp.json())


def test_put_rejects_unknown_field(client):
    """An extra / typo'd field is rejected at the Pydantic boundary
    (``extra='forbid'``) as a 422 with the bad key in ``detail`` so the
    form can flag it. 422 is FastAPI's standard validation-error code
    for malformed bodies -- the response body still names the field."""
    resp = client.put(
        "/api/admin/tts/config",
        json={"voice_id": "male-qn-qingse", "bogus_field": "x"},
    )
    assert resp.status_code == 422
    # FastAPI's default 422 body: ``detail`` is a list of dicts with
    # ``loc`` (field path) and ``msg``. The bogus key is named there.
    detail_blob = str(resp.json())
    assert "bogus_field" in detail_blob


def test_put_rejects_out_of_range_sample_rate(client):
    """``sample_rate`` must be one of MiniMax's supported rates."""
    resp = client.put(
        "/api/admin/tts/config",
        json={"sample_rate": 11025},  # not a MiniMax-supported rate
    )
    assert resp.status_code == 400
    detail = resp.json()["error"]["detail"]
    assert "sample_rate" in detail
    assert "8000" in detail["sample_rate"]


def test_put_rejects_invalid_base_url(client):
    """``base_url`` must look like an HTTP(S) URL -- the synthesise
    path concatenates ``/v1/t2a_v2`` onto it, so a bare host or scheme-
    less value would 404 on the wire.
    """
    resp = client.put(
        "/api/admin/tts/config",
        json={"base_url": "api.example.com"},  # missing scheme
    )
    assert resp.status_code == 400
    assert "base_url" in resp.json()["error"]["detail"]


def test_put_accepts_empty_api_key(client):
    """An empty ``api_key`` is a valid clear-the-credential state, NOT
    an error -- the synthesise endpoint surfaces 503 separately.

    The form's "保存" flow must be able to clear an old key.
    """
    resp = client.put(
        "/api/admin/tts/config",
        json={"api_key": ""},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["api_key_masked"] == "(未设置)"