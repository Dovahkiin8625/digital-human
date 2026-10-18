"""``GET /api/admin/llm/config`` + ``PUT /api/admin/llm/config``.

The endpoint pair surfaces the OpenAI-compatible LLM configuration to the
Models tab and persists operator edits to the SQLite ``settings`` table.
The chat path reads through :func:`server.llm.config.read_config` on every
call so a successful PUT takes effect on the next request without a server
restart.
"""
from __future__ import annotations


def test_get_returns_masked_api_key(client):
    """The GET response masks ``api_key`` as ``****<last4>`` so the UI
    never has to display the raw bearer token.

    Test env pins ``DH_LLM_API_KEY=""`` so the masked value is the
    "(未设置)" sentinel; the field's *presence* is what matters here.
    """
    body = client.get("/api/admin/llm/config").json()
    assert "api_key_masked" in body
    assert body["api_key_masked"] == "(未设置)"  # test env has empty key
    # The unmasked ``api_key`` is *not* in the GET response.
    assert "api_key" not in body


def test_get_includes_public_config_fields(client):
    """The GET response carries every operator-facing field. ``model``
    has a real default in Settings; ``base_url`` defaults to empty (the
    operator must wire it)."""
    body = client.get("/api/admin/llm/config").json()
    for field in ("model", "base_url"):
        assert field in body
    # Default model from Settings.
    assert body["model"] == "gpt-3.5-turbo"
    # base_url defaults to empty -- not configured out of the box.
    assert body["base_url"] == ""
    # ``configured`` is False: api_key + base_url both empty.
    assert body["configured"] is False


def test_put_updates_db_and_subsequent_get_reflects(client):
    """A PUT writes to the ``settings`` table; the next GET returns
    exactly what was sent (so the operator's "保存" feedback loop is
    honest)."""
    payload = {
        "api_key": "test-key-1234567890",
        "base_url": "https://api.example.com/v1",
        "model": "llama3",
    }
    resp = client.put("/api/admin/llm/config", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["base_url"] == "https://api.example.com/v1"
    assert body["model"] == "llama3"
    # api_key_masked reflects the new value: last 4 of "test-key-1234567890".
    assert body["api_key_masked"] == "****7890"
    # Raw api_key stays out of the response -- only the masked form leaks.
    assert "api_key" not in body
    # configured flips to True once both api_key and base_url are set.
    assert body["configured"] is True

    # And the next GET agrees.
    body2 = client.get("/api/admin/llm/config").json()
    assert body2["base_url"] == "https://api.example.com/v1"
    assert body2["model"] == "llama3"


def test_put_rejects_unknown_field(client):
    """An extra / typo'd field is rejected at the Pydantic boundary
    (``extra='forbid'``) as a 422 with the bad key in ``detail`` so the
    form can flag it."""
    resp = client.put(
        "/api/admin/llm/config",
        json={"base_url": "https://api.example.com/v1", "bogus_field": "x"},
    )
    assert resp.status_code == 422
    detail_blob = str(resp.json())
    assert "bogus_field" in detail_blob


def test_put_rejects_invalid_base_url(client):
    """``base_url`` must look like an HTTP(S) URL -- the chat path
    concatenates ``/chat/completions`` onto it, so a bare host or
    scheme-less value would 404 on the wire.
    """
    resp = client.put(
        "/api/admin/llm/config",
        json={"base_url": "api.example.com"},  # missing scheme
    )
    assert resp.status_code == 400
    assert "base_url" in resp.json()["error"]["detail"]


def test_put_rejects_empty_model(client):
    """``model`` must be a non-empty string -- an empty value would
    yield a 4xx upstream response and a confusing error to the operator.
    """
    resp = client.put(
        "/api/admin/llm/config",
        json={"model": ""},
    )
    assert resp.status_code == 400
    assert "model" in resp.json()["error"]["detail"]


def test_put_accepts_empty_api_key(client):
    """An empty ``api_key`` is a valid clear-the-credential state, NOT
    an error -- the chat endpoint surfaces 503 separately.

    The form's "保存" flow must be able to clear an old key.
    """
    resp = client.put(
        "/api/admin/llm/config",
        json={"api_key": ""},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["api_key_masked"] == "(未设置)"
    # configured flips back to False: no api_key.
    assert body["configured"] is False
