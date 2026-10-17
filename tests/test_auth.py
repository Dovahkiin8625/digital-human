"""Error-envelope shape tests that were once bundled with the now-deleted
auth/tenant code. Kept here so the response-shape contract still has
explicit coverage (the formatters are used by every error path)."""


def test_error_envelope_shape(client):
    """A route-handler 404 goes through the AppError -> envelope path."""
    resp = client.get("/api/v1/avatars/nonexistent-id-here")
    assert resp.status_code == 404
    body = resp.json()
    assert body == {
        "error": {"code": "not_found", "message": "avatar nonexistent-id-here not found", "detail": None}
    }


def test_internal_error_envelope_hides_exception_text():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from server.core.errors import register_handlers

    app = FastAPI()
    register_handlers(app)

    @app.get("/boom")
    def boom():
        raise RuntimeError("s3cret-boom")

    with TestClient(app, raise_server_exceptions=False) as c:
        resp = c.get("/boom")
    assert resp.status_code == 500
    body = resp.json()
    assert body == {
        "error": {"code": "internal_error", "message": "internal server error", "detail": None}
    }
    assert "s3cret-boom" not in resp.text