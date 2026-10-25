"""An avatar and the engine that drives it are a pair.

An avatar records the digital-human model that owns its pack. The server
drives it with that model's engine, and refuses anything else -- with a
message that says what is wrong, rather than letting the engine fail
later with a message about a missing pack.
"""

import io

import pytest
from PIL import Image
from sqlalchemy.orm import sessionmaker

from server.db.models import Avatar
from server.db.session import get_engine


def _png_bytes(w=640, h=800):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


def _seed(client, drive_model: str, status: str = "ready") -> str:
    db = sessionmaker(bind=get_engine())()
    try:
        row = Avatar(
            name="a", status=status, source_image_path="", drive_model=drive_model
        )
        db.add(row)
        db.commit()
        return row.id
    finally:
        db.close()


def test_a_created_avatar_records_its_drive_model(client):
    resp = client.post(
        "/api/admin/avatars",
        data={"name": "a", "model": "musetalk"},
        files={"file": ("a.png", _png_bytes(), "image/png")},
    )
    assert resp.status_code == 201, resp.text
    assert resp.json()["drive_model"] == "musetalk"

    avatar_id = resp.json()["avatar_id"]
    detail = client.get(f"/api/admin/avatars/{avatar_id}").json()
    assert detail["drive_model"] == "musetalk"
    assert detail["client_rendered"] is False


def test_an_unknown_model_is_refused_at_creation(client):
    resp = client.post(
        "/api/admin/avatars",
        data={"name": "a", "model": "nope"},
        files={"file": ("a.png", _png_bytes(), "image/png")},
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "unknown_model"


def test_a_client_rendered_avatar_cannot_start_a_server_session(client):
    """Live2D is drawn by the browser, so there is no server session for
    it. The refusal says so, instead of letting the engine complain about
    a pack it was never going to read."""
    avatar_id = _seed(client, "live2d")

    resp = client.post("/api/v1/sessions", json={"avatar_id": avatar_id})
    assert resp.status_code == 409
    body = resp.json()
    assert body["error"]["code"] == "avatar_not_server_driven"
    assert "Live2D" in body["error"]["message"]


def test_an_avatar_with_an_unregistered_model_is_refused(client):
    """A model that is no longer registered leaves the avatar undrivable.
    The refusal names the problem instead of failing obscurely later."""
    avatar_id = _seed(client, "ghost")

    resp = client.post("/api/v1/sessions", json={"avatar_id": avatar_id})
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "unknown_model"


def test_a_session_drives_the_avatar_with_its_own_model(client):
    """The engine comes from the avatar's model, not from whatever else
    the box happens to be running."""
    from server.main import _app

    avatar_id = _seed(client, "musetalk")
    resp = client.post("/api/v1/sessions", json={"avatar_id": avatar_id})
    assert resp.status_code == 201, resp.text

    engine = _app.state.session_manager.engines.engine_for("musetalk")
    render = _app.state.session_manager.get_render(resp.json()["session_id"])
    assert render.engine is engine
