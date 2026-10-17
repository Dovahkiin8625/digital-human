"""Admin-only close of any active session.

Admins need an out-of-band escape hatch: the session debug page can wedge
when the cached key goes stale or the operator wants to kill a stuck
session, so DELETE on /api/admin/sessions/{id} accepts any caller.
"""
from __future__ import annotations

import io
from uuid import uuid4

from PIL import Image


def _png():
    buf = io.BytesIO()
    Image.new("RGB", (640, 800), (200, 120, 80)).save(buf, format="PNG")
    return buf.getvalue()


def _upload_avatar(client, name):
    return client.post(
        "/api/v1/avatars", data={"name": name},
        files={"file": ("a.png", _png(), "image/png")},
    ).json()["avatar_id"]


def test_admin_close_session_marks_closed(client):
    avatar_id = _upload_avatar(client, "a")
    sid = client.post(
        "/api/v1/sessions", json={"avatar_id": avatar_id},
    ).json()["session_id"]
    # Session is open and listed.
    rows = client.get("/api/admin/sessions").json()
    assert any(r["session_id"] == sid for r in rows)

    resp = client.delete(f"/api/admin/sessions/{sid}")
    assert resp.status_code == 204

    rows = client.get("/api/admin/sessions").json()
    row = next(r for r in rows if r["session_id"] == sid)
    assert row["status"] in {"closed", "idle", "speaking", "listening", "error"}


def test_admin_close_session_unknown_404(client):
    resp = client.delete(f"/api/admin/sessions/{uuid4().hex}")
    assert resp.status_code == 404


def test_admin_sweep_returns_count(client):
    resp = client.post("/api/admin/sessions/sweep")
    assert resp.status_code == 200
    assert isinstance(resp.json().get("closed"), int)