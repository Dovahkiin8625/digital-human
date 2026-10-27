"""Admin-only session row management.

The admin console surfaces two actions on the System page sessions card:
the per-row "删除" button (DELETE /api/admin/sessions/{id}) and the
"清理已关闭会话" sweep button (POST /api/admin/sessions/sweep). Both
must remove the StreamSession row from the DB so the operator can purge
the accumulation of closed sessions the lifespan sweeper has been
marking closed over time. They must NOT remove the in-memory render of
a still-active session — that's the operation's escape-hatch side, and
breaking it would wedge real conversations.
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


def _session_ids(rows):
    return {r["session_id"] for r in rows}


def test_admin_close_session_deletes_row(client):
    """DELETE /api/admin/sessions/{id} removes the StreamSession row."""
    avatar_id = _upload_avatar(client, "a")
    sid = client.post(
        "/api/v1/sessions", json={"avatar_id": avatar_id},
    ).json()["session_id"]
    # Session is open and listed.
    rows = client.get("/api/admin/sessions").json()
    assert sid in _session_ids(rows)

    resp = client.delete(f"/api/admin/sessions/{sid}")
    assert resp.status_code == 204

    rows = client.get("/api/admin/sessions").json()
    assert sid not in _session_ids(rows), (
        "closed-but-undeleted rows accumulate forever and the System page "
        "never gets shorter, which is the bug this test pins down"
    )


def test_admin_close_session_unknown_404(client):
    resp = client.delete(f"/api/admin/sessions/{uuid4().hex}")
    assert resp.status_code == 404


def test_admin_sweep_returns_count(client):
    resp = client.post("/api/admin/sessions/sweep")
    assert resp.status_code == 200
    assert isinstance(resp.json().get("closed"), int)


def test_admin_sweep_purges_closed_session_rows(client):
    """Sweep deletes every StreamSession row whose status is 'closed'."""
    avatar_id = _upload_avatar(client, "a")
    sid = client.post(
        "/api/v1/sessions", json={"avatar_id": avatar_id},
    ).json()["session_id"]
    # Mark the row closed in the DB but leave the render alive: this matches
    # what the lifespan sweeper leaves behind after it force-closes a
    # wedge — and what the operator is asking the button to clean up.
    from sqlalchemy.orm import sessionmaker

    from server.db.session import get_engine

    db = sessionmaker(bind=get_engine())()
    try:
        from server.db.models import StreamSession

        row = db.get(StreamSession, sid)
        row.status = "closed"
        db.commit()
    finally:
        db.close()

    resp = client.post("/api/admin/sessions/sweep")
    assert resp.status_code == 200
    assert resp.json()["closed"] >= 1

    rows = client.get("/api/admin/sessions").json()
    assert sid not in _session_ids(rows)


def test_admin_sweep_keeps_active_session_rows(client):
    """Sweep must not delete rows whose render is still alive."""
    avatar_id = _upload_avatar(client, "a")
    sid = client.post(
        "/api/v1/sessions", json={"avatar_id": avatar_id},
    ).json()["session_id"]

    resp = client.post("/api/admin/sessions/sweep")
    assert resp.status_code == 200

    rows = client.get("/api/admin/sessions").json()
    assert sid in _session_ids(rows), "sweep must not touch active sessions"
