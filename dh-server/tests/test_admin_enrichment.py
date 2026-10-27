"""Admin list endpoints expose the data the console needs.

Jobs and sessions each carry one or two fields on top of what their flat row
already had (latency stats per session); the console just reads those keys.
"""
from __future__ import annotations

import io

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


def test_session_row_includes_avatar_id_and_latency_stats(client):
    avatar_id = _upload_avatar(client, "x")
    sid = client.post(
        "/api/v1/sessions", json={"avatar_id": avatar_id},
    ).json()["session_id"]

    rows = client.get("/api/admin/sessions").json()
    row = next(r for r in rows if r["session_id"] == sid)
    assert row["avatar_id"] == avatar_id
    # Live metrics the engine writes; values may be 0 at idle.
    for k in (
        "frames_out",
        "dropped_frames",
        "keepalive_frames",
        "avg_latency_ms",
        "p50_latency_ms",
        "p95_latency_ms",
        "max_latency_ms",
    ):
        assert k in row["metrics"], f"missing metric {k}"
    assert "latency_histogram" in row["metrics"]


def test_jobs_row_includes_avatar_id_and_created_at(client):
    avatar_id = _upload_avatar(client, "x")

    rows = client.get("/api/admin/jobs").json()
    row = next(r for r in rows if r["avatar_id"] == avatar_id)
    assert row["avatar_id"] == avatar_id
    assert "created_at" in row