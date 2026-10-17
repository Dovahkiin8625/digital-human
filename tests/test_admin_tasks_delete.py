"""DELETE /api/admin/tasks/{task_id}: remove a generation task + its scratch dir.

Covers the operator-facing "delete this record" action on the 图生视频 page.
The DELETE handler is expected to:
  * 404 on a non-existent or non-uuid task id
  * 204 on a finished / failed / queued task, removing the DB row
  * remove the on-disk scratch dir under ``data_dir/tmp/anim/<task_id>``
    so the operator's disk doesn't accumulate dead ends after a delete.
"""
from __future__ import annotations

from uuid import uuid4

def _seed_task(task_id, status="done", error=None):
    """Insert a GenerationTask row directly so the test doesn't depend on
    the i2v pipeline running. Mirrors the row shape ``admin_generate_i2v``
    writes via SQLAlchemy.
    """
    import json

    from sqlalchemy.orm import sessionmaker

    from server.db.models import GenerationTask
    from server.db.session import get_engine

    session = sessionmaker(bind=get_engine())()
    try:
        session.add(
            GenerationTask(
                id=task_id,
                type="i2v",
                status=status,
                stage="" if status == "queued" else "done",
                progress=100 if status == "done" else 0,
                error=error,
                params_json=json.dumps({"backend": "liveportrait"}),
                result_json=json.dumps({"mp4_path": "ignored"}) if status == "done" else None,
            )
        )
        session.commit()
    finally:
        session.close()


def test_admin_delete_task_not_found_returns_404(client):
    # A fresh uuid never existed in the DB: the handler must surface 404 so
    # the UI can show "该记录已不存在" rather than silently 204.
    assert client.delete(f"/api/admin/tasks/{uuid4().hex}", ).status_code == 404
    # A non-uuid id must also 404 (the validator rejects before the DB lookup).
    assert client.delete("/api/admin/tasks/not-a-uuid", ).status_code == 404


def test_admin_delete_task_removes_row_and_scratch_dir(client):
    import shutil

    from server.config import get_settings

    task_id = uuid4().hex
    scratch_root = get_settings().data_dir / "tmp" / "anim"
    scratch = scratch_root / task_id
    scratch.mkdir(parents=True, exist_ok=True)
    (scratch / "out.mp4").write_bytes(b"\x00" * 64)
    _seed_task(task_id, status="done")

    resp = client.delete(f"/api/admin/tasks/{task_id}", )
    assert resp.status_code == 204

    # The row is gone — the list endpoint should no longer report it.
    listed = client.get("/api/admin/tasks", ).json()["tasks"]
    assert all(t["task_id"] != task_id for t in listed)
    # The on-disk scratch dir is gone too — leftover dirs would clutter
    # ``data_dir/tmp/anim`` over the operator's career.
    assert not scratch.exists()


def test_admin_delete_task_works_for_failed_and_queued_rows(client):
    # The UI's 删除 button must work regardless of terminal state: an operator
    # clearing out a failed run (or a queued one that never started) is a
    # normal flow.
    for status, err in (("failed", "boom"), ("queued", None)):
        task_id = uuid4().hex
        _seed_task(task_id, status=status, error=err)
        resp = client.delete(f"/api/admin/tasks/{task_id}", )
        assert resp.status_code == 204, f"delete of {status} row must succeed"
        listed = client.get("/api/admin/tasks", ).json()["tasks"]
        assert all(t["task_id"] != task_id for t in listed)


def test_admin_delete_task_does_not_remove_other_tasks_scratch_dirs(client):
    # Guard against the obvious bug: an ``rmtree`` aimed at one task id
    # wiping the whole scratch root or a sibling task's dir.
    from server.config import get_settings

    scratch_root = get_settings().data_dir / "tmp" / "anim"
    keep_id = uuid4().hex
    drop_id = uuid4().hex
    keep_dir = scratch_root / keep_id
    drop_dir = scratch_root / drop_id
    keep_dir.mkdir(parents=True, exist_ok=True)
    drop_dir.mkdir(parents=True, exist_ok=True)
    (keep_dir / "out.mp4").write_bytes(b"\x00" * 16)
    (drop_dir / "out.mp4").write_bytes(b"\x00" * 16)
    _seed_task(keep_id, status="done")
    _seed_task(drop_id, status="done")

    assert client.delete(f"/api/admin/tasks/{drop_id}", ).status_code == 204

    assert keep_dir.exists(), "sibling scratch dir must survive a delete"
    assert not drop_dir.exists(), "deleted scratch dir must be gone"
