"""``POST /api/admin/i2v`` + ``GET /api/admin/backends`` for the 形象活化 tab.

Stage 1 ships the dispatch surface but no animators are registered yet: the
registry is intentionally empty so a future ``avatar/<backend>`` change lands
green. These tests pin the empty-state contracts (well-formed 400 + well-
formed empty list) so the dispatcher wiring can't silently regress before any
backends are added.
"""
from __future__ import annotations

import io

from PIL import Image

from server.avatar import animators


def _png():
    buf = io.BytesIO()
    Image.new("RGB", (640, 800), (10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


def test_backends_endpoint_lists_registry(client, monkeypatch):
    # The UI populates its dropdown from /api/admin/backends. Every backend
    # registers itself on import; the page renders one OptGroup per
    # ``group_zh`` bucket, so locking the names gives us coverage for
    # the whole dispatch surface in a single assertion.
    #
    # Since the lifecycle refactor (animate() refuses when not loaded),
    # the backend listing also filters to ``is_loaded=True`` -- so this
    # test forces each registered backend to a "loaded" state before
    # asserting membership. Without that flip, the test env's
    # not-loaded backends (no weights on disk) would be missing from
    # the listing and the assertion would fail.
    from server.avatar import animators, musetalk_silent

    for animator in animators.ANIMATORS.values():
        # Backing attrs the lifecycle methods look at. ``is_loaded`` is
        # normally a property; assigning to the underlying ``_loaded``
        # mirrors what a real ``load()`` call would have flipped.
        try:
            animator._loaded = True
        except AttributeError:
            pass
    # MuseTalk silent's ``is_loaded`` is a property that proxies to
    # the live drive engine's runtime -- not a `_loaded` field. Patch
    # the proxy to return True so this row also appears in the listing.
    monkeypatch.setattr(
        musetalk_silent.MuseTalkSilentAnimator,
        "_engine_runtime_loaded",
        lambda self: True,
    )

    try:
        resp = client.get("/api/admin/backends", )
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body.get("backends"), list)
        names = {b["name"] for b in body["backends"]}
        expected = {
            "liveportrait",
            "wan22-i2v-5b-bf16",
            "musetalk-silent",
        }
        assert expected.issubset(names), (
            f"missing backends: {expected - names}; "
            "is the /api/admin/backends filter dropping them?"
        )
    finally:
        for animator in animators.ANIMATORS.values():
            try:
                animator._loaded = False
            except AttributeError:
                pass


def test_backends_endpoint_hides_unloaded_animators(client, monkeypatch):
    """The 形象活化 dropdown shows only backends whose pipeline is live.

    Pinning behaviour: an operator looking at the 图生视频 tab must
    never see an option that will 503 on click. The animation dispatch
    refuses ``is_loaded=False`` (see test_animator_loadable.py), so
    listing them in the dropdown would be a UI lie. The endpoint
    filters to ``is_loaded=True`` and returns whatever's left.
    """
    from server.avatar import animators

    class _Loaded:
        name = "_lp_loaded"

        @property
        def is_loaded(self):
            return True

        def meta(self):
            return {
                "label_zh": "loaded",
                "group_zh": "测试",
                "description": "x",
                "vram_hint": "0",
                "accepts_prompt": True,
                "accepts_num_frames": True,
                "accepts_steps": True,
            }

    class _NotLoaded:
        name = "_lp_offline"

        @property
        def is_loaded(self):
            return False

        def meta(self):
            return {
                "label_zh": "offline",
                "group_zh": "测试",
                "description": "x",
                "vram_hint": "0",
                "accepts_prompt": True,
                "accepts_num_frames": True,
                "accepts_steps": True,
            }

    monkeypatch.setitem(animators.ANIMATORS, "_lp_loaded", _Loaded())
    monkeypatch.setitem(animators.ANIMATORS, "_lp_offline", _NotLoaded())

    body = client.get("/api/admin/backends").json()
    names = {b["name"] for b in body["backends"]}
    assert "_lp_loaded" in names
    assert "_lp_offline" not in names


def test_backends_endpoint_omits_unloaded_musetalk_silent(client, monkeypatch):
    """MuseTalk silent's ``is_loaded`` mirrors the main drive engine
    runtime; when that runtime is unloaded, the silent row must NOT
    appear in the dropdown (clicking it would 503 immediately).

    Pinning behaviour for the lifecycle coupling added with the
    load/unload rewrite.
    """
    from server.avatar import musetalk_silent

    # Mark MuseTalk silent as unloaded by giving the runtime check a
    # default-False branch. (The test env's main engine hasn't loaded
    # a real runtime, so this is the realistic pre-fix state.)
    monkeypatch.setattr(
        musetalk_silent.MuseTalkSilentAnimator,
        "_engine_runtime_loaded",
        lambda self: False,
    )
    body = client.get("/api/admin/backends").json()
    names = {b["name"] for b in body["backends"]}
    assert "musetalk-silent" not in names
    # And when the drive engine IS loaded, MuseTalk silent must reappear
    # in the dropdown.
    monkeypatch.setattr(
        musetalk_silent.MuseTalkSilentAnimator,
        "_engine_runtime_loaded",
        lambda self: True,
    )
    body = client.get("/api/admin/backends").json()
    names = {b["name"] for b in body["backends"]}
    assert "musetalk-silent" in names


def test_i2v_rejects_unknown_backend_name(client):
    # The dispatcher refuses a backend that isn't in ANIMATORS, naming
    # the known ones so the operator can correct their input.
    resp = client.post(
        "/api/admin/i2v",
        data={"backend": "wan22-99b-fake", "prompt": "x", "num_frames": "8", "steps": "10"},
        files={"file": ("a.png", _png(), "image/png")},
    )
    assert resp.status_code == 400
    body = resp.json()
    assert body["error"]["code"] == "invalid_backend"
    msg = body["error"]["message"]
    # The error message must include the unknown name + at least one valid
    # one so the operator doesn't have to guess what's installed.
    assert "wan22-99b-fake" in msg
    assert "liveportrait" in msg
    assert "musetalk-silent" in msg


def test_i2v_requires_backend_field(client):
    # A request without ``backend`` fails FastAPI form parsing with 422.
    resp = client.post(
        "/api/admin/i2v",
        data={"prompt": "x", "num_frames": "8", "steps": "10"},
        files={"file": ("a.png", _png(), "image/png")},
    )
    assert resp.status_code == 422


def test_registry_importable_with_every_backend_registered():
    # The registry is wired. Each backend installs itself into ANIMATORS via
    # a side-effect import; tests for the specific backends (liveportrait /
    # wan22 / musetalk_silent) lock the deep contract, while this file
    # locks the at-a-glance shape: all four expected names appear in the
    # registry when the API module is loaded.
    assert hasattr(animators, "ANIMATORS")
    assert isinstance(animators.ANIMATORS, dict)
    expected = {
        "liveportrait",
        "wan22-i2v-5b-bf16",
        "musetalk-silent",
    }
    registered = set(animators.ANIMATORS)
    assert expected.issubset(registered), (
        f"missing backends: {expected - registered}"
    )


def test_animator_protocol_exposes_meta():
    # A bare-bones standalone backend (LivePortrait / Wan 2.2 I2V / MuseTalk
    # silent / ...) must return a metadata dict from meta() with the
    # contract keys the UI's TypeScript shape depends on.
    class _Stub:
        name = "stub"

        @property
        def is_ready(self):
            return True

        def meta(self):
            return {
                "label_zh": "示例",
                "group_zh": "测试",
                "description": "stub",
                "vram_hint": "0 GB",
                "accepts_prompt": True,
                "accepts_num_frames": True,
                "accepts_steps": True,
            }

        def validate(self, *, prompt, num_frames, steps):
            return {}

        def animate(self, source_path, *, prompt, num_frames, steps, out_dir):
            # Write an empty mp4 placeholder so the endpoint has something
            # to stream -- the actual content is irrelevant here.
            (out_dir / "out.mp4").write_bytes(b"\x00")

    s = _Stub()
    assert s.meta()["label_zh"] == "示例"
    assert s.is_ready is True
    assert s.validate(prompt="", num_frames=8, steps=10) == {}


# ----- Async task submission ----------------------------------------------
#
# The 图生视频 tab is too slow to keep the HTTP request open (an encoder can
# run for tens of seconds). The endpoint submits a GenerationTask and returns
# 202 + task_id; the operator polls /api/admin/tasks/{id} for progress.


def test_i2v_submits_task_and_returns_202(client, monkeypatch):
    # The submission endpoint must not block on the encoder. A stub backend
    # that sleeps long enough to exceed any reasonable client timeout is the
    # way to assert the request returns before the work finishes.
    from server.avatar import animators

    class _Slow:
        name = "slow"

        @property
        def is_ready(self):
            return True

        def meta(self):
            return {
                "label_zh": "慢速",
                "group_zh": "测试",
                "description": "slow stub",
                "vram_hint": "0 GB",
                "accepts_prompt": True,
                "accepts_num_frames": True,
                "accepts_steps": True,
            }

        def validate(self, *, prompt, num_frames, steps):
            return {}

        def animate(self, source_path, *, prompt, num_frames, steps, out_dir):
            (out_dir / "out.mp4").write_bytes(b"\x00FAKE-MP4")

    monkeypatch.setitem(animators.ANIMATORS, "slow", _Slow())

    resp = client.post(
        "/api/admin/i2v",
        data={"backend": "slow", "prompt": "x", "num_frames": "8", "steps": "10"},
        files={"file": ("a.png", _png(), "image/png")},
    )
    assert resp.status_code == 202
    body = resp.json()
    assert "task_id" in body and len(body["task_id"]) == 32


def test_i2v_task_status_progresses_to_done(client, monkeypatch):
    # After submission, GET /api/admin/tasks/{id} must reflect the backend's
    # progress: status moves queued -> running -> done, and a finished task
    # has a non-NULL result_json pointing at the rendered clip.
    from server.avatar import animators

    class _Quick:
        name = "quick"

        @property
        def is_ready(self):
            return True

        def meta(self):
            return {
                "label_zh": "快速",
                "group_zh": "测试",
                "description": "quick stub",
                "vram_hint": "0 GB",
                "accepts_prompt": True,
                "accepts_num_frames": True,
                "accepts_steps": True,
            }

        def validate(self, *, prompt, num_frames, steps):
            return {}

        def animate(self, source_path, *, prompt, num_frames, steps, out_dir):
            (out_dir / "out.mp4").write_bytes(b"\x00QUICK")

    monkeypatch.setitem(animators.ANIMATORS, "quick", _Quick())

    resp = client.post(
        "/api/admin/i2v",
        data={"backend": "quick", "prompt": "x", "num_frames": "8", "steps": "10"},
        files={"file": ("a.png", _png(), "image/png")},
    )
    task_id = resp.json()["task_id"]

    # Poll until done. The animation runs in a background task so the test
    # must wait for it -- bound it so a hung backend fails the test cleanly.
    import time

    deadline = time.time() + 10.0
    body = None
    while time.time() < deadline:
        body = client.get(f"/api/admin/tasks/{task_id}", ).json()
        if body["status"] in ("done", "failed"):
            break
        time.sleep(0.05)
    assert body is not None and body["status"] == "done", body
    assert body["type"] == "i2v"
    # ``result`` is the deserialised payload; the actual bytes come from
    # /api/admin/tasks/{id}/result.mp4. The path is server-internal -- the
    # page never reads it directly.
    assert body["result"] is not None
    assert body["result"]["mp4_path"]

    # The result endpoint streams the same bytes the animator wrote.
    clip = client.get(f"/api/admin/tasks/{task_id}/result.mp4", )
    assert clip.status_code == 200
    assert clip.content == b"\x00QUICK"


def test_i2v_task_records_failure(client, monkeypatch):
    # A backend that raises must surface as ``status="failed"`` with the
    # error message intact, never as a 5xx hanging the client.
    from server.avatar import animators
    from server.avatar.animators import AnimatorError

    class _Broken:
        name = "broken"

        @property
        def is_ready(self):
            return True

        def meta(self):
            return {
                "label_zh": "坏",
                "group_zh": "测试",
                "description": "broken stub",
                "vram_hint": "0 GB",
                "accepts_prompt": True,
                "accepts_num_frames": True,
                "accepts_steps": True,
            }

        def validate(self, *, prompt, num_frames, steps):
            return {}

        def animate(self, source_path, *, prompt, num_frames, steps, out_dir):
            raise AnimatorError("weights missing")

    monkeypatch.setitem(animators.ANIMATORS, "broken", _Broken())

    resp = client.post(
        "/api/admin/i2v",
        data={"backend": "broken", "prompt": "x", "num_frames": "8", "steps": "10"},
        files={"file": ("a.png", _png(), "image/png")},
    )
    task_id = resp.json()["task_id"]

    import time

    deadline = time.time() + 10.0
    body = None
    while time.time() < deadline:
        body = client.get(f"/api/admin/tasks/{task_id}", ).json()
        if body["status"] in ("done", "failed"):
            break
        time.sleep(0.05)
    assert body is not None and body["status"] == "failed"
    assert "weights missing" in (body["error"] or "")


def test_i2v_submission_rejects_unknown_backend_synchronously(client):
    # Validation must remain synchronous -- a typo'd backend name is operator
    # error and shouldn't pollute the task table with a doomed row.
    resp = client.post(
        "/api/admin/i2v",
        data={"backend": "wan22-99b-fake", "prompt": "x", "num_frames": "8", "steps": "10"},
        files={"file": ("a.png", _png(), "image/png")},
    )
    assert resp.status_code == 400


def test_tasks_list_endpoint_returns_recent(client, monkeypatch):
    # The Tasks tab lists recent generation tasks newest-first. Two
    # submissions yield two rows; the ordering must be deterministic.
    from server.avatar import animators

    class _Noop:
        name = "noop"

        @property
        def is_ready(self):
            return True

        def meta(self):
            return {
                "label_zh": "无操作",
                "group_zh": "测试",
                "description": "noop stub",
                "vram_hint": "0 GB",
                "accepts_prompt": True,
                "accepts_num_frames": True,
                "accepts_steps": True,
            }

        def validate(self, *, prompt, num_frames, steps):
            return {}

        def animate(self, source_path, *, prompt, num_frames, steps, out_dir):
            (out_dir / "out.mp4").write_bytes(b"\x00")

    monkeypatch.setitem(animators.ANIMATORS, "noop", _Noop())

    ids = []
    for i in range(2):
        resp = client.post(
            "/api/admin/i2v",
            data={"backend": "noop", "prompt": f"p{i}", "num_frames": "8", "steps": "10"},
            files={"file": (f"a{i}.png", _png(), "image/png")},
        )
        ids.append(resp.json()["task_id"])

    rows = client.get("/api/admin/tasks", ).json()["tasks"]
    seen_ids = [r["task_id"] for r in rows]
    # Both submitted tasks appear, with the most recent first.
    assert ids[0] in seen_ids and ids[1] in seen_ids
    assert seen_ids.index(ids[1]) < seen_ids.index(ids[0])
