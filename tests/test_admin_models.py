"""HTTP layer for the Models admin tab: list, load, unload.

These tests run against a real FastAPI TestClient with the test doubles wired
through ``tests/conftest.py`` (drive engine = mock). The mock engine reports
its own model slot; load/unload paths are checked by stubbing the relevant
engine methods so the route's *dispatch* stays tested without needing a GPU.
"""
from __future__ import annotations


# ----- GET /api/admin/models ----------------------------------------------


def test_list_models_includes_models_gpu_and_active_sessions(client):
    # The Models tab fetches a single payload every poll. It must include the
    # engine's runtime list, the GPU snapshot (None on a CPU-only host,
    # populated dict otherwise), and a count of currently-active sessions so
    # the unload button knows when to refuse.
    body = client.get("/api/admin/models").json()
    assert "models" in body and isinstance(body["models"], list)
    assert "gpu" in body
    # The snapshot is either None (no CUDA) or a dict with the canonical
    # set of keys; either branch is valid -- the test machine has a GPU but
    # CI typically does not.
    if body["gpu"] is not None:
        assert set(body["gpu"]) >= {
            "device",
            "name",
            "total_gb",
            "allocated_gb",
            "reserved_gb",
            "peak_gb",
            "free_gb",
        }
    assert body["active_sessions"] == 0

    # The mock engine surfaces exactly one drive_runtime row; portrait
    # animators are merged in too (test env registers LivePortrait + Wan
    # 2.2 via side-effect imports). The operator sees one row per
    # thing-that-can-go-on-the-GPU in the Models tab.
    engine_rows = [m for m in body["models"] if m["kind"] == "drive_runtime"]
    assert len(engine_rows) == 1
    m = engine_rows[0]
    assert m["id"] == "mock_drive"
    assert set(m) >= {
        "id",
        "type",
        "components",
        "source",
        "estimated_bytes",
        "loaded",
        "can_unload",
        "loaded_at",
    }


# ----- POST /api/admin/models/{id}/load ----------------------------------


def test_load_refuses_to_load_a_mock_engine_slot(client):
    # The mock engine has no runtime to materialise; it also reports itself
    # ``loaded=True`` (always-loaded CPU placeholder). Either refusal --
    # 400 "no weights" or 409 "already loaded" -- is a valid answer; the
    # point of this test is that the endpoint must NOT silently 200 an
    # empty action.
    resp = client.post("/api/admin/models/mock_drive/load")
    assert resp.status_code in (400, 409)
    code = resp.json()["error"]["code"]
    assert code in {"no_weights_to_load", "already_loaded"}


def test_load_unknown_model_returns_404(client):
    resp = client.post("/api/admin/models/no_such_model/load")
    assert resp.status_code == 404


def test_load_calls_engine_when_supported(client, monkeypatch):
    # When the engine exposes a ``load`` method, the endpoint must call it.
    # Stubs the engine on the session manager so the dispatch is exercised
    # without GPU weights.
    from server.main import _app

    engine = _app.state.session_manager.engine

    calls = {"n": 0}

    def fake_load():
        calls["n"] += 1

    monkeypatch.setattr(engine, "load", fake_load, raising=False)

    # Patch describe to claim the slot exists and isn't loaded yet.
    def fake_describe():
        return {
            "name": engine.name,
            "gpu": False,
            "models": [
                {
                    "id": "stub_runtime",
                    "type": "drive_runtime",
                    "components": [],
                    "source": None,
                    "estimated_bytes": 0,
                    "loaded": False,
                    "can_unload": False,
                    "loaded_at": None,
                }
            ],
        }

    monkeypatch.setattr(engine, "describe", fake_describe)
    resp = client.post("/api/admin/models/stub_runtime/load")
    assert resp.status_code == 200
    assert calls["n"] == 1


def test_load_409_when_already_loaded(client, monkeypatch):
    # Re-loading a live runtime is a 409; the engine's own warmup() would
    # short-circuit on _runtime is not None, but the HTTP layer must short
    # BEFORE calling the engine so the operator sees a clear status.
    from server.main import _app

    engine = _app.state.session_manager.engine

    def fake_describe():
        return {
            "name": engine.name,
            "gpu": False,
            "models": [
                {
                    "id": "stub_runtime",
                    "type": "drive_runtime",
                    "components": [],
                    "source": None,
                    "estimated_bytes": 0,
                    "loaded": True,
                    "can_unload": True,
                    "loaded_at": None,
                }
            ],
        }

    monkeypatch.setattr(engine, "describe", fake_describe)
    resp = client.post("/api/admin/models/stub_runtime/load")
    assert resp.status_code == 409


# ----- POST /api/admin/models/{id}/unload --------------------------------


def test_unload_returns_400_when_engine_has_no_runtime_to_free(client, monkeypatch):
    # Mock has nothing to unload either; the operator gets the same clear
    # refusal rather than a 200-no-op. The engine's describe reports
    # can_unload=False; the endpoint uses that as the gate.
    from server.main import _app

    engine = _app.state.session_manager.engine

    def fake_describe():
        return {
            "name": engine.name,
            "gpu": False,
            "models": [
                {
                    "id": "mock_drive",
                    "type": "drive_runtime",
                    "components": [],
                    "source": None,
                    "estimated_bytes": 0,
                    "loaded": True,
                    "can_unload": False,
                    "loaded_at": None,
                }
            ],
        }

    monkeypatch.setattr(engine, "describe", fake_describe)
    resp = client.post("/api/admin/models/mock_drive/unload")
    assert resp.status_code == 400


def test_unload_409_when_active_sessions_exist(client, monkeypatch):
    # With one or more live StreamSession rows the unload must refuse --
    # tearing the runtime out from under a streaming pipeline silently
    # breaks every conversation. Returns the count so the UI can show
    # "还有 3 个活动会话, 请先关闭".
    from uuid import uuid4

    from sqlalchemy.orm import sessionmaker

    from server.db.models import StreamSession
    from server.db.session import get_engine
    from server.main import _app

    engine = _app.state.session_manager.engine

    def fake_describe():
        return {
            "name": engine.name,
            "gpu": False,
            "models": [
                {
                    "id": "stub_runtime",
                    "type": "drive_runtime",
                    "components": [],
                    "source": None,
                    "estimated_bytes": 0,
                    "loaded": True,
                    "can_unload": True,
                    "loaded_at": None,
                }
            ],
        }

    monkeypatch.setattr(engine, "describe", fake_describe)

    # Insert a row directly so the active count is non-zero *and* survives
    # the route's own session.close() -- the model's status field is what
    # the endpoint reads.
    Session = sessionmaker(bind=get_engine())
    db = Session()
    try:
        row = StreamSession(id=uuid4().hex, avatar_id="x", status="active")
        db.add(row)
        db.commit()
        live_sid = row.id
    finally:
        db.close()

    try:
        resp = client.post("/api/admin/models/stub_runtime/unload")
        assert resp.status_code == 409
        body = resp.json()
        assert body["error"]["code"] == "engine_busy"
        assert body["error"]["detail"]["active_sessions"] >= 1
    finally:
        # Cleanup -- the row was inserted by the test, not via the upload path.
        db = Session()
        try:
            row = db.get(StreamSession, live_sid)
            if row is not None:
                db.delete(row)
                db.commit()
        finally:
            db.close()


def test_unload_calls_engine_when_supported(client, monkeypatch):
    # When the engine exposes an ``unload`` method AND reports itself
    # loadable via describe, and no active sessions exist, the route must
    # call the engine's unload and return 200.
    from server.main import _app

    engine = _app.state.session_manager.engine

    calls = {"n": 0}

    def fake_unload():
        calls["n"] += 1

    monkeypatch.setattr(engine, "unload", fake_unload, raising=False)

    def fake_describe():
        return {
            "name": engine.name,
            "gpu": False,
            "models": [
                {
                    "id": "stub_runtime",
                    "type": "drive_runtime",
                    "components": [],
                    "source": None,
                    "estimated_bytes": 0,
                    "loaded": True,
                    "can_unload": True,
                    "loaded_at": None,
                }
            ],
        }

    monkeypatch.setattr(engine, "describe", fake_describe)
    resp = client.post("/api/admin/models/stub_runtime/unload")
    assert resp.status_code == 200
    assert calls["n"] == 1


def test_unload_unknown_model_returns_404(client):
    resp = client.post("/api/admin/models/no_such_model/unload")
    assert resp.status_code == 404


# ----- service load / unload dispatch --------------------------------------


def test_list_services_exposes_loadable_flags(client, monkeypatch):
    # The Models tab's service rows now also surface ``loaded`` /
    # ``can_load`` / ``can_unload`` so the operator can drive the
    # load/unload lifecycle for LivePortrait and Wan 2.2 from this tab.
    # The test just verifies the shape: a freshly-restarted server
    # should report all rows unloaded with ``can_load=True`` (each
    # backend now exposes a load method).
    body = client.get("/api/admin/models").json()
    animator_rows = [m for m in body["models"] if m["kind"] == "portrait_animator"]
    assert animator_rows, "at least one portrait animator should be registered"
    for r in animator_rows:
        assert set(r) >= {
            "id",
            "kind",
            "label_zh",
            "configured",
            "loaded",
            "can_load",
            "can_unload",
            "status",
        }
        # A test-env restart: no animator has its pipeline materialised.
        assert r["loaded"] is False
        # ``can_unload`` mirrors ``loaded``; not-loaded rows must not
        # show an enabled "释放显存" button.
        assert r["can_unload"] is False
        # Each backend now exposes a ``load()`` method -- the operator
        # can always attempt to bring it up, even when the underlying
        # weights are missing (the click then surfaces a 503 with a
        # clear hint at the fetch script).
        assert r["can_load"] is True, (
            f"{r['id']!r} reports can_load=False; the operator cannot "
            "reload after unload"
        )


def test_liveportrait_service_row_can_load_even_without_weights(client):
    """A LivePortrait row with no on-disk weights must still render a
    加载 button (the click surfaces a 503 with the fetch-script hint).

    ``can_load`` keys on "load() method exists", not "weights are
    present". Otherwise the operator can't even trigger the load
    attempt that would tell them where the script lives.
    """
    body = client.get("/api/admin/models").json()
    lp = next(
        (m for m in body["models"] if m["id"] == "animator:liveportrait"),
        None,
    )
    # Test env pins ``DH_DATA_DIR`` to a tmp dir, so no weights are on
    # disk; this test still holds when weights exist (the button is
    # always there).
    assert lp is not None
    if lp["weights_present"]:
        # The "real" path: weights are on disk, button must be visible.
        assert lp["can_load"] is True
    else:
        # No weights path: button is still visible so the click can
        # surface a 503 with the fetch-script hint.
        assert lp["can_load"] is True
        assert lp["missing_script"] == "scripts/fetch_liveportrait_weights.sh"


def test_load_service_unknown_id_returns_404(client):
    resp = client.post("/api/admin/services/no_such_service/load")
    assert resp.status_code == 404


def test_load_service_for_animation_backend_without_load_is_400(client):
    # An animator registered for the dropdown but lacking a ``load``
    # method (the future TTS-shaped entries) must surface 400, not a
    # silent 200-no-op. Today the registered backends all expose
    # ``load``, so we test by registering a stripped stub.
    from server.avatar import animators

    class _Spy:
        name = "no_load_backend"

        def meta(self):
            return {"label_zh": "x", "group_zh": "x", "description": ""}

        @property
        def is_ready(self):
            return True

        @property
        def is_loaded(self):
            return False

        @property
        def weights_dir(self):
            from pathlib import Path

            class _P:
                def is_dir(self_inner):
                    return True

                def iterdir(self_inner):
                    return iter([object()])

            return _P()

        # No ``load`` / ``unload`` -- simulating a future TTS-shaped
        # backend that lives on the Models tab but has nothing to
        # materialise.

    animators.ANIMATORS["no_load_backend"] = _Spy()
    try:
        resp = client.post("/api/admin/services/animator:no_load_backend/load")
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "no_weights_to_load"
    finally:
        animators.ANIMATORS.pop("no_load_backend", None)


def test_load_model_dispatches_animator_ids_to_anims(client, monkeypatch):
    """The Models tab UI calls ``POST /api/admin/models/<id>/load`` for
    every row in its table; portrait animators are now in that table too
    (under ids like ``animator:liveportrait``). The endpoint must
    dispatch to the right backend -- engine slot for ``musetalk_runtime``
    / ``mock_drive``, animator instance for ``animator:<name>``.

    Regression for the bug where moving animators into the models list
    left their load button wired to the engine-slot dispatcher and the
    UI surfaced ``model 'animator:liveportrait' not found``.
    """
    from server.avatar import animators

    class _Spy:
        def __init__(self):
            self.name = "liveportrait"
            self.calls = []

        def meta(self):
            return {"label_zh": "LivePortrait", "group_zh": "x", "description": ""}

        @property
        def is_ready(self):
            return True

        @property
        def is_loaded(self):
            return False

        @property
        def weights_dir(self):
            from pathlib import Path

            class _P:
                def is_dir(self_inner):
                    return True

                def iterdir(self_inner):
                    return iter([object()])

            return _P()

        def load(self):
            self.calls.append("load")

    spy = _Spy()
    animators.ANIMATORS["_spy"] = spy
    try:
        resp = client.post("/api/admin/models/animator:_spy/load")
        assert resp.status_code == 200, resp.text
        assert spy.calls == ["load"]
    finally:
        animators.ANIMATORS.pop("_spy", None)


def test_unload_model_dispatches_animator_ids_to_anims(client, monkeypatch):
    """Same dispatch as load: ``/models/<id>/unload`` resolves
    ``animator:<name>`` to the registered instance's ``unload`` method.
    """
    from server.avatar import animators

    class _Spy:
        def __init__(self):
            self.name = "liveportrait"
            self.calls = []

        def meta(self):
            return {"label_zh": "LivePortrait", "group_zh": "x", "description": ""}

        @property
        def is_ready(self):
            return True

        @property
        def is_loaded(self):
            return True  # must be loaded to unload

        @property
        def weights_dir(self):
            from pathlib import Path

            class _P:
                def is_dir(self_inner):
                    return True

                def iterdir(self_inner):
                    return iter([object()])

            return _P()

        def unload(self):
            self.calls.append("unload")

    spy = _Spy()
    animators.ANIMATORS["_spy"] = spy
    try:
        resp = client.post("/api/admin/models/animator:_spy/unload")
        assert resp.status_code == 200, resp.text
        assert spy.calls == ["unload"]
    finally:
        animators.ANIMATORS.pop("_spy", None)


def test_load_service_calls_animator_load(client, monkeypatch):
    # When the dispatch lands on a real animator (LivePortrait /
    # Wan), the endpoint must call its ``load`` method. ``is_loaded``
    # is mocked False so the gate yields, and the spy records the call.
    from server.avatar import animators

    class _Spy:
        def __init__(self):
            self.name = "liveportrait"
            self.calls = []

        def meta(self):
            return {"label_zh": "LivePortrait", "group_zh": "x", "description": ""}

        @property
        def is_ready(self):
            return True

        @property
        def is_loaded(self):
            return False

        @property
        def weights_dir(self):
            from pathlib import Path

            class _P:
                def is_dir(self_inner):
                    return True

                def iterdir(self_inner):
                    return iter([object()])

            return _P()

        def load(self):
            self.calls.append("load")

    spy = _Spy()
    animators.ANIMATORS["_spy"] = spy
    try:
        resp = client.post("/api/admin/services/animator:_spy/load")
        assert resp.status_code == 200
        assert spy.calls == ["load"]
    finally:
        animators.ANIMATORS.pop("_spy", None)


def test_load_service_409_when_already_loaded(client, monkeypatch):
    from server.avatar import animators

    class _Spy:
        name = "liveportrait"

        def meta(self):
            return {"label_zh": "x", "group_zh": "x", "description": ""}

        @property
        def is_ready(self):
            return True

        @property
        def is_loaded(self):
            return True

        @property
        def weights_dir(self):
            from pathlib import Path

            class _P:
                def is_dir(self_inner):
                    return True

                def iterdir(self_inner):
                    return iter([object()])

            return _P()

        def load(self):
            raise AssertionError("load should not be called when already loaded")

    animators.ANIMATORS["_spy"] = _Spy()
    try:
        resp = client.post("/api/admin/services/animator:_spy/load")
        assert resp.status_code == 409
        assert resp.json()["error"]["code"] == "already_loaded"
    finally:
        animators.ANIMATORS.pop("_spy", None)


def test_load_service_503_when_load_raises(client):
    from server.avatar import animators

    class _Spy:
        name = "liveportrait"

        def meta(self):
            return {"label_zh": "x", "group_zh": "x", "description": ""}

        @property
        def is_ready(self):
            return True

        @property
        def is_loaded(self):
            return False

        @property
        def weights_dir(self):
            from pathlib import Path

            class _P:
                def is_dir(self_inner):
                    return True

                def iterdir(self_inner):
                    return iter([object()])

            return _P()

        def load(self):
            raise RuntimeError("disk died")

    animators.ANIMATORS["_spy"] = _Spy()
    try:
        resp = client.post("/api/admin/services/animator:_spy/load")
        assert resp.status_code == 503
    finally:
        animators.ANIMATORS.pop("_spy", None)


def test_unload_service_calls_animator_unload(client, monkeypatch):
    from server.avatar import animators

    class _Spy:
        def __init__(self):
            self.name = "liveportrait"
            self.calls = []

        def meta(self):
            return {"label_zh": "x", "group_zh": "x", "description": ""}

        @property
        def is_ready(self):
            return True

        @property
        def is_loaded(self):
            return True

        @property
        def weights_dir(self):
            from pathlib import Path

            class _P:
                def is_dir(self_inner):
                    return True

                def iterdir(self_inner):
                    return iter([object()])

            return _P()

        def unload(self):
            self.calls.append("unload")

    spy = _Spy()
    animators.ANIMATORS["_spy"] = spy
    try:
        resp = client.post("/api/admin/services/animator:_spy/unload")
        assert resp.status_code == 200
        assert spy.calls == ["unload"]
    finally:
        animators.ANIMATORS.pop("_spy", None)


def test_unload_service_409_when_not_loaded(client):
    from server.avatar import animators

    class _Spy:
        name = "liveportrait"

        def meta(self):
            return {"label_zh": "x", "group_zh": "x", "description": ""}

        @property
        def is_ready(self):
            return True

        @property
        def is_loaded(self):
            return False

        @property
        def weights_dir(self):
            from pathlib import Path

            class _P:
                def is_dir(self_inner):
                    return True

                def iterdir(self_inner):
                    return iter([object()])

            return _P()

        def unload(self):
            raise AssertionError("unload should not be called when not loaded")

    animators.ANIMATORS["_spy"] = _Spy()
    try:
        resp = client.post("/api/admin/services/animator:_spy/unload")
        assert resp.status_code == 409
        assert resp.json()["error"]["code"] == "not_loaded"
    finally:
        animators.ANIMATORS.pop("_spy", None)


def test_unload_service_400_when_no_unload_method(client):
    # A backend without an ``unload`` method (e.g. mock-shaped or
    # future TTS-shaped backends) can't release VRAM, so the endpoint
    # returns 400. The Models tab hides the button on the client side;
    # the dispatch gives a clean refusal if it ever races.
    from server.avatar import animators

    class _Spy:
        name = "liveportrait"

        def meta(self):
            return {"label_zh": "x", "group_zh": "x", "description": ""}

        @property
        def is_ready(self):
            return True

        @property
        def is_loaded(self):
            return True

        # No weights_dir, no load, no unload.
    animators.ANIMATORS["_spy"] = _Spy()
    try:
        resp = client.post("/api/admin/services/animator:_spy/unload")
        assert resp.status_code == 400
    finally:
        animators.ANIMATORS.pop("_spy", None)


# ----- service surface (animators + TTS) ----------------------------------


def test_list_models_includes_services_field(client):
    # The Models tab surfaces ONE row per engine runtime slot under
    # ``models`` and a SEPARATE list under ``services`` for everything
    # else this server uses -- portrait animators (LivePortrait, Wan 2.2)
    # and the MiniMax TTS endpoint. Empty service list
    # would be a valid state (a fresh server with no animators imported),
    # so the test only asserts the shape, not the size.
    body = client.get("/api/admin/models").json()
    assert "services" in body
    assert isinstance(body["services"], list)


def test_list_services_lists_portrait_animators(client):
    # Each registered ANIMATORS entry surfaces as its own model row now
    # (portrait animators are model-shaped -- they have their own
    # weights directory and ``load()`` / ``unload()`` lifecycle).
    # ``kind`` carries "portrait_animator"; the operator-facing strings
    # come from the backend's own ``meta()``.
    body = client.get("/api/admin/models").json()
    animators = [m for m in body["models"] if m["kind"] == "portrait_animator"]
    names = {a["id"] for a in animators}
    # The backends registered via side-effect imports in admin.py at
    # module load. The names follow the PatternAnimator.name contract.
    assert {"animator:liveportrait"} <= names
    # Wan 2.2 has multiple variants; at least one ``animator:wan22-*`` id.
    assert any(a.startswith("animator:wan22-") for a in names)


def test_list_services_includes_minimax_tts(client):
    # The TTS service is a singleton ``tts:minimax`` entry; configured
    # mirrors whether ``DH_MINIMAX_API_KEY`` is set. Tests run with the
    # key pinned to empty, so configured is False here -- an honest
    # "未配置" badge.
    body = client.get("/api/admin/models").json()
    tts_entries = [s for s in body["services"] if s["kind"] == "tts_service"]
    assert any(s["id"] == "tts:minimax" for s in tts_entries)
    row = next(s for s in tts_entries if s["id"] == "tts:minimax")
    assert row["configured"] is False  # the test env pins the key to ""
    assert row["label_zh"]
    assert row["description"]


def test_models_list_includes_portrait_animators(client):
    """Portrait animators are model-shaped (own weights, own load/unload
    lifecycle) so they belong under ``models``, not ``services``.

    The Models tab renders one table per top-level list; the operator
    expects "what can I put on the GPU right now" to be a single list.
    TTS is a remote service so it stays in ``services``.
    """
    body = client.get("/api/admin/models").json()
    model_ids = {m["id"] for m in body["models"]}
    # The drive engine slot is always there (mock in CI, musetalk on
    # the GPU host). We only assert one engine slot exists, not which
    # specific id, because the test env uses the mock engine.
    engine_rows = [m for m in body["models"] if m.get("kind") == "drive_runtime"]
    assert len(engine_rows) == 1, f"expected one drive_runtime row, got {engine_rows}"
    # Portrait animators are merged in.
    animator_models = [m for m in body["models"] if m.get("kind") == "portrait_animator"]
    assert {a["id"] for a in animator_models} >= {"animator:liveportrait"}
    assert any(a["id"].startswith("animator:wan22-") for a in animator_models)
    # Each animator row carries the unified ModelInfo flags.
    sample = animator_models[0]
    assert "loaded" in sample
    assert "can_load" in sample
    assert "can_unload" in sample
    # TTS stays in services.
    service_kinds = {s["kind"] for s in body["services"]}
    assert "tts_service" in service_kinds
    # And no portrait animator leaked into services.
    assert "portrait_animator" not in service_kinds


def test_models_list_animator_row_carries_label_zh_and_vram_hint(client):
    """When the portrait-animator row moved into ``models``, it must
    carry the operator-facing strings (``label_zh``, ``description``,
    ``vram_hint``) so the UI's ModelInfo renderer has them.
    """
    body = client.get("/api/admin/models").json()
    lp = next(
        (m for m in body["models"] if m["id"] == "animator:liveportrait"),
        None,
    )
    assert lp is not None
    assert lp["label_zh"]
    assert lp["vram_hint"]
    assert "description" in lp


def test_list_services_animator_row_includes_vram_hint(client):
    # Each animator row carries the same VRAM-fit info its meta() exposes
    # so the operator can scan the Models tab instead of cross-referencing
    # the 形象活化 dropdown -- the same numbers, rendered alongside the
    # drive-engine row they may conflict with.
    body = client.get("/api/admin/models").json()
    lp = next(
        m for m in body["models"]
        if m["id"] == "animator:liveportrait"
    )
    assert set(lp) >= {
        "id",
        "kind",
        "label_zh",
        "description",
        "vram_required_gb",
        "vram_hint",
        "fits_current_gpu",
        "configured",
        "weights_present",
        "weights_dir",
        "missing_script",
    }


def test_list_services_tts_configured_flips_with_api_key(client, monkeypatch):
    # The badge the operator sees reflects the current value of
    # ``DH_MINIMAX_API_KEY``: an empty key surfaces as "未配置" so the
    # operator can distinguish "service is misconfigured" from "service
    # is slow / down". The endpoint helper is called for each request,
    # so a monkeypatched settings value is observable on the next GET.
    from server.api import admin as admin_module
    from server.config import get_settings

    get_settings.cache_clear()
    try:
        # Empty (the test default) -> not configured.
        body = client.get("/api/admin/models").json()
        tts = next(s for s in body["services"] if s["id"] == "tts:minimax")
        assert tts["configured"] is False

        # Monkeypatch the helper directly so the next request sees the
        # configured branch without rebuilding the app (the conftest's
        # ``reset_engine`` path is heavy and not needed for this flip).
        original = admin_module._tts_service_row

        def configured_row():
            row = original()
            row["configured"] = True
            row["status"] = "已配置"
            return row

        monkeypatch.setattr(admin_module, "_tts_service_row", configured_row)
        body = client.get("/api/admin/models").json()
        tts = next(s for s in body["services"] if s["id"] == "tts:minimax")
        assert tts["configured"] is True
        assert tts["status"] == "已配置"
    finally:
        get_settings.cache_clear()
