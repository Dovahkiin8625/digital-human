"""The digital-human model registry.

One model = one avatar pipeline + the drive engine that renders its pack.
The two are bound together, so the server can never build a pack with one
model and drive it with another.

Several models may be registered at once: an operator runs more than one
runtime on the same box, and the Models tab loads and unloads each of
them independently. There is no "the active model" any more -- an avatar
records the model that owns it.
"""
import pytest

from server.avatar.pipeline import PipelineError, create_pipeline
from server.engine.base import EngineUnavailable, create_engine
from server.model_registry import (
    DEFAULT_MODEL,
    MODELS,
    SOURCE_ARCHIVE,
    SOURCE_IMAGE_VIDEO,
    ModelUnavailable,
    catalog,
    default_model,
    get_model,
    list_models,
)


def test_registry_ships_musetalk_and_live2d():
    assert DEFAULT_MODEL in MODELS
    entry = MODELS[DEFAULT_MODEL]
    assert entry.name == DEFAULT_MODEL
    assert entry.label_zh
    assert entry.engine_factory is not None
    assert entry.pipeline_factory is not None
    assert entry.source_kind == SOURCE_IMAGE_VIDEO
    assert entry.server_rendered and not entry.client_rendered


def test_live2d_is_a_model_without_a_server_engine():
    """The client renders a Cubism project, so there is no engine to hold
    weights and no session to drive."""
    entry = get_model("live2d")
    assert entry.engine_factory is None
    assert entry.pipeline_factory is not None
    assert entry.source_kind == SOURCE_ARCHIVE
    assert entry.client_rendered and not entry.server_rendered


def test_catalog_describes_every_model():
    names = {m["name"] for m in catalog()}
    assert names == set(MODELS)
    by_name = {m["name"]: m for m in catalog()}
    assert by_name["musetalk"]["client_rendered"] is False
    assert by_name["musetalk"]["source_kind"] == SOURCE_IMAGE_VIDEO
    assert by_name["live2d"]["client_rendered"] is True
    assert by_name["live2d"]["source_kind"] == SOURCE_ARCHIVE
    assert all(m["label_zh"] for m in catalog())


def test_list_models_returns_the_registry_in_order():
    assert [m.name for m in list_models()] == list(MODELS)


def test_an_unknown_model_is_refused():
    with pytest.raises(ModelUnavailable):
        get_model("nope")
    # The two factories keep their own exception types so the callers that
    # already handle them do not have to learn a new one.
    with pytest.raises(EngineUnavailable):
        create_engine("nope")
    with pytest.raises(PipelineError):
        create_pipeline("nope")


def test_the_default_model_is_the_fallback():
    assert default_model().name == DEFAULT_MODEL
    # ``None`` means "the default" -- the public upload endpoint has no
    # model picker of its own.
    assert get_model(None).name == DEFAULT_MODEL


def test_a_registered_model_binds_its_own_engine_and_pipeline(monkeypatch):
    """The extension point: one class pair plus one registration."""
    import server.model_registry as registry

    built: list[str] = []

    class _Engine:
        name = "probe"

        def describe(self):
            return {"name": self.name, "gpu": False}

    class _Pipeline:
        pass

    monkeypatch.setitem(
        MODELS,
        "probe",
        registry.DigitalHumanModel(
            name="probe",
            label_zh="探针",
            engine_factory=lambda: built.append("engine") or _Engine(),
            pipeline_factory=lambda: built.append("pipeline") or _Pipeline(),
        ),
    )

    # Both halves come from the same model -- that is the whole point.
    assert create_engine("probe").name == "probe"
    assert isinstance(create_pipeline("probe"), _Pipeline)
    assert built == ["engine", "pipeline"]


def test_a_client_rendered_model_has_no_engine_to_build():
    """Asking for a client-rendered model's engine is an error, not an
    empty object: there is nothing to drive with."""
    from server.engine.pool import EnginePool

    with pytest.raises(EngineUnavailable):
        EnginePool().engine_for("live2d")


def test_the_pool_caches_one_engine_per_model():
    from server.engine.pool import EnginePool

    pool = EnginePool()
    first = pool.engine_for(DEFAULT_MODEL)
    second = pool.engine_for(DEFAULT_MODEL)
    assert first is second
    assert set(pool.engines()) == {DEFAULT_MODEL}


def test_the_registry_has_no_active_model():
    """The process-wide "active model" is gone: nothing to freeze, and no
    settings row to write."""
    import server.model_registry as registry

    assert not hasattr(registry, "activate")
    assert not hasattr(registry, "active")
    assert not hasattr(registry, "read_active_model")
    assert not hasattr(registry, "write_active_model")
    assert not hasattr(registry, "ACTIVE_KEY")
