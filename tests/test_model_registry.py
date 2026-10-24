"""The digital-human model registry.

One model = one avatar pipeline + the drive engine that renders its pack.
The two are bound together, so the server can never build a pack with one
model and drive it with another.
"""
import pytest

from server.avatar.pipeline import PipelineError, create_pipeline
from server.engine.base import EngineUnavailable, create_engine
from server.model_registry import (
    ACTIVE_KEY,
    DEFAULT_MODEL,
    MODELS,
    ModelUnavailable,
    activate,
    active,
    catalog,
    get_model,
    read_active_model,
    write_active_model,
)


@pytest.fixture()
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("DH_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DH_DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    from sqlalchemy.orm import sessionmaker

    from server.config import get_settings
    from server.db.session import get_engine, init_db, reset_engine

    get_settings.cache_clear()
    reset_engine()
    init_db(get_engine())
    session = sessionmaker(bind=get_engine())()
    try:
        yield session
    finally:
        session.close()
        reset_engine()
        get_settings.cache_clear()


@pytest.fixture()
def pristine_active(monkeypatch):
    """Restore the process-wide active model after the test.

    ``activate`` writes a module global on purpose -- the engine and the
    pipeline must resolve to the same model for the whole process -- so the
    tests that change it have to put it back.
    """
    import server.model_registry as registry

    monkeypatch.setattr(registry, "_active", None)
    return registry


def test_registry_ships_musetalk():
    assert DEFAULT_MODEL in MODELS
    entry = MODELS[DEFAULT_MODEL]
    assert entry.name == DEFAULT_MODEL
    assert entry.label_zh
    assert entry.engine_factory is not None
    assert entry.pipeline_factory is not None
    assert {"name": DEFAULT_MODEL, "label_zh": entry.label_zh} in catalog()


def test_an_unknown_model_is_refused():
    with pytest.raises(ModelUnavailable):
        get_model("nope")
    # The two factories keep their own exception types so the callers that
    # already handle them do not have to learn a new one.
    with pytest.raises(EngineUnavailable):
        create_engine("nope")
    with pytest.raises(PipelineError):
        create_pipeline("nope")


def test_active_falls_back_to_the_default_before_activation(pristine_active):
    assert active().name == DEFAULT_MODEL


def test_a_registered_model_binds_its_own_engine_and_pipeline(pristine_active, monkeypatch):
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

    activate("probe")

    assert active().name == "probe"
    # Both halves come from the same model -- that is the whole point.
    assert create_engine().name == "probe"
    assert isinstance(create_pipeline(), _Pipeline)
    assert built == ["engine", "pipeline"]


def test_activating_an_unknown_model_is_refused(pristine_active):
    with pytest.raises(ModelUnavailable):
        activate("nope")


def test_read_active_model_defaults_without_a_row(db):
    assert read_active_model(db) == DEFAULT_MODEL


def test_write_then_read_round_trips(db):
    from server.db.models import Setting

    write_active_model(db, DEFAULT_MODEL)
    assert read_active_model(db) == DEFAULT_MODEL
    assert db.get(Setting, ACTIVE_KEY) is not None


def test_write_active_model_refuses_an_unknown_name(db):
    with pytest.raises(ModelUnavailable):
        write_active_model(db, "nope")
    # Nothing was stored, so the default still stands.
    assert read_active_model(db) == DEFAULT_MODEL


def test_read_active_model_falls_back_on_a_malformed_row(db):
    from server.db.models import Setting

    db.add(Setting(key=ACTIVE_KEY, value="{not json"))
    db.commit()

    assert read_active_model(db) == DEFAULT_MODEL


def test_read_active_model_falls_back_on_an_unregistered_name(db):
    from server.db.models import Setting

    db.add(Setting(key=ACTIVE_KEY, value='"a-model-that-was-removed"'))
    db.commit()

    assert read_active_model(db) == DEFAULT_MODEL
