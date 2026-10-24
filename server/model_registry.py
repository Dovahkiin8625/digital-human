"""Registry of digital-human models.

A *digital human model* is one complete talking-avatar implementation: an
avatar preparation pipeline (upload -> pack) plus the drive engine that
renders that pack. The two are bound together on purpose -- a pack written
by one pipeline is only readable by the matching engine, which is exactly
why the two ``DH_*`` env vars this module replaces had to be kept in step
with each other.

Selection used to be ``DH_DRIVE_ENGINE`` / ``DH_AVATAR_PIPELINE``. Both are
gone. The operator's choice now lives in the SQLite ``settings`` table --
the same place the MiniMax TTS and LLM configs live -- and is edited from
the Models tab. See :func:`read_active_model` / :func:`write_active_model`.

MuseTalk is the only registered model today. Adding another is one class
pair plus one :func:`register_model` call::

    from server.model_registry import DigitalHumanModel, register_model

    register_model(
        DigitalHumanModel(
            name="mymodel",
            label_zh="我的模型",
            engine_factory=MyEngine,
            pipeline_factory=MyPipeline,
        )
    )

Both factories are called lazily, and the modules they import are
torch-free at import time, so this module stays importable in the
torch-free ``.venv``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from server.avatar.pipeline import AvatarPipeline
    from server.engine.base import DriveEngine


class ModelUnavailable(Exception):
    """Raised when a model name is not in the registry."""


@dataclass(frozen=True)
class DigitalHumanModel:
    """One digital-human model: a pipeline and the engine that drives it.

    ``engine_factory`` / ``pipeline_factory`` are called with no arguments
    and must return a fresh instance. They are invoked lazily so that
    importing this module never pulls in torch.
    """

    name: str
    label_zh: str
    engine_factory: Callable[[], "DriveEngine"]
    pipeline_factory: Callable[[], "AvatarPipeline"]


#: The registry. Ordered by insertion, so the Models tab lists the models
#: in the order they are registered here.
MODELS: dict[str, DigitalHumanModel] = {}

#: The model a box runs when the operator hasn't chosen one.
DEFAULT_MODEL = "musetalk"

#: The ``settings`` table row that holds the operator's choice. It is
#: deliberately NOT a tunable: ``GET /api/admin/settings`` is
#: unauthenticated and serves the tunables verbatim, so this key lives
#: here and gets its own endpoints.
ACTIVE_KEY = "dh_active_model"


def register_model(model: DigitalHumanModel) -> None:
    """Add a model to the registry. A later registration replaces an
    earlier one with the same name."""
    MODELS[model.name] = model


def catalog() -> list[dict]:
    """The registry as the API and the console see it."""
    return [{"name": m.name, "label_zh": m.label_zh} for m in MODELS.values()]


def get_model(name: str | None = None) -> DigitalHumanModel:
    """Look up a model by name.

    ``None`` means "the model this process is running" -- see
    :func:`active`. Raises :class:`ModelUnavailable` for an unknown name.
    """
    if name is None:
        return active()
    model = MODELS.get(name)
    if model is None:
        raise ModelUnavailable(f"unknown digital human model: {name}")
    return model


# ----- the process-wide active model ---------------------------------------

#: The model this process runs. Frozen by :func:`activate` at startup so
#: the engine and the pipeline are always the same model: a pack can never
#: be written by one model and driven by another.
_active: DigitalHumanModel | None = None


def activate(name: str) -> DigitalHumanModel:
    """Freeze the model this process runs.

    Called once from ``create_app()`` after the operator's choice has been
    read out of the settings table. A restart is what applies a change --
    the engine is a process-wide singleton holding the model's weights in
    VRAM, so swapping it under live traffic is not something this server
    does.
    """
    global _active
    model = MODELS.get(name)
    if model is None:
        raise ModelUnavailable(f"unknown digital human model: {name}")
    _active = model
    return model


def active() -> DigitalHumanModel:
    """The model this process runs.

    Falls back to :data:`DEFAULT_MODEL` before :func:`activate` is called
    (scripts, unit tests that build no app).
    """
    if _active is not None:
        return _active
    model = MODELS.get(DEFAULT_MODEL)
    if model is None:  # pragma: no cover -- the default is always registered
        raise ModelUnavailable("no digital human model is registered")
    return model


# ----- the operator's choice (SQLite settings table) -----------------------


def read_active_model(db) -> str:
    """The model name the operator chose, or :data:`DEFAULT_MODEL`.

    A missing row, a malformed value, or a name that is no longer
    registered all fall back to the default rather than failing the
    caller -- a stale row must not stop the server from booting.
    """
    from server.db.models import Setting

    row = db.get(Setting, ACTIVE_KEY)
    if row is None:
        return DEFAULT_MODEL
    try:
        name = json.loads(row.value)
    except (ValueError, TypeError):
        return DEFAULT_MODEL
    if not isinstance(name, str) or name not in MODELS:
        return DEFAULT_MODEL
    return name


def write_active_model(db, name: str) -> str:
    """Persist the operator's choice and return it.

    Raises :class:`ModelUnavailable` for a name that is not registered, so
    the API layer can turn it into a 400 instead of storing a value that
    would only fail at the next boot.
    """
    if name not in MODELS:
        raise ModelUnavailable(f"unknown digital human model: {name}")

    from datetime import datetime, timezone

    from server.db.models import Setting

    row = db.get(Setting, ACTIVE_KEY)
    if row is None:
        db.add(Setting(key=ACTIVE_KEY, value=json.dumps(name)))
    else:
        row.value = json.dumps(name)
        row.updated_at = datetime.now(timezone.utc)
    db.commit()
    return name


# ----- the built-in model ---------------------------------------------------


def _musetalk_engine() -> "DriveEngine":
    from server.engine.musetalk_engine import MuseTalkEngine

    return MuseTalkEngine()


def _musetalk_pipeline() -> "AvatarPipeline":
    from server.avatar.musetalk_pipeline import MuseTalkAvatarPipeline

    return MuseTalkAvatarPipeline()


register_model(
    DigitalHumanModel(
        name="musetalk",
        label_zh="MuseTalk 1.5",
        engine_factory=_musetalk_engine,
        pipeline_factory=_musetalk_pipeline,
    )
)
