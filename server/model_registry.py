"""Registry of digital-human models.

A *digital human model* is one complete talking-avatar implementation: an
avatar preparation pipeline (upload -> pack) plus the drive engine that
renders that pack. The two are bound together on purpose -- a pack written
by one pipeline is only readable by the matching engine, which is exactly
why the two ``DH_*`` env vars this module replaces had to be kept in step
with each other.

An avatar records the model that owns it (``Avatar.drive_model``), so the
pairing is a fact in the database and not merely a property of whatever
happened to be running when the pack was built. The operator therefore
never *switches* the box between models: several runtimes may be resident
at once, each with its own weights in VRAM, and the Models tab loads and
unloads them.

Two source kinds are registered today:

* ``image_video`` -- the operator uploads a photo or a clip; the pipeline
  builds a pack and the server drives the avatar over WebRTC.
* ``archive`` -- the operator uploads a zipped Cubism project; there is
  no server-side engine at all (the browser renders it), so the model
  registers a pipeline and no ``engine_factory``.

MuseTalk and Live2D are the two registered models. Adding another is one
class pair plus one :func:`register_model` call::

    from server.model_registry import DigitalHumanModel, register_model

    register_model(
        DigitalHumanModel(
            name="mymodel",
            label_zh="我的模型",
            pipeline_factory=MyPipeline,
            engine_factory=MyEngine,
        )
    )

Both factories are called lazily, and the modules they import are
torch-free at import time, so this module stays importable in the
torch-free ``.venv``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from server.avatar.pipeline import AvatarPipeline
    from server.engine.base import DriveEngine

#: The operator uploads a photo or a clip; the server drives the avatar.
SOURCE_IMAGE_VIDEO = "image_video"

#: The operator uploads an archive (a Cubism project); the client renders
#: the avatar, so the model has no server-side engine.
SOURCE_ARCHIVE = "archive"


class ModelUnavailable(Exception):
    """Raised when a model name is not in the registry."""


@dataclass(frozen=True)
class DigitalHumanModel:
    """One digital-human model: a pipeline and the engine that drives it.

    ``pipeline_factory`` / ``engine_factory`` are called with no arguments
    and must return a fresh instance. They are invoked lazily so that
    importing this module never pulls in torch.

    ``engine_factory`` is ``None`` for a model the *client* renders: the
    server hosts the files and the browser draws the frames, so there is
    no engine to hold weights and no session to drive.
    """

    name: str
    label_zh: str
    pipeline_factory: Callable[[], "AvatarPipeline"]
    engine_factory: Callable[[], "DriveEngine"] | None = None
    source_kind: str = SOURCE_IMAGE_VIDEO

    @property
    def client_rendered(self) -> bool:
        """True when the browser draws this model's avatars."""
        return self.engine_factory is None

    @property
    def server_rendered(self) -> bool:
        """True when the server drives this model's avatars over WebRTC."""
        return self.engine_factory is not None


#: The registry. Ordered by insertion, so the Models tab lists the models
#: in the order they are registered here.
MODELS: dict[str, DigitalHumanModel] = {}

#: The model used where the caller has no choice to make: the public
#: avatar-upload endpoint, and the model the server warms up at boot so
#: the first session does not pay a cold load. It is a *fallback*, not a
#: selection -- an operator runs several models at once.
DEFAULT_MODEL = "musetalk"


def register_model(model: DigitalHumanModel) -> None:
    """Add a model to the registry. A later registration replaces an
    earlier one with the same name."""
    MODELS[model.name] = model


def catalog() -> list[dict]:
    """The registry as the API and the console see it."""
    return [
        {
            "name": m.name,
            "label_zh": m.label_zh,
            "source_kind": m.source_kind,
            "client_rendered": m.client_rendered,
        }
        for m in MODELS.values()
    ]


def list_models() -> list[DigitalHumanModel]:
    """Every registered model, in registration order."""
    return list(MODELS.values())


def get_model(name: str | None = None) -> DigitalHumanModel:
    """Look up a model by name.

    ``None`` means "the default model" -- see :func:`default_model`.
    Raises :class:`ModelUnavailable` for an unknown name.
    """
    if name is None:
        return default_model()
    model = MODELS.get(name)
    if model is None:
        raise ModelUnavailable(f"unknown digital human model: {name}")
    return model


def default_model() -> DigitalHumanModel:
    """The fallback model. Never raises: the default is always registered."""
    model = MODELS.get(DEFAULT_MODEL)
    if model is None:  # pragma: no cover -- the default is always registered
        raise ModelUnavailable("no digital human model is registered")
    return model


def is_client_rendered(name: str) -> bool:
    """Whether the named model is rendered by the client.

    False for a model that is not registered any more: there is no
    runtime to speak of, and the caller has a bigger problem.
    """
    model = MODELS.get(name)
    return bool(model and model.client_rendered)


# ----- the built-in models --------------------------------------------------


def _musetalk_engine() -> "DriveEngine":
    from server.engine.musetalk_engine import MuseTalkEngine

    return MuseTalkEngine()


def _musetalk_pipeline() -> "AvatarPipeline":
    from server.avatar.musetalk_pipeline import MuseTalkAvatarPipeline

    return MuseTalkAvatarPipeline()


def _live2d_pipeline() -> "AvatarPipeline":
    from server.avatar.live2d_pipeline import Live2DAvatarPipeline

    return Live2DAvatarPipeline()


register_model(
    DigitalHumanModel(
        name="musetalk",
        label_zh="MuseTalk 1.5",
        pipeline_factory=_musetalk_pipeline,
        engine_factory=_musetalk_engine,
        source_kind=SOURCE_IMAGE_VIDEO,
    )
)

register_model(
    DigitalHumanModel(
        name="live2d",
        label_zh="Live2D",
        pipeline_factory=_live2d_pipeline,
        # No engine: the browser renders a Cubism project itself. The server
        # only hosts the project files and the SDK runtime.
        engine_factory=None,
        source_kind=SOURCE_ARCHIVE,
    )
)
