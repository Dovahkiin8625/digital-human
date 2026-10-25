"""The live drive engines, one per server-rendered digital-human model.

An engine holds its model's weights in VRAM, so an operator may keep
several resident at once -- one per model -- and load / unload them from
the Models tab. Engines are created on demand: a model nobody drives
costs nothing until something asks for it.

Which engine drives an avatar is decided by the avatar's own
``drive_model``, never by whatever happens to be running. A model the
*client* renders (Live2D) has no engine at all, so asking the pool for
one is an error rather than an empty object.
"""
from __future__ import annotations

import threading

from server.engine.base import DriveEngine, EngineUnavailable


class EnginePool:
    """The engines this process holds, keyed by digital-human model name."""

    def __init__(self) -> None:
        self._engines: dict[str, DriveEngine] = {}
        self._lock = threading.Lock()

    def engine_for(self, model_name: str) -> DriveEngine:
        """The engine of ``model_name``, created on first use.

        Raises :class:`EngineUnavailable` for a model that is not
        registered, and for one the client renders (it has no server
        engine to hand out).
        """
        # Imported here so the module attribute is resolved at call time:
        # the test suite replaces ``server.engine.base.create_engine`` with
        # a CPU-only double.
        from server.engine.base import create_engine
        from server.model_registry import ModelUnavailable, get_model

        try:
            model = get_model(model_name)
        except ModelUnavailable as exc:
            raise EngineUnavailable(str(exc)) from exc
        if model.engine_factory is None:
            raise EngineUnavailable(
                f"model {model_name!r} is rendered by the client; "
                "it has no server drive engine"
            )
        with self._lock:
            engine = self._engines.get(model_name)
            if engine is None:
                engine = create_engine(model_name)
                self._engines[model_name] = engine
            return engine

    def engines(self) -> dict[str, DriveEngine]:
        """The engines created so far, by model name."""
        with self._lock:
            return dict(self._engines)

    async def close_all(self) -> None:
        """Close every engine this pool created. Idempotent per engine."""
        for engine in self.engines().values():
            await engine.close()
