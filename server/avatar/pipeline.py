import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from server.avatar.store import AvatarStore


class PipelineError(Exception):
    pass


class AvatarPipeline(ABC):
    @abstractmethod
    def process(
        self,
        avatar_id: str,
        source_path: Path,
        out_dir: Path,
        progress: Callable[[str, int], None],
    ) -> dict:
        ...


def create_pipeline(name: str | None = None) -> AvatarPipeline:
    """Build the avatar pipeline of a registered digital-human model.

    ``name`` defaults to the model this process is running (see
    ``server.model_registry``). The pipeline and the drive engine always
    come from the same model, so a pack can never be written by one model
    and driven by another.

    Raises :class:`PipelineError` for a name that is not registered.
    """
    from server.model_registry import ModelUnavailable, get_model

    try:
        model = get_model(name)
    except ModelUnavailable as exc:
        raise PipelineError(str(exc)) from exc
    return model.pipeline_factory()


def run_avatar_job(db, store, pipeline: AvatarPipeline, job_id: str) -> None:
    from server.db.models import Avatar, AvatarJob

    job = db.get(AvatarJob, job_id)
    avatar = db.get(Avatar, job.avatar_id)

    def progress(stage: str, pct: int) -> None:
        job.stage = stage
        job.progress = pct
        db.commit()

    try:
        avatar.status = "processing"
        db.commit()
        from server.avatar.tunables import effective

        # Per-avatar overrides captured by the create form beat both module
        # defaults and operator-pushed Setting rows. The audit row reflects
        # exactly what the pipeline saw, so a later idle-video re-encode
        # (which reads from the same column) will produce the same result.
        effective_now = effective(db, overrides=avatar.tunable_overrides or None)
        config = pipeline.process(
            avatar.id,
            Path(avatar.source_image_path),
            store.avatar_dir(avatar.id),
            progress,
        )
        pack_json = store.avatar_dir(avatar.id) / "pack.json"
        pack_json.write_text(json.dumps(config))
        avatar.pack_path = str(pack_json)
        # Audit: write the tunables actually used. ``effective()`` resolves DB
        # overrides on top of module defaults, so this row is exactly what the
        # pipeline and the idle encoder saw at run time.
        config["_effective_params"] = effective_now
        avatar.config_json = config
        avatar.status = "ready"
        db.commit()
    except Exception as exc:  # noqa: BLE001 — 失败必须落库可见
        db.rollback()
        avatar.status = "failed"
        job.error = str(exc)[:1024]
        db.commit()


def run_job_in_new_session(job_id: str, store: "AvatarStore") -> None:
    """Run an avatar job to completion in its own DB session.

    Shared by the FastAPI BackgroundTasks call sites in api/avatars.py (tenant
    upload) and api/admin.py (admin retry); both previously carried a private,
    near-identical copy of this runner.
    """
    from sqlalchemy.orm import sessionmaker

    from server.db.session import get_engine

    db = sessionmaker(bind=get_engine())()
    try:
        run_avatar_job(db, store, create_pipeline(), job_id)
    finally:
        db.close()
