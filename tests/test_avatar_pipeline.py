import json

import pytest
from PIL import Image

from server.avatar.pipeline import MockAvatarPipeline, PipelineError, create_pipeline, run_avatar_job
from server.avatar.store import AvatarStore
from server.db.models import Avatar, AvatarJob


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


def test_pipeline_produces_pack(tmp_path, db):
    store = AvatarStore(tmp_path / "data")
    src = tmp_path / "photo.png"
    Image.new("RGB", (640, 800), (200, 120, 80)).save(src)

    avatar = Avatar(name="a", tenant_id="t", status="processing", source_image_path=str(src))
    db.add(avatar)
    db.flush()
    job = AvatarJob(avatar_id=avatar.id, stage="queued")
    db.add(job)
    db.commit()

    run_avatar_job(db, store, MockAvatarPipeline(), job.id)

    db.refresh(avatar)
    assert avatar.status == "ready"
    pack_dir = store.avatar_dir(avatar.id)
    assert avatar.pack_path == str(pack_dir / "pack.json")
    assert (pack_dir / "normalized.png").exists()
    cfg = json.loads((pack_dir / "pack.json").read_text())
    assert cfg["mouth_scale"] == 1.0
    pack = store.load_pack(avatar)
    assert pack.avatar_id == avatar.id
    assert pack.source_image_path.exists()


def test_run_avatar_job_reports_failure_when_pack_write_raises(tmp_path, monkeypatch, db):
    store = AvatarStore(tmp_path / "data")
    src = tmp_path / "photo.png"
    Image.new("RGB", (640, 800), (200, 120, 80)).save(src)

    avatar = Avatar(name="a", tenant_id="t", status="processing", source_image_path=str(src))
    db.add(avatar)
    db.flush()
    job = AvatarJob(avatar_id=avatar.id, stage="queued")
    db.add(job)
    db.commit()

    def raise_disk_full(self, *args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("pathlib.Path.write_text", raise_disk_full)

    run_avatar_job(db, store, MockAvatarPipeline(), job.id)

    db.refresh(avatar)
    db.refresh(job)
    assert avatar.status == "failed"
    assert "disk full" in job.error


def test_pipeline_rejects_tiny_image(tmp_path, db):
    store = AvatarStore(tmp_path / "data")
    src = tmp_path / "tiny.png"
    Image.new("RGB", (100, 100), (0, 0, 0)).save(src)

    avatar = Avatar(name="a", tenant_id="t", status="processing", source_image_path=str(src))
    db.add(avatar)
    db.flush()
    job = AvatarJob(avatar_id=avatar.id)
    db.add(job)
    db.commit()

    run_avatar_job(db, store, MockAvatarPipeline(), job.id)

    db.refresh(avatar)
    db.refresh(job)
    assert avatar.status == "failed"
    assert "256" in job.error


def test_mock_pipeline_stages_and_config(tmp_path):
    src = tmp_path / "photo.png"
    Image.new("RGB", (640, 800), (10, 20, 30)).save(src)
    out = tmp_path / "out"
    stages = []

    cfg = MockAvatarPipeline().process("a", src, out, lambda s, p: stages.append((s, p)))

    assert stages == [("detect_face", 10), ("align", 40), ("normalize", 70), ("build_pack", 100)]
    assert cfg == {"mouth_scale": 1.0, "expression_intensity": 1.0, "crop": {"mode": "center"}}
    with Image.open(out / "normalized.png") as img:
        assert img.size == (720, 1280)


def test_save_source_image_validates_decodable(tmp_path):
    store = AvatarStore(tmp_path / "data")
    ok = tmp_path / "ok.png"
    Image.new("RGB", (300, 300), (1, 2, 3)).save(ok)

    path = store.save_source_image("a1", ok.read_bytes(), ".png")
    assert path == store.avatar_dir("a1") / "source.png"
    assert path.exists()

    with pytest.raises(PipelineError):
        store.save_source_image("a2", b"not an image", ".png")


def test_delete_assets_removes_avatar_dir(tmp_path):
    store = AvatarStore(tmp_path / "data")
    d = store.avatar_dir("a3")
    d.mkdir(parents=True)
    (d / "normalized.png").write_bytes(b"x")

    store.delete_assets("a3")
    assert not d.exists()


def test_create_pipeline_mock_and_unknown():
    assert isinstance(create_pipeline("mock"), MockAvatarPipeline)
    with pytest.raises(PipelineError):
        create_pipeline("nope")
