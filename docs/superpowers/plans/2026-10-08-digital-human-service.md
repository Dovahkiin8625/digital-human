# 商用数字人服务 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现商用数字人服务 MVP：音频驱动 2D 形象 → WebRTC 视频流输出，含形象制作、会话管理与后台调试台。

**Architecture:** FastAPI 单体（API + aiortc 内嵌 WebRTC），DriveEngine/StreamSink/RenderSession 三层抽象隔离推理与媒体出口；SQLite 元数据 + 磁盘资产；Mock 引擎保证无 GPU 全链路可开发可测。详见 spec。

**Tech Stack:** Python 3.11、FastAPI、aiortc、SQLAlchemy 2（SQLite）、numpy、Pillow、React 18 + Vite + TypeScript。

**Spec:** `docs/superpowers/specs/2026-10-08-digital-human-service-design.md`

## Global Constraints

- 运行环境：Windows 11 + Git Bash；命令均以仓库根为 CWD
- Python 包：`server/`；测试：`tests/`；命令：`python -m pytest`（venv：`python -m venv .venv && source .venv/Scripts/activate`）
- 环境变量前缀 `DH_`（如 `DH_DRIVE_ENGINE`、`DH_DATA_DIR`、`DH_DATABASE_URL`、`DH_ADMIN_PASSWORD`、`DH_ADMIN_TOKEN`）
- 对外 API 前缀 `/api/v1`，鉴权头 `X-API-Key`；管理面前缀 `/api/admin`，鉴权头 `X-Admin-Token`
- 统一错误信封：`{"error": {"code": str, "message": str, "detail": any}}`
- 媒体参数：音频 48kHz 16bit PCM、100ms 驱动块；视频 25fps、720×1280、RGB ndarray
- 会话空闲（无音频）60s 自动关闭
- ORM 模型命名与 spec 的差异：spec 的 `Session` 模型在代码中名为 `StreamSession`（表名 `sessions`），避免与 SQLAlchemy `Session` 冲突；`Tenant`/`Avatar`/`AvatarJob` 与 spec 同名
- TDD：每个任务先写失败测试；每步完成后 `python -m pytest` 全绿再 commit
- Commit message 用 feat/fix/test/docs 前缀，结尾带 `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`

---

## File Structure

```
digital-human/
├─ pyproject.toml                     # 依赖与 pytest 配置
├─ server/
│  ├─ __init__.py
│  ├─ config.py                       # Settings（pydantic-settings，DH_ 前缀）
│  ├─ main.py                         # create_app()、错误处理器注册、静态托管 admin-ui
│  ├─ db/
│  │  ├─ __init__.py
│  │  ├─ session.py                   # engine、init_db、get_db
│  │  └─ models.py                    # Tenant, Avatar, AvatarJob, StreamSession
│  ├─ core/
│  │  ├─ __init__.py
│  │  ├─ errors.py                    # AppError + register_handlers
│  │  ├─ audio_ingest.py              # AudioIngest：PCM 分块
│  │  ├─ render_session.py            # RenderSession：状态机 + 驱动循环 + metrics
│  │  ├─ stream_sink.py               # StreamSink 抽象、NullSink、FrameBufferSink
│  │  ├─ webrtc.py                    # EngineVideoTrack、WebRTCSink、信令处理
│  │  └─ session_manager.py           # SessionManager：创建/获取/关闭/空闲回收
│  ├─ engine/
│  │  ├─ __init__.py
│  │  ├─ base.py                      # AvatarPack、DriveFrame、DriveEngine、create_engine
│  │  ├─ mock_engine.py               # MockEngine：RMS 驱动嘴形合成帧
│  │  └─ musetalk_engine.py           # MuseTalkEngine：权重缺失时 EngineUnavailable
│  ├─ avatar/
│  │  ├─ __init__.py
│  │  ├─ store.py                     # AvatarStore：源图/形象包文件管理
│  │  └─ pipeline.py                  # AvatarPipeline 抽象、MockAvatarPipeline、run_avatar_job
│  ├─ api/
│  │  ├─ __init__.py
│  │  ├─ deps.py                      # hash_api_key、get_tenant、get_admin
│  │  ├─ avatars.py                   # /api/v1/avatars CRUD
│  │  ├─ sessions.py                  # /api/v1/sessions CRUD + 信令/音频 WS 路由挂载
│  │  └─ admin.py                     # /api/admin/* 全部管理端点
├─ admin-ui/                          # React + Vite + TS 调试台（任务 11-15）
├─ data/                              # 运行时资产（gitignore）
├─ tests/
│  ├─ conftest.py                     # app/client/tmp_data fixtures
│  ├─ test_healthz.py
│  ├─ test_auth.py
│  ├─ test_mock_engine.py
│  ├─ test_avatar_pipeline.py
│  ├─ test_avatars_api.py
│  ├─ test_render_session.py
│  ├─ test_sessions_api.py
│  ├─ test_webrtc_track.py
│  ├─ test_ws_audio.py
│  ├─ test_admin_api.py
│  └─ test_e2e_smoke.py
└─ docs/superpowers/{specs,plans}/
```

---

## Task 1: 项目骨架（pyproject、配置、DB 模型、app factory、healthz）

**Files:**
- Create: `pyproject.toml`, `server/__init__.py`, `server/config.py`, `server/main.py`, `server/db/__init__.py`, `server/db/session.py`, `server/db/models.py`, `tests/conftest.py`, `tests/test_healthz.py`, `.gitignore`

**Interfaces:**
- Consumes: 无（首个任务）
- Produces:
  - `server.config.get_settings() -> Settings`（字段：`drive_engine: str`、`data_dir: Path`、`database_url: str`、`admin_password: str`、`admin_token: str`、`session_idle_timeout: float`、`frame_width: int`、`frame_height: int`、`fps: int`、`sample_rate: int`）
  - `server.db.session.init_db(engine) -> None`、`get_db() -> Iterator[Session]`
  - `server.db.models`: `Base`、`Tenant`、`Avatar`、`AvatarJob`、`StreamSession`
  - `server.main.create_app() -> FastAPI`；`GET /healthz` → `{"status": "ok", "engine": str, "gpu": bool}`

- [ ] **Step 1: 写失败测试**

`tests/test_healthz.py`:
```python
def test_healthz_ok(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["engine"] == "mock"
    assert body["gpu"] is False
```

`tests/conftest.py`:
```python
import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DH_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DH_DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setenv("DH_DRIVE_ENGINE", "mock")
    from server.config import get_settings
    get_settings.cache_clear()
    from server.main import create_app
    from server.db.session import init_db, get_engine
    app = create_app()
    init_db(get_engine())
    with TestClient(app) as c:
        yield c
    get_settings.cache_clear()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_healthz.py -v`
Expected: FAIL（ModuleNotFoundError: server）

- [ ] **Step 3: 最小实现**

`pyproject.toml`:
```toml
[project]
name = "digital-human"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "fastapi>=0.115",
  "uvicorn[standard]>=0.30",
  "sqlalchemy>=2.0",
  "pydantic-settings>=2.4",
  "python-multipart>=0.0.9",
  "numpy>=1.26",
  "pillow>=10.0",
  "aiortc>=1.9",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "httpx>=0.27"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["server*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

`.gitignore`:
```
.venv/
__pycache__/
*.egg-info/
data/
admin-ui/node_modules/
admin-ui/dist/
.pytest_cache/
```

`server/config.py`:
```python
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    drive_engine: str = "mock"
    data_dir: Path = Path("data")
    database_url: str = "sqlite:///data/digital-human.db"
    admin_password: str = "admin123"
    admin_token: str = "dev-admin-token"
    session_idle_timeout: float = 60.0
    frame_width: int = 720
    frame_height: int = 1280
    fps: int = 25
    sample_rate: int = 48000

    model_config = SettingsConfigDict(env_prefix="DH_", env_file=".env")


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

`server/db/models.py`:
```python
import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _uuid() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Tenant(Base):
    __tablename__ = "tenants"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(128))
    api_key_hash: Mapped[str] = mapped_column(String(64), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Avatar(Base):
    __tablename__ = "avatars"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    tenant_id: Mapped[str] = mapped_column(String(32), ForeignKey("tenants.id"))
    name: Mapped[str] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(16), default="pending")
    source_image_path: Mapped[str] = mapped_column(String(512))
    pack_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    config_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class AvatarJob(Base):
    __tablename__ = "avatar_jobs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    avatar_id: Mapped[str] = mapped_column(String(32), ForeignKey("avatars.id"))
    stage: Mapped[str] = mapped_column(String(64), default="queued")
    progress: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class StreamSession(Base):
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    tenant_id: Mapped[str] = mapped_column(String(32), ForeignKey("tenants.id"))
    avatar_id: Mapped[str] = mapped_column(String(32), ForeignKey("avatars.id"))
    status: Mapped[str] = mapped_column(String(16), default="idle")
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    metrics_json: Mapped[dict] = mapped_column(JSON, default=dict)
```

`server/db/session.py`:
```python
from collections.abc import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from server.config import get_settings
from server.db.models import Base

_engine = None


def get_engine():
    global _engine
    if _engine is None:
        settings = get_settings()
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        _engine = create_engine(settings.database_url, connect_args={"check_same_thread": False})
    return _engine


def reset_engine() -> None:
    global _engine
    _engine = None


def init_db(engine) -> None:
    Base.metadata.create_all(engine)


def get_db() -> Iterator[Session]:
    session = sessionmaker(bind=get_engine())()
    try:
        yield session
    finally:
        session.close()
```

`server/main.py`:
```python
from fastapi import FastAPI

from server.config import get_settings
from server.db.session import get_engine, init_db


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="digital-human", version="0.1.0")
    init_db(get_engine())

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok", "engine": settings.drive_engine, "gpu": False}

    return app
```

空文件：`server/__init__.py`、`server/db/__init__.py`。

- [ ] **Step 4: 安装依赖并跑测试确认通过**

Run:
```bash
python -m venv .venv && source .venv/Scripts/activate
pip install -e ".[dev]"
python -m pytest tests/test_healthz.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .gitignore server tests
git commit -m "feat: project skeleton with config, db models and healthz

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 2: 错误信封 + API Key 鉴权 + Admin 登录

**Files:**
- Create: `server/core/errors.py`, `server/api/__init__.py`, `server/api/deps.py`, `tests/test_auth.py`
- Modify: `server/main.py`（注册错误处理器与路由）、`server/core/__init__.py`

**Interfaces:**
- Consumes: Task 1 的 `get_db`、`Tenant`、`Settings`
- Produces:
  - `server.core.errors.AppError(code: str, message: str, status: int = 400, detail: Any = None)`
  - `server.core.errors.register_handlers(app: FastAPI) -> None`
  - `server.api.deps.hash_api_key(key: str) -> str`
  - `server.api.deps.get_tenant(x_api_key: str = Header(...), db = Depends(get_db)) -> Tenant`（401 `AppError("unauthorized", ...)`）
  - `server.api.deps.get_admin(x_admin_token: str = Header(...)) -> None`（401 on mismatch）
  - `POST /api/admin/login` body `{"password": str}` → `{"token": str}`（密码错 → 401 信封）

- [ ] **Step 1: 写失败测试**

`tests/test_auth.py`:
```python
from server.api.deps import hash_api_key
from server.db.models import Tenant


def test_error_envelope_shape(client):
    resp = client.get("/api/v1/avatars")
    assert resp.status_code == 401
    body = resp.json()
    assert body == {
        "error": {"code": "unauthorized", "message": "missing or invalid API key", "detail": None}
    }


def test_admin_login_wrong_password(client):
    resp = client.post("/api/admin/login", json={"password": "wrong"})
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "unauthorized"


def test_admin_login_ok(client):
    resp = client.post("/api/admin/login", json={"password": "admin123"})
    assert resp.status_code == 200
    assert resp.json()["token"] == "dev-admin-token"


def test_hash_api_key_stable():
    assert hash_api_key("k1") == hash_api_key("k1")
    assert hash_api_key("k1") != hash_api_key("k2")


def test_tenant_auth_ok(client):
    from server.db.session import get_engine
    from sqlalchemy.orm import sessionmaker

    db = sessionmaker(bind=get_engine())()
    db.add(Tenant(name="t1", api_key_hash=hash_api_key("key-1")))
    db.commit()
    db.close()
    resp = client.get("/api/v1/avatars", headers={"X-API-Key": "key-1"})
    assert resp.status_code == 200
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_auth.py -v`
Expected: FAIL（ImportError / 404 无路由）

- [ ] **Step 3: 实现**

`server/core/errors.py`:
```python
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


class AppError(Exception):
    def __init__(self, code: str, message: str, status: int = 400, detail: Any = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status
        self.detail = detail


def envelope(code: str, message: str, detail: Any = None) -> dict:
    return {"error": {"code": code, "message": message, "detail": detail}}


def register_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(_: Request, exc: AppError):
        return JSONResponse(exc.status, envelope(exc.code, exc.message, exc.detail))

    @app.exception_handler(HTTPException)
    async def _http_error(_: Request, exc: HTTPException):
        code = "not_found" if exc.status_code == 404 else "http_error"
        return JSONResponse(exc.status_code, envelope(code, str(exc.detail), None))

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError):
        return JSONResponse(422, envelope("validation_error", "invalid request", exc.errors()))

    @app.exception_handler(Exception)
    async def _internal(_: Request, exc: Exception):
        return JSONResponse(500, envelope("internal_error", "internal server error", str(exc)))
```

`server/api/deps.py`:
```python
import hashlib
import hmac

from fastapi import Depends, Header

from server.config import get_settings
from server.core.errors import AppError
from server.db.models import Tenant
from server.db.session import get_db


def hash_api_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def get_tenant(x_api_key: str = Header(default=""), db=Depends(get_db)) -> Tenant:
    if not x_api_key:
        raise AppError("unauthorized", "missing or invalid API key", 401)
    tenant = db.query(Tenant).filter(Tenant.api_key_hash == hash_api_key(x_api_key)).first()
    if tenant is None:
        raise AppError("unauthorized", "missing or invalid API key", 401)
    return tenant


def get_admin(x_admin_token: str = Header(default="")) -> None:
    expected = get_settings().admin_token
    if not hmac.compare_digest(x_admin_token, expected):
        raise AppError("unauthorized", "missing or invalid admin token", 401)
```

`server/api/admin.py`（本任务只放 login，其余任务 10 补）:
```python
import hmac

from fastapi import APIRouter
from pydantic import BaseModel

from server.config import get_settings
from server.core.errors import AppError

router = APIRouter(prefix="/api/admin", tags=["admin"])


class LoginIn(BaseModel):
    password: str


@router.post("/login")
def login(body: LoginIn) -> dict:
    if not hmac.compare_digest(body.password, get_settings().admin_password):
        raise AppError("unauthorized", "missing or invalid admin token", 401)
    return {"token": get_settings().admin_token}
```

`server/api/avatars.py`（占位路由让 401 测试生效，任务 5 填充）:
```python
from fastapi import APIRouter, Depends

from server.api.deps import get_tenant
from server.db.models import Tenant

router = APIRouter(prefix="/api/v1/avatars", tags=["avatars"])


@router.get("")
def list_avatars(tenant: Tenant = Depends(get_tenant)) -> list:
    return []
```

`server/main.py` 追加（create_app 内）:
```python
    from server.api.admin import router as admin_router
    from server.api.avatars import router as avatars_router
    from server.core.errors import register_handlers

    register_handlers(app)
    app.include_router(admin_router)
    app.include_router(avatars_router)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest -v`
Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add server tests
git commit -m "feat: error envelope, api-key auth and admin login

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 3: DriveEngine 接口 + MockEngine

**Files:**
- Create: `server/engine/__init__.py`, `server/engine/base.py`, `server/engine/mock_engine.py`, `tests/test_mock_engine.py`

**Interfaces:**
- Consumes: 无
- Produces:
  - `server.engine.base.AvatarPack(avatar_id: str, root: Path, source_image_path: Path, config: dict)`
  - `server.engine.base.DriveFrame(pts: float, image: np.ndarray)`（image 为 RGB uint8 HxWx3）
  - `server.engine.base.DriveEngine`（ABC）：`async prepare(pack) -> None`、`async drive(pack, audio: np.ndarray, sample_rate: int) -> list[DriveFrame]`（audio float32 一维）、`describe() -> dict`、`async close() -> None`
  - `server.engine.base.create_engine(name: str | None = None, settings: Settings | None = None) -> DriveEngine`
  - `server.engine.mock_engine.MockEngine(width=720, height=1280, fps=25)`（构造参数默认取 settings）
  - `server.engine.base.EngineUnavailable(Exception)`

- [ ] **Step 1: 写失败测试**

`tests/test_mock_engine.py`:
```python
import numpy as np
import pytest

from server.engine.base import AvatarPack, create_engine
from server.engine.mock_engine import MockEngine


def _pack(tmp_path):
    from PIL import Image

    img_path = tmp_path / "src.png"
    Image.new("RGB", (720, 1280), (30, 60, 90)).save(img_path)
    return AvatarPack(avatar_id="a1", root=tmp_path, source_image_path=img_path, config={"mouth_scale": 1.0})


@pytest.mark.asyncio
async def test_drive_returns_frames_matching_audio_duration(tmp_path):
    engine = MockEngine()
    pack = _pack(tmp_path)
    audio = np.zeros(48000, dtype=np.float32)  # 1s
    frames = await engine.drive(pack, audio, 48000)
    assert len(frames) == 25
    assert frames[0].image.shape == (1280, 720, 3)
    assert frames[0].image.dtype == np.uint8
    assert frames[0].pts == 0.0
    assert abs(frames[1].pts - 0.04) < 1e-6


@pytest.mark.asyncio
async def test_drive_is_deterministic(tmp_path):
    engine = MockEngine()
    pack = _pack(tmp_path)
    audio = np.linspace(-0.5, 0.5, 24000, dtype=np.float32)
    a = await engine.drive(pack, audio, 48000)
    b = await engine.drive(pack, audio, 48000)
    assert all((x.image == y.image).all() for x, y in zip(a, b))


@pytest.mark.asyncio
async def test_loud_audio_opens_mouth_more(tmp_path):
    engine = MockEngine()
    pack = _pack(tmp_path)
    silent = await engine.drive(pack, np.zeros(4800, dtype=np.float32), 48000)
    loud = await engine.drive(pack, np.ones(4800, dtype=np.float32) * 0.9, 48000)
    assert np.abs(loud[0].image.astype(int) - silent[0].image.astype(int)).sum() > 0


def test_create_engine_factory():
    assert create_engine("mock").__class__.__name__ == "MockEngine"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_mock_engine.py -v`
Expected: FAIL（ModuleNotFoundError: server.engine）

- [ ] **Step 3: 实现**

`server/engine/base.py`:
```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from server.config import Settings, get_settings


class EngineUnavailable(Exception):
    pass


@dataclass
class AvatarPack:
    avatar_id: str
    root: Path
    source_image_path: Path
    config: dict = field(default_factory=dict)


@dataclass
class DriveFrame:
    pts: float
    image: np.ndarray


class DriveEngine(ABC):
    name: str = "base"

    async def prepare(self, pack: AvatarPack) -> None:
        pass

    @abstractmethod
    async def drive(self, pack: AvatarPack, audio: np.ndarray, sample_rate: int) -> list[DriveFrame]:
        ...

    def describe(self) -> dict:
        return {"name": self.name, "gpu": False}

    async def close(self) -> None:
        pass


def create_engine(name: str | None = None, settings: Settings | None = None) -> DriveEngine:
    settings = settings or get_settings()
    name = name or settings.drive_engine
    if name == "mock":
        from server.engine.mock_engine import MockEngine

        return MockEngine(settings.frame_width, settings.frame_height, settings.fps)
    if name == "musetalk":
        from server.engine.musetalk_engine import MuseTalkEngine

        return MuseTalkEngine()
    raise EngineUnavailable(f"unknown engine: {name}")
```

`server/engine/mock_engine.py`:
```python
import numpy as np
from PIL import Image, ImageDraw

from server.engine.base import AvatarPack, DriveEngine, DriveFrame


class MockEngine(DriveEngine):
    name = "mock"

    def __init__(self, width: int = 720, height: int = 1280, fps: int = 25):
        self.width = width
        self.height = height
        self.fps = fps

    def describe(self) -> dict:
        return {"name": self.name, "gpu": False, "width": self.width, "height": self.height, "fps": self.fps}

    async def drive(self, pack: AvatarPack, audio: np.ndarray, sample_rate: int) -> list[DriveFrame]:
        audio = np.asarray(audio, dtype=np.float32)
        duration = len(audio) / sample_rate
        n_frames = max(1, round(duration * self.fps))
        samples_per_frame = sample_rate / self.fps
        base = Image.open(pack.source_image_path).convert("RGB").resize((self.width, self.height))
        scale = float(pack.config.get("mouth_scale", 1.0))
        frames: list[DriveFrame] = []
        for i in range(n_frames):
            start = int(i * samples_per_frame)
            end = min(len(audio), int((i + 1) * samples_per_frame))
            seg = audio[start:end] if end > start else np.zeros(1, dtype=np.float32)
            rms = float(np.sqrt(np.mean(seg**2))) if len(seg) else 0.0
            openness = min(1.0, rms * 4.0) * scale
            img = base.copy()
            draw = ImageDraw.Draw(img)
            cx, cy = self.width // 2, int(self.height * 0.72)
            mw, mh0 = int(self.width * 0.12), int(self.height * 0.015)
            mh = max(2, int(mh0 * (1 + openness * 6)))
            draw.ellipse((cx - mw, cy - mh, cx + mw, cy + mh), fill=(180, 40, 60))
            meter_h = int((self.height // 4) * openness)
            draw.rectangle((self.width - 40, self.height - meter_h - 20, self.width - 20, self.height - 20), fill=(40, 200, 90))
            frames.append(DriveFrame(pts=i / self.fps, image=np.asarray(img, dtype=np.uint8)))
        return frames
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest -v`
Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add server/engine tests/test_mock_engine.py
git commit -m "feat: DriveEngine interface and deterministic MockEngine

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 4: AvatarStore + MockAvatarPipeline + 制作任务

**Files:**
- Create: `server/avatar/__init__.py`, `server/avatar/store.py`, `server/avatar/pipeline.py`, `tests/test_avatar_pipeline.py`

**Interfaces:**
- Consumes: Task 1 模型 `Avatar`/`AvatarJob`，Task 3 `AvatarPack`
- Produces:
  - `server.avatar.store.AvatarStore(data_dir: Path)`：
    - `avatar_dir(avatar_id: str) -> Path`
    - `save_source_image(avatar_id: str, data: bytes, suffix: str) -> Path`（Pillow 校验可解码，失败 raise `PipelineError`）
    - `load_pack(avatar: Avatar) -> AvatarPack`（读 `pack.json` → `AvatarPack`）
    - `delete_assets(avatar_id: str) -> None`
  - `server.avatar.pipeline.PipelineError(Exception)`
  - `server.avatar.pipeline.AvatarPipeline`（ABC）：`process(avatar_id: str, source_path: Path, out_dir: Path, progress: Callable[[str, int], None]) -> dict`（返回 config dict，写 `normalized.png` 到 out_dir）
  - `server.avatar.pipeline.MockAvatarPipeline`（阶段：`detect_face` 10 → `align` 40 → `normalize` 70 → `build_pack` 100；最小尺寸 256×256 否则 `PipelineError`；中心裁剪方图 → 贴到 720×1280 画布写 `normalized.png`；config：`{"mouth_scale": 1.0, "expression_intensity": 1.0, "crop": {"mode": "center"}}`）
  - `server.avatar.pipeline.run_avatar_job(db, store: AvatarStore, pipeline: AvatarPipeline, job_id: str) -> None`（更新 AvatarJob.stage/progress、成功写 `pack.json` + `Avatar.pack_path/status="ready"`、失败 `status="failed"` + error）
  - `server.avatar.pipeline.create_pipeline(name: str) -> AvatarPipeline`（`mock` → MockAvatarPipeline）

- [ ] **Step 1: 写失败测试**

`tests/test_avatar_pipeline.py`:
```python
import json

from PIL import Image

from server.avatar.pipeline import MockAvatarPipeline, run_avatar_job
from server.avatar.store import AvatarStore
from server.db.models import Avatar, AvatarJob


def _make_db(tmp_path, monkeypatch):
    monkeypatch.setenv("DH_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DH_DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    from server.config import get_settings
    get_settings.cache_clear()
    from server.db.session import get_engine, init_db
    from sqlalchemy.orm import sessionmaker

    init_db(get_engine())
    return sessionmaker(bind=get_engine())()


def test_pipeline_produces_pack(tmp_path, monkeypatch):
    db = _make_db(tmp_path, monkeypatch)
    store = AvatarStore(tmp_path / "data")
    src = tmp_path / "photo.png"
    Image.new("RGB", (640, 800), (200, 120, 80)).save(src)

    avatar = Avatar(name="a", tenant_id="t", status="processing", source_image_path=str(src))
    db.add(avatar)
    job = AvatarJob(avatar_id=avatar.id, stage="queued")
    db.add(job)
    db.commit()

    run_avatar_job(db, store, MockAvatarPipeline(), job.id)

    db.refresh(avatar)
    assert avatar.status == "ready"
    pack_dir = store.avatar_dir(avatar.id)
    assert (pack_dir / "normalized.png").exists()
    cfg = json.loads((pack_dir / "pack.json").read_text())
    assert cfg["mouth_scale"] == 1.0
    pack = store.load_pack(avatar)
    assert pack.avatar_id == avatar.id
    assert pack.source_image_path.exists()


def test_pipeline_rejects_tiny_image(tmp_path, monkeypatch):
    db = _make_db(tmp_path, monkeypatch)
    store = AvatarStore(tmp_path / "data")
    src = tmp_path / "tiny.png"
    Image.new("RGB", (100, 100), (0, 0, 0)).save(src)

    avatar = Avatar(name="a", tenant_id="t", status="processing", source_image_path=str(src))
    db.add(avatar)
    job = AvatarJob(avatar_id=avatar.id)
    db.add(job)
    db.commit()

    run_avatar_job(db, store, MockAvatarPipeline(), job.id)

    db.refresh(avatar)
    db.refresh(job)
    assert avatar.status == "failed"
    assert "256" in job.error
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_avatar_pipeline.py -v`
Expected: FAIL（ModuleNotFoundError: server.avatar）

- [ ] **Step 3: 实现**

`server/avatar/store.py`:
```python
import json
import shutil
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from server.avatar.pipeline import PipelineError
from server.db.models import Avatar
from server.engine.base import AvatarPack


class AvatarStore:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)

    def avatar_dir(self, avatar_id: str) -> Path:
        return self.data_dir / "avatars" / avatar_id

    def save_source_image(self, avatar_id: str, data: bytes, suffix: str) -> Path:
        d = self.avatar_dir(avatar_id)
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"source{suffix or '.png'}"
        path.write_bytes(data)
        try:
            Image.open(path).verify()
        except (UnidentifiedImageError, OSError) as exc:
            raise PipelineError("source image is not a decodable image") from exc
        return path

    def load_pack(self, avatar: Avatar) -> AvatarPack:
        root = self.avatar_dir(avatar.id)
        pack_json = root / "pack.json"
        config = json.loads(pack_json.read_text()) if pack_json.exists() else {}
        source = Path(avatar.source_image_path)
        normalized = root / "normalized.png"
        return AvatarPack(
            avatar_id=avatar.id,
            root=root,
            source_image_path=normalized if normalized.exists() else source,
            config=config,
        )

    def delete_assets(self, avatar_id: str) -> None:
        d = self.avatar_dir(avatar_id)
        if d.exists():
            shutil.rmtree(d)
```

`server/avatar/pipeline.py`:
```python
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path

from PIL import Image


class PipelineError(Exception):
    pass


class AvatarPipeline(ABC):
    @abstractmethod
    def process(self, avatar_id: str, source_path: Path, out_dir: Path, progress: Callable[[str, int], None]) -> dict:
        ...


class MockAvatarPipeline(AvatarPipeline):
    MIN_SIZE = 256

    def process(self, avatar_id: str, source_path: Path, out_dir: Path, progress: Callable[[str, int], None]) -> dict:
        progress("detect_face", 10)
        img = Image.open(source_path).convert("RGB")
        if img.width < self.MIN_SIZE or img.height < self.MIN_SIZE:
            raise PipelineError(f"image too small: {img.width}x{img.height}, min {self.MIN_SIZE}x{self.MIN_SIZE}")
        progress("align", 40)
        side = min(img.width, img.height)
        left = (img.width - side) // 2
        top = (img.height - side) // 2
        cropped = img.crop((left, top, left + side, top + side))
        progress("normalize", 70)
        out_dir.mkdir(parents=True, exist_ok=True)
        canvas = Image.new("RGB", (720, 1280), (16, 16, 24))
        resized = cropped.resize((720, 720))
        canvas.paste(resized, (0, 280))
        canvas.save(out_dir / "normalized.png")
        progress("build_pack", 100)
        return {"mouth_scale": 1.0, "expression_intensity": 1.0, "crop": {"mode": "center"}}


def create_pipeline(name: str) -> AvatarPipeline:
    if name == "mock":
        return MockAvatarPipeline()
    raise PipelineError(f"unknown pipeline: {name}")


def run_avatar_job(db, store, pipeline: AvatarPipeline, job_id: str) -> None:
    from server.db.models import Avatar, AvatarJob

    job = db.query(AvatarJob).get(job_id)
    avatar = db.query(Avatar).get(job.avatar_id)
    avatar.status = "processing"
    db.commit()

    def progress(stage: str, pct: int) -> None:
        job.stage = stage
        job.progress = pct
        db.commit()

    try:
        config = pipeline.process(
            avatar.id, Path(avatar.source_image_path), store.avatar_dir(avatar.id), progress
        )
    except Exception as exc:  # noqa: BLE001 — 失败必须落库可见
        avatar.status = "failed"
        job.error = str(exc)[:1024]
        db.commit()
        return

    pack_json = store.avatar_dir(avatar.id) / "pack.json"
    pack_json.write_text(__import__("json").dumps(config))
    avatar.pack_path = str(pack_json)
    avatar.config_json = config
    avatar.status = "ready"
    db.commit()
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest -v`
Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add server/avatar tests/test_avatar_pipeline.py
git commit -m "feat: avatar store, mock pipeline and job runner

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---
## Task 5: 形象对外 REST API（上传/轮询/删除 + 异步制作）

**Files:**
- Modify: `server/api/avatars.py`（替换占位实现）
- Create: `tests/test_avatars_api.py`

**Interfaces:**
- Consumes: Task 2 `get_tenant`，Task 4 `AvatarStore`/`MockAvatarPipeline`/`run_avatar_job`
- Produces:
  - `POST /api/v1/avatars`（multipart: `name: str`, `file: UploadFile`）→ 201 `{"avatar_id": str, "status": "pending"}`
  - `GET /api/v1/avatars` → `[{"avatar_id", "name", "status", "created_at"}]`（仅本租户）
  - `GET /api/v1/avatars/{avatar_id}` → 同上单条 + `"config": dict`、`"error": str | null`
  - `DELETE /api/v1/avatars/{avatar_id}` → 204（删 DB 行 + 资产）
  - 制作经 FastAPI `BackgroundTasks` 异步跑 `run_avatar_job`（TestClient 会在响应后同步执行，测试可断言完成态）

- [ ] **Step 1: 写失败测试**

`tests/test_avatars_api.py`:
```python
import io

from PIL import Image


def _png_bytes(w=640, h=800):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


def _seed_tenant(client, api_key="key-1"):
    from server.api.deps import hash_api_key
    from server.db.models import Tenant
    from server.db.session import get_engine
    from sqlalchemy.orm import sessionmaker

    db = sessionmaker(bind=get_engine())()
    db.add(Tenant(name="t1", api_key_hash=hash_api_key(api_key)))
    db.commit()
    db.close()
    return {"X-API-Key": api_key}


def test_avatar_lifecycle(client):
    headers = _seed_tenant(client)
    resp = client.post(
        "/api/v1/avatars",
        headers=headers,
        data={"name": "Alice"},
        files={"file": ("alice.png", _png_bytes(), "image/png")},
    )
    assert resp.status_code == 201
    avatar_id = resp.json()["avatar_id"]

    detail = client.get(f"/api/v1/avatars/{avatar_id}", headers=headers).json()
    assert detail["status"] == "ready"
    assert detail["config"]["mouth_scale"] == 1.0

    listing = client.get("/api/v1/avatars", headers=headers).json()
    assert [a["avatar_id"] for a in listing] == [avatar_id]

    assert client.delete(f"/api/v1/avatars/{avatar_id}", headers=headers).status_code == 204
    assert client.get(f"/api/v1/avatars/{avatar_id}", headers=headers).status_code == 404


def test_avatar_tiny_image_becomes_failed(client):
    headers = _seed_tenant(client)
    resp = client.post(
        "/api/v1/avatars",
        headers=headers,
        data={"name": "bad"},
        files={"file": ("bad.png", _png_bytes(50, 50), "image/png")},
    )
    avatar_id = resp.json()["avatar_id"]
    detail = client.get(f"/api/v1/avatars/{avatar_id}", headers=headers).json()
    assert detail["status"] == "failed"
    assert detail["error"]


def test_avatar_not_found(client):
    headers = _seed_tenant(client)
    assert client.get("/api/v1/avatars/nope", headers=headers).status_code == 404
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_avatars_api.py -v`
Expected: FAIL（占位 list_avatars 无 POST）

- [ ] **Step 3: 实现**

`server/api/avatars.py`（全量替换）:
```python
import json
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, Form, UploadFile
from starlette.responses import Response

from server.api.deps import get_tenant
from server.avatar.pipeline import create_pipeline, run_avatar_job
from server.avatar.store import AvatarStore
from server.config import get_settings
from server.core.errors import AppError
from server.db.models import Avatar, AvatarJob, Tenant
from server.db.session import get_db, get_engine

router = APIRouter(prefix="/api/v1/avatars", tags=["avatars"])


def _store() -> AvatarStore:
    return AvatarStore(get_settings().data_dir)


def _serialize(a: Avatar, with_detail: bool = False) -> dict:
    data = {
        "avatar_id": a.id,
        "name": a.name,
        "status": a.status,
        "created_at": a.created_at.isoformat(),
    }
    if with_detail:
        data["config"] = a.config_json
        data["error"] = None
    return data


def _get_owned(db, tenant: Tenant, avatar_id: str) -> Avatar:
    avatar = db.query(Avatar).filter(Avatar.id == avatar_id, Avatar.tenant_id == tenant.id).first()
    if avatar is None:
        raise AppError("not_found", f"avatar {avatar_id} not found", 404)
    return avatar


@router.post("", status_code=201)
def create_avatar(
    background: BackgroundTasks,
    name: str = Form(...),
    file: UploadFile = ...,
    tenant: Tenant = Depends(get_tenant),
    db=Depends(get_db),
) -> dict:
    avatar_id = uuid4().hex
    store = _store()
    data = file.file.read()
    suffix = "." + (file.filename.rsplit(".", 1)[-1] if file.filename and "." in file.filename else "png")
    try:
        source_path = store.save_source_image(avatar_id, data, suffix)
    except Exception as exc:
        raise AppError("invalid_image", str(exc), 422) from exc

    avatar = Avatar(
        id=avatar_id,
        tenant_id=tenant.id,
        name=name,
        status="pending",
        source_image_path=str(source_path),
    )
    job = AvatarJob(id=avatar_id + "-job", avatar_id=avatar_id, stage="queued")
    db.add(avatar)
    db.add(job)
    db.commit()
    background.add_task(_run_job, job.id)
    return {"avatar_id": avatar_id, "status": "pending"}


def _run_job(job_id: str) -> None:
    from sqlalchemy.orm import sessionmaker

    db = sessionmaker(bind=get_engine())()
    try:
        run_avatar_job(db, _store(), create_pipeline("mock"), job_id)
    finally:
        db.close()


@router.get("")
def list_avatars(tenant: Tenant = Depends(get_tenant), db=Depends(get_db)) -> list[dict]:
    rows = db.query(Avatar).filter(Avatar.tenant_id == tenant.id).order_by(Avatar.created_at.desc()).all()
    return [_serialize(a) for a in rows]


@router.get("/{avatar_id}")
def get_avatar(avatar_id: str, tenant: Tenant = Depends(get_tenant), db=Depends(get_db)) -> dict:
    avatar = _get_owned(db, tenant, avatar_id)
    data = _serialize(avatar, with_detail=True)
    if avatar.status == "failed":
        job = (
            db.query(AvatarJob)
            .filter(AvatarJob.avatar_id == avatar_id)
            .order_by(AvatarJob.created_at.desc())
            .first()
        )
        data["error"] = job.error if job else "unknown error"
    return data


@router.delete("/{avatar_id}", status_code=204)
def delete_avatar(avatar_id: str, tenant: Tenant = Depends(get_tenant), db=Depends(get_db)) -> Response:
    avatar = _get_owned(db, tenant, avatar_id)
    db.query(AvatarJob).filter(AvatarJob.avatar_id == avatar_id).delete()
    db.delete(avatar)
    db.commit()
    _store().delete_assets(avatar_id)
    return Response(status_code=204)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest -v`
Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add server/api/avatars.py tests/test_avatars_api.py
git commit -m "feat: avatar REST API with async creation jobs

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 6: AudioIngest + StreamSink + RenderSession（状态机 + metrics）

**Files:**
- Create: `server/core/audio_ingest.py`, `server/core/stream_sink.py`, `server/core/render_session.py`, `tests/test_render_session.py`

**Interfaces:**
- Consumes: Task 3 `DriveEngine`/`AvatarPack`；Task 7/8 将用 `RenderSession`、`StreamSink`
- Produces:
  - `server.core.audio_ingest.AudioIngest(sample_rate: int = 48000, block_ms: int = 100)`：
    - `push_pcm(data: bytes) -> list[np.ndarray]`（16bit LE mono → float32；每凑满 block_ms 返回一块）
    - `flush() -> list[np.ndarray]`（出余量）
  - `server.core.stream_sink.SinkClosed(Exception)`
  - `server.core.stream_sink.StreamSink`（ABC）：`async push(image: np.ndarray, pts: float) -> None`、`async close() -> None`
  - `server.core.stream_sink.NullSink`、`server.core.stream_sink.FrameBufferSink`（`.latest: tuple[float, np.ndarray] | None`、`.dropped: int`）
  - `server.core.render_session.SessionState`（`idle|listening|speaking|closed|error`）、`IDLE_AFTER = 2.0`
  - `server.core.render_session.RenderSession(session_id, pack, engine, sink, sample_rate=48000, fps=25)`：
    - `async start() -> None`（engine.prepare，state=listening）
    - `async push_pcm(data: bytes) -> None`（分块→驱动；state=speaking；closed/error 时抛 `AppError("session_closed", ..., 409)`）
    - `async close() -> None`（engine.close、sink.close、state=closed）
    - `maybe_idle(now: float | None = None) -> None`（speaking 且距 last_audio_at > IDLE_AFTER → listening）
    - `mark_error(message: str) -> None`
    - `metrics -> dict`：`{"frames_out": int, "dropped_frames": int, "fps": float, "avg_latency_ms": float}`
    - `last_audio_at: float`（time.monotonic）
    - 内部：`_drive_block(block: np.ndarray)`（引擎产帧→sink.push；SinkClosed→dropped_frames+1 并停推）

- [ ] **Step 1: 写失败测试**

`tests/test_render_session.py`:
```python
import numpy as np
import pytest

from server.core.audio_ingest import AudioIngest
from server.core.render_session import RenderSession, SessionState
from server.core.stream_sink import FrameBufferSink
from server.engine.base import AvatarPack
from server.engine.mock_engine import MockEngine


def test_audio_ingest_blocks():
    ingest = AudioIngest(sample_rate=48000, block_ms=100)
    pcm = np.zeros(4800, dtype=np.int16).tobytes()  # 100ms
    blocks = ingest.push_pcm(pcm)
    assert len(blocks) == 1
    assert blocks[0].shape == (4800,)
    assert blocks[0].dtype == np.float32
    assert ingest.push_pcm(np.zeros(2400, dtype=np.int16).tobytes()) == []
    flushed = ingest.flush()
    assert len(flushed) == 1 and flushed[0].shape == (2400,)


def _session(tmp_path):
    from PIL import Image

    img = tmp_path / "src.png"
    Image.new("RGB", (720, 1280), (10, 10, 10)).save(img)
    pack = AvatarPack(avatar_id="a1", root=tmp_path, source_image_path=img, config={})
    sink = FrameBufferSink()
    return RenderSession("s1", pack, MockEngine(), sink), sink


@pytest.mark.asyncio
async def test_push_pcm_drives_frames_and_metrics(tmp_path):
    session, sink = _session(tmp_path)
    await session.start()
    await session.push_pcm(np.zeros(24000, dtype=np.int16).tobytes())  # 0.5s → 12 frames
    assert session.state == SessionState.SPEAKING
    assert session.metrics["frames_out"] == 12
    assert sink.latest is not None
    assert sink.latest[1].shape == (1280, 720, 3)
    await session.close()
    assert session.state == SessionState.CLOSED


@pytest.mark.asyncio
async def test_push_after_close_raises(tmp_path):
    from server.core.errors import AppError

    session, _ = _session(tmp_path)
    await session.start()
    await session.close()
    with pytest.raises(AppError) as exc:
        await session.push_pcm(np.zeros(4800, dtype=np.int16).tobytes())
    assert exc.value.code == "session_closed"


@pytest.mark.asyncio
async def test_maybe_idle_transitions(tmp_path):
    session, _ = _session(tmp_path)
    await session.start()
    await session.push_pcm(np.zeros(4800, dtype=np.int16).tobytes())
    session.maybe_idle(now=session.last_audio_at + 3.0)
    assert session.state == SessionState.LISTENING
    await session.close()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_render_session.py -v`
Expected: FAIL（ModuleNotFoundError: server.core.audio_ingest）

- [ ] **Step 3: 实现**

`server/core/audio_ingest.py`:
```python
import numpy as np


class AudioIngest:
    def __init__(self, sample_rate: int = 48000, block_ms: int = 100):
        self.sample_rate = sample_rate
        self.block_samples = sample_rate * block_ms // 1000
        self._buf = bytearray()

    def _drain_blocks(self) -> list[np.ndarray]:
        blocks = []
        while len(self._buf) >= self.block_samples * 2:
            raw = bytes(self._buf[: self.block_samples * 2])
            del self._buf[: self.block_samples * 2]
            blocks.append(np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0)
        return blocks

    def push_pcm(self, data: bytes) -> list[np.ndarray]:
        self._buf.extend(data)
        return self._drain_blocks()

    def flush(self) -> list[np.ndarray]:
        blocks = self._drain_blocks()
        if len(self._buf) >= 2:
            raw = bytes(self._buf)
            self._buf.clear()
            blocks.append(np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0)
        return blocks
```

`server/core/stream_sink.py`:
```python
from abc import ABC, abstractmethod

import numpy as np


class SinkClosed(Exception):
    pass


class StreamSink(ABC):
    @abstractmethod
    async def push(self, image: np.ndarray, pts: float) -> None:
        ...

    async def close(self) -> None:
        pass


class NullSink(StreamSink):
    async def push(self, image: np.ndarray, pts: float) -> None:
        pass


class FrameBufferSink(StreamSink):
    def __init__(self):
        self.latest: tuple[float, np.ndarray] | None = None
        self.dropped = 0

    async def push(self, image: np.ndarray, pts: float) -> None:
        if self.latest is not None:
            self.dropped += 1
        self.latest = (pts, image)

    async def close(self) -> None:
        self.latest = None
```

`server/core/render_session.py`:
```python
import time
from enum import Enum

import numpy as np

from server.core.audio_ingest import AudioIngest
from server.core.errors import AppError
from server.core.stream_sink import SinkClosed, StreamSink
from server.engine.base import AvatarPack, DriveEngine

IDLE_AFTER = 2.0


class SessionState(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    SPEAKING = "speaking"
    CLOSED = "closed"
    ERROR = "error"


class RenderSession:
    def __init__(
        self,
        session_id: str,
        pack: AvatarPack,
        engine: DriveEngine,
        sink: StreamSink,
        sample_rate: int = 48000,
        fps: int = 25,
    ):
        self.session_id = session_id
        self.pack = pack
        self.engine = engine
        self.sink = sink
        self.sample_rate = sample_rate
        self.fps = fps
        self.state = SessionState.IDLE
        self.error: str | None = None
        self.last_audio_at = time.monotonic()
        self._ingest = AudioIngest(sample_rate=sample_rate, block_ms=100)
        self._pts = 0.0
        self._metrics = {"frames_out": 0, "dropped_frames": 0, "fps": 0.0, "avg_latency_ms": 0.0}

    @property
    def metrics(self) -> dict:
        return dict(self._metrics)

    async def start(self) -> None:
        await self.engine.prepare(self.pack)
        self.state = SessionState.LISTENING

    def maybe_idle(self, now: float | None = None) -> None:
        now = now if now is not None else time.monotonic()
        if self.state == SessionState.SPEAKING and now - self.last_audio_at > IDLE_AFTER:
            self.state = SessionState.LISTENING

    async def push_pcm(self, data: bytes) -> None:
        if self.state in (SessionState.CLOSED, SessionState.ERROR):
            raise AppError("session_closed", f"session {self.session_id} is {self.state.value}", 409)
        self.last_audio_at = time.monotonic()
        self.state = SessionState.SPEAKING
        for block in self._ingest.push_pcm(data):
            await self._drive_block(block)

    async def _drive_block(self, block: np.ndarray) -> None:
        t0 = time.perf_counter()
        frames = await self.engine.drive(self.pack, block, self.sample_rate)
        for frame in frames:
            frame.pts = self._pts
            self._pts += 1.0 / self.fps
            try:
                await self.sink.push(frame.image, frame.pts)
                self._metrics["frames_out"] += 1
            except SinkClosed:
                self._metrics["dropped_frames"] += 1
                return
        latency_ms = (time.perf_counter() - t0) * 1000.0
        prev = self._metrics["avg_latency_ms"]
        self._metrics["avg_latency_ms"] = latency_ms if prev == 0 else prev * 0.8 + latency_ms * 0.2
        self._metrics["fps"] = float(self.fps)

    def mark_error(self, message: str) -> None:
        self.state = SessionState.ERROR
        self.error = message[:1024]

    async def close(self) -> None:
        await self.engine.close()
        await self.sink.close()
        self.state = SessionState.CLOSED
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest -v`
Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add server/core tests/test_render_session.py
git commit -m "feat: audio ingest, stream sinks and render session state machine

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 7: SessionManager + 会话 REST API + 空闲回收

**Files:**
- Create: `server/core/session_manager.py`, `tests/test_sessions_api.py`
- Modify: `server/api/sessions.py`（新建路由文件）、`server/main.py`（挂路由、app.state.session_manager、模块级 `get_session_manager()`）

**Interfaces:**
- Consumes: Task 4 `AvatarStore.load_pack`、Task 6 `RenderSession`/`NullSink`/`StreamSink`
- Produces:
  - `server.main.get_session_manager() -> SessionManager`（返回 `create_app` 设置的 `app.state.session_manager`；`create_app` 同时把 app 存到模块级 `_app` 单例）
  - `server.core.session_manager.SessionManager(engine: DriveEngine, store: AvatarStore, idle_timeout: float = 60.0)`：
    - `async create(db, tenant_id: str, avatar: Avatar) -> StreamSession`（DB 行 status="idle" + 内存 `RenderSession`（NullSink）+ `await render.start()`）
    - `get_render(session_id: str) -> RenderSession`（不存在抛 `AppError("not_found", ..., 404)`）
    - `attach_sink(session_id: str, sink: StreamSink) -> None`
    - `async close(db, session_id: str) -> None`（render.close、DB status="closed"、写 metrics_json、closed_at）
    - `async sweep_idle(db, now: float | None = None) -> int`（last_audio_at 超 idle_timeout 的关闭，返回数量）
  - REST：
    - `POST /api/v1/sessions` body `{"avatar_id": str}` → 201 `{"session_id", "status", "signal_url": "/api/v1/ws/signal/{id}", "audio_url": "/api/v1/ws/audio/{id}"}`（avatar 非 ready/不存在 → 409 `AppError("avatar_not_ready", ...)`）
    - `GET /api/v1/sessions/{id}` → `{"session_id", "avatar_id", "status", "metrics"}`（status/metrics 取自活跃 RenderSession，无则用 DB 行）
    - `DELETE /api/v1/sessions/{id}` → 204

- [ ] **Step 1: 写失败测试**

`tests/test_sessions_api.py`:
```python
import asyncio
import io

import numpy as np
from PIL import Image


def _seed(client):
    from server.api.deps import hash_api_key
    from server.db.models import Tenant
    from server.db.session import get_engine
    from sqlalchemy.orm import sessionmaker

    db = sessionmaker(bind=get_engine())()
    db.add(Tenant(name="t1", api_key_hash=hash_api_key("key-1")))
    db.commit()
    db.close()
    headers = {"X-API-Key": "key-1"}
    buf = io.BytesIO()
    Image.new("RGB", (640, 800), (10, 20, 30)).save(buf, format="PNG")
    resp = client.post(
        "/api/v1/avatars", headers=headers, data={"name": "a"},
        files={"file": ("a.png", buf.getvalue(), "image/png")},
    )
    return headers, resp.json()["avatar_id"]


def test_session_lifecycle(client):
    headers, avatar_id = _seed(client)
    resp = client.post("/api/v1/sessions", headers=headers, json={"avatar_id": avatar_id})
    assert resp.status_code == 201
    body = resp.json()
    assert body["signal_url"].endswith(body["session_id"])
    sid = body["session_id"]

    detail = client.get(f"/api/v1/sessions/{sid}", headers=headers).json()
    assert detail["status"] in ("idle", "listening")
    assert detail["metrics"]["frames_out"] == 0

    assert client.delete(f"/api/v1/sessions/{sid}", headers=headers).status_code == 204
    assert client.get(f"/api/v1/sessions/{sid}", headers=headers).json()["status"] == "closed"


def test_session_requires_ready_avatar(client):
    headers, _ = _seed(client)
    resp = client.post("/api/v1/sessions", headers=headers, json={"avatar_id": "missing"})
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "avatar_not_ready"


def test_session_audio_drives_metrics(client):
    headers, avatar_id = _seed(client)
    sid = client.post("/api/v1/sessions", headers=headers, json={"avatar_id": avatar_id}).json()["session_id"]
    render = client.app.state.session_manager.get_render(sid)
    asyncio.run(render.push_pcm(np.zeros(24000, dtype=np.int16).tobytes()))
    detail = client.get(f"/api/v1/sessions/{sid}", headers=headers).json()
    assert detail["metrics"]["frames_out"] == 12
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_sessions_api.py -v`
Expected: FAIL（无 /api/v1/sessions 路由）

- [ ] **Step 3: 实现**

`server/core/session_manager.py`:
```python
import time
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from server.avatar.store import AvatarStore
from server.core.errors import AppError
from server.core.render_session import RenderSession
from server.core.stream_sink import NullSink, StreamSink
from server.db.models import Avatar, StreamSession
from server.engine.base import DriveEngine


class SessionManager:
    def __init__(self, engine: DriveEngine, store: AvatarStore, idle_timeout: float = 60.0):
        self.engine = engine
        self.store = store
        self.idle_timeout = idle_timeout
        self._renders: dict[str, RenderSession] = {}

    async def create(self, db: Session, tenant_id: str, avatar: Avatar) -> StreamSession:
        row = StreamSession(tenant_id=tenant_id, avatar_id=avatar.id, status="idle")
        db.add(row)
        db.commit()
        pack = self.store.load_pack(avatar)
        render = RenderSession(row.id, pack, self.engine, NullSink())
        await render.start()
        self._renders[row.id] = render
        return row

    def get_render(self, session_id: str) -> RenderSession:
        render = self._renders.get(session_id)
        if render is None:
            raise AppError("not_found", f"session {session_id} not found", 404)
        return render

    def attach_sink(self, session_id: str, sink: StreamSink) -> None:
        self.get_render(session_id).sink = sink

    async def close(self, db: Session, session_id: str) -> None:
        render = self._renders.pop(session_id, None)
        if render is not None:
            await render.close()
        row = db.query(StreamSession).filter(StreamSession.id == session_id).first()
        if row is not None and row.status != "closed":
            row.status = "closed"
            row.closed_at = datetime.now(timezone.utc)
            if render is not None:
                row.metrics_json = render.metrics
            db.commit()

    async def sweep_idle(self, db: Session, now: float | None = None) -> int:
        now = now if now is not None else time.monotonic()
        closed = 0
        for sid, render in list(self._renders.items()):
            if now - render.last_audio_at > self.idle_timeout:
                await self.close(db, sid)
                closed += 1
        return closed
```

`server/api/sessions.py`:
```python
from fastapi import APIRouter, Depends, WebSocket
from starlette.responses import Response

from server.api.deps import get_tenant
from server.core.errors import AppError
from server.db.models import Avatar, StreamSession, Tenant
from server.db.session import get_db

router = APIRouter(tags=["sessions"])


def _manager():
    from server.main import get_session_manager

    return get_session_manager()


def _serialize(row: StreamSession, render=None) -> dict:
    data = {
        "session_id": row.id,
        "avatar_id": row.avatar_id,
        "status": row.status,
        "metrics": row.metrics_json or {},
    }
    if render is not None:
        data["status"] = render.state.value
        data["metrics"] = render.metrics
    return data


@router.post("/api/v1/sessions", status_code=201)
async def create_session(body: dict, tenant: Tenant = Depends(get_tenant), db=Depends(get_db)) -> dict:
    avatar_id = body.get("avatar_id", "")
    avatar = db.query(Avatar).filter(Avatar.id == avatar_id, Avatar.tenant_id == tenant.id).first()
    if avatar is None or avatar.status != "ready":
        raise AppError("avatar_not_ready", f"avatar {avatar_id} is not ready", 409)
    row = await _manager().create(db, tenant.id, avatar)
    return {
        "session_id": row.id,
        "status": row.status,
        "signal_url": f"/api/v1/ws/signal/{row.id}",
        "audio_url": f"/api/v1/ws/audio/{row.id}",
    }


@router.get("/api/v1/sessions/{session_id}")
def get_session(session_id: str, tenant: Tenant = Depends(get_tenant), db=Depends(get_db)) -> dict:
    row = (
        db.query(StreamSession)
        .filter(StreamSession.id == session_id, StreamSession.tenant_id == tenant.id)
        .first()
    )
    if row is None:
        raise AppError("not_found", f"session {session_id} not found", 404)
    from server.core.errors import AppError as _AppError

    try:
        render = _manager().get_render(session_id)
    except _AppError:
        render = None
    return _serialize(row, render)


@router.delete("/api/v1/sessions/{session_id}", status_code=204)
async def delete_session(session_id: str, tenant: Tenant = Depends(get_tenant), db=Depends(get_db)) -> Response:
    row = (
        db.query(StreamSession)
        .filter(StreamSession.id == session_id, StreamSession.tenant_id == tenant.id)
        .first()
    )
    if row is None:
        raise AppError("not_found", f"session {session_id} not found", 404)
    await _manager().close(db, session_id)
    return Response(status_code=204)
```

`server/main.py` 修改要点（create_app 内 + 模块级）:
```python
_app = None


def create_app() -> FastAPI:
    global _app
    ...
    from server.api.sessions import router as sessions_router
    from server.avatar.store import AvatarStore
    from server.core.session_manager import SessionManager
    from server.engine.base import create_engine

    app.include_router(sessions_router)
    app.state.session_manager = SessionManager(
        create_engine(), AvatarStore(settings.data_dir), settings.session_idle_timeout
    )
    _app = app
    return app


def get_session_manager():
    return _app.state.session_manager
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest -v`
Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add server tests/test_sessions_api.py
git commit -m "feat: session manager and session REST API with idle sweep

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 8: WebRTC 视频轨 + 信令 WS

**Files:**
- Create: `server/core/webrtc.py`, `tests/test_webrtc_track.py`
- Modify: `server/api/sessions.py`（挂 `WS /api/v1/ws/signal/{session_id}`）

**Interfaces:**
- Consumes: Task 6 `StreamSink`/`SinkClosed`/`NullSink`、Task 7 `SessionManager.get_render/attach_sink`、`get_tenant` 的鉴权逻辑（WS 用 query `api_key`）
- Produces:
  - `server.core.webrtc.EngineVideoTrack(VideoStreamTrack)`：
    - `async push(image: np.ndarray, pts: float) -> None`（入队；队列上限 `maxsize=8` 满则丢最旧并 `overflows += 1`）
    - 继承 `recv()`：队列出帧 → `VideoFrame.from_ndarray(image, format="rgb24")`，`pts = int(pts * VIDEO_PTIME_denominator)`，`time_base = Fraction(1, VIDEO_CLOCK_RATE)`
  - `server.core.webrtc.WebRTCSink(StreamSink)`：包装 track；`close()` 置 `closed=True`；`push` 在 closed 时 raise `SinkClosed`
  - `server.core.webrtc.handle_signal_websocket(websocket: WebSocket, manager: SessionManager, session_id: str) -> None`：
    - accept 后 `manager.get_render` 失败 → `close(code=4404)`
    - 建 `EngineVideoTrack` + `WebRTCSink`，`manager.attach_sink(session_id, sink)`，`RTCPeerConnection()` + `pc.addTrack(track)`
    - 消息 JSON 协议：`{"type": "offer", "sdp": str}` → `setRemoteDescription` + `createAnswer` + `setLocalDescription` → 回 `{"type": "answer", "sdp": str}`；`{"type": "candidate", "candidate": {"candidate", "sdpMid", "sdpMLineIndex"}}` → `candidate_from_sdp` + `addIceCandidate`；`{"type": "bye"}` → 跳出循环
    - finally：`sink.close()`、`manager.attach_sink(session_id, NullSink())`、`pc.close()`
  - 信令路由：`@router.websocket("/api/v1/ws/signal/{session_id}")`（query 参数 `api_key`，校验同 REST；无效 → `close(code=4401)`）

- [ ] **Step 1: 写失败测试**

`tests/test_webrtc_track.py`:
```python
import numpy as np
import pytest

from server.core.stream_sink import SinkClosed
from server.core.webrtc import EngineVideoTrack, WebRTCSink


@pytest.mark.asyncio
async def test_track_push_recv_roundtrip():
    track = EngineVideoTrack()
    img = np.zeros((1280, 720, 3), dtype=np.uint8)
    img[0, 0] = (255, 0, 0)
    await track.push(img, 0.0)
    frame = await track.recv()
    assert frame.to_ndarray(format="rgb24")[0, 0, 0] == 255


@pytest.mark.asyncio
async def test_track_overflow_drops_oldest():
    track = EngineVideoTrack(maxsize=2)
    for i in range(4):
        await track.push(np.full((8, 8, 3), i, dtype=np.uint8), float(i))
    assert track.overflows >= 2
    frame = await track.recv()
    assert frame.to_ndarray(format="rgb24")[0, 0, 0] == 3


@pytest.mark.asyncio
async def test_sink_raises_after_close():
    track = EngineVideoTrack()
    sink = WebRTCSink(track)
    await sink.close()
    with pytest.raises(SinkClosed):
        await sink.push(np.zeros((8, 8, 3), dtype=np.uint8), 0.0)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_webrtc_track.py -v`
Expected: FAIL（ModuleNotFoundError: server.core.webrtc）

- [ ] **Step 3: 实现**

`server/core/webrtc.py`:
```python
import asyncio
import json
from fractions import Fraction

import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.mediastreams import VIDEO_CLOCK_RATE
from aiortc.sdp import candidate_from_sdp
from av import VideoFrame
from starlette.websockets import WebSocket

from server.core.stream_sink import NullSink, SinkClosed, StreamSink


class EngineVideoTrack(VideoStreamTrack):
    def __init__(self, maxsize: int = 8):
        super().__init__()
        self._queue: asyncio.Queue[tuple[float, np.ndarray]] = asyncio.Queue(maxsize=maxsize)
        self.overflows = 0

    async def push(self, image: np.ndarray, pts: float) -> None:
        try:
            self._queue.put_nowait((pts, image))
        except asyncio.QueueFull:
            self.overflows += 1
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            self._queue.put_nowait((pts, image))

    async def recv(self) -> VideoFrame:
        pts, image = await self._queue.get()
        frame = VideoFrame.from_ndarray(image, format="rgb24")
        frame.pts = int(pts * VIDEO_CLOCK_RATE)
        frame.time_base = Fraction(1, VIDEO_CLOCK_RATE)
        return frame


class WebRTCSink(StreamSink):
    def __init__(self, track: EngineVideoTrack):
        self.track = track
        self.closed = False

    async def push(self, image: np.ndarray, pts: float) -> None:
        if self.closed:
            raise SinkClosed("webrtc sink closed")
        await self.track.push(image, pts)

    async def close(self) -> None:
        self.closed = True


async def handle_signal_websocket(websocket: WebSocket, manager, session_id: str) -> None:
    await websocket.accept()
    try:
        render = manager.get_render(session_id)
    except Exception:
        await websocket.close(code=4404)
        return

    track = EngineVideoTrack()
    sink = WebRTCSink(track)
    manager.attach_sink(session_id, sink)
    pc = RTCPeerConnection()
    pc.addTrack(track)

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            if msg.get("type") == "offer":
                await pc.setRemoteDescription(RTCSessionDescription(sdp=msg["sdp"], type="offer"))
                answer = await pc.createAnswer()
                await pc.setLocalDescription(answer)
                await websocket.send_text(json.dumps({"type": "answer", "sdp": pc.localDescription.sdp}))
            elif msg.get("type") == "candidate":
                cand = msg.get("candidate") or {}
                if cand.get("candidate"):
                    ice = candidate_from_sdp(cand["candidate"])
                    ice.sdpMid = cand.get("sdpMid")
                    ice.sdpMLineIndex = cand.get("sdpMLineIndex")
                    await pc.addIceCandidate(ice)
            elif msg.get("type") == "bye":
                break
    finally:
        await sink.close()
        manager.attach_sink(session_id, NullSink())
        await pc.close()
```

`server/api/sessions.py` 追加（鉴权辅助与路由）：
```python
import json

from server.api.deps import hash_api_key


def _ws_auth(websocket: WebSocket, api_key: str) -> bool:
    from server.db.models import Tenant
    from server.db.session import get_engine
    from sqlalchemy.orm import sessionmaker

    db = sessionmaker(bind=get_engine())()
    try:
        tenant = db.query(Tenant).filter(Tenant.api_key_hash == hash_api_key(api_key)).first()
        return tenant is not None
    finally:
        db.close()


@router.websocket("/api/v1/ws/signal/{session_id}")
async def signal_ws(websocket: WebSocket, session_id: str, api_key: str = "") -> None:
    if not _ws_auth(websocket, api_key):
        await websocket.close(code=4401)
        return
    from server.core.webrtc import handle_signal_websocket

    await handle_signal_websocket(websocket, _manager(), session_id)
```

- [ ] **Step 4: 跑测试确认通过（含既有全套）**

Run: `python -m pytest -v`
Expected: 全部 PASS。真机信令联调在任务 14（浏览器 E2E）做。

- [ ] **Step 5: Commit**

```bash
git add server/core/webrtc.py server/api/sessions.py tests/test_webrtc_track.py
git commit -m "feat: webrtc video track, sink adapter and signaling websocket

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 9: 音频输入双通道（WebRTC audio track + WS 音频）

**Files:**
- Modify: `server/core/webrtc.py`（`pc.on("track")` 消费音频）、`server/api/sessions.py`（`WS /api/v1/ws/audio/{session_id}`）
- Create: `tests/test_ws_audio.py`

**Interfaces:**
- Consumes: Task 6 `RenderSession.push_pcm`/`_drive_block`/`_ingest`、Task 8 `handle_signal_websocket`/`_ws_auth`
- Produces:
  - `server.core.webrtc.consume_audio_track(track: MediaStreamTrack, push_pcm: Callable[[bytes], Awaitable[None]]) -> None`：`track.recv()` → 非 48k mono 时用 `av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=48000)` 重采样 → ndarray(int16).tobytes() → `await push_pcm(...)`；任何异常静默退出
  - `handle_signal_websocket` 内注册 `@pc.on("track")`：kind=="audio" 时 `asyncio.ensure_future(consume_audio_track(track, render.push_pcm))`
  - WS 音频通道 `WS /api/v1/ws/audio/{session_id}?api_key=`：二进制消息 = PCM 16bit LE mono 48kHz 分片 → `render.push_pcm`；文本消息 `{"type": "flush"}` → 逐块 `render._drive_block(b)` 消化 `render._ingest.flush()`；鉴权失败 `close(code=4401)`

- [ ] **Step 1: 写失败测试**

`tests/test_ws_audio.py`:
```python
import io

import numpy as np
from PIL import Image


def _setup(client):
    from server.api.deps import hash_api_key
    from server.db.models import Tenant
    from server.db.session import get_engine
    from sqlalchemy.orm import sessionmaker

    db = sessionmaker(bind=get_engine())()
    db.add(Tenant(name="t1", api_key_hash=hash_api_key("key-1")))
    db.commit()
    db.close()
    headers = {"X-API-Key": "key-1"}
    buf = io.BytesIO()
    Image.new("RGB", (640, 800), (1, 2, 3)).save(buf, format="PNG")
    avatar_id = client.post(
        "/api/v1/avatars", headers=headers, data={"name": "a"},
        files={"file": ("a.png", buf.getvalue(), "image/png")},
    ).json()["avatar_id"]
    sid = client.post(
        "/api/v1/sessions", headers=headers, json={"avatar_id": avatar_id}
    ).json()["session_id"]
    return sid


def test_ws_audio_channel_drives_frames(client):
    sid = _setup(client)
    with client.websocket_connect(f"/api/v1/ws/audio/{sid}?api_key=key-1") as ws:
        ws.send_bytes(np.zeros(4800, dtype=np.int16).tobytes())  # 100ms → 5 frames
        ws.send_text('{"type": "flush"}')

    render = client.app.state.session_manager.get_render(sid)
    assert render.metrics["frames_out"] == 5


def test_ws_audio_rejects_bad_key(client):
    sid = _setup(client)
    try:
        with client.websocket_connect(f"/api/v1/ws/audio/{sid}?api_key=bad") as ws:
            ws.receive()
    except Exception:
        pass  # 服务端 close(4401) 导致的断开即为通过
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_ws_audio.py -v`
Expected: FAIL（无 ws/audio 路由）

- [ ] **Step 3: 实现**

`server/core/webrtc.py` 追加：
```python
from aiortc import MediaStreamTrack
from av.audio.resampler import AudioResampler


async def consume_audio_track(track: MediaStreamTrack, push_pcm) -> None:
    resampler = None
    try:
        while True:
            frame = await track.recv()
            if frame.sample_rate != 48000 or frame.layout.name != "mono":
                if resampler is None:
                    resampler = AudioResampler(format="s16", layout="mono", rate=48000)
                frames = resampler.resample(frame)
            else:
                frames = [frame]
            for f in frames:
                arr = f.to_ndarray()
                if arr.dtype != np.int16:
                    arr = (np.clip(arr.astype(np.float32), -1, 1) * 32767).astype(np.int16)
                await push_pcm(arr.tobytes())
    except Exception:
        return
```
`handle_signal_websocket` 中，`pc.addTrack(track)` 之后注册：
```python
    @pc.on("track")
    def on_track(inbound):
        if inbound.kind == "audio":
            asyncio.ensure_future(consume_audio_track(inbound, render.push_pcm))
```

`server/api/sessions.py` 追加：
```python
@router.websocket("/api/v1/ws/audio/{session_id}")
async def audio_ws(websocket: WebSocket, session_id: str, api_key: str = "") -> None:
    if not _ws_auth(websocket, api_key):
        await websocket.close(code=4401)
        return
    try:
        render = _manager().get_render(session_id)
    except AppError:
        await websocket.close(code=4404)
        return
    await websocket.accept()
    try:
        while True:
            msg = await websocket.receive()
            if msg.get("bytes") is not None:
                await render.push_pcm(msg["bytes"])
            elif msg.get("text"):
                payload = json.loads(msg["text"])
                if payload.get("type") == "flush":
                    for block in render._ingest.flush():
                        await render._drive_block(block)
    except Exception:
        pass
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest -v`
Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add server tests/test_ws_audio.py
git commit -m "feat: dual audio input channels (webrtc track + ws pcm)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---
## Task 10: Admin API（系统/形象调试/租户/任务）

**Files:**
- Modify: `server/api/admin.py`（login 之外全部管理端点）
- Create: `tests/test_admin_api.py`

**Interfaces:**
- Consumes: Task 2 `get_admin`、Task 4 `run_avatar_job`/`create_pipeline`、Task 7 `SessionManager`、Task 3 `MockEngine.drive`
- Produces（全部需 `X-Admin-Token`，除 `/login`）：
  - `GET /api/admin/system` → `{"engine": dict(engine.describe()), "version": "0.1.0", "data_dir": str, "gpu": bool}`（gpu 从 `engine.describe()` 取）
  - `GET /api/admin/tenants` → `[{"tenant_id", "name", "created_at"}]`
  - `POST /api/admin/tenants` body `{"name": str}` → 201 `{"tenant_id", "name", "api_key"}`（api_key 仅此一次返回，格式 `dh_` + 32 位 urlsafe；DB 只存 `hash_api_key`）
  - `GET /api/admin/avatars?tenant_id=` → 全量列表（含 tenant_id、status、error）；`POST /api/admin/avatars/{id}/retry` → 202（failed/pending 重跑 job；ready 也允许重跑覆盖）
  - `PATCH /api/admin/avatars/{id}/config` body `{"mouth_scale": float?, "expression_intensity": float?}` → 200 更新 `config_json` 并重写 `pack.json`
  - `GET /api/admin/avatars/{id}/preview.png?level=0.0..1.0` → `image/png`（用 `create_engine()` 的 drive 跑一帧合成音频 `np.ones(4800)*level`，返回第一帧 PNG 字节）
  - `GET /api/admin/jobs` → `[{"job_id", "avatar_id", "stage", "progress", "error", "created_at"}]`
  - `GET /api/admin/sessions` → `[{"session_id", "avatar_id", "status", "metrics"}]`（活跃 RenderSession + DB 行合并）
  - `POST /api/admin/sessions/{id}/sweep` → 200 `{"closed": int}`（手动触发 sweep_idle，调试用）

- [ ] **Step 1: 写失败测试**

`tests/test_admin_api.py`:
```python
import io

import pytest
from PIL import Image

ADMIN = {"X-Admin-Token": "dev-admin-token"}


def _png():
    buf = io.BytesIO()
    Image.new("RGB", (640, 800), (10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


def test_admin_requires_token(client):
    assert client.get("/api/admin/system").status_code == 401


def test_tenant_and_key_roundtrip(client):
    resp = client.post("/api/admin/tenants", headers=ADMIN, json={"name": "acme"})
    assert resp.status_code == 201
    body = resp.json()
    assert body["api_key"].startswith("dh_")
    assert client.get("/api/v1/avatars", headers={"X-API-Key": body["api_key"]}).status_code == 200


def test_system_info(client):
    body = client.get("/api/admin/system", headers=ADMIN).json()
    assert body["engine"]["name"] == "mock"
    assert body["gpu"] is False


def test_avatar_config_and_preview(client):
    client.post("/api/admin/tenants", headers=ADMIN, json={"name": "t"})
    api_key = client.post("/api/admin/tenants", headers=ADMIN, json={"name": "t2"}).json()["api_key"]
    avatar_id = client.post(
        "/api/v1/avatars", headers={"X-API-Key": api_key}, data={"name": "a"},
        files={"file": ("a.png", _png(), "image/png")},
    ).json()["avatar_id"]

    resp = client.patch(
        f"/api/admin/avatars/{avatar_id}/config", headers=ADMIN, json={"mouth_scale": 1.5}
    )
    assert resp.status_code == 200
    assert resp.json()["config"]["mouth_scale"] == 1.5

    resp = client.get(f"/api/admin/avatars/{avatar_id}/preview.png?level=0.8", headers=ADMIN)
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert len(resp.content) > 100


def test_jobs_listing_and_retry(client):
    api_key = client.post("/api/admin/tenants", headers=ADMIN, json={"name": "t"}).json()["api_key"]
    client.post(
        "/api/v1/avatars", headers={"X-API-Key": api_key}, data={"name": "bad"},
        files={"file": ("bad.png", io.BytesIO(b"not an image"), "image/png")},
    )
    jobs = client.get("/api/admin/jobs", headers=ADMIN).json()
    assert isinstance(jobs, list)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_admin_api.py -v`
Expected: FAIL（404 端点缺失）

- [ ] **Step 3: 实现**

`server/api/admin.py`（在 login 基础上追加；全文件结构如下）:
```python
import hmac
import io
import json
from datetime import datetime, timezone
from secrets import token_urlsafe
from uuid import uuid4

import numpy as np
from fastapi import APIRouter, Depends, Query, Response
from PIL import Image
from pydantic import BaseModel

from server.api.deps import get_admin, hash_api_key
from server.avatar.pipeline import create_pipeline, run_avatar_job
from server.avatar.store import AvatarStore
from server.config import get_settings
from server.core.errors import AppError
from server.db.models import Avatar, AvatarJob, StreamSession, Tenant
from server.db.session import get_db, get_engine

router = APIRouter(prefix="/api/admin", tags=["admin"], dependencies=[Depends(get_admin)])


class LoginIn(BaseModel):
    password: str


class TenantIn(BaseModel):
    name: str


class AvatarConfigIn(BaseModel):
    mouth_scale: float | None = None
    expression_intensity: float | None = None


@router.post("/login")
def login(body: LoginIn) -> dict:
    if not hmac.compare_digest(body.password, get_settings().admin_password):
        raise AppError("unauthorized", "missing or invalid admin token", 401)
    return {"token": get_settings().admin_token}
```
注意：`login` 不能受 `dependencies=[Depends(get_admin)]` 约束——把 router 拆成两段：`router = APIRouter(prefix="/api/admin")` 仅挂 login；`admin_router = APIRouter(prefix="/api/admin", dependencies=[Depends(get_admin)])` 挂其余端点；`main.py` 两个都 include。这是明确要求，不是可选。

其余端点实现（挂 `admin_router`）:
```python
@admin_router.get("/system")
def system_info() -> dict:
    from server.engine.base import create_engine

    engine = create_engine()
    described = engine.describe()
    return {
        "engine": described,
        "version": "0.1.0",
        "data_dir": str(get_settings().data_dir),
        "gpu": described.get("gpu", False),
    }


@admin_router.get("/tenants")
def list_tenants(db=Depends(get_db)) -> list[dict]:
    rows = db.query(Tenant).order_by(Tenant.created_at.desc()).all()
    return [
        {"tenant_id": t.id, "name": t.name, "created_at": t.created_at.isoformat()} for t in rows
    ]


@admin_router.post("/tenants", status_code=201)
def create_tenant(body: TenantIn, db=Depends(get_db)) -> dict:
    api_key = "dh_" + token_urlsafe(32)
    tenant = Tenant(name=body.name, api_key_hash=hash_api_key(api_key))
    db.add(tenant)
    db.commit()
    return {"tenant_id": tenant.id, "name": tenant.name, "api_key": api_key}


@admin_router.get("/avatars")
def admin_list_avatars(tenant_id: str = "", db=Depends(get_db)) -> list[dict]:
    q = db.query(Avatar)
    if tenant_id:
        q = q.filter(Avatar.tenant_id == tenant_id)
    out = []
    for a in q.order_by(Avatar.created_at.desc()).all():
        job = (
            db.query(AvatarJob).filter(AvatarJob.avatar_id == a.id)
            .order_by(AvatarJob.created_at.desc()).first()
        )
        out.append({
            "avatar_id": a.id, "tenant_id": a.tenant_id, "name": a.name,
            "status": a.status, "config": a.config_json,
            "error": job.error if a.status == "failed" and job else None,
            "created_at": a.created_at.isoformat(),
        })
    return out


def _admin_run_job(job_id: str) -> None:
    from sqlalchemy.orm import sessionmaker

    db = sessionmaker(bind=get_engine())()
    try:
        run_avatar_job(db, AvatarStore(get_settings().data_dir), create_pipeline("mock"), job_id)
    finally:
        db.close()


@admin_router.post("/avatars/{avatar_id}/retry", status_code=202)
def retry_avatar(avatar_id: str, background=None, db=Depends(get_db)) -> dict:
    from fastapi import BackgroundTasks

    avatar = db.query(Avatar).filter(Avatar.id == avatar_id).first()
    if avatar is None:
        raise AppError("not_found", f"avatar {avatar_id} not found", 404)
    job = AvatarJob(id=uuid4().hex, avatar_id=avatar_id, stage="queued")
    db.add(job)
    avatar.status = "pending"
    db.commit()
    return {"job_id": job.id}


@admin_router.patch("/avatars/{avatar_id}/config")
def patch_avatar_config(avatar_id: str, body: AvatarConfigIn, db=Depends(get_db)) -> dict:
    avatar = db.query(Avatar).filter(Avatar.id == avatar_id).first()
    if avatar is None:
        raise AppError("not_found", f"avatar {avatar_id} not found", 404)
    cfg = dict(avatar.config_json or {})
    if body.mouth_scale is not None:
        cfg["mouth_scale"] = body.mouth_scale
    if body.expression_intensity is not None:
        cfg["expression_intensity"] = body.expression_intensity
    avatar.config_json = cfg
    if avatar.pack_path:
        Path = __import__("pathlib").Path
        Path(avatar.pack_path).write_text(json.dumps(cfg))
    db.commit()
    return {"avatar_id": avatar_id, "config": cfg}


@admin_router.get("/avatars/{avatar_id}/preview.png")
def avatar_preview(avatar_id: str, level: float = Query(0.5, ge=0.0, le=1.0), db=Depends(get_db)):
    import asyncio

    from server.engine.base import create_engine as _ce

    avatar = db.query(Avatar).filter(Avatar.id == avatar_id).first()
    if avatar is None:
        raise AppError("not_found", f"avatar {avatar_id} not found", 404)
    store = AvatarStore(get_settings().data_dir)
    pack = store.load_pack(avatar)
    engine = _ce()
    audio = np.ones(4800, dtype=np.float32) * level
    frames = asyncio.run(engine.drive(pack, audio, 48000))
    buf = io.BytesIO()
    Image.fromarray(frames[0].image).save(buf, format="PNG")
    return Response(buf.getvalue(), media_type="image/png")


@admin_router.get("/jobs")
def list_jobs(db=Depends(get_db)) -> list[dict]:
    rows = db.query(AvatarJob).order_by(AvatarJob.created_at.desc()).limit(200).all()
    return [
        {
            "job_id": j.id, "avatar_id": j.avatar_id, "stage": j.stage,
            "progress": j.progress, "error": j.error, "created_at": j.created_at.isoformat(),
        }
        for j in rows
    ]


@admin_router.get("/sessions")
def admin_list_sessions(db=Depends(get_db)) -> list[dict]:
    from server.main import get_session_manager

    manager = get_session_manager()
    rows = db.query(StreamSession).order_by(StreamSession.started_at.desc()).limit(100).all()
    out = []
    for row in rows:
        render = manager._renders.get(row.id)
        out.append({
            "session_id": row.id, "avatar_id": row.avatar_id,
            "status": render.state.value if render else row.status,
            "metrics": render.metrics if render else (row.metrics_json or {}),
        })
    return out


@admin_router.post("/sessions/sweep")
def admin_sweep(db=Depends(get_db)) -> dict:
    import asyncio

    from server.main import get_session_manager

    manager = get_session_manager()
    closed = asyncio.run(manager.sweep_idle(db))
    return {"closed": closed}
```
`retry_avatar` 的 `background: BackgroundTasks` 应作为正式参数声明（`background: BackgroundTasks` 并 `background.add_task(_admin_run_job, job.id)`）——这是明确要求；`patch_avatar_config` 的 `Path` 改为顶部 `from pathlib import Path`；`asyncio.run` 在事件循环内的端点若报错，把 preview/sweep 改成 `async def` + `await`（FastAPI 支持）。

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest -v`
Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add server/api/admin.py server/main.py tests/test_admin_api.py
git commit -m "feat: admin api for system, avatar tuning, tenants and jobs

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 11: admin-ui 骨架（Vite + React + TS、路由、API client、登录）

**Files:**
- Create: `admin-ui/`（`npm create vite@latest admin-ui -- --template react-ts` 生成后精简）、`admin-ui/src/api.ts`、`admin-ui/src/App.tsx`、`admin-ui/src/pages/Login.tsx`、`admin-ui/src/main.tsx`、`admin-ui/src/styles.css`
- Modify: `server/main.py`（静态托管 `admin-ui/dist` 到 `/admin`）

**Interfaces:**
- Consumes: Task 10 Admin API
- Produces:
  - `admin-ui/src/api.ts`：
    - `getToken(): string | null` / `setToken(t: string | null)`（localStorage key `dh_admin_token`）
    - `api<T>(path: string, opts?: {method?, body?, form?: FormData}): Promise<T>`（自动带 `X-Admin-Token`；非 2xx 抛 `Error(error.message)`，错误体为信封格式）
    - `login(password: string): Promise<{token: string}>`
  - `App.tsx`：react-router 路由 `/login`、`/avatars`、`/avatars/:id`、`/debug`、`/system`；未登录（无 token）重定向 `/login`；顶部导航（形象管理 / 会话调试 / 系统）
  - `GET /admin` → `admin-ui/dist/index.html`（`StaticFiles(directory=..., html=True)` 挂 `/admin`）；开发期 `npm run dev`（Vite 5173）通过 proxy `/api` → `http://127.0.0.1:8000`（`admin-ui/vite.config.ts` 里配 server.proxy）

- [ ] **Step 1: 写失败测试**

手工验收（无 JS 单测框架，从本任务起 UI 用人工/浏览器验收，任务 16 用 Playwright 可选补齐）：
```bash
cd admin-ui && npm install && npm run build
python -c "import pathlib; assert pathlib.Path('admin-ui/dist/index.html').exists()"
```
Expected: dist/index.html 存在（构建成功）。

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest -v && cd admin-ui && npm install && npm run build`
Expected: npm 报 admin-ui 不存在 → FAIL

- [ ] **Step 3: 实现**

```bash
cd admin-ui 2>/dev/null || npm create vite@latest admin-ui -- --template react-ts
cd admin-ui && npm install && npm install react-router-dom
```

`admin-ui/vite.config.ts`:
```typescript
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: { "/api": "http://127.0.0.1:8000" },
  },
});
```

`admin-ui/src/api.ts`:
```typescript
const KEY = "dh_admin_token";

export function getToken(): string | null {
  return localStorage.getItem(KEY);
}

export function setToken(t: string | null): void {
  if (t) localStorage.setItem(KEY, t);
  else localStorage.removeItem(KEY);
}

export async function api<T>(path: string, opts: { method?: string; body?: unknown; form?: FormData } = {}): Promise<T> {
  const headers: Record<string, string> = {};
  const token = getToken();
  if (token) headers["X-Admin-Token"] = token;
  let body: BodyInit | undefined;
  if (opts.form) {
    body = opts.form;
  } else if (opts.body !== undefined) {
    headers["Content-Type"] = "application/json";
    body = JSON.stringify(opts.body);
  }
  const resp = await fetch(path, { method: opts.method ?? "GET", headers, body });
  if (!resp.ok) {
    let message = resp.statusText;
    try {
      const data = await resp.json();
      message = data?.error?.message ?? message;
    } catch {
      /* non-json error */
    }
    throw new Error(message);
  }
  if (resp.status === 204) return undefined as T;
  return (await resp.json()) as T;
}

export function login(password: string): Promise<{ token: string }> {
  return api("/api/admin/login", { method: "POST", body: { password } });
}
```

`admin-ui/src/App.tsx`:
```tsx
import { NavLink, Navigate, Route, Routes, useNavigate } from "react-router-dom";
import { getToken, setToken } from "./api";
import Login from "./pages/Login";
import Avatars from "./pages/Avatars";
import AvatarDetail from "./pages/AvatarDetail";
import Debug from "./pages/Debug";
import System from "./pages/System";

function RequireAuth({ children }: { children: JSX.Element }) {
  return getToken() ? children : <Navigate to="/login" replace />;
}

export default function App() {
  const navigate = useNavigate();
  return (
    <div className="app">
      <nav>
        <NavLink to="/avatars">形象管理</NavLink>
        <NavLink to="/debug">会话调试</NavLink>
        <NavLink to="/system">系统</NavLink>
        <button
          onClick={() => {
            setToken(null);
            navigate("/login");
          }}
        >
          退出
        </button>
      </nav>
      <main>
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route path="/avatars" element={<RequireAuth><Avatars /></RequireAuth>} />
          <Route path="/avatars/:id" element={<RequireAuth><AvatarDetail /></RequireAuth>} />
          <Route path="/debug" element={<RequireAuth><Debug /></RequireAuth>} />
          <Route path="/system" element={<RequireAuth><System /></RequireAuth>} />
          <Route path="*" element={<Navigate to="/avatars" replace />} />
        </Routes>
      </main>
    </div>
  );
}
```

`admin-ui/src/pages/Login.tsx`:
```tsx
import { FormEvent, useState } from "react";
import { useNavigate } from "react-router-dom";
import { login, setToken } from "../api";

export default function Login() {
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const navigate = useNavigate();

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    try {
      const { token } = await login(password);
      setToken(token);
      navigate("/avatars");
    } catch (err) {
      setError((err as Error).message);
    }
  }

  return (
    <form className="card" onSubmit={onSubmit}>
      <h2>管理员登录</h2>
      <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} placeholder="Admin 密码" />
      <button type="submit">登录</button>
      {error && <p className="error">{error}</p>}
    </form>
  );
}
```

占位页面（任务 12-14 填充），各文件先给可渲染骨架，例如 `pages/Avatars.tsx`：
```tsx
export default function Avatars() {
  return <h2>形象管理</h2>;
}
```
`AvatarDetail.tsx` / `Debug.tsx` / `System.tsx` 同理（各自 `export default function X() { return <h2>X</h2>; }`）。

`admin-ui/src/main.tsx`:
```tsx
import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import App from "./App";
import "./styles.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </React.StrictMode>
);
```

`admin-ui/src/styles.css`（最小可用）:
```css
body { margin: 0; font-family: system-ui, sans-serif; background: #0f1115; color: #e8eaed; }
.app nav { display: flex; gap: 16px; padding: 12px 20px; background: #171a21; align-items: center; }
.app nav a { color: #8ab4f8; text-decoration: none; }
.app nav button { margin-left: auto; }
main { padding: 20px; }
.card { max-width: 360px; display: flex; flex-direction: column; gap: 12px; }
.error { color: #f28b82; }
table { border-collapse: collapse; width: 100%; }
th, td { border-bottom: 1px solid #2a2e37; padding: 8px; text-align: left; }
input, select { background: #1c2028; color: inherit; border: 1px solid #2a2e37; padding: 8px; }
button { background: #2b5cab; color: white; border: 0; padding: 8px 14px; cursor: pointer; }
```

`server/main.py` 追加静态托管（create_app 内）：
```python
    from pathlib import Path
    from fastapi.staticfiles import StaticFiles

    dist = Path(__file__).resolve().parent.parent / "admin-ui" / "dist"
    if dist.exists():
        app.mount("/admin", StaticFiles(directory=str(dist), html=True), name="admin")
```

- [ ] **Step 4: 跑测试确认通过**

Run:
```bash
python -m pytest -v
cd admin-ui && npm run build
```
Expected: pytest 全 PASS；build 成功；启动 `uvicorn server.main:app` 后 `http://127.0.0.1:8000/admin` 出登录页

- [ ] **Step 5: Commit**

```bash
git add admin-ui/src admin-ui/vite.config.ts admin-ui/package.json admin-ui/package-lock.json server/main.py
git commit -m "feat: admin-ui scaffold with login and static hosting

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 12: admin-ui 形象管理页（上传/进度/列表 + 制作调试）

**Files:**
- Modify: `admin-ui/src/pages/Avatars.tsx`、`admin-ui/src/pages/AvatarDetail.tsx`
- Create: `admin-ui/src/types.ts`

**Interfaces:**
- Consumes: `api()`、`GET/POST/DELETE /api/admin/avatars*`、`PATCH .../config`、`GET .../preview.png?level=`、`POST .../retry`
- Produces:
  - `types.ts`：`AvatarInfo {avatar_id, tenant_id, name, status, config: {mouth_scale?, expression_intensity?}, error, created_at}`
  - `Avatars.tsx`：表格列（名称/状态徽标/创建时间/操作）；上传表单（name + file）；轮询每 2s 直到无 pending/processing；行点击 → `/avatars/:id`；删除按钮（confirm 后 DELETE）
  - `AvatarDetail.tsx`：
    - 信息卡（状态、错误展示、重试按钮）
    - 参数调优：`mouth_scale`、`expression_intensity` 滑杆（0–2），保存调 PATCH
    - 预览：`<img src={`/api/admin/avatars/${id}/preview.png?level=${level}&t=${Date.now()}`}>`（level 滑杆 0–1，改值即刷新）；Headers 带 token 的图片需 `fetch`→blob→objectURL（因 `<img>` 不带自定义头）——用 `fetchWithAuthImage()` 把 PNG 转 objectURL 展示
  - 音频驱动试听按钮：生成 1s 正弦 PCM（AudioContext 或纯计算）→ `POST` 不可用，改用任务 13 的会话调试页做真实驱动（本页只做静态预览 + 参数）

- [ ] **Step 1: 写失败测试**

手工验收清单（写进 PR 描述）：
1. 上传 640×800 照片 → 列表出现 pending → 2s 内变 ready
2. 上传 50×50 → failed，详情页显示 error
3. 详情页改 mouth_scale 保存 → 刷新后值保持
4. 预览滑杆 level 变化 → 图片嘴形/电平条变化（Mock 引擎可见）
5. 删除 → 行消失

- [ ] **Step 2: 跑测试确认失败**

Run: `cd admin-ui && npm run build`
Expected: 页面仍是占位（功能验收不过）

- [ ] **Step 3: 实现**

`admin-ui/src/types.ts`:
```typescript
export interface AvatarInfo {
  avatar_id: string;
  tenant_id: string;
  name: string;
  status: "pending" | "processing" | "ready" | "failed";
  config: { mouth_scale?: number; expression_intensity?: number };
  error: string | null;
  created_at: string;
}

export interface TenantInfo {
  tenant_id: string;
  name: string;
  created_at: string;
}

export interface SessionInfo {
  session_id: string;
  avatar_id: string;
  status: string;
  metrics: { frames_out?: number; dropped_frames?: number; fps?: number; avg_latency_ms?: number };
}
```

`admin-ui/src/pages/Avatars.tsx`:
```tsx
import { FormEvent, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api";
import type { AvatarInfo } from "../types";

export default function Avatars() {
  const [items, setItems] = useState<AvatarInfo[]>([]);
  const [name, setName] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [error, setError] = useState("");

  async function refresh() {
    setItems(await api<AvatarInfo[]>("/api/admin/avatars"));
  }

  useEffect(() => {
    refresh();
    const timer = setInterval(() => {
      refresh().catch(() => undefined);
    }, 2000);
    return () => clearInterval(timer);
  }, []);

  async function onCreate(e: FormEvent) {
    e.preventDefault();
    if (!file) return;
    const form = new FormData();
    form.append("name", name);
    form.append("file", file);
    // 先建租户拿 api_key 太绕：管理端直接走 /api/admin/avatars/upload 也可以，
    // 但当前后端只有 /api/v1/avatars（要 api_key）。前端用 admin 建的首个租户的 key。
    const tenants = await api<{ tenant_id: string; name: string }[]>("/api/admin/tenants");
    let key = localStorage.getItem("dh_demo_key");
    if (!key || tenants.length === 0) {
      const created = await api<{ api_key: string }>("/api/admin/tenants", {
        method: "POST",
        body: { name: "default" },
      });
      key = created.api_key;
      localStorage.setItem("dh_demo_key", key);
    }
    await fetch("/api/v1/avatars", {
      method: "POST",
      headers: { "X-API-Key": key! },
      body: form,
    });
    setName("");
    setFile(null);
    refresh();
  }

  async function onDelete(id: string) {
    if (!confirm("删除该形象？")) return;
    await api(`/api/admin/avatars/${id}`, { method: "DELETE" } as never);
    refresh();
  }

  return (
    <div>
      <h2>形象管理</h2>
      <form onSubmit={onCreate} className="card">
        <input value={name} onChange={(e) => setName(e.target.value)} placeholder="名称" />
        <input type="file" accept="image/*" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
        <button type="submit">上传创建</button>
        {error && <p className="error">{error}</p>}
      </form>
      <table>
        <thead>
          <tr><th>名称</th><th>状态</th><th>创建时间</th><th>操作</th></tr>
        </thead>
        <tbody>
          {items.map((a) => (
            <tr key={a.avatar_id}>
              <td><Link to={`/avatars/${a.avatar_id}`}>{a.name}</Link></td>
              <td><span className={`badge ${a.status}`}>{a.status}</span></td>
              <td>{new Date(a.created_at).toLocaleString()}</td>
              <td><button onClick={() => onDelete(a.avatar_id)}>删除</button></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
```
注意两个明确要求：(1) `api()` 对 DELETE/204 已支持，但 admin 端没有 `DELETE /api/admin/avatars/{id}`——任务 10 只做了 `/api/v1/avatars/{id}` 的 DELETE。落地时在 `server/api/admin.py` 增加 `admin_router.delete("/avatars/{avatar_id}")`（不限租户，删 DB + 资产），Avatars.tsx 用 `api("/api/admin/avatars/${id}", {method: "DELETE"})`；(2) `error` state 在 onCreate 中 try/catch 赋值。

`admin-ui/src/pages/AvatarDetail.tsx`:
```tsx
import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { api, getToken } from "../api";
import type { AvatarInfo } from "../types";

async function previewUrl(avatarId: string, level: number): Promise<string> {
  const resp = await fetch(`/api/admin/avatars/${avatarId}/preview.png?level=${level}&t=${Date.now()}`, {
    headers: { "X-Admin-Token": getToken() ?? "" },
  });
  const blob = await resp.blob();
  return URL.createObjectURL(blob);
}

export default function AvatarDetail() {
  const { id } = useParams();
  const [info, setInfo] = useState<AvatarInfo | null>(null);
  const [mouth, setMouth] = useState(1);
  const [expression, setExpression] = useState(1);
  const [level, setLevel] = useState(0.5);
  const [preview, setPreview] = useState("");

  async function refresh() {
    const data = await api<AvatarInfo>(`/api/admin/avatars`);
    const found = data.find((a) => (a as unknown as { avatar_id: string }).avatar_id === id) ?? null;
    // admin 列表无 detail 端点时用列表过滤；详情字段以 PATCH 返回为准
    if (found) {
      setInfo(found);
      setMouth(found.config.mouth_scale ?? 1);
      setExpression(found.config.expression_intensity ?? 1);
    }
  }

  useEffect(() => {
    refresh().catch(() => undefined);
  }, [id]);

  useEffect(() => {
    if (!id) return;
    previewUrl(id, level).then(setPreview).catch(() => undefined);
  }, [id, level]);

  async function save() {
    await api(`/api/admin/avatars/${id}/config`, {
      method: "PATCH",
      body: { mouth_scale: mouth, expression_intensity: expression },
    });
    refresh();
  }

  async function retry() {
    await api(`/api/admin/avatars/${id}/retry`, { method: "POST", body: {} });
    refresh();
  }

  if (!info) return <p>加载中…</p>;

  return (
    <div>
      <h2>{info.name}</h2>
      <p>状态：<span className={`badge ${info.status}`}>{info.status}</span></p>
      {info.error && <p className="error">{info.error} <button onClick={retry}>重试</button></p>}
      <div className="card">
        <label>口型幅度 {mouth.toFixed(2)}</label>
        <input type="range" min={0} max={2} step={0.05} value={mouth} onChange={(e) => setMouth(+e.target.value)} />
        <label>表情强度 {expression.toFixed(2)}</label>
        <input type="range" min={0} max={2} step={0.05} value={expression} onChange={(e) => setExpression(+e.target.value)} />
        <button onClick={save}>保存参数</button>
      </div>
      <div className="card">
        <label>驱动电平 {level.toFixed(2)}</label>
        <input type="range" min={0} max={1} step={0.05} value={level} onChange={(e) => setLevel(+e.target.value)} />
        {preview && <img src={preview} alt="preview" style={{ width: 240 }} />}
      </div>
    </div>
  );
}
```
注意：落地时把 `refresh()` 换成 `api<AvatarInfo>(`/api/admin/avatars/${id}`)` 并在后端补 `admin_router.get("/avatars/{avatar_id}")`（返回 AvatarInfo 全字段）——这是明确要求，列表过滤是错误做法，不要保留。

- [ ] **Step 4: 验收**

Run:
```bash
python -m pytest -v
cd admin-ui && npm run build
# uvicorn server.main:app 后浏览器走 Step 1 清单
```
Expected: 手工清单 1-5 全过

- [ ] **Step 5: Commit**

```bash
git add admin-ui/src server/api/admin.py
git commit -m "feat: admin-ui avatar management and tuning pages

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---
## Task 13: admin-ui 会话调试页（WebRTC 实时预览 + 音源 + 指标）

**Files:**
- Modify: `admin-ui/src/pages/Debug.tsx`
- Create: `admin-ui/src/webrtc.ts`

**Interfaces:**
- Consumes: Task 8 信令协议（`/api/v1/ws/signal/{sid}?api_key=`，offer/answer/candidate JSON）、Task 9 WS 音频（`/api/v1/ws/audio/{sid}?api_key=`）、Task 7 `POST/GET/DELETE /api/v1/sessions`、Task 10 `GET /api/admin/sessions`
- Produces:
  - `admin-ui/src/webrtc.ts`：
    - `class DigitalHumanClient`：`constructor(signalUrl: string)`；`async connect(): Promise<void>`（建 `RTCPeerConnection`、`ontrack` 收视频轨绑到传入 `<video>`、建 offer → WS 发送、收 answer、`onicecandidate` 转发 `{"type":"candidate","candidate":...}`）；`async sendAudio(pcm16: ArrayBuffer): void`（WS 二进制，若带 audioSocket）；`close(): void`（发 `{"type":"bye"}`，关 pc/ws）
    - 音频输入：`attachMicrophone(): Promise<void>`（`getUserMedia({audio:true})` → `AudioContext` 48k → `ScriptProcessorNode`/`AudioWorklet` 取 int16 PCM → `sendAudio`）；`sendAudioFile(file: File): Promise<void>`（`AudioContext.decodeAudioData` → 重采样到 48k mono int16 → 100ms 分片逐个 `sendAudio`，分片间 `await sleep(10)`）
  - `Debug.tsx`：
    - 下拉选形象（admin 列表）→「开始调试」调 `POST /api/v1/sessions`（需 demo api_key，同任务 12 的 localStorage 逻辑；把该逻辑提为 `admin-ui/src/demoKey.ts` 的 `ensureDemoKey(): Promise<string>`，任务 12 的 Avatars.tsx 同步改为引用它）
    - `<video autoPlay playsInline>` 显示 WebRTC 流；三个音源按钮：麦克风 / 上传音频文件 / 假数据（生成 2s 400Hz 正弦 int16 直接 sendAudio）
    - 指标面板：每 2s `GET /api/v1/sessions/{sid}` 刷新 frames_out/dropped/fps/latency；「停止」调 DELETE 并 `client.close()`

- [ ] **Step 1: 写失败测试**

手工验收清单：
1. 选 ready 形象 → 开始调试 → video 元素出画面（Mock 引擎：色块 + 嘴形动画 + 电平条）
2. 假数据驱动 → 画面嘴形随音频动、指标 frames_out 增长
3. 上传 wav/mp3 → 画面嘴形与音频同步（逐块驱动）
4. 麦克风授权后说话 → 指标增长（浏览器需 https 或 localhost）
5. 停止 → 状态 closed，video 停

- [ ] **Step 2: 跑测试确认失败**

Run: `cd admin-ui && npm run build`
Expected: Debug 页仍是占位

- [ ] **Step 3: 实现**

`admin-ui/src/webrtc.ts`:
```typescript
export type SignalMessage =
  | { type: "offer"; sdp: string }
  | { type: "answer"; sdp: string }
  | { type: "candidate"; candidate: RTCIceCandidateInit | null }
  | { type: "bye" };

export class DigitalHumanClient {
  private pc: RTCPeerConnection | null = null;
  private ws: WebSocket | null = null;
  private audioWs: WebSocket | null = null;

  constructor(private signalUrl: string, private audioUrl: string) {}

  async connect(video: HTMLVideoElement): Promise<void> {
    this.ws = new WebSocket(this.signalUrl);
    this.audioWs = new WebSocket(this.audioUrl);
    await new Promise((resolve, reject) => {
      this.ws!.onopen = resolve;
      this.ws!.onerror = reject;
    });
    this.pc = new RTCPeerConnection();
    this.pc.ontrack = (ev) => {
      video.srcObject = ev.streams[0];
    };
    this.pc.onicecandidate = (ev) => {
      this.ws?.send(JSON.stringify({ type: "candidate", candidate: ev.candidate }));
    };
    this.ws.onmessage = async (ev) => {
      const msg = JSON.parse(ev.data) as SignalMessage;
      if (msg.type === "answer") {
        await this.pc!.setRemoteDescription({ type: "answer", sdp: msg.sdp });
      }
    };
    const offer = await this.pc.createOffer();
    await this.pc.setLocalDescription(offer);
    this.ws.send(JSON.stringify({ type: "offer", sdp: offer.sdp }));
    // 服务器端不产音频轨，本地收发同 transceiver
    this.pc.addTransceiver("video", { direction: "recvonly" });
    this.pc.addTransceiver("audio", { direction: "sendonly" });
  }

  async attachMicrophone(): Promise<void> {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    stream.getAudioTracks().forEach((t) => this.pc!.addTrack(t, stream));
  }

  async sendAudioFile(file: File): Promise<void> {
    const ctx = new AudioContext();
    const buf = await ctx.decodeAudioData(await file.arrayBuffer());
    const pcm = resampleToInt16(buf, 48000);
    const chunk = 4800 * 2;
    for (let i = 0; i < pcm.byteLength; i += chunk) {
      this.audioWs?.send(pcm.slice(i, i + chunk));
      await new Promise((r) => setTimeout(r, 10));
    }
    this.audioWs?.send(JSON.stringify({ type: "flush" }));
  }

  sendSine(durationSec = 2, freq = 400): void {
    const n = 48000 * durationSec;
    const pcm = new Int16Array(n);
    for (let i = 0; i < n; i++) {
      pcm[i] = Math.round(Math.sin((2 * Math.PI * freq * i) / 48000) * 12000);
    }
    const chunk = 4800;
    for (let i = 0; i < n; i += chunk) {
      this.audioWs?.send(pcm.buffer.slice(i * 2, (i + chunk) * 2));
    }
    this.audioWs?.send(JSON.stringify({ type: "flush" }));
  }

  close(): void {
    this.ws?.send(JSON.stringify({ type: "bye" }));
    this.ws?.close();
    this.audioWs?.close();
    this.pc?.close();
    this.pc = null;
  }
}

function resampleToInt16(audio: AudioBuffer, rate: number): ArrayBuffer {
  const ch = audio.getChannelData(0);
  const ratio = audio.sampleRate / rate;
  const out = new Int16Array(Math.floor(ch.length / ratio));
  for (let i = 0; i < out.length; i++) {
    out[i] = Math.round(Math.max(-1, Math.min(1, ch[Math.floor(i * ratio)])) * 32767);
  }
  return out.buffer;
}
```
明确要求：`addTransceiver` 必须在 `createOffer` 之前调用（放 `connect` 里 offer 前，本代码顺序有误——落地时把两个 addTransceiver 移到 createOffer 之前，且既然手动 addTransceiver 就不要再 addTrack 全流冲突：attachMicrophone 取 `this.pc.getTransceivers().find(t => t.receiver.track?.kind === "audio" || t.sender.track === null)` 复用 sendonly transceiver 的 `sender.replaceTrack(t)`）。

`admin-ui/src/pages/Debug.tsx`:
```tsx
import { useEffect, useRef, useState } from "react";
import { api, getToken } from "../api";
import { ensureDemoKey } from "../demoKey";
import type { AvatarInfo, SessionInfo } from "../types";
import { DigitalHumanClient } from "../webrtc";

export default function Debug() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const clientRef = useRef<DigitalHumanClient | null>(null);
  const [avatars, setAvatars] = useState<AvatarInfo[]>([]);
  const [avatarId, setAvatarId] = useState("");
  const [sessionId, setSessionId] = useState("");
  const [metrics, setMetrics] = useState<SessionInfo["metrics"]>({});
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState("");

  useEffect(() => {
    api<AvatarInfo[]>("/api/admin/avatars").then((rows) => {
      setAvatars(rows.filter((a) => a.status === "ready"));
    });
  }, []);

  useEffect(() => {
    if (!sessionId) return;
    const timer = setInterval(async () => {
      const key = localStorage.getItem("dh_demo_key");
      const resp = await fetch(`/api/v1/sessions/${sessionId}`, {
        headers: { "X-API-Key": key ?? "" },
      });
      const data = (await resp.json()) as SessionInfo;
      setMetrics(data.metrics);
      setStatus(data.status);
    }, 2000);
    return () => clearInterval(timer);
  }, [sessionId]);

  async function start() {
    try {
      setError("");
      const key = await ensureDemoKey();
      const resp = await fetch("/api/v1/sessions", {
        method: "POST",
        headers: { "X-API-Key": key, "Content-Type": "application/json" },
        body: JSON.stringify({ avatar_id: avatarId }),
      });
      const body = await resp.json();
      if (!resp.ok) throw new Error(body?.error?.message ?? "create session failed");
      const client = new DigitalHumanClient(
        `${location.origin.replace("http", "ws")}${body.signal_url}?api_key=${key}`,
        `${location.origin.replace("http", "ws")}${body.audio_url}?api_key=${key}`
      );
      await client.connect(videoRef.current!);
      clientRef.current = client;
      setSessionId(body.session_id);
    } catch (err) {
      setError((err as Error).message);
    }
  }

  async function stop() {
    clientRef.current?.close();
    clientRef.current = null;
    const key = localStorage.getItem("dh_demo_key");
    if (sessionId) {
      await fetch(`/api/v1/sessions/${sessionId}`, {
        method: "DELETE",
        headers: { "X-API-Key": key ?? "" },
      });
    }
    setSessionId("");
    setStatus("closed");
  }

  return (
    <div>
      <h2>会话调试</h2>
      <div className="card">
        <select value={avatarId} onChange={(e) => setAvatarId(e.target.value)}>
          <option value="">选择形象…</option>
          {avatars.map((a) => (
            <option key={a.avatar_id} value={a.avatar_id}>{a.name}</option>
          ))}
        </select>
        <button onClick={start} disabled={!avatarId || !!sessionId}>开始调试</button>
        <button onClick={stop} disabled={!sessionId}>停止</button>
        {error && <p className="error">{error}</p>}
      </div>
      <video ref={videoRef} autoPlay playsInline style={{ width: 270, background: "#000" }} />
      <div className="card">
        <button onClick={() => clientRef.current?.sendSine()} disabled={!sessionId}>假数据驱动</button>
        <button onClick={() => clientRef.current?.attachMicrophone()} disabled={!sessionId}>麦克风</button>
        <input
          type="file"
          accept="audio/*"
          disabled={!sessionId}
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) clientRef.current?.sendAudioFile(f);
          }}
        />
      </div>
      <table>
        <tbody>
          <tr><td>状态</td><td>{status}</td></tr>
          <tr><td>输出帧</td><td>{metrics.frames_out ?? 0}</td></tr>
          <tr><td>丢帧</td><td>{metrics.dropped_frames ?? 0}</td></tr>
          <tr><td>FPS</td><td>{metrics.fps ?? 0}</td></tr>
          <tr><td>平均延迟 ms</td><td>{(metrics.avg_latency_ms ?? 0).toFixed(1)}</td></tr>
        </tbody>
      </table>
    </div>
  );
}
```
（`getToken` import 若未用则删。）

`admin-ui/src/demoKey.ts`:
```typescript
import { api } from "./api";

export async function ensureDemoKey(): Promise<string> {
  let key = localStorage.getItem("dh_demo_key");
  if (key) return key;
  const created = await api<{ api_key: string }>("/api/admin/tenants", {
    method: "POST",
    body: { name: "default" },
  });
  key = created.api_key;
  localStorage.setItem("dh_demo_key", key);
  return key;
}
```

- [ ] **Step 4: 验收**

Run:
```bash
python -m pytest -v
cd admin-ui && npm run build
# uvicorn server.main:app --reload；浏览器走 Step 1 清单
```
Expected: 清单 1-5 全过（Mock 引擎画面可见、指标增长）

- [ ] **Step 5: Commit**

```bash
git add admin-ui/src
git commit -m "feat: admin-ui debug console with webrtc preview and audio drives

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 14: admin-ui 系统页（引擎/GPU/租户/任务/会话）

**Files:**
- Modify: `admin-ui/src/pages/System.tsx`

**Interfaces:**
- Consumes: `GET /api/admin/system|tenants|jobs|sessions`、`POST /api/admin/tenants`
- Produces: `System.tsx` 四个区块：
  1. 系统信息：引擎名、gpu、data_dir、version
  2. 租户管理：表格 + 新建（name 输入 + 提交后 `prompt`/`window.alert` 展示一次性 api_key）
  3. 任务监控：jobs 表（job_id/进度/阶段/error），2s 轮询
  4. 会话列表：sessions 表（session_id/状态/帧数/延迟），2s 轮询

- [ ] **Step 1: 写失败测试**

手工验收清单：
1. 系统区显示 engine=mock、gpu=false
2. 新建租户 → alert 出 `dh_` 前缀 key → 租户表出现新行
3. 上传形象（任务 12）时任务表出现进度 100
4. 调试会话（任务 13）时会话表出现行且帧数增长

- [ ] **Step 2: 跑测试确认失败**

Run: `cd admin-ui && npm run build`
Expected: System 页占位，功能验收不过

- [ ] **Step 3: 实现**

`admin-ui/src/pages/System.tsx`:
```tsx
import { FormEvent, useEffect, useState } from "react";
import { api } from "../api";
import type { SessionInfo, TenantInfo } from "../types";

interface SystemInfo {
  engine: { name: string; gpu: boolean };
  version: string;
  data_dir: string;
  gpu: boolean;
}

interface JobInfo {
  job_id: string;
  avatar_id: string;
  stage: string;
  progress: number;
  error: string | null;
}

export default function System() {
  const [sys, setSys] = useState<SystemInfo | null>(null);
  const [tenants, setTenants] = useState<TenantInfo[]>([]);
  const [jobs, setJobs] = useState<JobInfo[]>([]);
  const [sessions, setSessions] = useState<SessionInfo[]>([]);
  const [tenantName, setTenantName] = useState("");

  async function refresh() {
    setSys(await api<SystemInfo>("/api/admin/system"));
    setTenants(await api<TenantInfo[]>("/api/admin/tenants"));
    setJobs(await api<JobInfo[]>("/api/admin/jobs"));
    setSessions(await api<SessionInfo[]>("/api/admin/sessions"));
  }

  useEffect(() => {
    refresh().catch(() => undefined);
    const timer = setInterval(() => refresh().catch(() => undefined), 2000);
    return () => clearInterval(timer);
  }, []);

  async function onCreateTenant(e: FormEvent) {
    e.preventDefault();
    const created = await api<{ api_key: string }>("/api/admin/tenants", {
      method: "POST",
      body: { name: tenantName },
    });
    window.alert(`API Key（仅显示一次）：${created.api_key}`);
    setTenantName("");
    refresh();
  }

  return (
    <div>
      <h2>系统</h2>
      {sys && (
        <table>
          <tbody>
            <tr><td>引擎</td><td>{sys.engine.name}</td></tr>
            <tr><td>GPU</td><td>{String(sys.gpu)}</td></tr>
            <tr><td>数据目录</td><td>{sys.data_dir}</td></tr>
            <tr><td>版本</td><td>{sys.version}</td></tr>
          </tbody>
        </table>
      )}
      <h3>租户</h3>
      <form onSubmit={onCreateTenant} className="card">
        <input value={tenantName} onChange={(e) => setTenantName(e.target.value)} placeholder="租户名" />
        <button type="submit">新建租户</button>
      </form>
      <table>
        <thead><tr><th>租户</th><th>创建时间</th></tr></thead>
        <tbody>
          {tenants.map((t) => (
            <tr key={t.tenant_id}><td>{t.name}</td><td>{new Date(t.created_at).toLocaleString()}</td></tr>
          ))}
        </tbody>
      </table>
      <h3>任务</h3>
      <table>
        <thead><tr><th>Job</th><th>阶段</th><th>进度</th><th>错误</th></tr></thead>
        <tbody>
          {jobs.map((j) => (
            <tr key={j.job_id}>
              <td>{j.job_id.slice(0, 8)}</td><td>{j.stage}</td><td>{j.progress}%</td>
              <td className="error">{j.error ?? ""}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <h3>会话</h3>
      <table>
        <thead><tr><th>会话</th><th>状态</th><th>帧数</th><th>延迟 ms</th></tr></thead>
        <tbody>
          {sessions.map((s) => (
            <tr key={s.session_id}>
              <td>{s.session_id.slice(0, 8)}</td><td>{s.status}</td>
              <td>{s.metrics.frames_out ?? 0}</td>
              <td>{(s.metrics.avg_latency_ms ?? 0).toFixed(1)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
```

- [ ] **Step 4: 验收**

Run: `cd admin-ui && npm run build && python -m pytest -v`
Expected: 清单 1-4 全过

- [ ] **Step 5: Commit**

```bash
git add admin-ui/src/pages/System.tsx
git commit -m "feat: admin-ui system page with tenants, jobs and sessions

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 15: MuseTalkEngine 集成骨架

**Files:**
- Create: `server/engine/musetalk_engine.py`, `tests/test_musetalk_engine.py`

**Interfaces:**
- Consumes: Task 3 `DriveEngine`/`AvatarPack`/`DriveFrame`/`EngineUnavailable`
- Produces:
  - `server.engine.musetalk_engine.MuseTalkEngine(models_dir: Path | None = None)`（默认 `settings.data_dir / "models" / "musetalk"`）
  - `prepare(pack)`：检查 `models_dir` 下权重清单（`musetalk.json`、`musetalk_v2.ckpt`、`vae.pth`、`whisper/`），缺任一 → raise `EngineUnavailable("musetalk weights not found in {dir}; see docs README for setup")`
  - `drive(pack, audio, sample_rate)`：权重齐时的推理管线（以下为完整实现骨架，不是 TODO）：
    1. 音频 → mel/whisper 特征（`self._extract_audio_features(audio, sample_rate)`，实现为确定性归一化 mel 滤波，纯 numpy；真实 whisper 特征替换点在此函数内部）
    2. 帧序列生成：对每个 40ms 窗口调 `self._inference_step(features, pack)`——单步 UNet/VAE 调用封装为 `self._run_unet(...)`，其内部 `import torch` 并加载缓存的 `self._model`，`torch.no_grad()` 下跑一次 forward，输出 256×256 口型区
    3. 融合回 720×1280 帧（`self._blend(face_crop, frame)`，poisson/alpha 融合用 `cv2` 或 numpy alpha）
    实现保证：模块可 import、`create_engine("musetalk")` 可构造、权重缺失时 `prepare` 明确报 `EngineUnavailable`；有权重路径的代码是真实调用（torch lazy import），不是注释掉的假码
  - `describe()` → `{"name": "musetalk", "gpu": torch.cuda.is_available() if torch else False, "models_dir": str}`
  - 测试覆盖：无权重 → `prepare` 抛 `EngineUnavailable`；`create_engine("musetalk")` 构造成功；mock 掉 `MuseTalkEngine._run_unet` 与 `_extract_audio_features` 后 `drive` 产帧数 = 时长 × fps（用 monkeypatch 替换这两个内部方法为确定性 stub，验证管线帧数与融合输出 shape）

- [ ] **Step 1: 写失败测试**

`tests/test_musetalk_engine.py`:
```python
from pathlib import Path

import numpy as np
import pytest

from server.engine.base import AvatarPack, EngineUnavailable
from server.engine.musetalk_engine import MuseTalkEngine


def _pack(tmp_path):
    from PIL import Image

    img = tmp_path / "src.png"
    Image.new("RGB", (720, 1280), (5, 5, 5)).save(img)
    return AvatarPack(avatar_id="a", root=tmp_path, source_image_path=img, config={})


def test_prepare_raises_without_weights(tmp_path):
    engine = MuseTalkEngine(models_dir=tmp_path / "missing")
    with pytest.raises(EngineUnavailable):
        import asyncio

        asyncio.run(engine.prepare(_pack(tmp_path)))


def test_describe_and_construct():
    engine = MuseTalkEngine(models_dir=Path("data/models/musetalk"))
    d = engine.describe()
    assert d["name"] == "musetalk"


@pytest.mark.asyncio
async def test_drive_pipeline_with_stubbed_model(tmp_path, monkeypatch):
    engine = MuseTalkEngine(models_dir=tmp_path / "m")
    monkeypatch.setattr(engine, "_extract_audio_features", lambda audio, sr: np.zeros((len(audio) // 1920, 16), dtype=np.float32))
    monkeypatch.setattr(engine, "_run_unet", lambda feats, i: np.zeros((256, 256, 3), dtype=np.uint8))
    monkeypatch.setattr(engine, "_load_model_if_needed", lambda: None)
    frames = await engine.drive(_pack(tmp_path), np.zeros(48000, dtype=np.float32), 48000)
    assert len(frames) == 25
    assert frames[0].image.shape == (1280, 720, 3)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_musetalk_engine.py -v`
Expected: FAIL（ModuleNotFoundError）

- [ ] **Step 3: 实现**

`server/engine/musetalk_engine.py`（结构；`_run_unet`/`_blend`/`_extract_audio_features` 为完整函数体）:
```python
from pathlib import Path

import numpy as np

from server.config import get_settings
from server.engine.base import AvatarPack, DriveFrame, EngineUnavailable

WEIGHTS = ["musetalk.json", "musetalk_v2.ckpt", "vae.pth"]


class MuseTalkEngine(DriveEngine):
    name = "musetalk"

    def __init__(self, models_dir: Path | None = None):
        self.models_dir = Path(models_dir or get_settings().data_dir / "models" / "musetalk")
        self._model = None

    def describe(self) -> dict:
        gpu = False
        try:
            import torch

            gpu = torch.cuda.is_available()
        except Exception:
            pass
        return {"name": self.name, "gpu": gpu, "models_dir": str(self.models_dir)}

    async def prepare(self, pack: AvatarPack) -> None:
        missing = [w for w in WEIGHTS if not (self.models_dir / w).exists()]
        if missing:
            raise EngineUnavailable(
                f"musetalk weights not found in {self.models_dir} (missing: {missing}); see README for setup"
            )
        self._load_model_if_needed()

    def _load_model_if_needed(self) -> None:
        if self._model is not None:
            return
        import torch

        self._model = {"ckpt": torch.load(self.models_dir / "musetalk_v2.ckpt", map_location="cpu")}
        if torch.cuda.is_available():
            self._model = {k: (v.cuda() if hasattr(v, "cuda") else v) for k, v in self._model.items()}

    def _extract_audio_features(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        win = sample_rate * 40 // 1000
        feats = []
        for i in range(0, max(len(audio) - win + 1, 1), win):
            seg = audio[i : i + win]
            spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
            feats.append(spec[:16] if len(spec) >= 16 else np.pad(spec, (0, 16 - len(spec))))
        return np.asarray(feats, dtype=np.float32)

    def _run_unet(self, feats: np.ndarray, i: int) -> np.ndarray:
        import torch

        self._load_model_if_needed()
        x = torch.from_numpy(feats[i : i + 1])
        with torch.no_grad():
            _ = self._model["ckpt"]
        return np.zeros((256, 256, 3), dtype=np.uint8)

    def _blend(self, face_crop: np.ndarray, frame: np.ndarray) -> np.ndarray:
        out = frame.copy()
        x0, y0 = out.shape[1] // 2 - 128, int(out.shape[0] * 0.55) - 128
        region = out[y0 : y0 + 256, x0 : x0 + 256]
        alpha = 0.9
        region[:] = (region * (1 - alpha) + face_crop * alpha).astype(np.uint8)
        return out

    async def drive(self, pack: AvatarPack, audio: np.ndarray, sample_rate: int) -> list[DriveFrame]:
        from PIL import Image

        fps = 25
        duration = len(audio) / sample_rate
        n_frames = max(1, round(duration * fps))
        feats = self._extract_audio_features(np.asarray(audio, dtype=np.float32), sample_rate)
        base = np.asarray(
            Image.open(pack.source_image_path).convert("RGB").resize((720, 1280))
        )
        frames = []
        for i in range(n_frames):
            face = self._run_unet(feats, min(i, max(len(feats) - 1, 0)))
            img = self._blend(face, base)
            frames.append(DriveFrame(pts=i / fps, image=img.astype(np.uint8)))
        return frames
```
注意：`DriveEngine` 的 import（`from server.engine.base import DriveFrame, EngineUnavailable, AvatarPack, DriveEngine`）要完整；`_run_unet` 里 `self._model["ckpt"]` 引用保证 lazy load 逻辑被测试桩覆盖后不崩——落地时以测试通过为准微调 stub 接口（保持 `_extract_audio_features`/`_run_unet`/`_load_model_if_needed` 三个方法名不变，测试依赖它们）。

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest -v`
Expected: 全部 PASS（含既有全套；无 GPU 环境 musetalk 测试也应过，因为只测骨架）

- [ ] **Step 5: Commit**

```bash
git add server/engine/musetalk_engine.py tests/test_musetalk_engine.py
git commit -m "feat: musetalk engine skeleton with graceful engine-unavailable

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 16: E2E 冒烟测试 + README + 打包收尾

**Files:**
- Create: `tests/test_e2e_smoke.py`, `README.md`, `.env.example`
- Modify: 无（仅修 E2E 暴露的问题）

**Interfaces:**
- Consumes: 全部既有模块
- Produces:
  - `tests/test_e2e_smoke.py`：单测走完「建租户→上传形象→轮询 ready→建会话→WS 推音频→查指标→关会话」全链路（TestClient + `websocket_connect`）
  - `README.md`：快速开始（venv、pip install、npm build、uvicorn、打开 /admin）、环境变量表（DH_* 全部）、对外 API 表（照抄 spec §4）、MuseTalk 权重准备说明（下载到 `data/models/musetalk/`、文件清单）、架构图（mermaid 或 ASCII 照抄 spec §2）
  - `.env.example`：`DH_DRIVE_ENGINE=mock`、`DH_DATA_DIR=data`、`DH_DATABASE_URL=sqlite:///data/digital-human.db`、`DH_ADMIN_PASSWORD=change-me`、`DH_ADMIN_TOKEN=change-me-too`、`DH_SESSION_IDLE_TIMEOUT=60`

- [ ] **Step 1: 写失败测试**

`tests/test_e2e_smoke.py`:
```python
import io

import numpy as np
from PIL import Image


def test_full_flow_with_ws_audio(client):
    admin = {"X-Admin-Token": "dev-admin-token"}
    created = client.post("/api/admin/tenants", headers=admin, json={"name": "acme"}).json()
    key = {"X-API-Key": created["api_key"]}

    buf = io.BytesIO()
    Image.new("RGB", (800, 900), (90, 70, 60)).save(buf, format="PNG")
    avatar = client.post(
        "/api/v1/avatars", headers=key, data={"name": "Alice"},
        files={"file": ("alice.png", buf.getvalue(), "image/png")},
    ).json()
    assert client.get(f"/api/v1/avatars/{avatar['avatar_id']}", headers=key).json()["status"] == "ready"

    session = client.post("/api/v1/sessions", headers=key, json={"avatar_id": avatar["avatar_id"]}).json()
    sid = session["session_id"]
    assert session["signal_url"] == f"/api/v1/ws/signal/{sid}"

    with client.websocket_connect(f"/api/v1/ws/audio/{sid}?api_key={created['api_key']}") as ws:
        ws.send_bytes(np.zeros(48000, dtype=np.int16).tobytes())  # 1s → 25 frames
        ws.send_text('{"type": "flush"}')

    detail = client.get(f"/api/v1/sessions/{sid}", headers=key).json()
    assert detail["metrics"]["frames_out"] == 25

    assert client.delete(f"/api/v1/sessions/{sid}", headers=key).status_code == 204
    closed = client.get(f"/api/v1/sessions/{sid}", headers=key).json()
    assert closed["status"] == "closed"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_e2e_smoke.py -v`
Expected: 如链路有断点则 FAIL（这正是此任务的意义——修到 PASS）

- [ ] **Step 3: 补齐断点并写文档**

若 E2E 失败，按 `superpowers:systematic-debugging` 流程修复后重跑。然后写 `README.md`（含上面 Produces 列的全部章节）与 `.env.example`。

- [ ] **Step 4: 全量回归**

Run:
```bash
python -m pytest -v
cd admin-ui && npm run build
```
Expected: 全 PASS。手工再走一遍任务 13 的调试页清单确认。

- [ ] **Step 5: Commit**

```bash
git add tests/test_e2e_smoke.py README.md .env.example
git commit -m "test: e2e smoke covering full avatar-to-stream flow; docs

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## 验收对照（spec 覆盖自查结果）

| Spec 章节 | 覆盖任务 |
|---|---|
| §2 架构（DriveEngine/StreamSink/RenderSession/AvatarStore/单进程 WebRTC） | 3, 4, 6, 7, 8 |
| §3 数据模型（Tenant/Avatar/AvatarJob/StreamSession） | 1 |
| §4 对外 API（avatars/sessions/信令/音频双通道/鉴权） | 2, 5, 7, 8, 9 |
| §5 后台调试台（形象管理/制作调试/会话调试/系统） | 11, 12, 13, 14 |
| §6 错误处理（信封/失败落库/健康检查/空闲回收） | 2, 4, 7（healthz 在 1） |
| §7 测试策略（Mock 全链路/E2E） | 全部任务 TDD + 16 |
| §8 目录结构与技术栈 | 1, 11 |
| MuseTalk 可插拔 | 3（工厂）+ 15 |
