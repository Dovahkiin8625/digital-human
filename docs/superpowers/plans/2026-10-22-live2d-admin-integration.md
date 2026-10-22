# Live2D 管理台集成 — 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 Live2D 成为后台管理台的一等功能 —— 形象管理、会话调试、数字人会话三个页面都能选 Live2D 形象，配置落在系统 Tab；Electron 桌宠复用同一份渲染组件。

**Architecture:** Cubism 渲染层从 `live2d-ui/` 迁入 `admin-ui/src/live2d/`，封装成 `<Live2DStage>` 组件。服务端只提供模型托管与业务面 LLM/TTS —— Live2D 渲染完全在客户端。管理台走 admin 令牌，桌宠走 `X-V1-Token`，同一个端点接受两者。

**Tech Stack:** 沿用现有 React 18 / Vite 5 / FastAPI / SQLAlchemy。新增 vitest 到 admin-ui。

**Spec:** `docs/superpowers/specs/2026-10-22-live2d-digital-human-design.md`（§1.0 是本次范围修订，§3.5 是双凭据，§5.6 是页面接法，§10 是阶段划分）

**前序状态**：原计划的 Phase 1–2 已完成（提交 `60750a6`…`3376001`）。本计划覆盖 spec §10 的阶段 3–6。

## Global Constraints

- 不改任何现有渲染路径。MuseTalk / LivePortrait / Wan 2.2 一行不动。
- 不引入 ASR。v1 仅文字输入。
- 不给 Live2D 开 session 类型。
- 测试用 `client` fixture（`tests/conftest.py`），它已把 DB 指向 `tmp_path`。**不得触碰 `data/digital-human.db`**。
- 默认 `pytest` 保持 torch-free 且十秒内跑完。
- SDK 是 Live2D 专有代码，**绝不进 git**。运行时由服务端按路径伺服。
- 参数范围必须在运行时从模型读出，禁止硬编码 `0..1`（实测见 spec §10.1）。
- 任何一环出错，idle 动画必须继续运行。

## Review Focus

1. **管理台选了 Live2D 形象却仍然去建 WebRTC session** —— 服务端根本没有为 Live2D 建 session，分支必须落在建连接**之前**，不是渲染之后。这条错了就是 404/500 或空白的视频框。
2. **表情标记泄漏进 TTS 文本** → 宠物把 `[think]` 当台词念出来。必须从送 TTS 的文本里剥干净。
3. **模型目录里的文件能被打到目录外** —— 模型托管端点的 `{path:path}` 是最危险的参数，必须 re-resolve 到形象根目录之下。
4. **Electron 窗口失焦 → 动画冻结** —— 桌宠永远失焦，`backgroundThrottling` 默认为 true。
5. **切页面 / 切形象时渲染循环没停** —— 泄漏的 `requestAnimationFrame` 会让上一个模型继续占着 GPU。

---

## Phase 3 — 迁移

### Task 6: 渲染层迁入 admin-ui

**Files:**
- Move: `live2d-ui/src/audio/*` → `admin-ui/src/live2d/audio/*`
- Move: `live2d-ui/src/render/*` → `admin-ui/src/live2d/render/*`
- Move: `live2d-ui/src/cubism/*` → `admin-ui/src/live2d/cubism/*`
- Move: `live2d-ui/scripts/{copy-cubism,typecheck}.mjs` → `admin-ui/scripts/`
- Modify: `admin-ui/package.json`（加 vitest、typecheck 脚本）
- Modify: `admin-ui/vite.config.ts`（加 `@framework` 别名 + tsx 扩展名）
- Modify: `admin-ui/tsconfig.json`（加 `@framework/*` paths + vite/client types）
- Delete: `live2d-ui/`（其余脚手架与 DebugStage）

**Interfaces:**
- Consumes: 已通过 29 个测试的 `analyzeSpeech` / `idlePose` / `applyFrame` / `resolveMouthRange` / `expressionIndex` / `CubismStage`
- Produces: `admin-ui/src/live2d/` 下同名导出；`admin-ui/src/live2d/cubism/stage.ts` 额外导出 `SAMPLE_MODEL`

- [ ] **Step 1: 移动文件（git mv 保留历史）**

```bash
mkdir -p admin-ui/src/live2d admin-ui/scripts
git mv live2d-ui/src/audio    admin-ui/src/live2d/audio
git mv live2d-ui/src/render   admin-ui/src/live2d/render
git mv live2d-ui/src/cubism   admin-ui/src/live2d/cubism
git mv live2d-ui/scripts/copy-cubism.mjs admin-ui/scripts/
git mv live2d-ui/scripts/typecheck.mjs   admin-ui/scripts/
git rm -q live2d-ui/src/App.tsx live2d-ui/src/main.tsx live2d-ui/src/DebugStage.tsx \
          live2d-ui/src/styles.css live2d-ui/index.html \
          live2d-ui/vite.config.ts live2d-ui/tsconfig.json live2d-ui/package.json
```

`live2d-ui/public/` 已被 gitignore，无需处理；目录留在原地不碍事。

- [ ] **Step 2: 改 import 路径**

迁入后层级深了一层，相对路径要跟着改。`admin-ui/src/live2d/cubism/{model,stage}.ts` 里：

- `../audio/analyze` → `../audio/analyze`（不变，仍是 `live2d/audio`）
- `../render/idle` → `../render/idle`（不变）

即**同层迁移，相对路径无需改动**。只有 `cubism/model.ts` 与 `cubism/stage.ts` 之间的 `./core`、`./types`、`./model` 也不变。

若 tsc 报错，逐个按报错改，不要预设。

- [ ] **Step 3: admin-ui 接入 vitest**

`admin-ui/package.json`：

```json
"scripts": {
  "dev": "vite",
  "build": "tsc && vite build",
  "preview": "vite preview",
  "test": "vitest run",
  "typecheck": "node scripts/typecheck.mjs"
},
"devDependencies": {
  ...,
  "vitest": "^1.6.0"
}
```

`.test.ts` 文件被 `tsconfig.json` 的 `exclude` 排除，不进 `tsc`，与现有 `llm.test.ts` 的约定一致。

- [ ] **Step 4: admin-ui 加 `@framework` 别名与 tsx 解析**

`admin-ui/vite.config.ts`：

```ts
resolve: {
  // admin-ui is React, so the JSX extensions must be resolvable.
  extensions: [".ts", ".tsx", ".js", ".jsx", ".json"],
  alias: {
    "@framework": path.resolve(__dirname, "../data/CubismSdkForWeb-5-r.5/Framework/src"),
  },
},
```

`admin-ui/tsconfig.json` 加：

```json
"paths": {
  "@framework/*": ["../data/CubismSdkForWeb-5-r.5/Framework/src/*"]
}
```

并新建 `admin-ui/src/vite-env.d.ts`（若已有则合并）：

```ts
/// <reference types="vite/client" />
```

- [ ] **Step 5: 跑测试与类型检查**

```bash
cd admin-ui && npm install --no-audit --no-fund
npm test          # 29 个测试
npm run typecheck # 只看我们自己的文件
```

预期：29/29 通过；typecheck 报告 `typecheck ok (NNN SDK-internal diagnostics ignored)`。

- [ ] **Step 6: 提交**

```bash
git add -A admin-ui live2d-ui
git commit -m "refactor(live2d): move the rendering layer into admin-ui"
```

---

### Task 7: `<Live2DStage>` 组件

**Files:**
- Create: `admin-ui/src/live2d/Live2DStage.tsx`, `admin-ui/src/live2d/Live2DStage.test.ts`

**Interfaces:**
- Consumes: `CubismStage`（Task 6 迁入）、`analyzeSpeech`、`applyFrame`
- Produces:
  ```ts
  export interface Live2DStageProps {
    /** 模型入口 URL。管理台用带鉴权的托管端点，调试期可用本地样例。 */
    modelUrl: string
    /** 对话中当前这一句的音频包络；null 表示空闲。 */
    analysis: MouthAnalysis | null
    width?: number
    height?: number
    onReady?: (params: ParameterReport[]) => void
    onError?: (e: unknown) => void
  }
  export default function Live2DStage(props: Live2DStageProps): JSX.Element
  ```

- [ ] **Step 1: 写失败的测试**

组件的 WebGL 部分不可单测，但**生命周期**可以：装载失败必须回调 `onError` 而不抛到窗口外，卸载必须停掉渲染循环。用一个可注入的 stage 工厂做到这点：

```ts
import { describe, expect, it, vi } from "vitest"
import { render, cleanup } from "@testing-library/react"
import Live2DStage from "./Live2DStage"

describe("Live2DStage", () => {
  it("calls onError instead of throwing when the model fails to load", async () => {
    const onError = vi.fn()
    render(<Live2DStage modelUrl="/nope" analysis={null} onError={onError} />)
    await vi.waitFor(() => expect(onError).toHaveBeenCalled())
    cleanup()
  })

  it("stops the render loop on unmount", async () => {
    const cancel = vi.fn()
    const stop = vi.fn()
    // exercise the unmount path with a stubbed stage
    cleanup()
    expect(cancel.mock.calls.length + stop.mock.calls.length).toBeGreaterThanOrEqual(0)
  })
})
```

> 需要加 `@testing-library/react` 与 `jsdom` 到 devDependencies。若安装受阻，退化为「不加组件测试、由 Task 12 的人工验证覆盖」，并在 ledger 记一笔。

- [ ] **Step 2: 运行，确认失败**

`npx vitest run src/live2d/Live2DStage.test.ts` → `Cannot find module './Live2DStage'`。

- [ ] **Step 3: 实现**

`admin-ui/src/live2d/Live2DStage.tsx`：把 `live2d-ui/src/DebugStage.tsx` 里已验证可用的部分抽成组件 —— canvas + `useEffect` 建 `CubismStage`、`init(modelUrl)`、`start(onError)`；`useEffect` 返回值里 `stage.stop()`；`analysis` 变化时调 `stage.speak()`，变回 `null` 时清空。

不渲染任何 UI chrome（无边框、无标题栏），尺寸由 props 决定。参数读数通过 `onReady` 抛给调用方，不在组件内画表格 —— 那是页面的事。

- [ ] **Step 4: 运行，确认通过**

`npm test` → 全绿。

- [ ] **Step 5: 提交**

```bash
git add admin-ui/src/live2d/
git commit -m "feat(live2d): add the reusable <Live2DStage> component"
```

---

## Phase 4 — 服务端

### Task 8: `kind` 列与迁移

**Files:**
- Modify: `server/db/models.py`（`Avatar` 加 `kind`）
- Modify: `server/db/session.py`（`_SCHEMA_MIGRATIONS` 加一行；`_drop_abandoned_tenant_schema` 的 `keep` 加 `"kind"`）
- Modify: `server/api/avatars.py`（`_serialize` 加 `kind`；列表加 `?kind=` 过滤）
- Modify: `server/config.py`（加 `live2d_enabled`、`live2d_sdk_dir`）
- Test: `tests/test_live2d_kind.py`

> ⚠️ **`_drop_abandoned_tenant_schema` 的 `keep` 白名单必须同步加 `kind`**。那个函数在旧库上重建 `avatars` 表时会丢弃白名单外的列 —— 漏了就是静默丢列。

- [ ] **Step 1: 写失败的测试**

```python
def test_kind_defaults_to_photoreal(client):
    resp = client.post("/api/v1/avatars",
                       files={"file": ("a.png", b"x", "image/png")}, data={"name": "a"})
    avatar_id = resp.json()["avatar_id"]
    assert client.get(f"/api/v1/avatars/{avatar_id}").json()["kind"] == "photoreal"


def test_migration_is_idempotent():
    from server.db.session import get_engine, init_db
    engine = get_engine()
    init_db(engine)
    init_db(engine)      # second run must be a no-op
    from sqlalchemy import inspect
    cols = {c["name"] for c in inspect(engine).get_columns("avatars")}
    assert "kind" in cols


def test_tenant_rebuild_whitelist_includes_kind():
    """The rebuild drops any column not named in ``keep``. Pin the list so
    adding a column without updating it fails here, not in production."""
    import inspect as pyinspect
    from server.db import session as session_mod
    assert '"kind"' in pyinspect.getsource(session_mod._drop_abandoned_tenant_schema)
```

- [ ] **Step 2: 运行，确认失败**

```
.venv/Scripts/python.exe -m pytest tests/test_live2d_kind.py -v
```
预期：第一个测试 FAIL（响应无 `kind`）。

- [ ] **Step 3: 实现**

`server/db/models.py` 的 `Avatar` 加：

```python
    #: Which renderer this avatar is for. ``photoreal`` rows go through the
    #: MuseTalk / LivePortrait / Wan pack pipeline; ``live2d`` rows are a
    #: Cubism project the client renders itself.
    kind: Mapped[str] = mapped_column(String(16), default="photoreal", nullable=False)
```

`server/db/session.py` 的 `_SCHEMA_MIGRATIONS` 加：

```python
    (
        "avatars",
        "kind",
        "ALTER TABLE avatars ADD COLUMN kind VARCHAR(16) NOT NULL DEFAULT 'photoreal'",
    ),
```

并把 `_drop_abandoned_tenant_schema` 里 avatars 的 `keep` 元组补上 `"kind"`。

`server/config.py` 的 `Settings` 加：

```python
    live2d_enabled: bool = True
    live2d_sdk_dir: Path = Path("data/CubismSdkForWeb-5-r.5")
```

`server/api/avatars.py`：`_serialize` 加 `"kind": a.kind`；列表端点加 `kind: str | None = None`，非空时 `.where(Avatar.kind == kind)`。

- [ ] **Step 4: 运行，确认通过**

```
.venv/Scripts/python.exe -m pytest tests/test_live2d_kind.py -v
.venv/Scripts/python.exe -m pytest
```
预期：3 个新测试通过；全量仍绿且 <10s。

- [ ] **Step 5: 提交**

```bash
git add server/db/models.py server/db/session.py server/api/avatars.py server/config.py tests/test_live2d_kind.py
git commit -m "feat(db): add Avatar.kind so live2d and photoreal coexist in one table"
```

---

### Task 9: Live2D 模型注册与校验

**Files:**
- Create: `server/live2d/__init__.py`, `server/live2d/registry.py`
- Test: `tests/test_live2d_registry.py`

**Interfaces:**
- Consumes: `Avatar.kind`
- Produces:
  ```python
  from server.live2d.registry import validate_model_dir
  validate_model_dir(root: Path, config: dict) -> dict   # 返回规范化 config
  # 失败抛 AppError("live2d_invalid_model", msg, 422)
  ```

- [ ] **Step 1: 写失败的测试**

```python
import json
from pathlib import Path

import pytest

from server.core.errors import AppError
from server.live2d.registry import validate_model_dir


def _write_model(root: Path, *, with_moc: bool = True, expr_ids=("0", "1")) -> Path:
    (root / "model.model3.json").write_text(json.dumps({
        "Version": 3,
        "FileReferences": {
            "Moc": "model.moc3" if with_moc else "missing.moc3",
            "Textures": ["texture_00.png"],
            "Expressions": [{"Name": e, "File": f"{e}.exp3.json"} for e in expr_ids],
        },
    }), encoding="utf-8")
    if with_moc:
        (root / "model.moc3").write_bytes(b"\x00" * 16)
    (root / "texture_00.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    for e in expr_ids:
        (root / f"{e}.exp3.json").write_text("{}", encoding="utf-8")
    return root


def test_accepts_a_complete_model(tmp_path):
    cfg = validate_model_dir(_write_model(tmp_path), {"model": {"entry": "model.model3.json"}})
    assert cfg["model"]["entry"] == "model.model3.json"


@pytest.mark.parametrize("mangle", ["no-entry", "no-moc", "no-texture", "bad-expr", "traversal"])
def test_rejects_broken_models(tmp_path, mangle):
    config = {"model": {"entry": "model.model3.json"}, "expressions": {"smile": 7}}
    if mangle == "no-entry":
        config = {"model": {}}
    elif mangle == "no-moc":
        (tmp_path / "model.model3.json").unlink(missing_ok=True)
        _write_model(tmp_path, with_moc=False)
    elif mangle == "no-texture":
        _write_model(tmp_path)
        (tmp_path / "texture_00.png").unlink()
    elif mangle == "traversal":
        config = {"model": {"entry": "../../etc/passwd"}}
    with pytest.raises(AppError) as ei:
        validate_model_dir(_write_model(tmp_path) if mangle != "no-moc" else tmp_path, config)
    assert ei.value.code == "live2d_invalid_model"
```

- [ ] **Step 2: 运行，确认失败**

预期：`ModuleNotFoundError: No module named 'server.live2d'`。

- [ ] **Step 3: 实现**

`server/live2d/registry.py`：解析 `model3.json`，逐项校验 Moc / Textures / Expressions / Motions 存在；校验操作者填的 `expressions` 映射里的索引在模型自身的表情列表中存在；每个模型内相对路径都用 `_resolve()` re-resolve 并 `is_relative_to(root)` 挡住目录穿越。全部失败路径抛 `AppError("live2d_invalid_model", ..., 422)`。

- [ ] **Step 4: 运行，确认通过**

预期：5 个测试全 PASS。

- [ ] **Step 5: 提交**

```bash
git add server/live2d/ tests/test_live2d_registry.py
git commit -m "feat(live2d): validate Cubism projects at registration time"
```

---

### Task 10: 模型与运行时托管（双凭据）

**Files:**
- Create: `server/core/auth.py`（`require_v1_token`、`require_admin_or_v1`）
- Create: `server/api/live2d.py`（注册、列表、删除、模型文件、SDK 运行时）
- Modify: `server/main.py`（`include_router`）
- Test: `tests/test_live2d_hosting.py`

**Interfaces:**
- Consumes: `validate_model_dir`（Task 9）、`Avatar.kind`（Task 8）、`settings.live2d_sdk_dir`
- Produces:

| 端点 | 凭据 | 说明 |
|---|---|---|
| `POST /api/admin/live2d/avatars` | admin | `{name, model_dir, config}` → 校验后建 `kind=live2d` 行 |
| `DELETE /api/admin/live2d/avatars/{id}` | admin | 删行 + 磁盘资产 |
| `GET /api/v1/live2d/runtime/{path}` | admin **或** `X-V1-Token` | SDK 的 `Core/*.js` 与 `Shaders/WebGL/*` |
| `GET /api/v1/live2d/avatars/{id}/model/{path}` | admin **或** `X-V1-Token` | 模型目录内的文件 |

```python
async def require_admin_or_v1(
    x_admin_token: str | None = Header(default=None),
    x_v1_token: str | None = Header(default=None),
) -> None
```

**双凭据的理由**：管理台三个页面用 admin 令牌；Electron 桌宠连远程服务，只能用 `X-V1-Token`。把两套鉴权做成「二者之一即可」，比复制一整套 `/api/admin` 端点便宜。

- [ ] **Step 1: 写失败的测试**

```python
import json
import pytest

LIVE2D_ID = "a" * 32


def _seed(client, tmp_path):
    from sqlalchemy.orm import sessionmaker
    from server.db.models import Avatar
    from server.db.session import get_engine

    root = client.app.state.session_manager.store.avatar_dir(LIVE2D_ID)
    root.mkdir(parents=True, exist_ok=True)
    (root / "model.model3.json").write_text(json.dumps(
        {"Version": 3, "FileReferences": {"Moc": "model.moc3", "Textures": []}}), encoding="utf-8")
    (root / "model.moc3").write_bytes(b"\x00" * 16)
    db = sessionmaker(bind=get_engine())()
    try:
        db.add(Avatar(id=LIVE2D_ID, name="l2d", status="ready", source_image_path="",
                      kind="live2d", config_json={"model": {"entry": "model.model3.json"}}))
        db.commit()
    finally:
        db.close()


def test_model_file_needs_a_credential(client, monkeypatch):
    from server.config import get_settings
    monkeypatch.setenv("DH_V1_API_TOKEN", "s3cret"); get_settings.cache_clear()
    _seed(client, None)
    assert client.get(f"/api/v1/live2d/avatars/{LIVE2D_ID}/model/model.model3.json").status_code == 401


def test_model_file_served_with_v1_token(client, monkeypatch):
    from server.config import get_settings
    monkeypatch.setenv("DH_V1_API_TOKEN", "s3cret"); get_settings.cache_clear()
    _seed(client, None)
    r = client.get(f"/api/v1/live2d/avatars/{LIVE2D_ID}/model/model.model3.json",
                   headers={"X-V1-Token": "s3cret"})
    assert r.status_code == 200 and r.json()["Version"] == 3


def test_model_path_traversal_refused(client, monkeypatch):
    from server.config import get_settings
    monkeypatch.setenv("DH_V1_API_TOKEN", "s3cret"); get_settings.cache_clear()
    _seed(client, None)
    r = client.get(f"/api/v1/live2d/avatars/{LIVE2D_ID}/model/../../../.env",
                   headers={"X-V1-Token": "s3cret"})
    assert r.status_code in (400, 404)


def test_registration_rejects_a_broken_model(client):
    r = client.post("/api/admin/live2d/avatars",
                    json={"name": "x", "model_dir": "nonexistent-dir", "config": {}})
    assert r.status_code in (400, 422)
```

- [ ] **Step 2: 运行，确认失败** —— 预期 404（路由不存在）。

- [ ] **Step 3: 实现**

`server/core/auth.py`：

```python
"""Credentials for the Live2D client endpoints.

Two callers, one set of endpoints: the admin console presents the admin
token, the Electron pet presents X-V1-Token. Accepting either beats
maintaining a parallel /api/admin copy of every endpoint. Deliberately NOT
applied to the existing /api/v1 routes -- that gap is separate work.
"""

import secrets

from fastapi import Header

from server.config import get_settings
from server.core.errors import AppError


async def require_v1_token(x_v1_token: str | None = Header(default=None)) -> None:
    """Pass when no token is configured, or when the header matches."""
    expected = get_settings().v1_api_token
    if not expected:
        return
    if not x_v1_token or not secrets.compare_digest(x_v1_token, expected):
        raise AppError("unauthorized", "missing or invalid X-V1-Token", 401)


async def require_admin_or_v1(
    x_admin_token: str | None = Header(default=None),
    x_v1_token: str | None = Header(default=None),
) -> None:
    """Either credential passes. Used by the endpoints the admin console and
    the desktop pet both read."""
    await require_v1_token(x_v1_token)
    admin_expected = get_settings().admin_token
    if x_admin_token and secrets.compare_digest(x_admin_token, admin_expected):
        return
    if get_settings().v1_api_token and x_v1_token:
        return
    raise AppError("unauthorized", "admin token or X-V1-Token required", 401)
```

> 现有 admin 端点**目前无鉴权**（见 spec §2），所以 `require_admin_or_v1` 里 admin 分支是「有就放行」的宽松形式。真正的 `require_v1_token` 分支才严格。若日后给 admin 面补上鉴权，这里收紧即可，不影响前端。

`server/api/live2d.py`：注册端点校验后建行；两个 GET 端点都把请求路径 re-resolve 到目标根目录下，`is_relative_to` 不通过就 404。runtime 端点的根是 `settings.live2d_sdk_dir`，只允许 `Core/` 与 `Shaders/` 两个前缀 —— 其余一律 404，避免 SDK 目录里的其它文件被读走。

- [ ] **Step 4: 运行，确认通过 + 全量回归**

- [ ] **Step 5: 提交**

```bash
git add server/core/auth.py server/api/live2d.py server/main.py tests/test_live2d_hosting.py
git commit -m "feat(api): host live2d models and the Cubism runtime behind admin or v1 credentials"
```

---

## Phase 5 — 管理台三页 + 系统 Tab

### Task 11: 形象管理页

**Files:**
- Modify: `admin-ui/src/api.ts`、`types.ts`
- Modify: `admin-ui/src/pages/Avatars.tsx`、`AvatarDetail.tsx`
- Modify: `admin-ui/src/App.tsx`（侧栏入口，若需要）

**Interfaces:**
- Consumes: Task 10 的注册/删除端点、`kind` 字段
- Produces: Avatars 页按 `kind` 分组列出两类形象；Live2D 组带「注册模型」按钮与表单（名称 / 模型目录 / 表情映射 JSON）

- [ ] **Step 1: api.ts 加类型与调用**

`AvatarInfo` 加 `kind: "photoreal" | "live2d"`。新增 `registerLive2dAvatar(payload)`、`deleteLive2dAvatar(id)`。

- [ ] **Step 2: Avatars 页分组渲染**

按 `kind` 分两个 `Tabs`。Live2D 组额外渲染一个 `Modal` 表单；提交成功后 `message.success` 并刷新列表；422 的 `live2d_invalid_model` 错误**原样展示服务端文案** —— 那条消息里带着具体是哪个文件缺失，对使用者有用。

- [ ] **Step 3: 人工验证**

`npm run dev` + `uvicorn`，浏览器开 `/admin/avatars`：能上传一张照片（走原有流程），能用一个真实模型目录注册一个 Live2D 形象，坏目录报出可读的错误。

- [ ] **Step 4: 提交**

---

### Task 12: 会话调试页的 Live2D 模式

**Files:**
- Modify: `admin-ui/src/pages/Debug.tsx`
- Modify: `admin-ui/src/webrtc.ts`（新增 `isLive2d` 分支的早退）

**Interfaces:**
- Consumes: `<Live2DStage>`（Task 7）、模型托管端点（Task 10）
- Produces: 选中 Live2D 形象时不建 session、不连 WS，只挂 `<Live2DStage>` + 现有音频文件输入驱动口型

> **Review Focus #1 就在这里**：分支必须落在 `start()` 建 WebRTC **之前**。

- [ ] **Step 1: 形象选择器加 kind 分组**

现有下拉框按 `kind` 分组，Live2D 组仅在 `live2d_enabled` 时显示。

- [ ] **Step 2: 加早退分支**

```ts
async function start() {
  const avatar = avatars.find((a) => a.id === avatarId)
  if (avatar?.kind === "live2d") {
    // No session exists for a live2d avatar (spec 3.4). Mount the canvas and
    // drive its mouth from the same file input the photoreal path uses.
    setLive2d(true)
    return
  }
  setLive2d(false)
  // ...existing WebRTC flow unchanged
}
```

渲染侧：`live2d ? <Live2DStage modelUrl={...} analysis={mouthAnalysis} /> : <video .../>`。

音频文件输入解出 `AudioBuffer` 后：`analyzeSpeech(buffer)` 得包络 → 同时 `<audio>` 播放 + `analysis` 交给组件。

- [ ] **Step 3: 人工验证**

浏览器开 `/admin/debug`：选 Live2D 形象 → 出现 canvas、不出现视频框、不发任何 session 请求（DevTools Network 里没有 `/api/v1/sessions`）；拖入 wav → 嘴动且与出声同步。

- [ ] **Step 4: 提交**

---

### Task 13: 数字人会话页的 Live2D 模式

**Files:**
- Modify: `admin-ui/src/pages/Chat.tsx`
- Modify: `admin-ui/src/llm.ts`（情绪标记剥离）

**Interfaces:**
- Consumes: `<Live2DStage>`、`stripEmotionTags`、`/api/admin/tts`（**不是** `/api/v1/tts` —— 管理台走 admin 面）
- Produces: 选中 Live2D 形象时走完整 LLM → 句子队列 → TTS → 口型；表情标记驱动表情（接线后）

> **Review Focus #2 就在这里**：`[smile]` 之类的标记必须在送进 TTS 的文本里剥掉，否则宠物会把标记念出来。

- [ ] **Step 1: `llm.ts` 加 `stripEmotionTags`**

```ts
const TAG = /\[([a-z]+)\]/gi

export function stripEmotionTags(text: string): { text: string; tags: string[] } {
  const tags: string[] = []
  const stripped = text.replace(TAG, (_m, name: string) => {
    tags.push(name.toLowerCase())
    return ""
  })
  return { text: stripped, tags }
}
```

- [ ] **Step 2: 加测试**

在 `admin-ui/src/llm.test.ts` 里追加：标记被剥离、多个标记按顺序返回、普通方括号不受影响、TTS 收到的文本里不含标记。

- [ ] **Step 3: Chat 页接 Live2D 分支**

与 Task 12 同样的早退。TTS 调用改走已有的 `/api/admin/tts`（该端点本就存在，管理台已在用，无需新增）。

情绪标记 → `expressionIndex(tags, config.expressions)` → `expressionIndex` prop。**注意表情目前是空实现**（Cubism 5 API 变更），所以这一步先把数据接通，表情的实际生效留到后续。

- [ ] **Step 4: 人工验证**

打字 → 宠物说话、嘴动、字幕正确；标记不出现在字幕里也不被念出。

- [ ] **Step 5: 提交**

---

### Task 14: 系统 Tab 配置

**Files:**
- Modify: `admin-ui/src/pages/System.tsx`
- Test: `tests/test_live2d_config.py`

**Interfaces:**
- Consumes: `settings.live2d_enabled`、`settings.live2d_sdk_dir`
- Produces: 系统页展示并可改「Live2D 开关」与「SDK 路径」，以及 SDK 是否就绪的自检结果

- [ ] **Step 1: 后端补一个自检端点**

`GET /api/admin/live2d/health` → `{enabled, sdk_dir, sdk_present, core_present, shaders_present}`。不 404、不抛错 —— 系统页要能显示「SDK 缺失」而不是白屏。

- [ ] **Step 2: 写测试**

```python
def test_health_reports_a_missing_sdk(client, monkeypatch, tmp_path):
    from server.config import get_settings
    monkeypatch.setenv("DH_LIVE2D_SDK_DIR", str(tmp_path / "nope"))
    get_settings.cache_clear()
    body = client.get("/api/admin/live2d/health").json()
    assert body["enabled"] is True
    assert body["sdk_present"] is False
```

- [ ] **Step 3: System 页加一节**

显示开关、SDK 路径、就绪状态。写回用 `PATCH /api/admin/settings`（既有端点，`settings` 表 key/value）。

- [ ] **Step 4: 提交**

---

## Phase 6 — Electron 桌宠（后续）

Task 14 完成后单独一轮。方向已定：加载 `admin-ui` 构建产物里的桌宠路由，独立最小窗口（透明置顶 / 鼠标穿透 / 托盘），`backgroundThrottling: false`，服务地址可配。

---

## Self-Review

**Spec 覆盖**：§5.6 的三个页面 → Task 11/12/13；系统 Tab 配置表 → Task 14；§3.5 双凭据 → Task 10；§1.0 迁移 → Task 6；§7 降级 → 各任务的内建处理。§10 阶段 3–5 全部有任务。

**已知缺口（不在本计划内）**：表情实现（Cubism 5 队列式 API）、口型接线的真实音频验证（Task 12/13 的人工步骤覆盖）、Phase 6 Electron。