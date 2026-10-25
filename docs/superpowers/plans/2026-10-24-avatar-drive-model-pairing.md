# 形象 ↔ 驱动模型配套 + 多运行时并存 Implementation Plan

> **状态：已实现**（2026-10-24）。步骤用 checkbox 记录实际完成情况。

**Goal:** 把「形象 ↔ 驱动引擎」变成数据库里的一条事实（`avatars.drive_model`），删掉管理台的「当前数字人模型」卡片，让多个运行时同时驻留（各自加载 / 卸载），并把 Live2D 变成驱动模型之一 —— 形象创建时可以选。

**Architecture:** 注册表（`server/model_registry.py`）里的一条 = 一个驱动模型 = 一条流水线 + 一个引擎（客户端渲染的模型没有引擎）。形象行记住自己的模型，造包用这个模型的流水线，开会话用这个模型的引擎。引擎实例由 `server/engine/pool.py` 按模型名懒建并缓存，多个可以同时驻留。

**Tech Stack:** Python 3.11+ / FastAPI / SQLAlchemy；React 18 + Vite + TS + antd 5 + vitest。

## Global Constraints

- **一个模型 = 一条流水线 + 一个引擎**，`engine_factory=None` 表示客户端渲染（Live2D）。
- **形象 ↔ 模型是配套的**：`Avatar.drive_model` 是必填列，造包与会话都按它解析，绝不按「机器上恰好在跑什么」。
- **没有「当前模型」**：`dh_active_model` 设置行、`activate()/active()`、`PUT /api/admin/models/active` 全部删除。
- **多个模型可以同时驻留**：每个模型一份权重，在 Models 页逐个加载 / 卸载；开机仍预热默认模型（今天的首个会话不用等冷加载）。
- **Live2D 是驱动模型之一**：`source_kind="archive"`，走同一个上传端点、同一个删除端点。
- 前端所有用户可见文案用中文，与现有页面一致。
- **不引入 `@testing-library/react` / jsdom**（与既有约束一致）。

## Review Focus

规格是愿景文档，它没有逐条穷举输入。下面几类是任何测试都不会自动覆盖、且最可能咬到真实用户的输入。

1. **老库里 `avatars.kind` 的值。** 迁移必须把 `live2d` 搬进 `drive_model` 并删掉旧列，否则升级后 Live2D 形象会被当成 MuseTalk 形象驱动。
2. **租户重建白名单。** `_drop_abandoned_columns` 按列名白名单重建 `avatars`；白名单漏了 `drive_model`，老库触发重建时这一列会**静默消失**。
3. **模型 id 与权重槽位 id 混淆。** Models 页的行现在按**模型名**寻址（`musetalk`），不再是引擎自报的槽位 id（`musetalk_runtime`）；加载 / 卸载端点同理。
4. **客户端渲染的形象建服务端会话。** 服务端必须说人话（409 + 模型名），而不是让引擎抛一句私有的「包不对」。

---

### Task 1: 注册表与数据列

`server/model_registry.py` 改成模型 = 运行时；`Avatar.kind` → `Avatar.drive_model`；`server/db/session.py` 加迁移 + 回填 + 白名单。

- [x] `DigitalHumanModel` 增加 `engine_factory=None`（客户端渲染）与 `source_kind`；删除 `activate/active/read_active_model/write_active_model/ACTIVE_KEY`，保留 `DEFAULT_MODEL` 作兜底。
- [x] 注册 `musetalk`（有引擎）与 `live2d`（无引擎，`source_kind="archive"`）。
- [x] `Avatar.drive_model`（NOT NULL，默认 `musetalk`）；`_SCHEMA_MIGRATIONS` 加一条 ALTER。
- [x] `_backfill_drive_model`：`kind='live2d'` → `drive_model='live2d'`，跑在删列之前。
- [x] `_drop_abandoned_tenant_schema` → `_drop_abandoned_columns`，白名单换成 `drive_model`，`drop` 集合加上 `kind`。
- [x] 测试：`tests/test_model_registry.py` 重写；`tests/test_live2d_kind.py` → `tests/test_live2d_drive_model.py`（含迁移回填与删列的用例）。

### Task 2: 流水线与引擎池

- [x] `AvatarPipeline.process(..., *, options=None)`；`Live2DAvatarPipeline`（新文件 `server/avatar/live2d_pipeline.py`）解压 + 探测入口 + 校验工程，`inline_build = True`。
- [x] `expressions` 的解析与 422 从 `server/api/live2d.py` 搬进流水线。
- [x] `create_pipeline(name)` 走注册表；`run_job_in_new_session` 用**形象自己的模型**。
- [x] `server/engine/pool.py::EnginePool`：`engine_for(model_name)` 懒建并缓存，`close_all()`。
- [x] `SessionManager` 改持 pool；`create()` 按 `avatar.drive_model` 取引擎。
- [x] `POST /api/v1/sessions`：客户端渲染的形象 → 409 + 人话；模型未注册 → 409。
- [x] `server/main.py`：删掉 active 解析；开机预热默认模型；`/healthz` 列出全部模型。
- [x] 测试：`tests/test_drive_model_pairing.py`（新建，5 个用例）。

### Task 3: 模型列表与创建端点

- [x] `GET /api/admin/models`：每个注册模型一行（服务端渲染的取引擎 describe，客户端渲染的报 SDK 状态），行 id = 模型名。
- [x] `POST /api/admin/models/{model}/load|unload` 按模型名寻址；`animator:<name>` 保持原派发。
- [x] 删掉 `PUT /api/admin/models/active`。
- [x] 新增 `GET /api/admin/drive-models`。
- [x] `POST /api/admin/avatars` 加 `model` 字段：`image_video` 走后台 job，`archive` 请求内同步建包（失败不留行、不留文件）。
- [x] 删掉 `POST /api/admin/live2d/avatars` 与 `DELETE /api/admin/live2d/avatars/{id}`；托管路由的守卫改成能力判断。
- [x] `AvatarStore.save_archive`：压缩包存在**形象目录之外**，解压后的工程才是模型。

### Task 4: 前端

- [x] `types.ts`：`AvatarKind` → `DriveModelId`；`AvatarInfo.drive_model` + `client_rendered`；`ModelsPayload` 去掉三个 active 字段；`SystemInfo.runtimes`。
- [x] `api.ts`：`createAvatar` / `deleteAvatar` / `listDriveModels`。
- [x] `avatars/driveModels.ts`：注册表清单的一次性拉取与缓存（`useDriveModels`）。
- [x] `avatars/KindTag.tsx` → `avatars/DriveModelTag.tsx`。
- [x] `avatars/describe.ts`：`driveLabel` 读 `config[avatar.drive_model]?.mode`。
- [x] `pages/Models.tsx`：**删掉 `ActiveModelCard`**，模型表多出「客户端渲染」这一类行。
- [x] `pages/Avatars.tsx`：`Segmented` → 驱动模型下拉，表单按 `source_kind` 分支，提交/删除统一端点。
- [x] `pages/AvatarDetail.tsx` / `Debug.tsx` / `Chat.tsx` / `Pet.tsx` / `System.tsx` / `App.tsx`：按驱动模型分派。
- [x] 测试：`describe.test.ts` 改用 `drive_model`；`tests/test_admin_ui.py` 改为钉住「没有 active 卡片」与「模型下拉存在」。

### Task 5: 文档

- [x] `README.md`：架构、可插拔模型一节、API 表、健康检查示例。
- [x] 本计划文档。
