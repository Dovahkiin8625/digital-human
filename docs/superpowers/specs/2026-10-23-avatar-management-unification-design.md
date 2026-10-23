# 形象管理统一 + Live2D 模型上传 — 设计文档

日期：2026-10-23
状态：已与需求方对齐（zip 上传 / 含详情页分叉 / 删服务端路径注册 / 单表 + 类型分段控件，四项均经确认）
关联：`docs/superpowers/specs/2026-10-22-live2d-digital-human-design.md`

## 1. 目标与范围

把**形象管理**页从四张并列卡片收成**一张表 + 一个创建入口**，并让用户**直接上传 Live2D 模型**。

两种形象在产品上是同一个东西——一个可被会话选中、可被驱动、可被删除的形象；区别只是 `Avatar.kind`。今天前端把它们做成了两套：两个创建表单、两个列表、两个删除端点。这份设计把 UI 收成一份骨架，`kind` 只驱动差异部分。

### 1.1 已确认的四项决策

| # | 决策 | 内容 |
|---|---|---|
| 1 | 上传方式 | **zip 压缩包**。用户在浏览器打包整个 Cubism 工程拖进来 |
| 2 | 重构范围 | **列表页 + 详情页分叉**。不做删除端点合并，不动 photoreal 现有行为 |
| 3 | 旧入口 | **删掉**「填服务端路径注册」。只留 zip |
| 4 | 页面结构 | **单表 + 类型 Segmented**。不用 Tabs，不用卡片列表 |

### 1.2 非目标（YAGNI）

- **不合并后端端点**。`POST /api/admin/avatars`（视频）与 `POST /api/admin/live2d/avatars`（zip）是两个端点，**各自原地演进**。「同一个功能」是产品语义，不要求一个 HTTP 路由。
- **不合并 DELETE**。`DELETE /api/admin/avatars/{id}` 与 `DELETE /api/admin/live2d/avatars/{id}` 都保留。后者对 `photoreal` 返回 404，这个行为是刻意的（它保证删除 Live2D 不会误伤视频形象），改它要另开一件事。
- **不改 `/api/v1/live2d/avatars/{id}/model/{path}`**。托管路径、`_safe_join`、token 门全部不动。
- **不改 photoreal 的创建/删除/详情行为**。`AvatarDetail` 的 photoreal 分支一行不动。
- **不为 zip 上传引入新的鉴权**。当前 admin 面无鉴权是既有决定，本设计不碰。
- **不引入 `@testing-library/react` / jsdom**。理由见 §6.2。

## 2. 背景：现状与差距

后端**已经是一个功能了**：同一张 `avatar` 表、同一个列表端点 `/api/admin/avatars?kind=`（`server/api/admin.py:909`），靠 `Avatar.kind` 区分。分开的只有 UI 和写入路径。

**现状差距**：

| 现状 | 差距 |
|---|---|
| Live2D 只能注册服务端路径（`server/api/live2d.py:82`） | 用户必须先 SCP 文件到机器上、填绝对路径。没有浏览器上传路径 |
| `Avatars.tsx` 713 行，四张卡片 | 新建形象 / 形象列表 / 注册 Live2D 模型 / Live2D 模型 |
| `CopyableId`、状态 `StatusTag`、创建时间列、删除 `Popconfirm`、表格外壳 | 两份逐字重复（`Avatars.tsx:360-449` 与 `:635-705`） |
| `AvatarDetail.tsx` 写死 photoreal | 拉 `idle.mp4` / `normalized.png` / `artifacts` / `timeline`，Live2D 全部 404 或无意义；「按当前设置重建」点了会 500（Live2D 没有 `AvatarJob`，`retry_avatar` 会跑流水线） |

## 3. 后端

### 3.1 唯一改动：注册端点原地换成 multipart

```
POST /api/admin/live2d/avatars        (multipart/form-data)

  name         str    required
  file         zip    required      整个 Cubism 工程
  entry        str    optional     model3.json 的相对路径；留空则自动探测
  expressions  str    optional     JSON 文本

  → 201 { avatar_id, status: "ready", config: { model: { entry } , ...} }
  → 422 live2d_invalid_model，消息指出具体缺失文件（沿用 validate_model_dir 的措辞）
  → 413 file_too_large
```

流程与现在的 `register_live2d_avatar`（`server/api/live2d.py:83-113`）一致，只把 `shutil.copytree` 换成安全解压：

1. `avatar_id = uuid4().hex`，目标目录 `data/avatars/<id>/`
2. 解压 `file` 到该目录（§3.3）
3. `validate_model_dir(target, config)` 校验解压后的**副本**（不是 zip 里的原始字节——那是将被服务的目录）
4. 校验通过才 `db.add(Avatar(kind="live2d", status="ready", ...))`
5. **任何一步失败：`shutil.rmtree(target, ignore_errors=True)` 且不建行**

`RegisterIn`（`server/api/live2d.py:71`）与 JSON 版本函数删除。`delete_live2d_avatar` 与 `_store()` 保留。

### 3.2 入口文件自动探测

`entry` 留空时，遍历解压后的树找 `*.model3.json`：恰好一个就用它，零个或多个则 422 并列出候选路径。

这不是锦上添花，是**实际会踩的坑**：用户在 Windows 上右键压缩 `Hiyori/` 文件夹，得到 `Hiyori/model.model3.json`，写死 `model.model3.json` 会直接失败。探测出的相对路径原样存进 `config.model.entry`，客户端按它拼 URL（`admin-ui/src/live2d/cubism/core.ts:40`），Cubism 的相对引用自然解析——文件留在子目录里也没关系，`_safe_join` 是相对形象根解析的。

### 3.3 解压安全

| 防护 | 做法 |
|---|---|
| **zip-slip** | 每个条目目标 `(root / name).resolve()` 后必须 `is_relative_to(root.resolve())`。绝对路径、`..`、盘符（`C:\`）一律拒 |
| **符号链接** | 读 `ZipInfo.external_attr` 高 16 位的 `S_IFMT`，`S_IFLNK` 直接拒。zip 里的 symlink 不解压就不会被 `extractall` 跟随，但它会在后续构建路径时越界 |
| **解压炸弹（条目数）** | `live2d_max_archive_entries`（默认 512） |
| **解压炸弹（累计字节）** | `live2d_max_unpacked_bytes`（默认 512 MB）。逐条目累加**声明的** `file_size`，超限即中止 |
| **解压炸弹（单条目）** | `live2d_max_entry_bytes`（默认 64 MB） |

压缩包本身沿用现有 `avatar_max_upload_bytes`（300 MB，`server/config.py:35`）——不解压就能拒。

累计字节上限**不能**沿用同一个数：300 MB 的 zip 能轻松展开成 30 GB。三个新值放 `server/config.py`，可由 `.env` 覆盖。

实现用 `zipfile.ZipFile` 逐条目手动 `open()` → `shutil.copyfileobj`，**不用 `extractall`**——`extractall` 在 Python 3.12 之前不做路径检查。

### 3.4 删除的代码

| 位置 | 内容 |
|---|---|
| `server/api/live2d.py:71-79` | `RegisterIn` |
| `server/api/live2d.py:82-113` | `register_live2d_avatar` |
| `admin-ui/src/api.ts:77-86` | `Live2DRegisterIn` + `registerLive2dAvatar` 的 JSON 版本 |
| `admin-ui/src/pages/Avatars.tsx:586-632` | 「注册 Live2D 模型」整张卡片 |
| `README.md:213` | 端点表格里那一行改成 multipart |
| `tests/test_live2d_hosting.py:78-127` | 注册端点的四个用例改写成上传 zip |

### 3.5 不动的代码

`DELETE /api/admin/live2d/avatars/{id}`、`GET /api/v1/live2d/avatars/{id}/model/{path}`、`GET /api/admin/live2d/health`、`_safe_join`、`_runtime_trees`、`validate_model_dir` 本身（自动探测在它**之前**决定 `entry`，不侵入它）。

## 4. 前端

### 4.1 新增两个共享件

```
admin-ui/src/avatars/KindTag.tsx      真人 → 蓝 Tag；Live2D → 紫 Tag
admin-ui/src/avatars/UploadField.tsx Upload.Dragger 外壳
                                     props: accept / dragText / hint / fileList / onChange
```

`UploadField` 是真正被复用两次的东西（图片视频一次、zip 一次）；`KindTag` 被列表页与详情页头各用一次。

**这就是全部的抽取。** 其余都是删重复：合并后的表格只有一份列定义，所以不需要抽列模块；`CopyableId` 合并后只有一处调用方，留在 `Avatars.tsx` 里。

### 4.2 `Avatars.tsx`（713 → 约 380 行）

四张卡片 → 两张。

**新建形象**（一张 Card）

- 顶部 `Segmented [真人视频 | Live2D]`，绑定 `kind` state
- `名称`：`Form.Item`，两个 kind 都要
- 以下按 `kind` 整块替换：
  - `photoreal`：`UploadField(accept="image/*,video/*")` + 「高级参数」Collapse（空闲视频 CRF / x264 preset）
  - `live2d`：`UploadField(accept=".zip,application/zip")` + 入口文件 `Input` + 表情映射 `TextArea`
- 提交按钮统一叫「上传创建」
- 两个 `UploadField` **各有独立的 `fileList` state**，切类型不会把选中的 zip 带进照片分支
- `onFinish` 按 `kind` 分派：`photoreal` 打现有 `POST /api/admin/avatars`（FormData，含 idle 覆盖），`live2d` 打 `POST /api/admin/live2d/avatars`（FormData，含 entry / expressions）
- `creatingGuard` 一个实例守两边（`reentryGuard.ts`，同步取锁，两次点击只落一个请求）
- 视频时长 ≤15 秒的前端预检只对 `photoreal` 生效

**形象列表**（一张表）

8 列 = 形象 ID / 名称 / 类型 / 驱动方式 / 警告 / 状态 / 创建时间 / 操作。

只有两列按 `kind` 分叉：

| 列 | photoreal | live2d |
|---|---|---|
| 驱动方式 | `图片驱动` / `视频驱动`（读 `config.musetalk.mode`，构建完成前是 `—`） | `Cubism` |
| 警告 | 现有的 Badge | `—`（没有 pack 质量警告这个概念） |

其余五列完全共用。删除 `photoItems` / `live2dItems` 两个 `useMemo`，`items` 直接进 `dataSource`。行点击仍导航到 `/avatars/:id`，删除按钮仍走各自的端点（`DELETE /api/admin/avatars/{id}` vs `/api/admin/live2d/avatars/{id}`）——合并前端表格不等于合并后端删除语义。

### 4.3 `AvatarDetail.tsx` 按 kind 分叉

共用头部：返回按钮 / 名称 / `KindTag` / `StatusTag` / Descriptions(形象 ID, 类型, 创建时间) / error Alert。下面二选一。

**`photoreal` 分支：现有内容一行不改**（预览卡、构建过程 Steps、产物表、底层配置、「按当前设置重建」按钮）。

**`live2d` 分支**：

| 卡片 | 内容 |
|---|---|
| 预览 | `<Live2DStage modelUrl={live2dModelUrl(id, entry)} audio={null} />`，走 idle 层，嘴不动。组件已被 Debug / Chat / Pet 四个页面共用，这里直接套 |
| 模型信息 | Descriptions：入口文件、表情映射（从 `config.expressions` 渲染成 Tag）、文件数、总大小 |
| 模型文件 | **复用现有的产物表**。`/api/admin/avatars/{id}/artifacts` 就是 `rglob` 整个形象目录（`server/api/admin.py:40-66`），Live2D 的 moc3 / 纹理 / 动作就在那儿，下载按钮照旧能用 |

Live2D 分支**明确不渲染**：

- 「按当前设置重建」按钮——Live2D 没有 `AvatarJob`，`retry_avatar`（`server/api/admin.py:1027`）会建一个 job 并跑流水线，必然失败
- 「构建过程」Steps——没有流水线阶段
- 「底层配置」卡——没有 `config.musetalk`
- 不调用 `/timeline` 与 `/jobs` 两个请求

## 5. 数据流

```
用户拖 zip → POST /api/admin/live2d/avatars (multipart)
  → 解压到 data/avatars/<id>/
  → validate_model_dir  ← 逐项核对 model3.json 的 FileReferences
  → INSERT avatar(kind='live2d', status='ready', config_json={model:{entry}})
  → 201 { avatar_id, config }
  → 前端 refresh() 拉 GET /api/admin/avatars，新行出现在同一张表
  → 点行进详情 → <Live2DStage> 拉 GET /api/v1/live2d/avatars/<id>/model/<entry>
```

失败路径：解压/校验任一失败 → `rmtree` 目录 → 422 带具体文件名 → 前端 `message.error` 原样显示，不刷新列表。

## 6. 测试

### 6.1 后端（pytest）

`tests/test_live2d_hosting.py` 现有的四个注册用例改写成「上传 zip」，另加：

- zip-slip 条目（`../../.env`）被拒，且**目录已清理、无 Avatar 行**
- 符号链接条目被拒
- 超 `live2d_max_archive_entries` 被拒
- 超 `live2d_max_unpacked_bytes` 被拒
- 超 `avatar_max_upload_bytes` 的 zip 直接 413
- entry 自动探测：根目录 / 子目录 / 零候选 / 多候选
- 坏 zip（非 zip 字节流）被拒

### 6.2 前端（vitest）

`admin-ui` **没有** `@testing-library/react` 与 jsdom——`vite.config.ts:34-41` 的 `test` 段没设 `environment`，`package.json` 的 devDependencies 里也没有。现有前端测试（`reentryGuard` / `analyze` / `frame` / `params` / `idle` / `expressions`）全是纯逻辑。

按 `reentryGuard.ts` 的既有先例——它当初就是为了能被 `node --test` 跑（见 `vite.config.ts:36-40` 的注释）而从页面里抽出来的纯函数——把判断逻辑抽成纯函数再测，不引入新依赖：

```
admin-ui/src/avatars/describe.ts + describe.test.ts
  driveLabel(avatar)   -> "图片驱动" | "视频驱动" | "Cubism"
  parseExpressions(raw) -> { ok: true, value } | { ok: false, error }
```

`parseExpressions` 是从现在的 `Avatars.tsx:239-256` 原样搬出来的（JSON 语法检查 + 必须是非数组对象）。组件本身不测渲染。

### 6.3 人工验证

上传一个真实 Hiyori zip → 列表出现新行 → 详情页画布出现会呼吸的模型 → 会话调试页选中它能说话。

## 7. 交付顺序

1. **后端**：新 setting → 安全解压 → multipart 端点 → 删 JSON 端点 → pytest
2. **前端纯逻辑**：`describe.ts` + 测试
3. **前端页面**：`KindTag` / `UploadField` → `Avatars.tsx` 改写 → `AvatarDetail.tsx` 分叉
4. **文档**：README 端点表

每步各自可验证、可提交。1 和 3 之间没有耦合，2 必须在 3 之前。