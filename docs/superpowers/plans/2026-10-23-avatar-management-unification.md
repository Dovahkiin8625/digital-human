# 形象管理统一 + Live2D 模型上传 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把形象管理页从四张卡片收成一张表 + 一个创建入口，并让用户直接在浏览器上传 Live2D 模型压缩包。

**Architecture:** 后端唯一改动是把 `POST /api/admin/live2d/avatars` 从 JSON 换成 multipart，新增一个安全解压模块；photoreal 端点一行不动。前端新增两个共享组件（`KindTag` / `UploadField`）和两个纯函数（`describe.ts`），`Avatars.tsx` 与 `AvatarDetail.tsx` 按 `Avatar.kind` 分叉渲染。

**Tech Stack:** Python 3.11+ / FastAPI / `zipfile` / SQLAlchemy；React 18 + Vite + TS + antd 5 + vitest。

**Spec:** `docs/superpowers/specs/2026-10-23-avatar-management-unification-design.md`

## Global Constraints

- `Avatar.kind` 取值只有 `"photoreal"` 与 `"live2d"`，前端 `AvatarKind` 必须与之一致。
- Live2D 的 `config_json` 形状固定为 `{ model: { entry: string }, expressions?: Record<string, number> }`。`entry` 是**相对形象目录**的 POSIX 路径，存进 `config.model.entry` 供 `core.ts:modelFileUrl` 拼 URL。
- 任何注册失败都必须：`shutil.rmtree` 目标目录 + **不建 Avatar 行** + 返回 422 且 `error.message` 指名具体缺失文件。
- 解压安全三档上限（`server/config.py`）：`live2d_max_archive_entries = 512`、`live2d_max_unpacked_bytes = 512_000_000`、`live2d_max_entry_bytes = 64_000_000`。
- 压缩包体积上限沿用 `avatar_max_upload_bytes`（300 MB），不解压即拒。
- 前端所有用户可见文案用中文，与现有页面一致。
- **不合并后端端点**：`POST /api/admin/avatars` 与 `POST /api/admin/live2d/avatars` 并存；`DELETE /api/admin/avatars/{id}` 与 `DELETE /api/admin/live2d/avatars/{id}` 并存，合并后的前端表格按行的 `kind` 选端点。
- **不引入 `@testing-library/react` / jsdom**。`vite.config.ts` 的 `test` 段没有 `environment`，前端测试只能是纯逻辑。
- photoreal 的行为不得改变：创建流程、删除流程、详情页内容与现有截图级一致。

## Review Focus

规格是愿景文档，它没有逐条穷举输入。下面五类是规格隐含、但任何测试都不会自动覆盖、且最可能咬到真实用户的输入——每一类在下方都有钉住它的测试。

1. **macOS 压缩包里的 `__MACOSX/._model.model3.json`。** Finder 压缩会附带 AppleDouble 影子文件，它同样以 `.model3.json` 结尾。不过滤就会让自动探测数出两个候选，把一次完全合法的上传变成 422。→ Task 1 Step 4
2. **工程在压缩包子目录里**（Windows 右键压缩 `Hiyori/` 得到 `Hiyori/model.model3.json`）。这是最常见的真实形态，写死 `model.model3.json` 必然失败。→ Task 1 Step 4、Task 2 Step 4
3. **高压缩比压缩包**（几 KB 展开成几百 MB 的全零文件）。只挡压缩包体积挡不住这个。→ Task 1 Step 3
4. **条目名为 `../../.env` 或 `C:\Windows\x`** 的压缩包。zip-slip 与 Windows 盘符路径是两种不同的逃逸，都要拒。→ Task 1 Step 3
5. **切换类型后残留的选中文件**。选了照片再切到 Live2D，若两个上传区共用一份 `fileList`，点创建会把 zip 当图片发出去（反之亦然）。组件测试在这个仓库不可用（无 jsdom），所以这条靠 Task 5 Step 2 的「两份独立 state」不变量 + 文末人工验证第 6 条钉住。

---

### Task 1: 安全解压与入口探测

`server/live2d/archive.py` 全新模块。`extract_archive` 逐条目解压并在每次写入前校验目标路径；`find_entries` / `resolve_entry` 在解压后的目录里定位 `model3.json`。配套三个 setting 在 `server/config.py`。本任务纯单元测试，不碰 HTTP。

**Files:**
- Create: `server/live2d/archive.py`
- Modify: `server/config.py:68-72`（在 `live2d_sdk_dir` 之后追加三个字段）
- Test: `tests/test_live2d_archive.py`

**Interfaces:**
- Consumes: `server.core.errors.AppError`, `server.config.get_settings()`。
- Produces:
  - `extract_archive(src: Path, dest: Path, *, max_entries: int, max_unpacked_bytes: int, max_entry_bytes: int) -> None`
  - `find_entries(root: Path) -> list[str]` —— 相对 root 的 POSIX 路径，已排序
  - `resolve_entry(root: Path, requested: str | None) -> str` —— 失败抛 `AppError("live2d_invalid_model", ..., 422)`
  - `Settings.live2d_max_archive_entries: int`、`Settings.live2d_max_unpacked_bytes: int`、`Settings.live2d_max_entry_bytes: int`

- [ ] **Step 1: 写失败的测试**

创建 `tests/test_live2d_archive.py`：

```python
"""Safe extraction of an uploaded Cubism project.

An avatar upload is operator input an attacker can shape: a zip can name a
member ``../../.env``, ship a symlink aimed at the data dir, or expand a
few hundred KB into hundreds of MB. These tests pin the boundary -- what
lands on disk, and what never does.
"""

import io
import json
import stat
import zipfile
from pathlib import Path

import pytest

from server.core.errors import AppError
from server.live2d.archive import extract_archive, find_entries, resolve_entry

LIMITS = dict(max_entries=512, max_unpacked_bytes=512_000_000, max_entry_bytes=64_000_000)

MODEL = {
    "Version": 3,
    "FileReferences": {"Moc": "model.moc3", "Textures": ["texture_00.png"]},
}


def _zip_bytes(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _archive(tmp_path: Path, entries: dict[str, bytes], name: str = "m.zip") -> Path:
    path = tmp_path / name
    path.write_bytes(_zip_bytes(entries))
    return path


def _flat_project(**extra: str) -> dict[str, bytes]:
    files = {
        "model.model3.json": json.dumps(MODEL).encode(),
        "model.moc3": b"\x00" * 16,
        "texture_00.png": b"\x89PNG",
    }
    files.update(extra)
    return files


def test_extracts_a_flat_project(tmp_path):
    src = _archive(tmp_path, _flat_project())
    dest = tmp_path / "out"
    extract_archive(src, dest, **LIMITS)
    assert (dest / "model.moc3").read_bytes() == b"\x00" * 16
    assert json.loads((dest / "model.model3.json").read_text())["Version"] == 3


def test_refuses_a_parent_escape(tmp_path):
    src = _archive(tmp_path, {**{f"ok{i}.bin": b"x" for i in range(3)}, "../../.env": b"secret"})
    with pytest.raises(AppError) as exc:
        extract_archive(src, tmp_path / "out", **LIMITS)
    assert "illegal path" in exc.value.message
    assert not (tmp_path / ".env").exists()


@pytest.mark.parametrize("name", ["/etc/passwd", "C:\\Windows\\win.ini", "\\\\server\\share\\x"])
def test_refuses_an_absolute_name(tmp_path, name):
    """A zip written on one OS can carry a path the other one would honour.
    Whichever spelling arrives, only names that start under the destination
    are written."""
    src = _archive(tmp_path, {name: b"secret"})
    with pytest.raises(AppError) as exc:
        extract_archive(src, tmp_path / "out", **LIMITS)
    assert "illegal path" in exc.value.message


def test_refuses_a_symlink(tmp_path):
    """A planted link is a write primitive for a LATER member, and a trap
    for anything that later follows it. One attribute read to refuse."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        info = zipfile.ZipInfo("link")
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        zf.writestr(info, "/etc/passwd")
    src = tmp_path / "m.zip"
    src.write_bytes(buf.getvalue())
    with pytest.raises(AppError) as exc:
        extract_archive(src, tmp_path / "out", **LIMITS)
    assert "symbolic link" in exc.value.message


def test_refuses_too_many_entries(tmp_path):
    entries = {f"f{i}.bin": b"x" for i in range(5)}
    src = _archive(tmp_path, entries)
    with pytest.raises(AppError) as exc:
        extract_archive(src, tmp_path / "out", **{**LIMITS, "max_entries": 4})
    assert "5 entries" in exc.value.message


def test_refuses_an_oversized_total(tmp_path):
    """The whole point of the unpacked cap: 4 MB of zeros compresses to a
    few hundred bytes, so a 300 MB archive-size limit does nothing here."""
    src = _archive(tmp_path, {"bomb.bin": b"\0" * 4_000_000})
    with pytest.raises(AppError) as exc:
        extract_archive(src, tmp_path / "out", **{**LIMITS, "max_unpacked_bytes": 1_000})
    assert "1000 bytes" in exc.value.message


def test_refuses_an_oversized_single_entry(tmp_path):
    src = _archive(tmp_path, {"big.bin": b"\0" * 2_000_000})
    with pytest.raises(AppError) as exc:
        extract_archive(src, tmp_path / "out", **{**LIMITS, "max_entry_bytes": 1_000})
    assert "big.bin" in exc.value.message


def test_a_lying_header_cannot_inflate_the_archive(tmp_path):
    """The uncompressed-size field is attacker-controlled: a hostile zip
    claims 8 bytes and hides a deflate bomb behind the claim. Whatever the
    archive does, no more than the declared size may reach the disk."""
    payload = b"\0" * 4_000_000
    raw = bytearray(_zip_bytes({"bomb.bin": payload}))
    # Local file header carries the size at 22; the central-directory
    # record ("PK\\x01\\x02") carries it at +24. Patch both so the header
    # and the directory agree on the smaller lie.
    raw[22:26] = (8).to_bytes(4, "little")
    cd = bytes(raw).index(b"PK\x01\x02")
    raw[cd + 24: cd + 28] = (8).to_bytes(4, "little")
    src = tmp_path / "m.zip"
    src.write_bytes(bytes(raw))

    dest = tmp_path / "out"
    try:
        extract_archive(src, dest, **LIMITS)
    except AppError:
        pass  # a counter caught it -- also acceptable
    target = dest / "bomb.bin"
    assert not target.exists() or target.stat().st_size <= 8


def test_reports_a_body_that_is_not_a_zip(tmp_path):
    src = tmp_path / "m.zip"
    src.write_bytes(b"this is not a zip archive")
    with pytest.raises(AppError) as exc:
        extract_archive(src, tmp_path / "out", **LIMITS)
    assert "not a readable zip" in exc.value.message


# ----- entry discovery ----------------------------------------------------

def test_finds_a_root_level_entry(tmp_path):
    root = tmp_path / "m"
    for name, data in _flat_project().items():
        (root / name).write_bytes(data)
    assert find_entries(root) == ["model.model3.json"]


def test_finds_an_entry_inside_the_folder_a_zip_keeps(tmp_path):
    """Right-clicking Hiyori/ in Explorer zips the folder too, so the entry
    lands at Hiyori/model.model3.json. This is the common real-world shape;
    a hardcoded model.model3.json would fail on it."""
    root = tmp_path / "m"
    for name, data in _flat_project().items():
        target = root / "Hiyori" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    assert find_entries(root) == ["Hiyori/model.model3.json"]


def test_ignores_macos_resource_forks(tmp_path):
    """Finder adds an __MACOSX/ tree of AppleDouble files named ``._<original>``.
    Those still end in .json, so unfiltered they double the candidate count
    and turn a valid upload into an ambiguous one."""
    root = tmp_path / "m"
    for name, data in _flat_project().items():
        (root / name).write_bytes(data)
    macosx = root / "__MACOSX"
    macosx.mkdir()
    (macosx / "._model.model3.json").write_bytes(b"\x00\x05\x16\x07")
    assert find_entries(root) == ["model.model3.json"]


def test_resolve_entry_returns_an_explicit_choice(tmp_path):
    root = tmp_path / "m"
    root.mkdir()
    (root / "a.model3.json").write_text("{}")
    (root / "b.model3.json").write_text("{}")
    assert resolve_entry(root, "b.model3.json") == "b.model3.json"


def test_resolve_entry_names_the_candidates_when_ambiguous(tmp_path):
    root = tmp_path / "m"
    root.mkdir()
    (root / "a.model3.json").write_text("{}")
    (root / "b.model3.json").write_text("{}")
    with pytest.raises(AppError) as exc:
        resolve_entry(root, None)
    assert "a.model3.json" in exc.value.message
    assert "b.model3.json" in exc.value.message


def test_resolve_entry_reports_an_archive_with_no_model(tmp_path):
    root = tmp_path / "m"
    root.mkdir()
    (root / "readme.txt").write_text("hi")
    with pytest.raises(AppError) as exc:
        resolve_entry(root, None)
    assert "model3.json" in exc.value.message
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd E:/project/digital-human && .venv/Scripts/python -m pytest tests/test_live2d_archive.py -q
```

Expected: FAIL — `ModuleNotFoundError: No module named 'server.live2d.archive'`

- [ ] **Step 3: 写实现**

创建 `server/live2d/archive.py`：

```python
"""Safe extraction of an uploaded Cubism project archive.

A zip is the only sane way to hand a browser a Cubism project -- it is a
dozen interlinked files -- but it is also attacker-controlled input. A
member can be named ``../../.env``, be a symlink aimed at the data dir, or
expand a few hundred KB into hundreds of MB. ``ZipFile.extractall`` checks
none of these before Python 3.12, so extraction here is member by member:
every write is validated against the destination root, and every bomb
metric is counted as it goes.
"""

import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath

from server.core.errors import AppError

_BAD = "live2d_invalid_model"


def _fail(message: str) -> AppError:
    return AppError(_BAD, message, 422)


def _is_absolute(name: str) -> bool:
    """True when ``name`` names somewhere outside any destination.

    Both flavours are checked because the two spellings mean different
    things on different hosts: ``PurePosixPath("/etc/passwd")`` is
    absolute but ``PureWindowsPath("/etc/passwd")`` is not, while
    ``PureWindowsPath("C:/x")`` is absolute and ``PurePosixPath("C:/x")``
    is a perfectly legal relative name. An archive written on one OS must
    not be honoured on the other.
    """
    return PurePosixPath(name).is_absolute() or PureWindowsPath(name).is_absolute()


def _member_target(dest: Path, name: str) -> Path:
    """Resolve one member under ``dest``, or refuse.

    Three escapes, and ``is_relative_to`` alone catches only the last: a
    ``..`` segment, an absolute or drive-qualified name, and a name that
    lands on a symlink an earlier member planted. The first two are
    checked textually because on some hosts they do not even survive
    ``dest / name``.
    """
    if (
        not name
        or _is_absolute(name)
        or ".." in PurePosixPath(name).parts
        or ".." in PureWindowsPath(name).parts
    ):
        raise _fail(f"illegal path in archive: {name!r}")
    target = (dest / name).resolve()
    if not target.is_relative_to(dest.resolve()):
        raise _fail(f"illegal path in archive: {name!r}")
    return target


def _reject_link(info: zipfile.ZipInfo) -> None:
    """Refuse a member that is a symbolic link.

    ``ZipFile.open`` on a link member yields whatever bytes the archiver
    stored after the link path, not the pointed-at file, so extraction
    alone does not read through it. The danger is a planted link that a
    LATER member writes through, and a member any later build step
    follows. Neither is worth carrying, and the check costs one read.
    """
    if stat.S_ISLNK(info.external_attr >> 16):
        raise _fail(f"archive contains a symbolic link: {info.filename!r}")


def extract_archive(
    src: Path,
    dest: Path,
    *,
    max_entries: int,
    max_unpacked_bytes: int,
    max_entry_bytes: int,
) -> None:
    """Extract ``src`` into ``dest``, refusing every unsafe shape.

    The declared uncompressed size is counted before anything is written,
    so a bomb is rejected without first filling the disk with it. The
    copy loop counts the bytes that actually arrive as well: the header
    is attacker-controlled, and while CPython's ``ZipExtFile`` bounds its
    reads by the declared size (so a liar cannot inflate), that is a
    property of one implementation and not the reason to be correct.

    On any failure the partially written tree is left in place for the
    caller to remove -- this function does not own ``dest``.
    """
    dest = Path(dest)
    try:
        with zipfile.ZipFile(src) as zf:
            members = zf.infolist()
            if len(members) > max_entries:
                raise _fail(f"archive has {len(members)} entries; limit is {max_entries}")

            declared_total = 0
            written_total = 0
            for info in members:
                _reject_link(info)
                if info.is_dir():
                    # Directory entries carry no bytes; parents are created
                    # on demand by the member writes below.
                    continue
                name = info.filename
                if info.file_size > max_entry_bytes:
                    raise _fail(
                        f"{name} unpacks to {info.file_size} bytes; "
                        f"per-file limit is {max_entry_bytes}"
                    )
                declared_total += info.file_size
                if declared_total > max_unpacked_bytes:
                    raise _fail(f"archive unpacks to more than {max_unpacked_bytes} bytes")

                target = _member_target(dest, name)
                target.parent.mkdir(parents=True, exist_ok=True)
                written = 0
                with zf.open(info) as member, target.open("wb") as out:
                    while True:
                        chunk = member.read(1024 * 1024)
                        if not chunk:
                            break
                        written += len(chunk)
                        if written > max_entry_bytes:
                            raise _fail(
                                f"{name} unpacks past the per-file limit of "
                                f"{max_entry_bytes} bytes"
                            )
                        out.write(chunk)
                written_total += written
                if written_total > max_unpacked_bytes:
                    raise _fail(f"archive unpacks to more than {max_unpacked_bytes} bytes")
    except zipfile.BadZipFile as exc:
        raise _fail(f"not a readable zip archive: {exc}") from exc


def find_entries(root: Path) -> list[str]:
    """Every ``*.model3.json`` in an extracted project, as sorted POSIX paths.

    macOS archives add an ``__MACOSX/`` sibling tree of AppleDouble files
    named ``._<original>``. Those names still end in ``.json``, so
    without the filter they double the count and make a valid upload look
    ambiguous.
    """
    root = Path(root)
    return sorted(
        p.relative_to(root).as_posix()
        for p in root.rglob("*.model3.json")
        if p.is_file()
        and not p.name.startswith("._")
        and "__MACOSX" not in p.relative_to(root).parts
    )


def resolve_entry(root: Path, requested: str | None) -> str:
    """The ``model3.json`` to register, or an error naming the candidates.

    An operator who leaves the field blank gets auto-detection, which is
    what makes the common right-click-compress shape work: the folder is
    inside the zip, so the entry is ``Hiyori/model.model3.json`` and not
    the ``model.model3.json`` a hardcoded default would claim.
    """
    found = find_entries(root)
    if requested:
        return requested
    if len(found) == 1:
        return found[0]
    if not found:
        raise _fail("no *.model3.json in the archive; is this a Cubism project?")
    raise _fail(
        "the archive holds several models -- name one explicitly: " + ", ".join(found[:10])
    )
```

在 `server/config.py` 的 `live2d_sdk_dir: Path = Path("data/CubismSdkForWeb-5-r.5")` 之后、`model_config` 之前插入：

```python
    # A Live2D upload is an operator-supplied zip of a whole Cubism project.
    # Three limits, because the archive size alone does not bound what an
    # upload does: a few hundred KB of zeros can expand to hundreds of MB,
    # and a small archive can declare a million members.
    live2d_max_archive_entries: int = 512
    live2d_max_unpacked_bytes: int = 512_000_000
    live2d_max_entry_bytes: int = 64_000_000
```

- [ ] **Step 4: 运行测试确认通过**

```bash
cd E:/project/digital-human && .venv/Scripts/python -m pytest tests/test_live2d_archive.py -q
```

Expected: PASS（17 passed）

- [ ] **Step 5: 提交**

```bash
cd E:/project/digital-human && git add server/live2d/archive.py server/config.py tests/test_live2d_archive.py && git commit -m "feat(live2d): extract uploaded Cubism projects under a zip-slip guard"
```

---

### Task 2: multipart 注册端点

把 `POST /api/admin/live2d/avatars` 从 JSON 换成 multipart，用 Task 1 的解压与探测；删掉 `RegisterIn` 和旧的注册函数。`tests/test_live2d_hosting.py` 的注册用例改写成上传 zip。README 的端点表同步。

**Files:**
- Modify: `server/api/live2d.py:14-16`（imports）、`:69-113`（端点）
- Modify: `tests/test_live2d_hosting.py:75-129`（注册用例）
- Modify: `README.md:213`（端点表那一行）

**Interfaces:**
- Consumes: `server.live2d.archive.extract_archive / resolve_entry`（Task 1）、`server.avatar.store.AvatarStore`、`server.live2d.registry.validate_model_dir`、`server.api.avatars._serialize`。
- Produces: `POST /api/admin/live2d/avatars` multipart，字段 `name: str`、`file: UploadFile`、`entry: str = ""`、`expressions: str = ""`；响应 `_serialize(row, with_detail=True)`。

- [ ] **Step 1: 写失败的测试**

在 `tests/test_live2d_hosting.py` 顶部（`LIVE2D_ID = "a" * 32` 之后）加辅助函数：

```python
def _project_zip(prefix: str = "") -> bytes:
    """A Cubism project as a browser would send it.

    ``prefix`` puts every file inside a folder, which is what right-click
    compress produces and what the server has to cope with.
    """
    import io
    import zipfile

    entries = {
        "model.model3.json": json.dumps({
            "Version": 3,
            "FileReferences": {
                "Moc": "model.moc3",
                "Textures": ["texture_00.png"],
                "Expressions": [{"Name": "0", "File": "0.exp3.json"}],
            },
        }).encode(),
        "model.moc3": b"\x00" * 16,
        "texture_00.png": b"\x89PNG\r\n\x1a\n",
        "0.exp3.json": b"{}",
    }
    if prefix:
        entries = {f"{prefix}/{k}": v for k, v in entries.items()}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _zip_without(filename: str) -> bytes:
    """A project zip with one member missing, for the validation failures."""
    entries = {
        "model.model3.json": json.dumps({
            "Version": 3,
            "FileReferences": {"Moc": "model.moc3", "Textures": ["texture_00.png"]},
        }).encode(),
        "model.moc3": b"\x00" * 16,
        "texture_00.png": b"\x89PNG",
    }
    entries.pop(filename)
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _upload(client, name="Hiyori", prefix="", entry="", expressions="", raw=None):
    data = _project_zip(prefix) if raw is None else raw
    form = {"name": name}
    if entry:
        form["entry"] = entry
    if expressions:
        form["expressions"] = expressions
    return client.post(
        "/api/admin/live2d/avatars",
        data=form,
        files={"file": ("model.zip", data, "application/zip")},
    )
```

把 `test_live2d_hosting.py:77-129` 的四个注册用例整体替换为：

```python
def test_upload_extracts_and_validates(client):
    resp = _upload(client, expressions='{"smile": 0}')
    assert resp.status_code == 201, resp.text
    avatar_id = resp.json()["avatar_id"]
    body = resp.json()
    assert body["kind"] == "live2d"
    assert body["status"] == "ready"
    # The response carries the config so the console can render the entry
    # without a second round trip.
    assert body["config"]["model"]["entry"] == "model.model3.json"
    root = client.app.state.session_manager.store.avatar_dir(avatar_id)
    assert (root / "model.moc3").read_bytes() == b"\x00" * 16


def test_upload_detects_an_entry_inside_a_subfolder(client):
    """The shape right-click compress produces. A hardcoded entry name
    would fail on exactly this."""
    resp = _upload(client, prefix="Hiyori")
    assert resp.status_code == 201, resp.text
    assert resp.json()["config"]["model"]["entry"] == "Hiyori/model.model3.json"
    avatar_id = resp.json()["avatar_id"]
    served = client.get(
        f"/api/v1/live2d/avatars/{avatar_id}/model/Hiyori/model.model3.json"
    )
    assert served.status_code == 200


def test_upload_rejects_a_broken_model(client):
    resp = _upload(client, name="broken", raw=_zip_without("model.moc3"))
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "live2d_invalid_model"
    assert "moc" in resp.json()["error"]["message"]


def test_upload_rejects_a_body_that_is_not_a_zip(client):
    resp = _upload(client, raw=b"not a zip at all")
    assert resp.status_code == 422
    assert "zip" in resp.json()["error"]["message"]


def test_upload_rejects_an_expression_the_model_lacks(client):
    resp = _upload(client, expressions='{"smile": 9}')
    assert resp.status_code == 422


def test_upload_rejects_malformed_expressions(client):
    resp = _upload(client, expressions="{not json")
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "live2d_invalid_config"


def test_a_rejected_upload_leaves_nothing_behind(client):
    store = client.app.state.session_manager.store
    before = set(p.name for p in (store.data_dir / "avatars").glob("*")) \
        if (store.data_dir / "avatars").exists() else set()
    resp = _upload(client, raw=_zip_without("model.moc3"))
    assert resp.status_code == 422
    after = set(p.name for p in (store.data_dir / "avatars").glob("*")) \
        if (store.data_dir / "avatars").exists() else set()
    assert after == before
    # And no row was written.
    assert client.get("/api/admin/avatars?kind=live2d").json() == []


def test_upload_refuses_a_zip_slip_entry(client):
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("model.model3.json", "{}")
        zf.writestr("../../.env", "secret")
    resp = _upload(client, raw=buf.getvalue())
    assert resp.status_code == 422
    assert "illegal path" in resp.json()["error"]["message"]


def test_upload_refuses_an_oversized_body(client, monkeypatch):
    monkeypatch.setenv("DH_AVATAR_MAX_UPLOAD_BYTES", "1024")
    get_settings.cache_clear()
    resp = _upload(client, raw=b"\x00" * 4096)
    assert resp.status_code == 413
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd E:/project/digital-human && .venv/Scripts/python -m pytest tests/test_live2d_hosting.py -q
```

Expected: FAIL —— 用例全部 422/415（multipart 的 `file` 字段当前只接受 JSON body，`RegisterIn` 的 `model_config = {"extra": "forbid"}` 会把表单字段当额外键拒掉）

- [ ] **Step 3: 写实现**

改 `server/api/live2d.py` 的 imports（`:14-20`）：在 `from fastapi import APIRouter, Depends` 改为

```python
import json
import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, Form, UploadFile
```

并加 `from server.live2d.archive import extract_archive, resolve_entry`。

把 `server/api/live2d.py:69-113` 整段（`class RegisterIn` 到 `register_live2d_avatar` 结束）替换为：

```python
# ----- console surface ----------------------------------------------------

def _save_upload(dest: Path, file: UploadFile) -> None:
    """Stream the archive to ``dest``, refusing an oversized body mid-copy.

    The declared length can lie and is absent entirely on a chunked body,
    so the copy is counted rather than trusted -- the same shape as
    ``AvatarStore.save_source``, which is why the limit is shared too.
    """
    limit = get_settings().avatar_max_upload_bytes
    written = 0
    with dest.open("wb") as out:
        while True:
            chunk = file.file.read(1024 * 1024)
            if not chunk:
                break
            written += len(chunk)
            if written > limit:
                raise AppError("file_too_large", f"upload is larger than {limit} bytes", 413)
            out.write(chunk)


def _parse_expressions(raw: str) -> dict:
    """Parse the operator's ``{"smile": 0}`` map, or refuse it.

    The console rejects a malformed map before submitting, so a failure
    here means a hand-rolled client -- but it must still fail loudly
    rather than register a model whose emotion map silently does nothing.
    """
    raw = (raw or "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except ValueError as exc:
        raise AppError(
            "live2d_invalid_config", f"expressions is not valid JSON: {exc}", 422
        ) from exc
    if not isinstance(parsed, dict) or not all(
        isinstance(k, str) and isinstance(v, int) and not isinstance(v, bool)
        for k, v in parsed.items()
    ):
        raise AppError(
            "live2d_invalid_config",
            'expressions must be an object of {"tag": index}',
            422,
        )
    return parsed


@router.post("/admin/live2d/avatars", status_code=201)
def upload_live2d_avatar(
    name: str = Form(...),
    file: UploadFile = ...,
    entry: str = Form(""),
    expressions: str = Form(""),
    db=Depends(get_db),
) -> dict:
    """Register a Live2D avatar from an uploaded Cubism project archive.

    The archive is extracted into the avatar's own directory, checked with
    the same ``validate_model_dir`` the server-side registration used, and
    only then written to the database -- so a rejected upload leaves no row
    and no files behind, and the shape that IS served is the shape that was
    validated.

    ``entry`` may be blank: the usual way an operator produces a zip is to
    right-click the exported folder, which puts it at
    ``Hiyori/model.model3.json``, so auto-detection is what makes the common
    upload work without the operator retyping a path they never knew.
    """
    config: dict = {"expressions": _parse_expressions(expressions)}

    settings = get_settings()
    avatar_id = uuid4().hex
    store = _store()
    target = store.avatar_dir(avatar_id)
    scratch = Path(settings.data_dir) / "tmp" / "live2d_upload"
    scratch.mkdir(parents=True, exist_ok=True)
    archive = scratch / f"{avatar_id}.zip"

    target.mkdir(parents=True, exist_ok=True)
    try:
        _save_upload(archive, file)
        extract_archive(
            archive,
            target,
            max_entries=settings.live2d_max_archive_entries,
            max_unpacked_bytes=settings.live2d_max_unpacked_bytes,
            max_entry_bytes=settings.live2d_max_entry_bytes,
        )
        config["model"] = {"entry": resolve_entry(target, entry.strip() or None)}
        validate_model_dir(target, config)
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        raise
    finally:
        archive.unlink(missing_ok=True)

    row = Avatar(
        id=avatar_id,
        name=name,
        status="ready",
        source_image_path="",
        pack_path=None,
        kind="live2d",
        config_json=config,
    )
    db.add(row)
    db.commit()
    return _serialize(row, with_detail=True)
```

- [ ] **Step 4: 运行测试确认通过**

```bash
cd E:/project/digital-human && .venv/Scripts/python -m pytest tests/test_live2d_hosting.py tests/test_live2d_archive.py tests/test_live2d_registry.py -q
```

Expected: PASS（全部通过，含 `test_a_rejected_upload_leaves_nothing_behind`）

- [ ] **Step 5: 更新 README 端点表**

把 `README.md:213` 那一行：

```
| POST | `/api/admin/live2d/avatars` | JSON `{name, model_dir, config}` → 复制并校验 Cubism 工程后建 `kind=live2d` 行。校验失败 422，消息指出具体缺失文件 |
```

替换为：

```
| POST | `/api/admin/live2d/avatars` | multipart `name` + `file`(整个 Cubism 工程的 zip) + `entry`(可选) + `expressions`(可选 JSON) → 解压并校验后建 `kind=live2d` 行，返回值含解析后的 `config.model.entry`。`entry` 留空则自动探测 `*.model3.json`（工程压缩在子目录里也能找到）。校验失败 422 并清理已解压文件，消息指出具体缺失文件 |
```

- [ ] **Step 6: 提交**

```bash
cd E:/project/digital-human && git add server/api/live2d.py tests/test_live2d_hosting.py README.md && git commit -m "feat(live2d): accept a Cubism project as an uploaded zip"
```

---

### Task 3: 前端纯逻辑

`describe.ts` 是列表页「驱动方式」列与创建表单表情校验的判断逻辑，抽成纯函数以便 vitest 覆盖——`admin-ui` 没有 jsdom，组件本身测不了。

**Files:**
- Create: `admin-ui/src/avatars/describe.ts`
- Test: `admin-ui/src/avatars/describe.test.ts`

**Interfaces:**
- Consumes: `admin-ui/src/types.ts` 的 `AvatarInfo`、`Live2DModelConfig`。
- Produces:
  - `driveLabel(avatar: Pick<AvatarInfo, "kind" | "config">): "图片驱动" | "视频驱动" | "Cubism" | null`
  - `parseExpressions(raw: string): { ok: true; value: Record<string, number> } | { ok: false; error: string }`

- [ ] **Step 1: 写失败的测试**

创建 `admin-ui/src/avatars/describe.test.ts`：

```ts
import { describe, expect, it } from "vitest"
import { driveLabel, parseExpressions } from "./describe"

function avatar(config: Record<string, unknown>, kind = "photoreal") {
  return { kind, config } as never
}

describe("driveLabel", () => {
  // The one cell that differs between the two kinds, so it is the one
  // worth testing: a wrong label here mislabels every row in the table.
  it("names the Cubism runtime for a live2d avatar", () => {
    expect(driveLabel(avatar({}, "live2d"))).toBe("Cubism")
  })

  it("names the drive mode the pipeline recorded", () => {
    expect(driveLabel(avatar({ musetalk: { mode: "video" } }))).toBe("视频驱动")
    expect(driveLabel(avatar({ musetalk: { mode: "image" } }))).toBe("图片驱动")
  })

  // pack.json is written by the build; before it lands there is genuinely
  // no mode to report, and an em dash is the honest cell.
  it("returns null before the pack exists", () => {
    expect(driveLabel(avatar({}))).toBeNull()
    expect(driveLabel(avatar({ musetalk: {} }))).toBeNull()
  })

  it("survives a live2d row with no config block at all", () => {
    expect(driveLabel({ kind: "live2d" } as never)).toBe("Cubism")
  })
})

describe("parseExpressions", () => {
  it("accepts an empty field as an empty map", () => {
    expect(parseExpressions("")).toEqual({ ok: true, value: {} })
    expect(parseExpressions("   ")).toEqual({ ok: true, value: {} })
  })

  it("accepts a well-formed map", () => {
    expect(parseExpressions('{"smile": 0, "think": 2}')).toEqual({
      ok: true,
      value: { smile: 0, think: 2 },
    })
  })

  // Caught here rather than server-side: a JSON syntax error is the
  // operator's typo, and the server's 422 would say only that the map is
  // wrong, not that the braces are unbalanced.
  it("names the syntax error", () => {
    expect(parseExpressions("{not json")).toEqual({
      ok: false,
      error: "表情映射不是合法 JSON",
    })
  })

  it("rejects a non-object", () => {
    expect(parseExpressions("[0, 1]")).toEqual({
      ok: false,
      error: '表情映射必须是形如 {"smile": 0} 的对象',
    })
    expect(parseExpressions('"smile"').ok).toBe(false)
  })

  // A non-integer here becomes an expressionIndex the Cubism stage cannot
  // use, and the failure surfaces as a blank face rather than an error.
  it("rejects a value that is not a non-negative integer", () => {
    expect(parseExpressions('{"smile": -1}').ok).toBe(false)
    expect(parseExpressions('{"smile": 1.5}').ok).toBe(false)
    expect(parseExpressions('{"smile": "0"}').ok).toBe(false)
  })
})
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd E:/project/digital-human/admin-ui && npx vitest run src/avatars/describe.test.ts
```

Expected: FAIL — `Cannot find module './describe'`

- [ ] **Step 3: 写实现**

创建 `admin-ui/src/avatars/describe.ts`：

```ts
import type { AvatarInfo } from "../types";

/** How one avatar is driven, as the list's 驱动方式 column reads it. */
export type DriveLabel = "图片驱动" | "视频驱动" | "Cubism";

/**
 * The single cell that differs between the two avatar kinds.
 *
 * A Live2D model has no source pack and no drive mode -- it is a Cubism
 * project the browser draws itself -- so it gets the runtime name. A
 * photoreal avatar's mode is only known once the pipeline has written
 * pack.json, so before that this returns null and the table renders a
 * dash rather than guessing from the file suffix.
 */
export function driveLabel(
  avatar: Pick<AvatarInfo, "kind" | "config">
): DriveLabel | null {
  if (avatar.kind === "live2d") return "Cubism";
  const cfg = avatar.config as Record<string, unknown> | null | undefined;
  const prep = (cfg?.musetalk as Record<string, unknown> | undefined) ?? {};
  const mode = prep.mode;
  if (mode === "video") return "视频驱动";
  if (mode === "image") return "图片驱动";
  return null;
}

export type ExpressionsResult =
  | { ok: true; value: Record<string, number> }
  | { ok: false; error: string };

/**
 * Validate the operator's emotion map before it is submitted.
 *
 * The index is checked here rather than left to the server's
 * `validate_model_dir` because a bad value does not fail at registration --
 * it becomes an `expressionIndex` the Cubism stage cannot use and surfaces
 * minutes later as a face that never changes.
 */
export function parseExpressions(raw: string): ExpressionsResult {
  const trimmed = raw.trim();
  if (!trimmed) return { ok: true, value: {} };

  let parsed: unknown;
  try {
    parsed = JSON.parse(trimmed);
  } catch {
    // A syntax error is the operator's typo. The server's 422 would say
    // only that the expression map is wrong, not that the braces are
    // unbalanced, which is the half of the message that helps.
    return { ok: false, error: "表情映射不是合法 JSON" };
  }
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
    return { ok: false, error: '表情映射必须是形如 {"smile": 0} 的对象' };
  }

  const value: Record<string, number> = {};
  for (const [tag, idx] of Object.entries(parsed as Record<string, unknown>)) {
    if (typeof idx !== "number" || !Number.isInteger(idx) || idx < 0) {
      return { ok: false, error: `表情映射「${tag}」的取值必须是非负整数` };
    }
    value[tag] = idx;
  }
  return { ok: true, value };
}
```

- [ ] **Step 4: 运行测试确认通过**

```bash
cd E:/project/digital-human/admin-ui && npx vitest run src/avatars/describe.test.ts
```

Expected: PASS（9 passed）

- [ ] **Step 5: 提交**

```bash
cd E:/project/digital-human && git add admin-ui/src/avatars/describe.ts admin-ui/src/avatars/describe.test.ts && git commit -m "refactor(admin-ui): extract the avatar row and expression-map logic"
```

---

### Task 4: 共享组件与 API 客户端

`KindTag` 被列表页与详情页头各用一次，`UploadField` 是创建表单两条分支共用的拖拽区。同时把 `api.ts` 的 `registerLive2dAvatar`（JSON）换成 `uploadLive2dAvatar`（multipart）。

**Files:**
- Create: `admin-ui/src/avatars/KindTag.tsx`
- Create: `admin-ui/src/avatars/UploadField.tsx`
- Modify: `admin-ui/src/api.ts:77-90`

**Interfaces:**
- Consumes: `admin-ui/src/types.ts` 的 `AvatarKind`、`AvatarInfo`、`Live2DModelConfig`。
- Produces:
  - `KindTag({ kind: AvatarKind })` —— 默认导出
  - `UploadField({ accept, dragText, hint, fileList, onChange })` —— 默认导出
  - `uploadLive2dAvatar(body: { name: string; file: File; entry?: string; expressions?: Record<string, number> }): Promise<AvatarInfo>`
  - `deleteLive2dAvatar(avatarId: string): Promise<void>` —— 不变

- [ ] **Step 1: 创建 `KindTag.tsx`**

```tsx
import { Tag } from "antd";
import type { AvatarKind } from "../types";

// One place that decides how a kind reads. The two kinds are the same
// feature -- a selectable, drivable, deletable avatar -- so this tag is
// the only thing telling them apart, and it must not drift between the
// list and the detail header.
const KINDS: Record<AvatarKind, { color: string; text: string }> = {
  photoreal: { color: "blue", text: "真人" },
  live2d: { color: "purple", text: "Live2D" },
};

export default function KindTag({ kind }: { kind: AvatarKind }) {
  const s = KINDS[kind] ?? { color: "default", text: kind };
  return <Tag color={s.color}>{s.text}</Tag>;
}
```

- [ ] **Step 2: 创建 `UploadField.tsx`**

```tsx
import type { ReactNode } from "react";
import { Upload } from "antd";
import { InboxOutlined } from "@ant-design/icons";
import type { UploadFile } from "antd/es/upload/interface";

export interface UploadFieldProps {
  /** The `accept` filter, e.g. "image/*,video/*" or ".zip,application/zip". */
  accept: string;
  dragText: ReactNode;
  hint: ReactNode;
  fileList: UploadFile[];
  onChange: (next: UploadFile[]) => void;
}

/**
 * The drop zone both create branches use.
 *
 * The two branches take genuinely different files -- a photo or clip
 * versus a zipped Cubism project -- but the shell around them is the
 * same control with the same one-file limit and the same never-upload
 * behaviour, so it lives here once.
 */
export default function UploadField({
  accept,
  dragText,
  hint,
  fileList,
  onChange,
}: UploadFieldProps) {
  return (
    <Upload.Dragger
      accept={accept}
      maxCount={1}
      listType="picture"
      beforeUpload={() => false}
      fileList={fileList}
      onChange={({ fileList: next }) => onChange(next)}
    >
      <p className="ant-upload-drag-icon">
        <InboxOutlined />
      </p>
      <p className="ant-upload-drag-text">{dragText}</p>
      <p className="ant-upload-hint">{hint}</p>
    </Upload.Dragger>
  );
}
```

- [ ] **Step 3: 改 `api.ts`**

把 `admin-ui/src/api.ts:77-90` 整段替换为：

```ts
export interface Live2DUploadIn {
  name: string;
  /** The whole Cubism project, zipped. */
  file: File;
  /** model3.json's path inside the archive. Blank lets the server detect
   *  it, which is what makes a zip of the exported folder work. */
  entry?: string;
  /** Emotion tag -> index into the model's own expression list. */
  expressions?: Record<string, number>;
}

/**
 * Register a Live2D avatar from an uploaded project archive.
 *
 * Replaces the old server-path registration: an operator used to have to
 * SCP the project onto the box and type an absolute path, which is not a
 * thing a browser can do.
 */
export function uploadLive2dAvatar(body: Live2DUploadIn): Promise<AvatarInfo> {
  const form = new FormData();
  form.append("name", body.name);
  form.append("file", body.file);
  if (body.entry) form.append("entry", body.entry);
  if (body.expressions && Object.keys(body.expressions).length > 0) {
    form.append("expressions", JSON.stringify(body.expressions));
  }
  return api<AvatarInfo>("/api/admin/live2d/avatars", { method: "POST", form });
}

export function deleteLive2dAvatar(avatarId: string): Promise<void> {
  return api<void>(`/api/admin/live2d/avatars/${avatarId}`, { method: "DELETE" });
}
```

- [ ] **Step 4: 类型检查**

```bash
cd E:/project/digital-human/admin-ui && npm run typecheck
```

Expected: PASS（`registerLive2dAvatar` 的唯一调用方 `Avatars.tsx` 还没改，此时应**报错**——把它当作预期的红灯，继续 Task 5）

若这一步报的是 `Avatars.tsx` 找不到 `registerLive2dAvatar` 之外的任何错误，先修掉再往下走。

- [ ] **Step 5: 提交**

```bash
cd E:/project/digital-human && git add admin-ui/src/avatars/KindTag.tsx admin-ui/src/avatars/UploadField.tsx admin-ui/src/api.ts && git commit -m "refactor(admin-ui): share the kind tag and drop zone, upload Live2D models"
```

---

### Task 5: 形象管理页重写

四张卡片 → 两张。一张创建表单用 `Segmented` 切类型，一张表用 `kind` 驱动两列。

**Files:**
- Modify: `admin-ui/src/pages/Avatars.tsx`（整体重写）

**Interfaces:**
- Consumes: `KindTag`、`UploadField`（Task 4），`driveLabel`、`parseExpressions`（Task 3），`uploadLive2dAvatar`、`deleteLive2dAvatar`（Task 4），`api`（未变）。
- Produces: 无新导出。`Avatars` 仍是 `App.tsx` 路由 `/avatars` 的默认导出。

- [ ] **Step 1: 改 imports**

`admin-ui/src/pages/Avatars.tsx` 的 antd import 列表里加 `Segmented`；把

```ts
import { api, deleteLive2dAvatar, registerLive2dAvatar } from "../api";
```

改为

```ts
import { api, deleteLive2dAvatar, uploadLive2dAvatar } from "../api";
```

并在 `import StatusTag from "../components/StatusTag";` 之后加：

```ts
import KindTag from "../avatars/KindTag";
import UploadField from "../avatars/UploadField";
import { driveLabel, parseExpressions } from "../avatars/describe";
```

类型 import 改为：

```ts
import type { AvatarInfo, AvatarKind, TunableParams } from "../types";
```

（`Live2DModelConfig` 不再需要——表情映射的解析已交给 `parseExpressions`。）

- [ ] **Step 2: 换 state 与创建表单的类型**

把 `[fileList, setFileList]` 重命名为 `[photoFiles, setPhotoFiles]`（文件里 `fileList` 与 `setFileList` 一共出现 6 处：`useState` 的声明与初始化闭包、`Upload.Dragger` 的两个 prop、以及 `onCreate` 里的读取和清空）。

把 `l2dCreating` / `l2dForm` 两段 state 删掉，换成：

```tsx
  // Both avatar kinds are the same feature with different inputs, so the
  // page has ONE create form and one list. `kind` is the only thing that
  // selects the endpoint, the drop zone, and the extra fields.
  const [kind, setKind] = useState<AvatarKind>("photoreal");
  // Two independent selections, not one: switching the Segmented control
  // must not carry a selected zip into the photo branch (or the reverse),
  // where it would be submitted to the wrong endpoint as a wrong file.
  const [zipFiles, setZipFiles] = useState<UploadFile[]>([]);
```

把 `[form] = Form.useForm<...>()` 的泛型改为：

```tsx
  const [form] = Form.useForm<{
    name: string;
    idle_crf?: number;
    idle_preset?: string;
    entry?: string;
    expressions?: string;
  }>();
```

- [ ] **Step 3: 替换提交逻辑**

把 `onRegisterLive2d`（`:232-276`）整个删掉。把 `onCreate`（`:288-348`）替换为：

```tsx
  async function submitPhotoreal(values: FormValues) {
    const file = photoFiles[0]?.originFileObj;
    if (!file) {
      message.warning("请先选择图片或视频");
      return;
    }
    setCreating(true);
    try {
      if (file.type.startsWith("video/")) {
        try {
          const seconds = await videoSeconds(file);
          if (seconds > 15) {
            message.error(`视频 ${seconds.toFixed(1)} 秒，超过 15 秒上限，请先剪短`);
            return;
          }
        } catch {
          // Duration is only a convenience; the server is authoritative.
        }
      }
      const body = new FormData();
      body.append("name", values.name.trim());
      body.append("file", file);
      // Per-avatar idle-video overrides. Empty fields fall through to the
      // server-side defaults; the server validates against tunables.ParamSpec.
      if (typeof values.idle_crf === "number") body.append("idle_crf", String(values.idle_crf));
      if (typeof values.idle_preset === "string" && values.idle_preset !== "") {
        body.append("idle_preset", values.idle_preset);
      }
      await api("/api/admin/avatars", { method: "POST", form: body });
      message.success("已提交，正在生成");
      form.resetFields();
      setPhotoFiles([]);
      await refresh();
    } catch (err) {
      message.error((err as Error).message);
    } finally {
      setCreating(false);
    }
  }

  async function submitLive2d(values: FormValues) {
    const file = zipFiles[0]?.originFileObj;
    if (!file) {
      message.warning("请先选择 Live2D 模型压缩包");
      return;
    }
    const parsed = parseExpressions(values.expressions ?? "");
    if (!parsed.ok) {
      message.error(parsed.error);
      return;
    }
    setCreating(true);
    try {
      await uploadLive2dAvatar({
        name: values.name.trim(),
        file,
        entry: (values.entry ?? "").trim(),
        expressions: parsed.value,
      });
      message.success("已注册");
      form.resetFields();
      setZipFiles([]);
      await refresh();
    } catch (err) {
      // Shown verbatim: the server's message names the specific missing
      // file ("missing moc: model.moc3"), which is exactly what tells the
      // operator what to fix.
      message.error((err as Error).message);
    } finally {
      setCreating(false);
    }
  }

  async function onFinish(values: FormValues) {
    // Synchronous re-entry guard: `creating` state flips too late to stop a
    // second click -- while video metadata is awaited or in the tick before
    // React re-renders -- and two POSTs would land, with two GPU prepare
    // jobs behind them. One guard covers both branches, because both kinds
    // cost the operator real work to redo.
    const { ran } = await creatingGuard.run(() =>
      kind === "live2d" ? submitLive2d(values) : submitPhotoreal(values),
    );
    if (!ran) return;
  }
```

并在该文件顶部（组件之外）加类型别名：

```tsx
type FormValues = {
  name: string;
  idle_crf?: number;
  idle_preset?: string;
  entry?: string;
  expressions?: string;
};
```

- [ ] **Step 4: 删掉拆表的两个 useMemo，合并删除分派**

把 `photoItems` / `live2dItems`（`:222-230`）两个 `useMemo` 删掉，保留 `visibleItems`。

把 `onDeleteLive2d` 删掉，`onDelete` 替换为：

```tsx
  async function onDelete(id: string, avatarKind: AvatarKind) {
    try {
      // Two delete routes for one feature: the live2d one refuses a
      // photoreal id outright rather than deleting it. Merging the table
      // does not merge that guard, so the row's kind picks the route.
      await (avatarKind === "live2d"
        ? deleteLive2dAvatar(id)
        : api(`/api/admin/avatars/${id}`, { method: "DELETE" }));
      message.success("已删除");
      await refresh();
    } catch (err) {
      message.error((err as Error).message);
    }
  }
```

- [ ] **Step 5: 重写列定义**

把 `columns`（`:360-449`）替换为：

```tsx
  const columns: ColumnsType<AvatarInfo> = [
    {
      title: "形象 ID",
      dataIndex: "avatar_id",
      width: 110,
      render: (v: string) => <CopyableId id={v} />,
    },
    {
      title: "名称",
      dataIndex: "name",
      render: (_, a) => (
        <Link to={`/avatars/${a.avatar_id}`} onClick={(e) => e.stopPropagation()}>
          {a.name}
        </Link>
      ),
    },
    {
      title: "类型",
      dataIndex: "kind",
      width: 100,
      render: (k: AvatarKind) => <KindTag kind={k} />,
    },
    {
      title: "驱动方式",
      width: 110,
      render: (_, a) => {
        const label = driveLabel(a);
        if (!label) return <Tag>—</Tag>;
        const color =
          a.kind === "live2d" ? "purple" : label === "视频驱动" ? "cyan" : "geekblue";
        return <Tag color={color}>{label}</Tag>;
      },
    },
    {
      title: "警告",
      width: 90,
      render: (_, a) => {
        // Live2D has no source pack, so there is no source-quality
        // assessment to report. A dash reads truer than "无" on a row that
        // was never assessed.
        if (a.kind === "live2d") {
          return <Typography.Text type="secondary">—</Typography.Text>;
        }
        const warnings = a.config.warnings ?? [];
        if (warnings.length === 0) return <Typography.Text type="secondary">无</Typography.Text>;
        return (
          <Tooltip title={warnings.join("\n")}>
            <Badge count={warnings.length} color="orange" />
          </Tooltip>
        );
      },
    },
    {
      title: "状态",
      dataIndex: "status",
      width: 110,
      render: (status: string) => <StatusTag status={status} />,
    },
    {
      title: "创建时间",
      dataIndex: "created_at",
      width: 200,
      render: (v: string) => new Date(v).toLocaleString(),
    },
    {
      title: "操作",
      width: 100,
      align: "right",
      render: (_, a) => (
        // The row itself navigates on click, and the Popconfirm's popup is a
        // React portal whose events still bubble up the React tree -- so the
        // whole cell has to swallow clicks, not just the button.
        <div onClick={(e) => e.stopPropagation()}>
          <Popconfirm
            title="删除该形象？"
            description={
              a.kind === "live2d"
                ? "删除后不可恢复，模型文件也会一并移除。"
                : "删除后不可恢复。"
            }
            okText="删除"
            cancelText="取消"
            okButtonProps={{ danger: true }}
            onConfirm={() => onDelete(a.avatar_id, a.kind)}
          >
            <Button danger size="small">
              删除
            </Button>
          </Popconfirm>
        </div>
      ),
    },
  ];
```

- [ ] **Step 6: 重写 render**

把 `return (` 到文件末尾（`:451-712`）整段替换为：

```tsx
  return (
    <Space direction="vertical" size={16} style={{ width: "100%" }}>
      <Typography.Title level={3} style={{ margin: 0 }}>
        形象管理
      </Typography.Title>

      <Card title="新建形象">
        <Segmented
          value={kind}
          onChange={(v) => setKind(v as AvatarKind)}
          options={[
            { label: "真人视频", value: "photoreal" },
            { label: "Live2D", value: "live2d" },
          ]}
          style={{ marginBottom: 16 }}
        />
        <Form
          form={form}
          layout="vertical"
          onFinish={onFinish}
          style={{ maxWidth: 560 }}
          // Idle knobs default to the server's live values so an operator
          // who never opens the Advanced section still gets what the box is
          // set to. `entry` defaults to the conventional filename; leaving
          // it blank makes the server detect it, which is what the folder-in-
          // zip shape needs.
          initialValues={{
            idle_crf: idleDefaults?.idle_crf ?? IDLE_CRF_DEFAULT,
            idle_preset: idleDefaults?.idle_preset ?? IDLE_PRESET_DEFAULT,
            entry: "model.model3.json",
          }}
        >
          <Form.Item
            name="name"
            label="名称"
            rules={[{ required: true, message: "请输入名称" }]}
          >
            <Input placeholder={kind === "live2d" ? "例如：Hiyori" : "例如：小美"} />
          </Form.Item>

          {kind === "photoreal" ? (
            <>
              <Form.Item label="驱动图片或视频" required>
                <UploadField
                  accept="image/*,video/*"
                  dragText="点击或拖拽图片 / 视频到此处"
                  hint="支持 JPG / PNG / MP4 / MOV，视频 ≤15 秒，建议清晰正脸半身；视频中请保持闭嘴、不说话"
                  fileList={photoFiles}
                  onChange={setPhotoFiles}
                />
              </Form.Item>
              {/*
                Idle-video knobs. Collapsed by default so the form stays calm
                for the common "just upload a photo" path; expand to tweak the
                quality / encode-speed trade-off per avatar.
              */}
              <Form.Item label="高级参数（可选）" style={{ marginBottom: 12 }}>
                <Collapse
                  ghost
                  items={[
                    {
                      key: "idle",
                      label: "空闲视频编码",
                      children: (
                        <Space direction="vertical" size={12} style={{ width: "100%" }}>
                          <Form.Item
                            name="idle_crf"
                            label="视频质量 (CRF)"
                            tooltip="0=无损 51=最小体积，建议 16~23；调高可减小文件但降低画质"
                            style={{ marginBottom: 0 }}
                          >
                            <InputNumber
                              min={IDLE_CRF_MIN}
                              max={IDLE_CRF_MAX}
                              step={1}
                              precision={0}
                              style={{ width: 200 }}
                            />
                          </Form.Item>
                          <Form.Item
                            name="idle_preset"
                            label="x264 编码速度"
                            tooltip="veryslow 文件更小但编码更慢；ultrafast 相反"
                            style={{ marginBottom: 0 }}
                          >
                            <Select
                              style={{ width: 200 }}
                              options={X264_PRESETS.map((p) => ({ value: p, label: p }))}
                            />
                          </Form.Item>
                        </Space>
                      ),
                    },
                  ]}
                />
              </Form.Item>
            </>
          ) : (
            <>
              <Form.Item label="Live2D 模型压缩包" required>
                <UploadField
                  accept=".zip,application/zip"
                  dragText="点击或拖拽 .zip 到此处"
                  hint="把整个 Cubism 工程打包成 zip。工程放在子目录里也能用，服务端会自动找到 model3.json"
                  fileList={zipFiles}
                  onChange={setZipFiles}
                />
              </Form.Item>
              <Form.Item
                name="entry"
                label="入口文件"
                tooltip="工程内的 model3.json 路径。留空则由服务端自动探测；其余文件（moc3、纹理、动作）由它引用，无需单独填写。"
              >
                <Input placeholder="model.model3.json" />
              </Form.Item>
              <Form.Item
                name="expressions"
                label="表情映射（可选，JSON）"
                tooltip='形如 {"smile": 0, "think": 1}，数值是模型自身表情列表的下标。注册时会校验下标是否存在。'
              >
                <Input.TextArea rows={2} placeholder={'{"smile": 0}'} />
              </Form.Item>
            </>
          )}

          <Button type="primary" htmlType="submit" loading={creating} icon={<PlusOutlined />}>
            上传创建
          </Button>
        </Form>
      </Card>

      {error && <Alert type="error" message={error} showIcon />}

      <Card
        title="形象列表"
        extra={
          <Space>
            <Input
              allowClear
              prefix={<SearchOutlined />}
              placeholder="搜索名称"
              style={{ width: 200 }}
              value={nameQuery}
              onChange={(e) => setNameQuery(e.target.value)}
            />
            <Button
              icon={<ReloadOutlined />}
              onClick={() => {
                setError("");
                refresh().catch((err) => setError((err as Error).message));
              }}
            >
              刷新
            </Button>
          </Space>
        }
      >
        <Table
          rowKey="avatar_id"
          columns={columns}
          dataSource={visibleItems}
          loading={loading}
          scroll={{ x: 1000 }}
          pagination={{ pageSize: 10, hideOnSinglePage: true }}
          locale={{ emptyText: <Empty description="暂无形象，先上传一个吧" /> }}
          onRow={(a) => ({
            onClick: () => navigate(`/avatars/${a.avatar_id}`),
            style: { cursor: "pointer" },
          })}
        />
      </Card>
    </Space>
  );
}
```

- [ ] **Step 7: 类型检查与前端测试**

```bash
cd E:/project/digital-human/admin-ui && npm run typecheck && npx vitest run
```

Expected: PASS —— 无类型错误，全部 vitest 用例通过

若 `typecheck` 报 `Segmented` 的 `onChange` 参数类型不匹配，改成 `onChange={(v) => setKind(v as AvatarKind)}`（若已是该写法仍报错，用 `String(v) as AvatarKind`）。

- [ ] **Step 8: 提交**

```bash
cd E:/project/digital-human && git add admin-ui/src/pages/Avatars.tsx && git commit -m "refactor(admin-ui): one table and one create form for both avatar kinds"
```

---

### Task 6: 详情页按 kind 分叉

Live2D 形象点进详情页不该看到一屏 404。抽出 `Live2dAvatarDetail.tsx`，`AvatarDetail.tsx` 按 `info.kind` 选主体。

**Files:**
- Create: `admin-ui/src/pages/Live2dAvatarDetail.tsx`
- Modify: `admin-ui/src/pages/AvatarDetail.tsx`

**Interfaces:**
- Consumes: `Live2DStage`、`live2dModelUrl`、`KindTag`、`ArtifactInfo`、`AvatarInfo`。
- Produces: `Live2dAvatarDetail({ info, artifacts }: { info: AvatarInfo; artifacts: ArtifactInfo[] })` —— 默认导出。`AvatarDetail.tsx` 里的 `formatBytes` 改为 `export function formatBytes`（供新文件复用）。

- [ ] **Step 1: 让 `formatBytes` 可导出**

`admin-ui/src/pages/AvatarDetail.tsx:79` 改为：

```tsx
export function formatBytes(b: number): string {
```

- [ ] **Step 2: 创建 `Live2dAvatarDetail.tsx`**

```tsx
import { useMemo, useState } from "react";
import { Button, Card, Col, Descriptions, Empty, Row, Space, Tag, Typography } from "antd";
import { ReloadOutlined } from "@ant-design/icons";
import Live2DStage from "../live2d/Live2DStage";
import { live2dModelUrl } from "../api";
import { formatBytes } from "./AvatarDetail";
import type { ArtifactInfo, AvatarInfo, Live2DModelConfig } from "../types";

/**
 * The Live2D half of the avatar detail page.
 *
 * Everything here is a different *kind* of the same thing, not a
 * different feature: the model files come from the same artifact listing
 * a photoreal avatar uses, and the preview is the same <Live2DStage>
 * component the debug page, the chat page and the desktop pet mount.
 * What a Live2D avatar does not have is a build pipeline -- so there is
 * no idle clip to fetch, no timeline, and nothing to retry.
 */
export default function Live2dAvatarDetail({
  info,
  artifacts,
}: {
  info: AvatarInfo;
  artifacts: ArtifactInfo[];
}) {
  const config = (info.config ?? {}) as Live2DModelConfig;
  const entry = config.model?.entry ?? "";
  // An avatar registered without an emotion map is normal, not broken --
  // render nothing rather than iterating undefined.
  const tags = Object.entries(config.expressions ?? {});
  // A canvas cannot raise this itself: a failed model load is a blank
  // rectangle with no error anywhere. The stage hands us the exception,
  // so the page has to be the one that shows it.
  const [stageError, setStageError] = useState("");

  const totalBytes = useMemo(
    () => artifacts.reduce((n, a) => n + a.size, 0),
    [artifacts],
  );

  return (
    <Space direction="vertical" size={16} style={{ width: "100%" }}>
      <Row gutter={[16, 16]}>
        <Col xs={24} md={12}>
          <Card title="预览">
            {stageError ? (
              <Empty description={`模型加载失败（${stageError}）`}>
                <Button icon={<ReloadOutlined />} onClick={() => window.location.reload()}>
                  重试
                </Button>
              </Empty>
            ) : entry ? (
              // audio=null: this is the idle layer only. The mouth is driven
              // by the AudioBuffer actually playing, and nothing is playing
              // here, so the model breathes and blinks without a mouth.
              <Live2DStage
                modelUrl={live2dModelUrl(info.avatar_id, entry)}
                audio={null}
                width={240}
                height={360}
                onError={(e) => setStageError(e instanceof Error ? e.message : String(e))}
              />
            ) : (
              <Empty description="该形象没有入口文件，无法预览" />
            )}
          </Card>
        </Col>

        <Col xs={24} md={12}>
          <Card title="模型信息">
            <Descriptions
              size="small"
              column={1}
              bordered
              items={[
                { key: "entry", label: "入口文件", children: entry || "—" },
                {
                  key: "expressions",
                  label: "表情映射",
                  children:
                    tags.length === 0 ? (
                      <Typography.Text type="secondary">无</Typography.Text>
                    ) : (
                      <Space size={4} wrap>
                        {tags.map(([tag, idx]) => (
                          <Tag key={tag}>
                            {tag} → {idx}
                          </Tag>
                        ))}
                      </Space>
                    ),
                },
                { key: "files", label: "文件数", children: artifacts.length },
                { key: "size", label: "总大小", children: formatBytes(totalBytes) },
              ]}
            />
            <Typography.Paragraph type="secondary" style={{ marginTop: 12, marginBottom: 0 }}>
              Live2D 形象在客户端用 WebGL 渲染，服务端不产出任何画面，因此没有构建过程、也没有可重建的产物。
            </Typography.Paragraph>
          </Card>
        </Col>
      </Row>
    </Space>
  );
}
```

- [ ] **Step 3: 让 `AvatarDetail` 按 kind 选主体**

在 `admin-ui/src/pages/AvatarDetail.tsx` 的 imports 里加：

```tsx
import KindTag from "../avatars/KindTag";
import Live2dAvatarDetail from "./Live2dAvatarDetail";
```

把 `refresh` 改为返回数据（分支判断要用）：

```tsx
  async function refresh() {
    if (!id) return null;
    const data = await api<AvatarInfo>(`/api/admin/avatars/${id}`);
    setInfo(data);
    return data;
  }
```

把挂载 effect（`:150-158`）替换为：

```tsx
  useEffect(() => {
    const signal = { ignored: false };
    refresh()
      .then((data) => {
        if (signal.ignored) return;
        // A Live2D avatar has no build pipeline, so there is no timeline to
        // read and no job to poll -- both requests would only 404. The
        // artifact listing IS shared and is fetched for both kinds.
        if (data?.kind === "live2d") return;
        void loadTimeline();
        void loadJob();
      })
      .catch((err) => {
        if (!signal.ignored) setError((err as Error).message);
      })
      .finally(() => {
        if (!signal.ignored) setLoading(false);
      });
    void loadArtifacts();
    return () => {
      signal.ignored = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id]);
```

把预览 effect（`:201-209`）替换为：

```tsx
  useEffect(() => {
    // Waits for `info` so the kind is known. A Live2D avatar has no
    // idle.mp4 and no normalized.png -- Live2dAvatarDetail mounts its own
    // canvas instead -- and fetching those two would just 404 twice.
    if (!info || info.kind === "live2d") return;
    const signal = { ignored: false };
    void loadPreview(signal);
    return () => {
      signal.ignored = true;
      if (previewRef.current) URL.revokeObjectURL(previewRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id, info?.kind]);
```

在 `if (!info) { ... }` 早退之后加：

```tsx
  const isLive2d = info.kind === "live2d";
```

把头部里的「按当前设置重建」按钮（`:337-339`）包上条件：

```tsx
        {!isLive2d && (
          <Button icon={<ReloadOutlined />} onClick={() => void retry()} type="primary" ghost>
            按当前设置重建
          </Button>
        )}
```

在头部的 `Descriptions` 里加一项类型：

```tsx
        { key: "kind", label: "类型", children: <KindTag kind={info.kind} /> },
```

把 `{warningList.length > 0 && (...)}` 到 `</Row>` 这一段（`:371-510`）连同 `<Card title="底层配置">`（`:527-572`）替换为：

```tsx
      {isLive2d ? (
        <Live2dAvatarDetail info={info} artifacts={artifacts} />
      ) : (
        <>
          {warningList.length > 0 && (
            <Alert
              type="warning"
              showIcon
              message="素材质量评估（形象已生成，更换更合适的素材可进一步提升效果）"
              description={
                <Space direction="vertical" size={6} style={{ width: "100%" }}>
                  {warningList.map((w, i) => {
                    const kind = classifyWarning(w);
                    const hint = THRESHOLD_HINTS[kind] ?? "";
                    return (
                      <Tooltip key={i} title={hint}>
                        <span style={{ cursor: "help" }}>· {w}</span>
                      </Tooltip>
                    );
                  })}
                </Space>
              }
            />
          )}

          <Row gutter={[16, 16]}>
            <Col xs={24} md={12}>
              <Card title="预览">
                {previewError ? (
                  <Empty description={`预览不可用（${previewError}）`}>
                    <Button
                      icon={<ReloadOutlined />}
                      onClick={() => {
                        void loadPreview();
                      }}
                    >
                      重试
                    </Button>
                  </Empty>
                ) : (
                  <Spin spinning={previewLoading}>
                    <div style={{ minHeight: 120 }}>
                      {preview && previewKind === "video" ? (
                        <video
                          src={preview}
                          controls
                          muted
                          autoPlay
                          playsInline
                          loop
                          style={{ width: 240 }}
                          onError={() =>
                            setPreviewError("视频无法解码播放")
                          }
                        />
                      ) : preview ? (
                        <img
                          src={preview}
                          alt="preview"
                          width={240}
                          onError={() => setPreviewError("图片无法解码显示")}
                        />
                      ) : null}
                    </div>
                  </Spin>
                )}
              </Card>
            </Col>

            <Col xs={24} md={12}>
              <Card title="构建过程">
                <Steps
                  size="small"
                  current={Math.max(0, stageCurrent)}
                  status={info.status === "failed" ? "error" : undefined}
                  items={STAGES.map((s, i) => ({
                    title: <StatusTag status={s} />,
                    status: stepStatus(i),
                  }))}
                />
                <Descriptions
                  size="small"
                  column={2}
                  style={{ marginTop: 12 }}
                  items={[
                    {
                      key: "mode",
                      label: "模式",
                      children:
                        timeline?.mode === "video"
                          ? "视频驱动"
                          : timeline?.mode === "image"
                            ? "图片驱动"
                            : "—",
                    },
                    { key: "frames", label: "帧数", children: timeline?.frame_count ?? "—" },
                    { key: "kept", label: "保留帧", children: timeline?.kept_count ?? "—" },
                    { key: "fps", label: "输出 FPS", children: timeline?.fps ?? "—" },
                    { key: "srcfps", label: "源 FPS", children: timeline?.source_fps ?? "—" },
                    { key: "detector", label: "检测器", children: timeline?.detector ?? "—" },
                    { key: "version", label: "版本", children: timeline?.version ?? "—" },
                  ]}
                />
                {latestJob && ACTIVE_JOB_STAGES.has(latestJob.stage) && (
                  <div style={{ marginTop: 12 }}>
                    <Typography.Text type="secondary">当前会话进度</Typography.Text>
                    <div style={{ marginTop: 4 }}>
                      <Space>
                        <StatusTag status={latestJob.stage} />
                        <Progress
                          percent={latestJob.progress}
                          size="small"
                          style={{ width: 200 }}
                          status={latestJob.progress >= 100 ? "success" : "active"}
                        />
                      </Space>
                    </div>
                  </div>
                )}
              </Card>
            </Col>
          </Row>

          <Card title="底层配置">
            <Descriptions
              size="small"
              column={2}
              bordered
              items={[
                { key: "bbox", label: "bbox", children: JSON.stringify(musetalk.bbox ?? "—") },
                { key: "crop", label: "crop", children: JSON.stringify(musetalk.crop ?? "—") },
                { key: "detector", label: "detector", children: String(musetalk.detector ?? "—") },
                {
                  key: "version",
                  label: "version",
                  children: String(musetalk.version ?? timeline?.version ?? "—"),
                },
                {
                  key: "frames",
                  label: "frame_count",
                  children: String(musetalk.frame_count ?? timeline?.frame_count ?? "—"),
                },
                { key: "fps", label: "fps", children: String(musetalk.fps ?? timeline?.fps ?? "—") },
                {
                  key: "srcfps",
                  label: "source_fps",
                  children: String(musetalk.source_fps ?? timeline?.source_fps ?? "—"),
                },
                { key: "mode", label: "mode", children: String(musetalk.mode ?? timeline?.mode ?? "—") },
                { key: "base", label: "base_image", children: baseImage ?? "—" },
              ]}
            />
            <Collapse
              ghost
              style={{ marginTop: 12 }}
              items={[
                {
                  key: "eff",
                  label: "实际生效参数（_effective_params）",
                  children:
                    effectiveParams && Object.keys(effectiveParams).length > 0 ? (
                      <Descriptions
                        size="small"
                        column={2}
                        bordered
                        items={Object.entries(effectiveParams).map(([k, v]) => ({
                          key: k,
                          label: k,
                          children: JSON.stringify(v),
                        }))}
                      />
                    ) : (
                      <Empty description="此形象尚未记录生效参数" />
                    ),
                },
              ]}
            />
            <Typography.Text type="secondary" style={{ display: "block", marginTop: 8 }}>
              调整「参数设置」中的全局值后，下次重建即生效。
            </Typography.Text>
          </Card>
        </>
      )}
```

把产物卡的标题改为随 kind 变化：

```tsx
      <Card title={isLive2d ? "模型文件" : "素材与产物"}>
```

- [ ] **Step 4: 类型检查与前端测试**

```bash
cd E:/project/digital-human/admin-ui && npm run typecheck && npx vitest run
```

Expected: PASS

- [ ] **Step 5: 跑完整 Python 测试确认没有回归**

```bash
cd E:/project/digital-human && .venv/Scripts/python -m pytest -q
```

Expected: PASS

- [ ] **Step 6: 提交**

```bash
cd E:/project/digital-human && git add admin-ui/src/pages/AvatarDetail.tsx admin-ui/src/pages/Live2dAvatarDetail.tsx && git commit -m "feat(live2d): render the Live2D model on the avatar detail page"
```

---

## 人工验证（Task 6 之后）

单测覆盖不到的两件事，必须在浏览器里确认：

```bash
# 起服务（mock 引擎，无需 GPU）
cd E:/project/digital-human && .venv/Scripts/python -m uvicorn --factory server.main:create_app --host 0.0.0.0 --port 8000
```

1. `/admin/avatars` → 类型切到 Live2D → 拖一个真实 Hiyori zip（工程在子目录里的那种）→ 创建 → 列表同一张表里出现新行，类型列显示 Live2D，驱动方式显示 Cubism。
2. 点进该行 → 画布出现会呼吸的模型；模型文件表列出 moc3 / 纹理 / 动作且能下载；页面上**没有**「按当前设置重建」和「构建过程」。
3. 会话调试页选中它 → 能说话、嘴跟着动。
4. 回到列表 → 删除该行 → 模型文件一并消失。
5. 切回真人视频 → 上传一张照片 → 行为与改动前一致。
6. 先在真人分支选一张图，**不创建**，切到 Live2D 分支 → 照片从上传区消失，Live2D 分支是空的；反过来再试一次。两个分支各挑一个文件后创建，确认各自发往正确的端点（Network 面板里看请求 URL 与 `file` 的文件名）。