"""Live2D model registration and hosting.

Two surfaces:

* ``/api/admin/live2d/*`` -- the console: register a model, delete one, ask
  whether the SDK is present.
* ``/api/v1/live2d/*`` -- read-only hosting for whichever client renders it
  (today the Electron pet), gated by ``X-V1-Token`` when one is configured.

The server never renders a Cubism model. It owns three things only: the row
that says a model exists, the files on disk, and the bytes to hand out.
"""

import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field
from sqlalchemy import select

from server.api.avatars import _require_uuid_hex, _serialize
from server.avatar.store import AvatarStore
from server.config import get_settings
from server.core.auth import require_v1_token
from server.core.errors import AppError
from server.db.models import Avatar
from server.db.session import get_db
from server.live2d.registry import validate_model_dir

router = APIRouter(prefix="/api", tags=["live2d"])

_MEDIA = {
    ".json": "application/json",
    ".moc3": "application/octet-stream",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".js": "application/javascript",
    ".vert": "text/plain",
    ".frag": "text/plain",
}


def _media_type(path: Path) -> str:
    return _MEDIA.get(path.suffix.lower(), "application/octet-stream")


def _store() -> AvatarStore:
    return AvatarStore(get_settings().data_dir)


def _safe_join(root: Path, relative: str) -> Path:
    """Resolve a request path under ``root``, or 404.

    ``{path:path}`` is the most dangerous parameter in the service: it takes
    anything, including ``..`` segments and absolute paths. Everything is
    re-resolved and checked against the root, and the caller never learns
    whether the escape landed on an existing file.
    """
    if not relative:
        raise AppError("not_found", "file not found", 404)
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()) or not target.is_file():
        raise AppError("not_found", "file not found", 404)
    return target


# ----- console surface ----------------------------------------------------

class RegisterIn(BaseModel):
    model_config = {"extra": "forbid"}

    name: str = Field(min_length=1, max_length=128)
    #: Server-side path to a Cubism project directory. Copied into the
    #: avatar's own storage so the avatar owns its files and deleting it
    #: takes the model with it -- the source directory is left untouched.
    model_dir: str
    config: dict = Field(default_factory=dict)


@router.post("/admin/live2d/avatars", status_code=201)
def register_live2d_avatar(body: RegisterIn, db=Depends(get_db)) -> dict:
    source = Path(body.model_dir)
    if not source.is_dir():
        raise AppError("live2d_invalid_model", f"model_dir not found: {body.model_dir}", 422)

    avatar_id = uuid4().hex
    store = _store()
    target = store.avatar_dir(avatar_id)
    shutil.copytree(source, target)

    # Validate the copy, not the source: that is the directory that will be
    # served, and a source that changes afterwards must not silently
    # invalidate the registration.
    try:
        validate_model_dir(target, body.config)
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        raise

    row = Avatar(
        id=avatar_id,
        name=body.name,
        status="ready",
        source_image_path="",
        pack_path=None,
        kind="live2d",
        config_json=body.config,
    )
    db.add(row)
    db.commit()
    return _serialize(row, with_detail=True)


@router.delete("/admin/live2d/avatars/{avatar_id}")
def delete_live2d_avatar(avatar_id: str, db=Depends(get_db)) -> Response:
    _require_uuid_hex(avatar_id)
    row = db.get(Avatar, avatar_id)
    # Refuse rather than delete: this endpoint must not become a second,
    # differently behaved way to delete a video avatar.
    if row is None or row.kind != "live2d":
        raise AppError("not_found", f"avatar {avatar_id} not found", 404)
    _store().delete_assets(avatar_id)
    db.delete(row)
    db.commit()
    return Response(status_code=204)


@router.get("/admin/live2d/health")
def live2d_health() -> dict:
    """Report whether the SDK is staged.

    Never raises: the System tab shows this, and an SDK that is missing is
    something to display, not an exception to propagate.
    """
    settings = get_settings()
    sdk = Path(settings.live2d_sdk_dir)
    core = sdk / "Core" / "live2dcubismcore.js"
    trees = _runtime_trees(sdk.resolve())
    return {
        "enabled": settings.live2d_enabled,
        "sdk_dir": str(sdk),
        "sdk_present": sdk.is_dir(),
        "core_present": core.is_file(),
        # Resolved the same way the runtime route resolves it, so this
        # reports what the server can actually serve.
        "shaders_present": "Shaders" in trees,
    }


# ----- client surface -----------------------------------------------------

@router.get("/v1/live2d/avatars/{avatar_id}/model/{path:path}")
def get_model_file(
    avatar_id: str,
    path: str,
    db=Depends(get_db),
    _token=Depends(require_v1_token),
):
    _require_uuid_hex(avatar_id)
    row = db.execute(select(Avatar).where(Avatar.id == avatar_id)).scalar_one_or_none()
    # A photoreal avatar has no model directory to serve. Same 404 as a
    # missing avatar, so this cannot be used to probe which ids exist.
    if row is None or row.kind != "live2d":
        raise AppError("not_found", f"avatar {avatar_id} not found", 404)
    target = _safe_join(_store().avatar_dir(avatar_id), path)
    return FileResponse(target, media_type=_media_type(target))


#: The SDK directory holds the Core, the Shaders, the licence, the samples
#: and the framework source. Only the first two are ever fetched by a
#: client; serving the rest would publish the licence text and the
#: framework source to anyone who can reach the port.
#:
#: The two are not siblings. In CubismSdkForWeb-5-r.5 the core sits at the
#: root of the SDK and the GLSL sits under ``Framework/``, beside the
#: framework source -- so the directory the SDK actually ships is not the
#: one the old docs (and this file) assumed. The URL namespace stays
#: ``Core/`` and ``Shaders/`` regardless: the client never learns the SDK's
#: internal layout, and the server maps one onto the other.
_RUNTIME_TREES: dict[str, tuple[str, ...]] = {
    "Core": ("Core",),
    "Shaders": ("Framework/Shaders", "Shaders"),
}


def _runtime_trees(root: Path) -> dict[str, Path]:
    """Map each runtime tree onto the directory that actually holds it.

    A tree that is not on disk is simply absent from the mapping, so an
    SDK that is missing (or laid out differently) produces a 404 rather
    than a path that resolves somewhere unexpected.
    """
    trees: dict[str, Path] = {}
    for name, candidates in _RUNTIME_TREES.items():
        for candidate in candidates:
            tree = (root / candidate).resolve()
            if tree.is_dir():
                trees[name] = tree
                break
    return trees


@router.get("/v1/live2d/runtime/{path:path}")
def get_runtime_file(path: str):
    """Serve the Cubism core and shaders. Deliberately NOT token-gated.

    Two reasons, both mechanical rather than policy:

    * The core is loaded by a ``<script>`` tag, and a script tag cannot
      carry a custom header. Requiring ``X-V1-Token`` would make it
      unloadable rather than protected.
    * The framework fetches its GLSL itself, with a plain ``fetch`` it
      does not let the caller configure with headers.

    So the boundary is drawn here instead: model files -- the
    operator-supplied content, and the thing a traversal would actually
    reach -- are token-gated, while the SDK runtime is a fixed,
    user-independent asset gated only by the Core/Shaders trees.

    The boundary is a *tree*, not a string prefix. The path is split on
    its first segment, that segment names one of the two trees, and the
    rest is joined inside that tree -- so ``Core/../LICENSE.md`` resolves
    outside the Core tree and is refused, rather than passing a
    ``startswith("Core/")`` check and then serving whatever it landed on.
    """
    root = Path(get_settings().live2d_sdk_dir).resolve()
    head, _, rest = path.partition("/")
    tree = _runtime_trees(root).get(head)
    if tree is None:
        raise AppError("not_found", "file not found", 404)
    target = _safe_join(tree, rest)
    return FileResponse(target, media_type=_media_type(target))