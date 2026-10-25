"""Live2D model hosting.

One surface:

* ``/api/v1/live2d/*`` -- read-only hosting for whichever client renders
  it (today the Electron pet), gated by ``X-V1-Token`` when one is
  configured.

Registering a Live2D avatar is not here any more: Live2D is a
digital-human *model* like any other (see ``server/model_registry``), so
its avatars are created through the one console upload endpoint
(``POST /api/admin/avatars``) and its project is built by the Live2D
avatar pipeline (``server/avatar/live2d_pipeline.py``).

The server never renders a Cubism model. It owns three things only: the
row that says a model exists, the files on disk, and the bytes to hand
out.
"""

from pathlib import Path

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse
from sqlalchemy import select

from server.api.avatars import _require_uuid_hex
from server.avatar.store import AvatarStore
from server.config import get_settings
from server.core.auth import require_v1_token
from server.core.errors import AppError
from server.db.models import Avatar
from server.db.session import get_db

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

#: The extensions the model host may serve.
#:
#: An uploaded archive is operator input and can carry anything beside the
#: model files. The console is mounted on the SAME origin as this route, so
#: a member served as ``application/javascript`` -- ``_MEDIA`` maps ``.js``
#: to it for the SDK runtime -- would be a script-planting primitive on an
#: admin plane that has no auth by design. A Cubism project needs none of
#: the executable types, so the host serves the model formats and nothing
#: else; anything else the operator uploaded is still downloadable through
#: the artifact route, which always sends ``application/octet-stream``.
_MODEL_SUFFIXES = frozenset(
    {".json", ".moc3", ".moc", ".png", ".jpg", ".jpeg", ".webp"}
)


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


def _is_client_rendered(row) -> bool:
    """Whether this avatar is driven by a model the *client* renders.

    Live2D is such a model today: the browser draws the Cubism project, so
    the server only hosts its files. A model that is no longer registered
    is not client-rendered either -- it has no files to serve.
    """
    from server.model_registry import ModelUnavailable, get_model

    try:
        return get_model(row.drive_model).client_rendered
    except ModelUnavailable:
        return False


@router.get("/admin/live2d/health")
def live2d_health() -> dict:
    """Report whether the SDK is staged.

    Never raises: the System tab shows this, and an SDK that is missing is
    something to display, not to throw on.
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
    # An avatar the server drives has no model directory to serve. Same
    # 404 as a missing avatar, so this cannot be used to probe which ids
    # exist.
    if row is None or not _is_client_rendered(row):
        raise AppError("not_found", f"avatar {avatar_id} not found", 404)
    target = _safe_join(_store().avatar_dir(avatar_id), path)
    # Only the model formats are served. Everything else an archive
    # carried is still on disk and still downloadable through the artifact
    # route; what it must not be is served from the console's own origin
    # as something the browser will execute.
    if target.suffix.lower() not in _MODEL_SUFFIXES:
        raise AppError("not_found", "file not found", 404)
    return FileResponse(
        target,
        media_type=_media_type(target),
        headers={"X-Content-Type-Options": "nosniff"},
    )


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
