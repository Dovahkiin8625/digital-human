"""Live2D model registration and hosting.

Hosting is gated by X-V1-Token when one is configured. The admin console
carries no credential of its own -- the admin plane has no auth since the
no-tenant refactor -- so an unset token means open, which is exactly what
every other admin endpoint already does. See the ledger ruling: the
alternative (a dual admin-or-v1 credential) cannot be implemented until
the admin plane is protected, and pretending otherwise would be security
theatre.
"""

import json

import pytest

from server.config import get_settings

LIVE2D_ID = "a" * 32


def _project_members(prefix: str = "") -> dict:
    """The files of a Cubism project.

    ``prefix`` puts every file inside a folder, which is what right-click
    compress produces and what the server has to cope with.
    """
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
    return entries


def _project_zip(prefix: str = "") -> bytes:
    """A Cubism project as a browser would send it."""
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in _project_members(prefix).items():
            zf.writestr(name, data)
    return buf.getvalue()


def _zip_with_extra(extra_name: str, extra: bytes) -> bytes:
    """A valid project zip plus one extra member of any kind.

    An archive is operator input: it can carry anything beside the model
    files, and the server has to decide what to serve.
    """
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in _project_members().items():
            zf.writestr(name, data)
        zf.writestr(extra_name, extra)
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
    """Register a Live2D avatar through the one console upload endpoint.

    Live2D is a digital-human model like any other, so the avatar is
    created for the ``live2d`` model and its project is built by that
    model's pipeline.
    """
    data = _project_zip(prefix) if raw is None else raw
    form = {"name": name, "model": "live2d"}
    if entry:
        form["entry"] = entry
    if expressions:
        form["expressions"] = expressions
    return client.post(
        "/api/admin/avatars",
        data=form,
        files={"file": ("model.zip", data, "application/zip")},
    )


def _write_model(root):
    root.mkdir(parents=True, exist_ok=True)
    (root / "model.model3.json").write_text(
        json.dumps({
            "Version": 3,
            "FileReferences": {
                "Moc": "model.moc3",
                "Textures": ["texture_00.png"],
                "Expressions": [{"Name": "0", "File": "0.exp3.json"}],
            },
        }),
        encoding="utf-8",
    )
    (root / "model.moc3").write_bytes(b"\x00" * 16)
    (root / "texture_00.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (root / "0.exp3.json").write_text("{}", encoding="utf-8")
    return root


def _seed_live2d(client):
    from sqlalchemy.orm import sessionmaker

    from server.db.models import Avatar
    from server.db.session import get_engine

    root = client.app.state.session_manager.store.avatar_dir(LIVE2D_ID)
    _write_model(root)
    db = sessionmaker(bind=get_engine())()
    try:
        db.add(Avatar(
            id=LIVE2D_ID, name="l2d", status="ready", source_image_path="",
            drive_model="live2d", config_json={"model": {"entry": "model.model3.json"}},
        ))
        db.commit()
    finally:
        db.close()


def _token(monkeypatch, value):
    monkeypatch.setenv("DH_V1_API_TOKEN", value)
    get_settings.cache_clear()


def _assert_refused(resp):
    """A 404 from OUR handler, not FastAPI's unknown-route 404.

    Without this, every "must refuse" test passes vacuously while the route
    does not exist yet.
    """
    assert resp.status_code == 404, resp.text
    assert "error" in resp.json(), f"expected our error envelope, got {resp.text}"
    assert resp.json()["error"]["code"] == "not_found"


# ----- registration -------------------------------------------------------

def test_upload_extracts_and_validates(client):
    resp = _upload(client, expressions='{"smile": 0}')
    assert resp.status_code == 201, resp.text
    avatar_id = resp.json()["avatar_id"]
    body = resp.json()
    # The avatar records the model that owns it.
    assert body["drive_model"] == "live2d"
    assert body["status"] == "ready"
    # The response carries the config so the console can render the entry
    # without a second round trip.
    assert body["config"]["model"]["entry"] == "model.model3.json"
    root = client.app.state.session_manager.store.avatar_dir(avatar_id)
    assert (root / "model.moc3").read_bytes() == b"\x00" * 16
    # The archive itself is not left in the avatar's directory: the
    # unpacked project is the model.
    assert not (root / "model.zip").exists()


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


def test_upload_rejects_an_empty_name(client):
    """A nameless upload must not become a row nobody can tell apart."""
    resp = _upload(client, name="")
    assert resp.status_code == 422, resp.text


def test_upload_rejects_an_overlong_name(client):
    resp = _upload(client, name="x" * 129)
    assert resp.status_code == 422


def test_a_rejected_upload_leaves_nothing_behind(client):
    store = client.app.state.session_manager.store
    avatars_dir = store.data_dir / "avatars"

    def names():
        return set(p.name for p in avatars_dir.glob("*")) if avatars_dir.exists() else set()

    before = names()
    resp = _upload(client, raw=_zip_without("model.moc3"))
    assert resp.status_code == 422
    assert names() == before
    # And no row was written.
    assert client.get("/api/admin/avatars?drive_model=live2d").json() == []


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


def test_delete_removes_row_and_assets(client):
    _seed_live2d(client)
    store = client.app.state.session_manager.store
    assert client.delete(f"/api/admin/avatars/{LIVE2D_ID}").status_code == 204
    assert client.get(f"/api/v1/avatars/{LIVE2D_ID}").status_code == 404
    assert not store.avatar_dir(LIVE2D_ID).exists()


def test_one_delete_endpoint_handles_every_model(client):
    """There is a single delete route now, so a Live2D avatar and a
    server-driven one are removed the same way."""
    import io

    from PIL import Image

    _seed_live2d(client)
    buf = io.BytesIO()
    Image.new("RGB", (640, 800), (10, 20, 30)).save(buf, format="PNG")
    created = client.post("/api/v1/avatars", data={"name": "p"},
                          files={"file": ("p.png", buf.getvalue(), "image/png")})
    photo_id = created.json()["avatar_id"]
    assert client.delete(f"/api/admin/avatars/{photo_id}").status_code == 204
    assert client.get(f"/api/v1/avatars/{photo_id}").status_code == 404


def test_delete_refuses_an_unknown_avatar(client):
    _assert_refused(client.delete(f"/api/admin/avatars/{LIVE2D_ID}"))


# ----- model hosting ------------------------------------------------------

def test_model_host_refuses_a_non_model_file(client):
    """An archive is operator input and can carry anything beside the model
    files. The console is served from the SAME origin as the model host, so
    a member that comes back as application/javascript is a script-planting
    primitive on an admin plane that has no auth by design.
    """
    resp = _upload(client, raw=_zip_with_extra("evil.js", b"alert(document.cookie)"))
    assert resp.status_code == 201, resp.text
    avatar_id = resp.json()["avatar_id"]
    _assert_refused(
        client.get(f"/api/v1/live2d/avatars/{avatar_id}/model/evil.js")
    )


def test_model_file_is_served_with_nosniff(client):
    """Belt to the extension allowlist: even a model file must not be
    re-interpreted by the browser as something it is not."""
    _seed_live2d(client)
    resp = client.get(f"/api/v1/live2d/avatars/{LIVE2D_ID}/model/model.model3.json")
    assert resp.status_code == 200
    assert resp.headers.get("x-content-type-options") == "nosniff"


def test_model_file_is_open_when_no_token_configured(client):
    _seed_live2d(client)
    resp = client.get(f"/api/v1/live2d/avatars/{LIVE2D_ID}/model/model.model3.json")
    assert resp.status_code == 200
    assert resp.json()["Version"] == 3


def test_model_file_requires_the_token_when_configured(client, monkeypatch):
    _seed_live2d(client)
    _token(monkeypatch, "s3cret")
    resp = client.get(f"/api/v1/live2d/avatars/{LIVE2D_ID}/model/model.model3.json")
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "unauthorized"


def test_model_file_served_with_the_token(client, monkeypatch):
    _seed_live2d(client)
    _token(monkeypatch, "s3cret")
    resp = client.get(
        f"/api/v1/live2d/avatars/{LIVE2D_ID}/model/model.moc3",
        headers={"X-V1-Token": "s3cret"},
    )
    assert resp.status_code == 200
    assert resp.content == b"\x00" * 16


def test_model_path_traversal_refused(client, monkeypatch):
    # Review Focus: {path:path} is the most dangerous parameter in the app.
    #
    # Percent-encoded, because that is the form that actually reaches a
    # handler: an HTTP client normalises a literal "../" out of the URL
    # before it is sent, so the naive spelling never exercises the check.
    # Starlette decodes the escapes back into ".." for the path param.
    _seed_live2d(client)
    _token(monkeypatch, "s3cret")
    resp = client.get(
        f"/api/v1/live2d/avatars/{LIVE2D_ID}/model/%2e%2e%2f%2e%2e%2f%2e%2e%2f.env",
        headers={"X-V1-Token": "s3cret"},
    )
    _assert_refused(resp)


def test_model_route_refuses_a_photoreal_avatar(client):
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (640, 800), (10, 20, 30)).save(buf, format="PNG")
    created = client.post("/api/v1/avatars", data={"name": "p"},
                          files={"file": ("p.png", buf.getvalue(), "image/png")})
    avatar_id = created.json()["avatar_id"]
    store = client.app.state.session_manager.store
    (store.avatar_dir(avatar_id) / "leak.txt").write_text("secret", encoding="utf-8")
    _assert_refused(client.get(f"/api/v1/live2d/avatars/{avatar_id}/model/leak.txt"))


# ----- SDK runtime --------------------------------------------------------

def test_health_reports_a_missing_sdk(client, monkeypatch, tmp_path):
    monkeypatch.setenv("DH_LIVE2D_SDK_DIR", str(tmp_path / "nope"))
    get_settings.cache_clear()
    body = client.get("/api/admin/live2d/health").json()
    assert body["sdk_present"] is False
    assert body["core_present"] is False


def _write_sdk(sdk, shaders_at="Framework"):
    """A Cubism SDK laid out the way CubismSdkForWeb-5-r.5 actually ships.

    The core is at the root, but the GLSL is NOT beside it: it sits under
    Framework/, next to the framework source. Every shader 404'd until the
    runtime endpoint learned that.
    """
    (sdk / "Core").mkdir(parents=True)
    (sdk / "Core" / "live2dcubismcore.js").write_text("core", encoding="utf-8")
    shaders = sdk / shaders_at / "Shaders" / "WebGL"
    shaders.mkdir(parents=True)
    (shaders / "a.vert").write_text("vert", encoding="utf-8")
    return sdk


def test_health_reports_a_present_sdk(client, monkeypatch, tmp_path):
    sdk = _write_sdk(tmp_path / "sdk")
    monkeypatch.setenv("DH_LIVE2D_SDK_DIR", str(sdk))
    get_settings.cache_clear()
    body = client.get("/api/admin/live2d/health").json()
    assert body["sdk_present"] is True
    assert body["core_present"] is True
    assert body["shaders_present"] is True


def test_runtime_serves_core_and_shaders(client, monkeypatch, tmp_path):
    sdk = _write_sdk(tmp_path / "sdk")
    monkeypatch.setenv("DH_LIVE2D_SDK_DIR", str(sdk))
    get_settings.cache_clear()
    assert client.get("/api/v1/live2d/runtime/Core/live2dcubismcore.js").text == "core"
    assert client.get("/api/v1/live2d/runtime/Shaders/WebGL/a.vert").text == "vert"


def test_runtime_serves_shaders_from_a_root_level_shaders_dir(client, monkeypatch, tmp_path):
    """The URL namespace is Core/ and Shaders/ regardless of how the SDK is
    unpacked, so a SDK with its shaders at the root still works."""
    sdk = _write_sdk(tmp_path / "sdk", shaders_at="")
    monkeypatch.setenv("DH_LIVE2D_SDK_DIR", str(sdk))
    get_settings.cache_clear()
    assert client.get("/api/v1/live2d/runtime/Shaders/WebGL/a.vert").text == "vert"


def test_runtime_is_served_without_a_token_even_when_one_is_configured(client, monkeypatch, tmp_path):
    """The SDK runtime is deliberately not token-gated.

    The core loads via a <script> tag, which cannot carry a custom header,
    and the framework fetches its own GLSL with a fetch the caller cannot
    configure. Requiring a token would make the runtime unloadable rather
    than protected. Model files -- the operator-supplied content -- are
    gated separately, and that is the boundary this pins.
    """
    sdk = tmp_path / "sdk"
    (sdk / "Core").mkdir(parents=True)
    (sdk / "Core" / "live2dcubismcore.js").write_text("core", encoding="utf-8")
    monkeypatch.setenv("DH_LIVE2D_SDK_DIR", str(sdk))
    _token(monkeypatch, "s3cret")
    assert client.get("/api/v1/live2d/runtime/Core/live2dcubismcore.js").text == "core"


@pytest.mark.parametrize(
    "path",
    [
        # Percent-encoded escapes reach the handler; a literal "../" is
        # normalised away by the client before the request is sent.
        "Core/%2e%2e%2f%2e%2e%2f%2e%2e%2f.env",
        "LICENSE.md",
        "Samples/Resources/Hiyori/Hiyori.model3.json",
        "Framework/src/live2dcubismframework.ts",
    ],
)
def test_runtime_refuses_anything_outside_core_and_shaders(client, monkeypatch, tmp_path, path):
    sdk = tmp_path / "sdk"
    (sdk / "Core").mkdir(parents=True)
    (sdk / "Core" / "live2dcubismcore.js").write_text("core", encoding="utf-8")
    (sdk / "LICENSE.md").write_text("secret license", encoding="utf-8")
    (sdk / "Framework").mkdir()
    (sdk / "Framework" / "src").mkdir()
    (sdk / "Framework" / "src" / "x.ts").write_text("secret", encoding="utf-8")
    monkeypatch.setenv("DH_LIVE2D_SDK_DIR", str(sdk))
    get_settings.cache_clear()
    _assert_refused(client.get(f"/api/v1/live2d/runtime/{path}"))


@pytest.mark.parametrize(
    "path",
    [
        # The allowlist is a boundary, not a string prefix: a path that
        # starts with an allowed name but resolves outside its tree must
        # still be refused. These reach the handler percent-encoded, which
        # is the form a client actually sends -- a literal "../" is
        # normalised away before it leaves the client.
        "Core/%2e%2e/LICENSE.md",
        "Core/%2e%2e/Framework/src/live2dcubismframework.ts",
        "Core/%2e%2e/Samples/Resources/Hiyori/Hiyori.model3.json",
        "Shaders/%2e%2e/Core/live2dcubismcore.js",
    ],
)
def test_runtime_refuses_to_escape_its_trees(client, monkeypatch, tmp_path, path):
    sdk = _write_sdk(tmp_path / "sdk")
    (sdk / "LICENSE.md").write_text("secret license", encoding="utf-8")
    (sdk / "Framework" / "src").mkdir(parents=True)
    (sdk / "Framework" / "src" / "live2dcubismframework.ts").write_text("secret", encoding="utf-8")
    hiyori = sdk / "Samples" / "Resources" / "Hiyori"
    hiyori.mkdir(parents=True)
    (hiyori / "Hiyori.model3.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("DH_LIVE2D_SDK_DIR", str(sdk))
    get_settings.cache_clear()
    _assert_refused(client.get(f"/api/v1/live2d/runtime/{path}"))