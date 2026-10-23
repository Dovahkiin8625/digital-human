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
            kind="live2d", config_json={"model": {"entry": "model.model3.json"}},
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

def test_registration_copies_and_validates(client, tmp_path):
    src = _write_model(tmp_path / "src")
    resp = client.post(
        "/api/admin/live2d/avatars",
        json={
            "name": "Hiyori",
            "model_dir": str(src),
            "config": {"model": {"entry": "model.model3.json"}, "expressions": {"smile": 0}},
        },
    )
    assert resp.status_code == 201, resp.text
    avatar_id = resp.json()["avatar_id"]

    detail = client.get(f"/api/v1/avatars/{avatar_id}").json()
    assert detail["kind"] == "live2d"
    assert detail["status"] == "ready"
    # The avatar owns its files, so deleting it takes the model with it.
    root = client.app.state.session_manager.store.avatar_dir(avatar_id)
    assert (root / "model.moc3").is_file()
    assert src.is_dir(), "the source directory must be left alone"


def test_registration_rejects_a_broken_model(client, tmp_path):
    src = _write_model(tmp_path / "src")
    (src / "model.moc3").unlink()
    resp = client.post(
        "/api/admin/live2d/avatars",
        json={"name": "broken", "model_dir": str(src),
              "config": {"model": {"entry": "model.model3.json"}}},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "live2d_invalid_model"
    assert "moc" in resp.json()["error"]["message"]


def test_registration_rejects_a_missing_directory(client, tmp_path):
    resp = client.post(
        "/api/admin/live2d/avatars",
        json={"name": "nope", "model_dir": str(tmp_path / "absent"),
              "config": {"model": {"entry": "model.model3.json"}}},
    )
    assert resp.status_code == 422


def test_registration_rejects_an_expression_the_model_lacks(client, tmp_path):
    src = _write_model(tmp_path / "src")
    resp = client.post(
        "/api/admin/live2d/avatars",
        json={"name": "bad-expr", "model_dir": str(src),
              "config": {"model": {"entry": "model.model3.json"},
                         "expressions": {"smile": 9}}},
    )
    assert resp.status_code == 422


def test_delete_removes_row_and_assets(client):
    _seed_live2d(client)
    store = client.app.state.session_manager.store
    assert client.delete(f"/api/admin/live2d/avatars/{LIVE2D_ID}").status_code == 204
    assert client.get(f"/api/v1/avatars/{LIVE2D_ID}").status_code == 404
    assert not store.avatar_dir(LIVE2D_ID).exists()


def test_delete_refuses_a_photoreal_avatar(client, tmp_path):
    """The live2d delete endpoint must not become a second, differently
    behaved way to delete a video avatar."""
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (640, 800), (10, 20, 30)).save(buf, format="PNG")
    created = client.post("/api/v1/avatars", data={"name": "p"},
                          files={"file": ("p.png", buf.getvalue(), "image/png")})
    _assert_refused(client.delete(f"/api/admin/live2d/avatars/{created.json()['avatar_id']}"))


# ----- model hosting ------------------------------------------------------

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


def test_health_reports_a_present_sdk(client, monkeypatch, tmp_path):
    sdk = tmp_path / "sdk"
    (sdk / "Core").mkdir(parents=True)
    (sdk / "Core" / "live2dcubismcore.js").write_text("core", encoding="utf-8")
    (sdk / "Shaders" / "WebGL").mkdir(parents=True)
    monkeypatch.setenv("DH_LIVE2D_SDK_DIR", str(sdk))
    get_settings.cache_clear()
    body = client.get("/api/admin/live2d/health").json()
    assert body["sdk_present"] is True
    assert body["core_present"] is True
    assert body["shaders_present"] is True


def test_runtime_serves_core_and_shaders(client, monkeypatch, tmp_path):
    sdk = tmp_path / "sdk"
    (sdk / "Core").mkdir(parents=True)
    (sdk / "Core" / "live2dcubismcore.js").write_text("core", encoding="utf-8")
    (sdk / "Shaders" / "WebGL").mkdir(parents=True)
    (sdk / "Shaders" / "WebGL" / "a.vert").write_text("vert", encoding="utf-8")
    monkeypatch.setenv("DH_LIVE2D_SDK_DIR", str(sdk))
    get_settings.cache_clear()
    assert client.get("/api/v1/live2d/runtime/Core/live2dcubismcore.js").text == "core"
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