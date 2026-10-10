import pathlib

import pytest

DIST = pathlib.Path(__file__).resolve().parent.parent / "admin-ui" / "dist"
INDEX = DIST / "index.html"
_MINIMAL = '<!doctype html><html><body><div id="root"></div></body></html>'


@pytest.fixture()
def spa_dist():
    """Guarantee a built SPA index exists so create_app() registers the /admin mount.

    The real ``npm run build`` output is used when present; otherwise a minimal
    index.html is synthesized (and cleaned up) so the mount wiring is still
    exercised on a source-only checkout.
    """
    created = not INDEX.exists()
    if created:
        DIST.mkdir(parents=True, exist_ok=True)
        INDEX.write_text(_MINIMAL, encoding="utf-8")
    yield INDEX
    if created:
        INDEX.unlink(missing_ok=True)
        try:
            DIST.rmdir()
        except OSError:
            pass


@pytest.fixture()
def ui_client(spa_dist, client):
    # spa_dist is set up before client, so create_app() sees the dist directory.
    return client


def test_admin_mount_serves_spa_index(ui_client):
    resp = ui_client.get("/admin/")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert 'id="root"' in resp.text


def test_admin_root_redirects_to_index(ui_client):
    # Assert the redirect itself, not just the followed 200: a regression that
    # stops /admin redirecting to /admin/ must fail this test.
    resp = ui_client.get("/admin", follow_redirects=False)
    assert resp.status_code in (301, 302, 307, 308)
    assert resp.headers["location"].endswith("/admin/")


def test_admin_deep_link_serves_spa_index(ui_client):
    # A client-side route (no file extension) must fall back to the SPA index
    # so a refresh / deep link under the /admin mount does not 404.
    resp = ui_client.get("/admin/avatars")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert 'id="root"' in resp.text


def test_admin_missing_asset_still_404s(ui_client):
    # A request for a real file (has an extension) must NOT fall back to index.
    resp = ui_client.get("/admin/assets/does-not-exist.js")
    assert resp.status_code == 404


def test_browser_router_uses_admin_basename():
    # The SPA is served under /admin, so client URLs must be scoped to that
    # mount; otherwise refresh/deep-links escape it. Not observable from the
    # server, so guard the source directly.
    source = (
        pathlib.Path(__file__).resolve().parent.parent
        / "admin-ui"
        / "src"
        / "main.tsx"
    ).read_text(encoding="utf-8")
    assert 'basename="/admin"' in source


def _avatars_source():
    return (
        pathlib.Path(__file__).resolve().parent.parent
        / "admin-ui"
        / "src"
        / "pages"
        / "Avatars.tsx"
    ).read_text(encoding="utf-8")


def test_avatar_upload_accepts_video():
    # The upload control must offer videos as well as images; the server enforces
    # the real limits, but the widget has to let a video be chosen at all.
    assert 'accept="image/*,video/*"' in _avatars_source()


def test_avatar_create_has_synchronous_reentry_guard():
    # A second click while video metadata is awaited (or before React re-renders)
    # must not land a second POST; state alone flips too late, so a ref guards.
    source = _avatars_source()
    assert "const creatingRef = useRef(false);" in source
    assert "if (creatingRef.current) return;" in source
    # The ref is taken only after the early validations, and always reset here:
    assert (
        "    } finally {\n"
        "      creatingRef.current = false;\n"
        "      setCreating(false);\n"
        "    }"
    ) in source
