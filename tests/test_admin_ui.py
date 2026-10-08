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
    resp = ui_client.get("/admin")
    assert resp.status_code == 200
    assert 'id="root"' in resp.text
