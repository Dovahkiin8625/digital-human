"""The console <-> server contract for Live2D.

The console's three pages all read ``/api/admin/avatars`` and branch on
``kind``. The pet reads ``/api/v1/avatars?kind=live2d``. Each endpoint has
its own serialiser, and each was missing a field the other side needs --
which is how a Live2D avatar could be registered successfully and then be
invisible to every page that can select one.
"""

from sqlalchemy.orm import sessionmaker

from server.db.models import Avatar
from server.db.session import get_engine


def _seed(client):
    db = sessionmaker(bind=get_engine())()
    try:
        db.add(Avatar(name="l2d", status="ready", source_image_path="",
                      kind="live2d",
                      config_json={"model": {"entry": "model.model3.json"},
                                   "expressions": {"smile": 0}}))
        db.commit()
    finally:
        db.close()


def test_admin_list_carries_kind(client):
    """Debug, Chat and Avatars all branch on this. Without it the branch
    before the session POST never fires and the page 409s."""
    _seed(client)
    rows = client.get("/api/admin/avatars").json()
    assert [r["kind"] for r in rows] == ["live2d"]


def test_admin_list_carries_config(client):
    """The console needs ``config.model.entry`` to build the model URL."""
    _seed(client)
    row = client.get("/api/admin/avatars").json()[0]
    assert row["config"]["model"]["entry"] == "model.model3.json"


def test_admin_detail_carries_kind(client):
    _seed(client)
    avatar_id = client.get("/api/admin/avatars").json()[0]["avatar_id"]
    assert client.get(f"/api/admin/avatars/{avatar_id}").json()["kind"] == "live2d"


def test_v1_list_carries_kind_and_config(client):
    """The pet lists avatars off /api/v1 and needs both: the model URL is
    built from config.model.entry, and the emotion allow-list comes from
    config.expressions."""
    _seed(client)
    row = client.get("/api/v1/avatars", params={"kind": "live2d"}).json()[0]
    assert row["kind"] == "live2d"
    assert row["config"]["model"]["entry"] == "model.model3.json"
    assert row["config"]["expressions"] == {"smile": 0}


def test_kind_filter_on_admin_list(client):
    _seed(client)
    assert client.get("/api/admin/avatars", params={"kind": "photoreal"}).json() == []
    assert len(client.get("/api/admin/avatars", params={"kind": "live2d"}).json()) == 1