"""The Live2D avatar's editable configuration.

The emotion map is the operator's only handle on the model's face: it
says which position in the model's OWN expression list each emotion tag
wears. It could only be set at upload time otherwise, so an avatar that
was uploaded without one could never be given a face -- and the model
would sit there with the same expression forever.
"""

import io
import json
import zipfile


def _project_zip() -> bytes:
    """A Cubism project with realistic expression names and motion groups.

    The names matter: shipped models call their expressions "Angry" and
    "Smile", never "0", so a validation that only accepted numeric names
    would have passed every synthetic fixture and every real model.

    The file CONTENTS are placeholders: registration checks that the
    files exist and that the entry file parses, not that a moc3 is a
    valid moc3.
    """
    entries = {
        "model.model3.json": json.dumps({
            "Version": 3,
            "FileReferences": {
                "Moc": "model.moc3",
                "Textures": ["texture_00.png"],
                "Expressions": [
                    {"Name": "Angry", "File": "exp/Angry.exp3.json"},
                    {"Name": "Normal", "File": "exp/Normal.exp3.json"},
                    {"Name": "Smile", "File": "exp/Smile.exp3.json"},
                ],
                "Motions": {
                    "Idle": [
                        {"File": "motions/mtn_00.motion3.json"},
                        {"File": "motions/mtn_01.motion3.json"},
                    ],
                    "TapBody": [
                        {"File": "motions/mtn_02.motion3.json"},
                    ],
                },
            },
        }).encode(),
        "model.moc3": b"moc-placeholder",
        "texture_00.png": b"png-placeholder",
        "exp/Angry.exp3.json": b"{}",
        "exp/Normal.exp3.json": b"{}",
        "exp/Smile.exp3.json": b"{}",
        "motions/mtn_00.motion3.json": b"{}",
        "motions/mtn_01.motion3.json": b"{}",
        "motions/mtn_02.motion3.json": b"{}",
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _upload(client, name="Natori", expressions=None):
    form = {"name": name, "model": "live2d"}
    if expressions:
        form["expressions"] = expressions
    return client.post(
        "/api/admin/avatars",
        data=form,
        files={"file": ("model.zip", _project_zip(), "application/zip")},
    )


def _png(size=(640, 800)):
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", size, (10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


def test_the_inventory_reports_the_models_own_expressions(client):
    """The console offers a choice, so it needs the names behind the
    positions -- the map is written in positions, and a bare number means
    nothing to an operator."""
    avatar_id = _upload(client).json()["avatar_id"]

    resp = client.get(f"/api/admin/avatars/{avatar_id}/live2d/model")
    assert resp.status_code == 200
    body = resp.json()
    assert body["entry"] == "model.model3.json"
    assert body["expressions"] == [
        {"index": 0, "name": "Angry"},
        {"index": 1, "name": "Normal"},
        {"index": 2, "name": "Smile"},
    ]
    assert body["motions"] == {"Idle": 2, "TapBody": 1}


def test_the_emotion_map_can_be_set_after_the_upload(client):
    """An avatar registered without a map can still be given a face."""
    avatar_id = _upload(client).json()["avatar_id"]

    resp = client.patch(
        f"/api/admin/avatars/{avatar_id}/live2d",
        json={"expressions": {"smile": 2, "sad": 0, "neutral": 1}},
    )
    assert resp.status_code == 200
    assert resp.json()["config"]["expressions"] == {
        "smile": 2,
        "sad": 0,
        "neutral": 1,
    }

    # And it sticks: the detail route is what the console reads.
    detail = client.get(f"/api/admin/avatars/{avatar_id}").json()
    assert detail["config"]["expressions"]["smile"] == 2


def test_the_emotion_map_can_be_cleared(client):
    avatar_id = _upload(client, expressions='{"smile": 2}').json()["avatar_id"]

    resp = client.patch(
        f"/api/admin/avatars/{avatar_id}/live2d", json={"expressions": {}}
    )
    assert resp.status_code == 200
    assert resp.json()["config"]["expressions"] == {}


def test_an_index_the_model_does_not_declare_is_refused(client):
    """The map is validated against the project on disk. An index the
    model does not have would become an expressionIndex the client cannot
    use, and the symptom -- a face that never changes -- would only show
    up minutes later, in the browser."""
    avatar_id = _upload(client, expressions='{"smile": 2}').json()["avatar_id"]

    resp = client.patch(
        f"/api/admin/avatars/{avatar_id}/live2d",
        json={"expressions": {"smile": 9}},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "live2d_invalid_model"

    # The stored map is untouched.
    detail = client.get(f"/api/admin/avatars/{avatar_id}").json()
    assert detail["config"]["expressions"] == {"smile": 2}


def test_a_rejected_map_leaves_the_stored_one_alone(client):
    avatar_id = _upload(client).json()["avatar_id"]

    resp = client.patch(
        f"/api/admin/avatars/{avatar_id}/live2d",
        json={"expressions": {"smile": "two"}},
    )
    assert resp.status_code == 422

    detail = client.get(f"/api/admin/avatars/{avatar_id}").json()
    assert detail["config"]["expressions"] == {}


def test_a_non_live2d_avatar_is_refused(client):
    """These endpoints act on a Cubism project; there is nothing sensible
    to do to a MuseTalk avatar here."""
    resp = client.post(
        "/api/admin/avatars",
        data={"name": "photo"},
        files={"file": ("a.png", _png(), "image/png")},
    )
    avatar_id = resp.json()["avatar_id"]

    assert client.get(f"/api/admin/avatars/{avatar_id}/live2d/model").status_code == 400
    assert client.patch(
        f"/api/admin/avatars/{avatar_id}/live2d", json={"expressions": {}}
    ).status_code == 400


def test_a_missing_avatar_is_a_404(client):
    missing = "b" * 32
    assert client.get(f"/api/admin/avatars/{missing}/live2d/model").status_code == 404
    assert client.patch(
        f"/api/admin/avatars/{missing}/live2d", json={"expressions": {}}
    ).status_code == 404
