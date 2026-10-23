"""Registering a Live2D model is the only cheap moment to check it is whole.

A Cubism project is a dozen interlinked files: ``model3.json`` names the moc3,
the textures, every expression and every motion. A missing one otherwise
surfaces as a blank canvas in the browser, minutes later, with no server-side
trace.
"""

import json
from pathlib import Path

import pytest

from server.core.errors import AppError
from server.live2d.registry import validate_model_dir


def _write_model(root: Path, *, with_moc: bool = True, expr_ids=("0", "1"),
                 with_motion: bool = True) -> Path:
    files = {
        "Moc": "model.moc3" if with_moc else "missing.moc3",
        "Textures": ["texture_00.png"],
        "Expressions": [{"Name": e, "File": f"{e}.exp3.json"} for e in expr_ids],
        # Always declared. `with_motion=False` means the reference is
        # present but the file is not -- a model with no Motions key at all
        # is perfectly valid and must not be rejected.
        "Motions": {"Idle": [{"File": "motions/idle.motion3.json"}]},
    }
    (root / "model.model3.json").write_text(
        json.dumps({"Version": 3, "FileReferences": files}), encoding="utf-8"
    )
    if with_moc:
        (root / "model.moc3").write_bytes(b"\x00" * 16)
    (root / "texture_00.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    for e in expr_ids:
        (root / f"{e}.exp3.json").write_text("{}", encoding="utf-8")
    if with_motion:
        (root / "motions").mkdir(exist_ok=True)
        (root / "motions" / "idle.motion3.json").write_text("{}", encoding="utf-8")
    return root


ENTRY = {"model": {"entry": "model.model3.json"}}


def test_accepts_a_complete_model(tmp_path):
    cfg = validate_model_dir(_write_model(tmp_path), dict(ENTRY))
    assert cfg["model"]["entry"] == "model.model3.json"


def test_rejects_a_config_with_no_entry(tmp_path):
    with pytest.raises(AppError) as ei:
        validate_model_dir(_write_model(tmp_path), {"model": {}})
    assert ei.value.code == "live2d_invalid_model"


def test_rejects_a_missing_entry_file(tmp_path):
    with pytest.raises(AppError) as ei:
        validate_model_dir(_write_model(tmp_path), {"model": {"entry": "nope.json"}})
    assert ei.value.code == "live2d_invalid_model"


def test_rejects_a_missing_moc(tmp_path):
    with pytest.raises(AppError) as ei:
        validate_model_dir(_write_model(tmp_path, with_moc=False), dict(ENTRY))
    assert ei.value.code == "live2d_invalid_model"


def test_rejects_a_missing_texture(tmp_path):
    root = _write_model(tmp_path)
    (root / "texture_00.png").unlink()
    with pytest.raises(AppError) as ei:
        validate_model_dir(root, dict(ENTRY))
    assert ei.value.code == "live2d_invalid_model"


def test_rejects_a_missing_expression_file(tmp_path):
    root = _write_model(tmp_path)
    (root / "1.exp3.json").unlink()
    with pytest.raises(AppError) as ei:
        validate_model_dir(root, dict(ENTRY))
    assert ei.value.code == "live2d_invalid_model"


def test_rejects_a_missing_motion_file(tmp_path):
    with pytest.raises(AppError) as ei:
        validate_model_dir(_write_model(tmp_path, with_motion=False), dict(ENTRY))
    assert ei.value.code == "live2d_invalid_model"


def test_accepts_a_model_with_no_motions_key(tmp_path):
    # Plenty of models ship without motions. An absent Motions key is not a
    # broken model, only a dangling reference is.
    root = _write_model(tmp_path)
    spec = json.loads((root / "model.model3.json").read_text(encoding="utf-8"))
    del spec["FileReferences"]["Motions"]
    (root / "model.model3.json").write_text(json.dumps(spec), encoding="utf-8")
    assert validate_model_dir(root, dict(ENTRY))["model"]["entry"] == "model.model3.json"


def test_rejects_an_expression_id_the_model_does_not_declare(tmp_path):
    # The operator's emotion map indexes into the model's own expression
    # list. An off-by-one here is invisible until the pet tries to make a
    # face that does not exist.
    with pytest.raises(AppError) as ei:
        validate_model_dir(
            _write_model(tmp_path, expr_ids=("0",)),
            {"model": {"entry": "model.model3.json"}, "expressions": {"smile": 7}},
        )
    assert ei.value.code == "live2d_invalid_model"
    assert "7" in ei.value.message


def test_accepts_an_expression_id_the_model_declares(tmp_path):
    cfg = validate_model_dir(
        _write_model(tmp_path, expr_ids=("0", "1")),
        {"model": {"entry": "model.model3.json"}, "expressions": {"smile": 1}},
    )
    assert cfg["expressions"]["smile"] == 1


@pytest.mark.parametrize(
    "entry", ["../../etc/passwd", "/etc/passwd", "sub/../../outside.json"]
)
def test_rejects_a_path_traversal_entry(tmp_path, entry):
    with pytest.raises(AppError) as ei:
        validate_model_dir(_write_model(tmp_path), {"model": {"entry": entry}})
    assert ei.value.code == "live2d_invalid_model"


def test_rejects_a_traversal_inside_file_references(tmp_path):
    root = _write_model(tmp_path)
    spec = json.loads((root / "model.model3.json").read_text(encoding="utf-8"))
    spec["FileReferences"]["Textures"] = ["../../../.env"]
    (root / "model.model3.json").write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(AppError) as ei:
        validate_model_dir(root, dict(ENTRY))
    assert ei.value.code == "live2d_invalid_model"