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


def _write_model(root: Path, *, with_moc: bool = True, expr_ids=("exp_01", "F01"),
                 with_motion: bool = True) -> Path:
    files = {
        "Moc": "model.moc3" if with_moc else "missing.moc3",
        "Textures": ["texture_00.png"],
        # Realistic names on purpose. Shipped Cubism models name their
        # expressions "exp_01", "F01", "Smile" -- never "0". A fixture with
        # numeric names hid a bug where the registry validated the map's
        # index against the names and so rejected every real model.
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
    (root / "F01.exp3.json").unlink()
    with pytest.raises(AppError) as ei:
        validate_model_dir(root, dict(ENTRY))
    assert ei.value.code == "live2d_invalid_model"


def test_rejects_a_missing_motion_file(tmp_path):
    with pytest.raises(AppError) as ei:
        validate_model_dir(_write_model(tmp_path, with_motion=False), dict(ENTRY))
    assert ei.value.code == "live2d_invalid_model"


def test_resolves_file_references_next_to_the_entry(tmp_path):
    """Cubism resolves every FileReferences path relative to the model3.json
    that declares them -- and so does the console's own loader, which
    fetches the moc from beside the entry rather than from the project root.

    A model in a subfolder (what right-click compress produces) names its
    moc "model.moc3" while the file sits next to the entry, not at the
    root. Validating against the root instead rejects every such model.
    """
    root = tmp_path / "Hiyori"
    root.mkdir(parents=True)
    _write_model(root)
    cfg = validate_model_dir(tmp_path, {"model": {"entry": "Hiyori/model.model3.json"}})
    assert cfg["model"]["entry"] == "Hiyori/model.model3.json"


def test_rejects_a_missing_moc_beside_a_subfolder_entry(tmp_path):
    """The subfolder case has to fail loudly too: a moc that is missing
    next to the entry is missing, even if some other file of that name
    happens to sit at the project root."""
    root = tmp_path / "Hiyori"
    root.mkdir(parents=True)
    _write_model(root)
    (tmp_path / "model.moc3").write_bytes(b"\x00" * 16)
    (root / "model.moc3").unlink()
    with pytest.raises(AppError) as ei:
        validate_model_dir(tmp_path, {"model": {"entry": "Hiyori/model.model3.json"}})
    assert ei.value.code == "live2d_invalid_model"


def test_accepts_a_model_with_no_motions_key(tmp_path):
    # Plenty of models ship without motions. An absent Motions key is not a
    # broken model, only a dangling reference is.
    root = _write_model(tmp_path)
    spec = json.loads((root / "model.model3.json").read_text(encoding="utf-8"))
    del spec["FileReferences"]["Motions"]
    (root / "model.model3.json").write_text(json.dumps(spec), encoding="utf-8")
    assert validate_model_dir(root, dict(ENTRY))["model"]["entry"] == "model.model3.json"


def test_rejects_an_expression_index_past_the_end_of_the_list(tmp_path):
    # The map's value is a POSITION in the model's expression list -- what
    # expressionIndex() consumes. An off-by-one here is invisible until the
    # pet tries to make a face that does not exist.
    with pytest.raises(AppError) as ei:
        validate_model_dir(
            _write_model(tmp_path, expr_ids=("exp_01",)),
            {"model": {"entry": "model.model3.json"}, "expressions": {"smile": 7}},
        )
    assert ei.value.code == "live2d_invalid_model"
    assert "7" in ei.value.message


def test_rejects_a_non_numeric_expression_index(tmp_path):
    with pytest.raises(AppError) as ei:
        validate_model_dir(
            _write_model(tmp_path),
            {"model": {"entry": "model.model3.json"}, "expressions": {"smile": "F01"}},
        )
    assert ei.value.code == "live2d_invalid_model"


def test_accepts_an_expression_id_the_model_declares(tmp_path):
    cfg = validate_model_dir(
        _write_model(tmp_path, expr_ids=("exp_01", "F01")),
        {"model": {"entry": "model.model3.json"}, "expressions": {"smile": 1}},
    )
    assert cfg["expressions"]["smile"] == 1


def test_accepts_a_realistic_model_with_named_expressions(tmp_path):
    """The whole point: shipped models name expressions "exp_01"/"F01", and
    an operator entering {"smile": 0} must be accepted."""
    cfg = validate_model_dir(
        _write_model(tmp_path),
        {"model": {"entry": "model.model3.json"}, "expressions": {"smile": 0, "sad": 1}},
    )
    assert cfg["expressions"] == {"smile": 0, "sad": 1}


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