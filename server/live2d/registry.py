"""Validate a Live2D Cubism project directory at registration time.

Registration is the only cheap moment to check a model is whole. A Cubism
project is a dozen interlinked files -- ``model3.json`` names the moc3, the
textures, every expression and every motion -- and a missing one otherwise
surfaces as a blank canvas in the browser, minutes later, with no
server-side trace.
"""

import json
from pathlib import Path

from server.core.errors import AppError

_BAD = "live2d_invalid_model"


def _fail(message: str) -> AppError:
    return AppError(_BAD, message, 422)


def _resolve(root: Path, relative: str) -> Path:
    """Resolve a model-relative path, refusing anything that leaves root.

    Two checks, because ``..`` alone is not sufficient: an absolute path is
    also an escape, and so is a symlink. Both are cheap here.
    """
    if not relative or relative.startswith(("/", "\\")) or ".." in Path(relative).parts:
        raise _fail(f"illegal path in model: {relative!r}")
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()):
        raise _fail(f"illegal path in model: {relative!r}")
    return target


def _require_file(root: Path, relative: str, what: str) -> Path:
    target = _resolve(root, relative)
    if not target.is_file():
        raise _fail(f"missing {what}: {relative}")
    return target


def validate_model_dir(path: Path, config: dict) -> dict:
    """Validate ``config`` against the Cubism project at ``path``.

    Returns the config unchanged so callers can ``store.save_pack``-style
    chain it. Raises :class:`AppError` (``live2d_invalid_model``, 422) with a
    message naming the specific file at fault -- the console shows it
    verbatim, and "invalid model" alone would not tell an operator what to fix.
    """
    root = Path(path)
    entry_name = (config.get("model") or {}).get("entry")
    if not entry_name:
        raise _fail("config.model.entry is required")
    _require_file(root, entry_name, "entry file")

    try:
        spec = json.loads((root / entry_name).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise _fail(f"entry file is not readable JSON: {exc}") from exc

    refs = spec.get("FileReferences") or {}

    moc = refs.get("Moc")
    if not moc:
        raise _fail("entry file declares no Moc")
    _require_file(root, moc, "moc")

    for texture in refs.get("Textures") or []:
        _require_file(root, texture, "texture")

    expression_names: list[str] = []
    for expr in refs.get("Expressions") or []:
        name = str(expr.get("Name"))
        expression_names.append(name)
        _require_file(root, expr.get("File", ""), f"expression {name}")

    for group in (refs.get("Motions") or {}).values():
        for motion in group or []:
            _require_file(root, motion.get("File", ""), "motion")

    # The operator's emotion map is {emotion tag: POSITION in the model's
    # expression list} -- that is what expressionIndex() consumes on the
    # client, and what the console's form tooltip tells them to enter.
    #
    # Validating the number against the expression NAMES would reject every
    # real model: shipped Cubism models name their expressions "exp_01",
    # "F01", "Smile" and so on, never "0". Only synthetic fixtures with
    # numeric names would have passed that check.
    for tag, idx in (config.get("expressions") or {}).items():
        if not isinstance(idx, int) or isinstance(idx, bool) or not 0 <= idx < len(expression_names):
            raise _fail(
                f"expressions[{tag!r}] = {idx!r} but the model declares "
                f"{len(expression_names)} expression(s) "
                f"({', '.join(expression_names) or 'none'}), so the value must be an "
                f"index in 0..{max(len(expression_names) - 1, 0)}"
            )

    return config