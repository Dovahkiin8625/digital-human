"""Admin-tunable avatar-creation knobs.

This module owns the *schema* for the small set of parameters that affect
avatar build quality. The pipeline reads from a DB-backed key/value store at
``avatarserver.db.models.Setting``; missing keys fall back to the defaults
defined here. The admin console writes via ``PATCH /api/admin/settings``.

Picking which knobs are tunable is a deliberate product call: every entry below
has a real, observable effect on the produced avatar. Keep this list short --
adding a knob for its own sake grows the surface area without helping anyone.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import json

# The presets are all x264 ones; PyAV rejects anything not on its list.
X264_PRESETS = (
    "ultrafast",
    "superfast",
    "veryfast",
    "faster",
    "fast",
    "medium",
    "slow",
    "slower",
    "veryslow",
)


@dataclass(frozen=True)
class ParamSpec:
    name: str
    type: Literal["int", "float", "enum", "str"]
    default: Any
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[str, ...] | None = None


#: Every admin-tunable knob: name, JSON value type, default, allowed range.
#: Pipeline and idle-video code import this list to know which keys to honour.
TUNABLES: tuple[ParamSpec, ...] = (
    # -- Production pipeline (server/avatar/musetalk_pipeline.py) --
    ParamSpec("face_crop_size", "int", 256, minimum=128, maximum=512),
    ParamSpec("bbox_shift", "int", 0, minimum=-100, maximum=100),
    ParamSpec("min_face_width_px", "float", 160.0, minimum=32.0, maximum=720.0),
    ParamSpec("min_nose_ratio", "float", 0.55, minimum=0.0, maximum=1.0),
    ParamSpec("min_sharpness", "float", 60.0, minimum=0.0, maximum=2000.0),
    ParamSpec("min_occupancy", "float", 0.92, minimum=0.0, maximum=1.0),
    ParamSpec("video_min_frames", "int", 8, minimum=1, maximum=375),
    ParamSpec("min_detection_confidence", "float", 0.5, minimum=0.0, maximum=1.0),
    # -- Idle video encoder (server/avatar/video.py) --
    ParamSpec("idle_crf", "int", 18, minimum=0, maximum=51),
    ParamSpec("idle_preset", "enum", "veryfast", choices=X264_PRESETS),
    # -- portrait activation (server/avatar/animators.py) --
    # No global tunables here. The new tab "形象活化" lets the operator pick a
    # backend per generation, each of which has its own per-call knobs. Per-
    # call overrides beat a global tunable -- they make one-off experiments
    # possible without leaving fingerprints on every other avatar's build.
    # -- portrait activation (server/avatar/liveportrait.py) --
    # Default LivePortrait motion template (driver name). The "形象活化" tab
    # pre-selects this for new runs; the operator can still override per-
    # run. Choices are dynamic (whatever's in ``liveportrait_drivers/``) so
    # we don't bind the set here -- the admin UI fetches the live list and
    # validates against it.
    ParamSpec("default_liveportrait_driver", "str", "subtle"),
)


def defaults() -> dict[str, Any]:
    """All defaults as a dict; the source of truth for ``GET /api/admin/settings``."""
    return {p.name: p.default for p in TUNABLES}


def by_name(name: str) -> ParamSpec:
    for p in TUNABLES:
        if p.name == name:
            return p
    raise KeyError(name)


def validate(payload: dict[str, Any]) -> dict[str, str]:
    """Reject unknown keys and out-of-range values; return ``{name: msg}`` errors.

    Empty mapping means OK. ``payload`` is whatever the operator sent; partial
    updates are fine because each key validates independently.
    """
    errors: dict[str, str] = {}
    for name, value in payload.items():
        try:
            spec = by_name(name)
        except KeyError:
            errors[name] = "unknown tunable"
            continue
        if spec.type == "enum":
            if value not in (spec.choices or ()):
                errors[name] = f"must be one of {list(spec.choices)}"
        elif spec.type in ("int", "float"):
            try:
                v = float(value)
            except (TypeError, ValueError):
                errors[name] = "must be numeric"
                continue
            if spec.type == "int" and not v.is_integer():
                errors[name] = "must be an integer"
                continue
            if spec.minimum is not None and v < spec.minimum:
                errors[name] = f"must be >= {spec.minimum}"
                continue
            if spec.maximum is not None and v > spec.maximum:
                errors[name] = f"must be <= {spec.maximum}"
    return errors


def coerce_json(value: str) -> Any:
    """Decode a stored JSON value back to its native type."""
    return json.loads(value)


def dump_json(value: Any) -> str:
    """Encode a value for SQLite storage; preserves bool/int/float/str exactly."""
    return json.dumps(value)


def effective(db, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the *currently effective* tunable values for one DB session.

    Defaults are always present; stored overrides win on each key. The helper
    is called per pipeline invocation, so a console update is picked up on the
    very next avatar without a server restart. ``db`` is whatever the caller
    holds -- the FastAPI Depends(get_db) generator, a BackgroundTasks session,
    or a unit test's hand-rolled session.

    ``overrides`` lets one caller (typically the avatar pipeline, reading the
    per-avatar ``tunable_overrides`` column at build time) take precedence
    over both the module defaults AND the operator-pushed Setting rows.
    Layering: defaults < Setting rows < overrides. Keys absent from the
    override dict fall through to the next layer; passing ``None`` is the
    same as omitting the argument.
    """
    from sqlalchemy import select

    from server.db.models import Setting

    out = dict(defaults())
    rows = db.execute(select(Setting)).scalars().all()
    for row in rows:
        try:
            out[row.key] = coerce_json(row.value)
        except (ValueError, TypeError):
            # A malformed value falls back to the default rather than 500-ing the
            # caller; the admin can re-PATCH to fix it.
            continue
    if overrides:
        out.update(overrides)
    return out
