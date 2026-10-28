"""Admin-tunable avatar-creation knobs.

This module owns the *schema* for the parameters that affect how an
avatar is built. The list is deliberately short: every entry here has a
real, observable effect on the produced avatar. A knob that nothing reads
is worse than no knob at all -- it looks like a setting, and it is not.

Today there are two, and they both belong to the idle video encoder:
``idle_crf`` and ``idle_preset``. The create form captures them per
avatar, and they are read when the idle clip is encoded
(``dh_server/api/admin.py``).

Which digital-human model drives an avatar is NOT a tunable: each avatar
records its own model (``avatars.drive_model``), and the Models tab
loads and unloads each model's weights on its own.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

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
TUNABLES: tuple[ParamSpec, ...] = (
    # -- Idle video encoder (dh_server/avatar/video.py) --
    ParamSpec("idle_crf", "int", 18, minimum=0, maximum=51),
    ParamSpec("idle_preset", "enum", "veryfast", choices=X264_PRESETS),
)


def defaults() -> dict[str, Any]:
    """All defaults as a dict."""
    return {p.name: p.default for p in TUNABLES}


def by_name(name: str) -> ParamSpec:
    for p in TUNABLES:
        if p.name == name:
            return p
    raise KeyError(name)


def validate(payload: dict[str, Any]) -> dict[str, str]:
    """Reject unknown keys and out-of-range values; return ``{name: msg}`` errors.

    Empty mapping means OK. ``payload`` is whatever the operator sent;
    partial updates are fine because each key validates independently.
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
                continue
    return errors


def effective(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """The values that apply to one avatar.

    Defaults first, then the per-avatar overrides the create form
    captured (``avatars.tunable_overrides``). There is no global layer
    above the defaults: nothing could write one.

    ``overrides`` may be ``None`` or ``{}``, which both mean "no
    override".
    """
    out = defaults()
    if overrides:
        out.update(overrides)
    return out
