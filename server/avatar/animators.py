"""Portrait animator registry for the "形象活化" tab.

A portrait animator takes a static portrait photo, possibly a textual prompt,
and a few generation knobs, and writes an mp4 file ``<out_dir>/out.mp4``. Each
backend (LivePortrait / Wan 2.2 I2V variants / MuseTalk silent / ...) keeps its
own constructor + dependencies + per-knob constraints; the registry gives the
admin endpoint and the UI a single dispatch point that hides those details.

The protocol deliberately keeps the parameter set fixed (prompt, num_frames,
steps). A backend that doesn't care about a knob returns an empty validation
error for it -- so the UI can show all three inputs unconditionally and only
the relevant ones actually fire when the user clicks 生成.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol


class AnimatorError(Exception):
    """Raised by an animator when it cannot complete (missing weights, OOM, ...).

    The admin endpoint catches it and turns it into a 502. Backends should
    raise rather than return -- that way the page always sees an empty body
    in the success case.
    """


class PortraitAnimator(Protocol):
    """Anything that turns a photo + knobs into ``<out_dir>/out.mp4``."""

    #: Stable identifier surfaced to the UI and the form payload.
    name: str

    def meta(self) -> dict:
        """Surface to the ``/api/admin/backends`` listing.

        Required keys (matching ``admin-ui/src/pages/ActivatePortrait.tsx``):
        ``label_zh``, ``group_zh``, ``description``, ``vram_hint``,
        ``accepts_prompt``, ``accepts_num_frames``, ``accepts_steps``.
        Optional numeric bounds: ``num_frames_min``, ``num_frames_max``,
        ``steps_min``, ``steps_max``.
        """
        ...

    #: True when the backend's weights + runtime are usable. Calling
    #: ``animate()`` when False must raise ``AnimatorError`` with a hint at
    #: which fetch script to run, mirroring the legacy behaviour.
    @property
    def is_ready(self) -> bool: ...

    def validate(self, *, prompt: str, num_frames: int, steps: int) -> dict[str, str]:
        """Return ``{field: msg}`` for invalid knobs; empty dict if all good."""
        ...

    def animate(
        self,
        source_path: Path,
        *,
        prompt: str,
        num_frames: int,
        steps: int,
        out_dir: Path,
    ) -> None:
        """Render ``<out_dir>/out.mp4``. Must raise ``AnimatorError`` on failure."""
        ...


#: The registry. Populated by side-effect imports in each backend module:
#:
#:     from server.avatar.animators import ANIMATORS
#:     from server.avatar.liveportrait import LivePortraitAnimator
#:     ANIMATORS[LivePortraitAnimator.name] = LivePortraitAnimator()
#:
#: The endpoint imports this dict at request time; ordering is insertion
#: order, so backends should appear grouped (local / oss / infrastructural).
ANIMATORS: dict[str, PortraitAnimator] = {}
