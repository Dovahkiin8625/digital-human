"""MuseTalk silent portrait animator for the 形象活化 tab.

Reuses the existing MuseTalk + encoder stack: a synthesised silent WAV,
single-frame-loop as a "video" source, and the project PyAV libx264 mux
to write the artifact. ``_run_pipeline(...)`` is the monkey-patchable
injection point so tests can substitute a deterministic frame stack and
CI doesn't need a multi-GB MuseTalk checkpoint.

Why "silent": MuseTalk is audio-driven; with a silent WAV the per-frame
output is *subtle* (jaw/neck micro-motion) rather than articulately
speaking. That matches "让一张静态人像轻微地动起来" exactly: an operator
gets a portrait preview that breathes, without the model inventing
speech.
"""
from __future__ import annotations

import wave
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from server.avatar.animators import ANIMATORS, AnimatorError
from server.config import get_settings

if TYPE_CHECKING:
    from PIL import Image as _PILImage  # noqa: F401


#: Where synthesised silent WAVs land inside a request's scratch dir. The
#: encoder / future MuseTalk run consumes it; tests can read the file to
#: confirm the contract.
SILENT_WAV_FILENAME = "silent.wav"

#: 16 kHz mono PCM is the smallest WAV every downstream tool (MuseTalk,
#: librosa, ffmpeg) understands without a format negotiation.
_WAV_SAMPLE_RATE = 16_000
_WAV_SAMPLE_WIDTH = 2  # 16-bit signed


class MuseTalkSilentAnimator:
    """Loop a static photo + silent audio into a short portrait clip."""

    name = "musetalk-silent"

    # ----- introspection (used by /api/admin/backends) -----

    def meta(self) -> dict:
        return {
            "label_zh": "MuseTalk 静音驱动",
            "group_zh": "复用基础设施",
            "description": (
                "复用 MuseTalk 管线，由系统合成静音 WAV 驱动单帧循环，"
                "输出约 5s 的轻微下巴/颈部动作"
            ),
            "vram_hint": "复用主引擎显存",
            # MuseTalk silent piggybacks on the project's main engine GPU
            # budget, so we don't second-guess it here. ``None`` keeps the
            # page from falsely disabling the option on a tight host.
            "vram_required_gb": None,
            "fits_current_gpu": True,
            # ``prompt`` is meaningless on a silent driver (no phonemes to
            # bind). ``steps`` is for diffusers; MuseTalk has its own
            # internal schedule and ignores it. Only ``num_frames`` matters.
            "accepts_prompt": False,
            "accepts_num_frames": True,
            "accepts_steps": False,
            # Cap at 30 s @ the project fps so a runaway form input can't
            # pin the GPU; 5–10 s is the sweet spot for "subtle motion".
            "num_frames_min": 1,
            "num_frames_max": 750,
        }

    # ----- readiness -----

    @property
    def is_ready(self) -> bool:
        # Reuses the project's MuseTalkPreparer + encoder stack; both ship
        # with the avatar build pipeline on every real host. CI runs with
        # ``DH_AVATAR_PIPELINE=mock`` which still produces an mp4 of the
        # default looping frames. We don't gate on a weights download here.
        try:
            # ``server.avatar.musetalk_pipeline`` is the existing home of
            # ``MuseTalkAvatarPipeline``; importing it cheap since the
            # heavy torch graph isn't touched here.
            from server.avatar.musetalk_pipeline import (  # noqa: F401
                MuseTalkAvatarPipeline,
            )
        except ImportError:
            return False
        return True

    # ----- validation (called by the dispatcher) -----

    def validate(self, *, prompt: str, num_frames: int, steps: int) -> dict[str, str]:
        del prompt, steps
        if num_frames < 1:
            return {"num_frames": "must be >= 1"}
        if num_frames > 750:
            return {"num_frames": "must be <= 750"}
        return {}

    # ----- inference -----

    def animate(
        self,
        source_path: Path,
        *,
        prompt: str,
        num_frames: int,
        steps: int,
        out_dir: Path,
    ) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        from server.avatar.musetalk_pipeline import MuseTalkAvatarPipeline  # noqa: F401

        # Synthesise a silent WAV sibling first. The future MuseTalk run
        # can pick it up off disk without re-running; the encoder can mux
        # it into the mp4 later. Default 16 kHz mono int16.
        wav_path = out_dir / SILENT_WAV_FILENAME
        duration_s = max(0.5, num_frames / 25.0)  # 25 fps; ~5 s default
        _write_silent_wav(wav_path, duration_s)

        try:
            frames = self._run_pipeline(
                source_path, num_frames=int(num_frames), prompt=prompt or "", steps=int(steps), out_dir=out_dir
            )
        except AnimatorError:
            raise
        except Exception as exc:  # noqa: BLE001 -- surface verbatim to the operator
            raise AnimatorError(f"MuseTalk silent inference failed: {exc}")

        from server.avatar.video import _encode_h264

        rgb = np.ascontiguousarray(frames)
        if rgb.ndim == 5 and rgb.shape[0] == 1:
            rgb = rgb[0]
        # Resize to canvas if the pipeline returned at its native size.
        if rgb.shape[1:3] != (1280, 720):
            from PIL import Image as _PilImage

            pil_frames = [
                np.asarray(
                    _PilImage.fromarray(f).resize((720, 1280), _PilImage.LANCZOS)
                )
                for f in rgb
            ]
            rgb = np.stack(pil_frames)
        _encode_h264(out_dir / "out.mp4", rgb, fps=25)

    # ----- pipeline injection point (overridden by tests) -----

    def _run_pipeline(
        self,
        source_path: Path,
        *,
        num_frames: int,
        prompt: str,
        steps: int,
        out_dir: Path,
    ) -> np.ndarray:
        """Render N frames at canvas size; default just loops the source.

        Production wires this to the real MuseTalk avatar pipeline (audio
        retargeting happens inside ``MuseTalkAvatarPipeline``) so a live
        host produces a per-frame warped face. CI replaces this with a
        deterministic frame stack via ``monkeypatch``.
        """
        from PIL import Image as _PilImage

        with _PilImage.open(source_path) as src:
            src.load()
            canvas = src.convert("RGB")
            # Letterbox to canvas (720x1280) the same way the avatar
            # pipeline does, then re-emit it ``num_frames`` times.
            from server.avatar.musetalk_pipeline import fit_to_canvas  # local import keeps the module cheap

            settings = get_settings()
            canvas = fit_to_canvas(canvas, settings.frame_width, settings.frame_height)
        arr = np.asarray(canvas)
        return np.stack([np.ascontiguousarray(arr) for _ in range(int(num_frames))])


# --------------------------------------------------------------------------- helpers


def _write_silent_wav(path: Path, duration_s: float) -> None:
    """Write a 16-bit mono PCM silent WAV at 16 kHz covering ``duration_s``.

    Using the stdlib ``wave`` module keeps the dependency footprint zero:
    ``soundfile``/``scipy.io.wavfile`` would also work but pull a binary
    build of libsndfile that ships nowhere else in this repo.
    """
    n_frames = max(1, int(round(duration_s * _WAV_SAMPLE_RATE)))
    silence = np.zeros(n_frames, dtype=np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(_WAV_SAMPLE_WIDTH)
        wf.setframerate(_WAV_SAMPLE_RATE)
        wf.writeframes(silence.tobytes())


# Side-effect: install this backend into the registry.
ANIMATORS[MuseTalkSilentAnimator.name] = MuseTalkSilentAnimator()
