"""Wan 2.2 I2V portrait animator (the "形象活化" tab's OSS-i2v slot).

A single 5 B dense checkpoint (``Wan-AI/Wan2.2-TI2V-5B-Diffusers``) drives
the OSS I2V backend, registered as ``wan22-i2v-5b-bf16`` in
:data:`server.avatar.animators.ANIMATORS`.

Wan-AI did not release an officially-quantised FP8 variant of this
checkpoint on either Hugging Face or ModelScope, so the fp8 backend that
was originally planned alongside bf16 is intentionally absent: shipping a
Comfy-Org or bullerwins community quant would force a non-diffusers loader
and break the rest of the pipeline. Re-evaluate when Wan-AI ships an
official FP8 release (or when diffusers gains runtime bf16->fp8
quantisation).

The model is a dense 5 B hybrid that diffusers exposes via
``WanImageToVideoPipeline``. Same 16N+1 motion grid as Wan 2.1, so the
form's InputNumber constraints carry over verbatim. New knobs vs Wan 2.1:

* ``negative_prompt`` (optional, empty here)
* ``guidance_scale`` (constant 5.0; not operator-tunable yet)
* explicit ``height`` / ``width`` (720P @ 24 fps is the sweet spot)

The diffusers import is deferred to first call so the test suite (and any
host that has only LivePortrait set up) doesn't pull a multi-GB torch
graph into the import graph.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from server.avatar.animators import ANIMATORS, AnimatorError
from server.config import get_settings

if TYPE_CHECKING:
    from PIL import Image as _PILImage  # noqa: F401


#: Root directory under which the weights live.
#: ``scripts/fetch_wan22_weights.sh`` populates this from the ModelScope
#: mirror of ``Wan-AI/Wan2.2-TI2V-5B-Diffusers``.
WAN22_WEIGHTS_ROOT = Path(get_settings().data_dir) / "models"


#: Single-variant spec table. Adding another Wan 2.2 release (e.g. an FP8
#: variant once Wan-AI ships one) is one row + an ``ANIMATORS[...]``
#: registration in :func:`all_animators`.
#:
#: ``vram_min_gb`` is a coarse estimate from the community model cards; we
#: reject hosts under it rather than let diffusers OOM mid-frame. Off-by-1 GB
#: triggers a clean hint, OOM at frame 3 is a confusing crash.
#:
#: ``torch_dtype`` and ``guidance_scale`` are kept here for symmetry with
#: future variants; today they're the only one of their kind so the spec
#: effectively pins them.
VARIANTS: dict[str, dict] = {
    "5b-bf16": {
        "weights_subdir": "Wan2.2-TI2V-5B-Diffusers",
        "label_zh": "Wan 2.2 I2V 5B (bf16)",
        "vram_hint": "~12 GB",
        # Peak VRAM measured on RTX 5070 Ti 16 GB with sequential_cpu_offload
        # at 704x1280 / 20 steps: 10.04 GB allocated. Threshold = peak + 2 GB
        # headroom so a 16 GB consumer card qualifies.
        "vram_min_gb": 12,
        "torch_dtype": "bfloat16",
        "guidance_scale": 5.0,
        "num_frames_min": 17,
        # 720P @ 24 fps: 113 frames = 4.71 s. Same on the 16N+1 grid as
        # Wan 2.1's 81 (which at 16 fps was ~5.06 s), so the form UX stays
        # uniform even though the underlying fps shifts.
        "num_frames_max": 113,
    },
}

#: Visible order in the dropdown. Just one variant today, kept as a tuple
#: so adding a sibling later is a one-line change.
_VARIANT_ORDER = ("5b-bf16",)


class Wan22I2VAnimator:
    """A Wan 2.2 I2V animator.

    Instances are constructed by :func:`all_animators`; tests instantiate
    a specific variant directly to scope fixtures (e.g. by writing weights
    for one variant only).
    """

    #: Map of internal variant id (used in ``VARIANTS``) to spec dict.
    #: Re-exported under ``VARIANTS`` so tests can request a specific
    #: variant without reaching into the private dict.
    VARIANTS = VARIANTS

    def __init__(self, *, variant: str, weights_root: Path | None = None):
        if variant not in VARIANTS:
            raise ValueError(f"unknown wan22 variant: {variant}")
        self.variant = variant
        self.spec = VARIANTS[variant]
        self._weights_root = (
            Path(weights_root) if weights_root else WAN22_WEIGHTS_ROOT
        )

    # ----- identity -----

    @property
    def name(self) -> str:
        return f"wan22-i2v-{self.variant}"

    @property
    def weights_dir(self) -> Path:
        return self._weights_root / self.spec["weights_subdir"]

    # ----- meta (used by /api/admin/backends) -----

    def meta(self) -> dict:
        from server.avatar.vram import get_vram_gb

        s = self.spec
        needed = int(s["vram_min_gb"])
        free_gb = get_vram_gb()
        return {
            "label_zh": s["label_zh"],
            "group_zh": "开源参考生视频",
            "description": (
                f"开源 Wan 2.2 I2V 视频生成，模型权重 {s['weights_subdir']}；"
                "生成效果强但与本机人像相关性低于 LivePortrait"
            ),
            "vram_hint": s["vram_hint"],
            "vram_required_gb": needed,
            "fits_current_gpu": None if free_gb is None else free_gb >= needed,
            "accepts_prompt": True,
            "accepts_num_frames": True,
            "accepts_steps": True,
            "num_frames_min": s["num_frames_min"],
            "num_frames_max": s["num_frames_max"],
            "steps_min": 10,
            "steps_max": 50,
        }

    # ----- readiness -----

    @property
    def is_ready(self) -> bool:
        # The diffusers Wan 2.2 release demands a CUDA host; tests stub
        # ``torch.cuda`` to simulate VRAM budgets. The constructor's
        # preflight below is the authoritative check the dispatcher uses
        # when ``animate()`` lands here.
        if not (self.weights_dir.is_dir() and any(self.weights_dir.iterdir())):
            return False
        try:
            import torch
        except ImportError:
            return False
        if not torch.cuda.is_available():
            return False
        try:
            free_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        except Exception:  # noqa: BLE001 -- cuda init failure must not crash readiness
            return False
        return free_gb >= self.spec["vram_min_gb"]

    # ----- validation -----

    def validate(self, *, prompt: str, num_frames: int, steps: int) -> dict[str, str]:
        errs: dict[str, str] = {}
        if num_frames < 1 or (num_frames - 1) % 16 != 0:
            errs["num_frames"] = "must be 16N+1 (e.g. 17, 33, 49, 65, 81, 97, 113)"
        if steps < 10 or steps > 50:
            errs["steps"] = "must be between 10 and 50"
        # ``prompt`` is optional: Wan supports empty strings. No constraint.
        del prompt
        return errs

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
        # Validate first so a bad input fails *before* any weights read or
        # GPU probe. The dispatcher already calls ``validate``; this is a
        # belt-and-braces check for direct callers (tests + future paths).
        field_errs = self.validate(prompt=prompt, num_frames=num_frames, steps=steps)
        if field_errs:
            raise AnimatorError(
                "wan22 rejected one or more inputs: " + "; ".join(
                    f"{k}={v}" for k, v in field_errs.items()
                )
            )

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if not (self.weights_dir.is_dir() and any(self.weights_dir.iterdir())):
            raise AnimatorError(
                f"wan22 weights missing under {self.weights_dir}; "
                f"run scripts/fetch_wan22_weights.sh"
            )

        # Defensive VRAM recheck: a host can technically pass ``is_ready``
        # once, then have other workloads eat the GPU. A second check here
        # surfaces the failure cleanly instead of mid-inference OOM.
        try:
            import torch

            free_gb = (
                torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
                if torch.cuda.is_available() else 0
            )
        except Exception:
            free_gb = 0
        if free_gb < self.spec["vram_min_gb"]:
            raise AnimatorError(
                f"wan22 {self.spec['label_zh']} needs ~{self.spec['vram_min_gb']} GB VRAM; "
                f"this host has {free_gb:.1f} GB"
            )

        # Side-channel: snap to grid so a slightly off input doesn't crash diffusers.
        if (num_frames - 1) % 16 != 0:
            rounded = ((num_frames - 1) // 16) * 16 + 1
            num_frames = max(self.spec["num_frames_min"], rounded)

        try:
            frames = self._run_pipeline(
                source_path, prompt=prompt, num_frames=num_frames, steps=steps, out_dir=out_dir
            )
        except AnimatorError:
            raise
        except Exception as exc:  # noqa: BLE001 -- surface verbatim to the operator
            raise AnimatorError(f"Wan 2.2 inference failed: {exc}")

        from server.avatar.video import _encode_h264

        # ``_run_pipeline`` returns RGB uint8 at canvas size (1280x720 Wan I2V
        # default; we resize below if it doesn't match the project canvas).
        rgb = np.ascontiguousarray(frames)
        if rgb.ndim == 5 and rgb.shape[0] == 1:
            rgb = rgb[0]
        if rgb.shape[1:3] != (1280, 720):
            from PIL import Image as _PilImage

            pil_frames = [
                np.asarray(
                    _PilImage.fromarray(f).resize((720, 1280), _PilImage.LANCZOS)
                )
                for f in rgb
            ]
            rgb = np.stack(pil_frames)
        _encode_h264(out_dir / "out.mp4", rgb, fps=16)

    # ----- pipeline injection point (overridden by tests) -----

    def _run_pipeline(
        self,
        source_path: Path,
        *,
        prompt: str,
        num_frames: int,
        steps: int,
        out_dir: Path,
    ) -> np.ndarray:
        """Run the real Wan 2.2 I2V pipeline.

        Lazy-imports the diffusers stack so CI doesn't pay for torch's
        2 GB init graph when only LivePortrait is configured.
        """
        try:
            import torch
            from diffusers import WanImageToVideoPipeline
        except ImportError as exc:
            raise AnimatorError(
                f"diffusers + torch not installed ({exc}); pip install diffusers torch"
            )

        dtype_name = self.spec["torch_dtype"]
        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float8_e4m3fn": torch.float8_e4m3fn,
        }.get(dtype_name)
        if torch_dtype is None:
            # Defensive: an unknown dtype string would crash diffusers with a
            # confusing AttributeError deep inside the loader.
            raise AnimatorError(
                f"wan22 variant {self.variant!r} has unknown torch_dtype {dtype_name!r}"
            )

        try:
            pipe = WanImageToVideoPipeline.from_pretrained(
                str(self.weights_dir),
                torch_dtype=torch_dtype,
            )
        except (OSError, ValueError) as exc:
            raise AnimatorError(
                f"could not load Wan 2.2 weights from {self.weights_dir}: {exc}"
            )

        # Sequential CPU offload keeps peak GPU memory under control on
        # consumer cards. The 5 B bf16 model fits a 24 GB card natively; on
        # smaller hosts the offload path slides it into 16 GB at the cost
        # of inference wall time. If we add an FP8 variant later this is
        # where it gets the additional VRAM headroom.
        try:
            pipe.enable_sequential_cpu_offload()
        except (AttributeError, ValueError):
            pipe.to("cuda" if torch.cuda.is_available() else "cpu")

        from PIL import Image as _PilImage

        with _PilImage.open(source_path) as src:
            src.load()
            output = pipe(
                image=src.convert("RGB"),
                prompt=prompt or "",
                negative_prompt="",
                height=704,  # 720P minus a 32-multiple snap; diffusers warns + adjusts otherwise
                width=1280,
                num_inference_steps=int(steps),
                num_frames=int(num_frames),
                guidance_scale=float(self.spec["guidance_scale"]),
                generator=torch.Generator(device="cpu").manual_seed(0),
            ).frames[0]
        del pipe  # release GPU memory before re-encode

        # Return as RGB uint8 stack; ``animate()`` handles resize + encode.
        # diffusers' Wan I2V pipeline returns float32 [0, 1] arrays when
        # output_type="np" (the default); PIL.fromarray refuses float, so we
        # quantise to uint8 here. Tests stub ``_run_pipeline`` to return
        # uint8 directly, so the contract stays consistent.
        frames = [
            np.ascontiguousarray((np.asarray(f) * 255).clip(0, 255).astype(np.uint8))
            for f in output
        ]
        return np.stack(frames)


def all_animators() -> list[Wan22I2VAnimator]:
    """Return one animator per variant, in the order the dropdown should show them."""
    return [Wan22I2VAnimator(variant=v) for v in _VARIANT_ORDER]


# Side-effect: populate the registry. The admin endpoint imports this module
# before serving ``/api/admin/backends``, so the page sees the variant.
for _a in all_animators():
    ANIMATORS[_a.name] = _a
