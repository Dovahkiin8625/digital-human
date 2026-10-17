"""LivePortrait portrait animator (the "形象活化" tab's local backend).

The real LivePortrait pipeline (KwaiVGI, MIT) is heavyweight: ``torch`` +
``onnxruntime`` + ``insightface`` + ≈1.5 GB of weights. We wrap it in a
:class:`LivePortraitAnimator` instance that registers itself into the
:data:`animators.ANIMATORS` registry when this module is imported.

Design contract:

* The module itself is importable **without** the heavy deps: we lazy-import
  them inside :meth:`LivePortraitAnimator._run_pipeline`. That keeps the test
  suite and the rest of the avatar pipeline from paying for models the user
  may not have installed.
* The runtime ships one bundled neutral-motion driving clip at
  ``LIVE_PORTRAIT_DRIVERS_DIR/subtle.mp4``. The fetch script
  ``scripts/fetch_liveportrait_weights.sh`` populates that directory. When
  the clip is missing, :attr:`is_ready` is False and :meth:`animate` raises
  :class:`AnimatorError` with a hint at the fetch script.
* ``promote`` does not take a textual prompt or diffusion steps -- LivePortrait
  retargets motion from the driving clip to the source face. ``meta()`` says
  so and ``validate(prompt=, steps=)`` therefore accept those knobs as
  no-ops.
* The driving clip is selectable: ``animate(..., driver="subtle")``. The
  UI lists available drivers via :func:`list_drivers` (canonical names
  populated by ``scripts/fetch_liveportrait_weights.sh`` plus anything
  the operator dropped in themselves). Each driver may live on disk as
  either ``<name>.mp4`` (raw video) or ``<name>.pkl`` (a pre-extracted
  motion template -- upstream dumps these next to the driving video at
  ``live_portrait_pipeline.py:170``). The resolver prefers ``.pkl`` when
  present so repeat runs skip crop + feature-extraction.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from server.avatar.animators import ANIMATORS, AnimatorError
from server.config import get_settings

if TYPE_CHECKING:
    from PIL import Image  # noqa: F401 -- type-only


#: Directory containing bundled driving clips. ``scripts/fetch_liveportrait_weights.sh``
#: populates this from the vendored upstream checkout at
#: ``vendor/LivePortrait-upstream/assets/examples/driving/`` (see :data:`_DRIVER_SOURCES`).
#: When empty, the animator reports ``is_ready=False`` and ``animate()`` raises
#: ``AnimatorError``.
LIVE_PORTRAIT_DRIVERS_DIR = (
    Path(get_settings().data_dir) / "liveportrait_drivers"
)

#: Upper bound on output frames. The bundled driving clips are < 200 frames,
#: but we clamp defensively so a runaway input can't allocate gigabytes of
#: tensors during inference. 1500 ≈ 60 s @ 25 fps.
_MAX_NUM_FRAMES = 1500


#: Canonical driver names → upstream source filename. Each entry maps a
#: friendly id (what the UI/API speaks) to the upstream ``assets/examples/driving/``
#: filename that ``scripts/fetch_liveportrait_weights.sh`` copies. The list
#: mixes raw mp4s and pre-extracted ``.pkl`` motion templates -- upstream
#: handles either via its ``is_template`` check at ``live_portrait_pipeline.py:102``.
#: A ``.pkl`` driver skips the crop + feature-extraction step on every run.
_DRIVER_SOURCES: dict[str, str] = {
    "subtle":   "d0.mp4",
    "motion_a": "d3.mp4",
    "motion_b": "d6.mp4",
    "motion_c": "d9.mp4",
    "motion_d": "d10.mp4",
    "motion_e": "d13.mp4",
}

#: Friendly Chinese labels for the UI. ``None`` for any canonical name
#: means "use the id itself" -- keeps the table compact for the ones
#: whose English id is already a recognisable motion verb.
_DRIVER_LABELS_ZH: dict[str, str] = {
    "subtle":   "轻微动作（默认）",
    "motion_a": "动作 A",
    "motion_b": "动作 B",
    "motion_c": "动作 C",
    "motion_d": "动作 D",
    "motion_e": "动作 E",
}

#: Stable ordering for the UI's dropdown: canonical ids first (in the
#: declared order), then any user-uploaded extras in lexical order.
_CANONICAL_DRIVER_ORDER: tuple[str, ...] = tuple(_DRIVER_SOURCES.keys())


# ----- driver resolution ---------------------------------------------------

def _driver_file(stem: str) -> tuple[str, int] | None:
    """Return ``(filename, size_bytes)`` of the effective on-disk file for a driver
    stem, or ``None`` if neither ``.pkl`` nor ``.mp4`` is present.

    Preference order: ``<stem>.pkl`` (pre-extracted template, fastest) over
    ``<stem>.mp4`` (raw video; upstream will crop + extract on first run).
    """
    drivers = LIVE_PORTRAIT_DRIVERS_DIR
    pkl = drivers / f"{stem}.pkl"
    if pkl.exists():
        return (pkl.name, pkl.stat().st_size)
    mp4 = drivers / f"{stem}.mp4"
    if mp4.exists():
        return (mp4.name, mp4.stat().st_size)
    return None


def _read_frame_count_sidecar(stem: str) -> int | None:
    """Return the cached frame count for a driver, or ``None`` if not yet known.

    The sidecar lives at ``<stem>.frames`` next to the mp4/pkl. It's a plain
    text file holding the integer frame count -- cheap to read, easy for
    ``scripts/fetch_liveportrait_weights.sh`` and the extract endpoint to
    write without needing a full VideoCapture pass on every API request.
    """
    sidecar = LIVE_PORTRAIT_DRIVERS_DIR / f"{stem}.frames"
    if not sidecar.is_file():
        return None
    try:
        return int(sidecar.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def write_frame_count_sidecar(stem: str, frames: int) -> None:
    """Persist the frame count for a driver so the API can surface it
    without re-decoding. Public so the extract endpoint and the fetch
    script can both write it.
    """
    if frames <= 0:
        return
    sidecar = LIVE_PORTRAIT_DRIVERS_DIR / f"{stem}.frames"
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(str(int(frames)), encoding="utf-8")


def list_drivers() -> list[dict]:
    """Every driver present under :data:`LIVE_PORTRAIT_DRIVERS_DIR`.

    Each entry: ``{name, filename, kind, size_bytes, cached, canonical,
    label_zh, frames}``. ``kind`` is ``"pkl"`` or ``"mp4"`` -- whichever the
    resolver would use. ``cached`` is ``True`` iff a ``.pkl`` exists for
    this stem (so an ``mp4``-kind entry that has a companion ``.pkl`` is
    "extracted"). ``canonical`` is ``True`` for the names in
    :data:`_DRIVER_SOURCES`; user-uploaded extras are listed too so the
    UI can offer them without forking the registry. ``frames`` is the
    template's frame count (``None`` until either the ``.pkl`` is read
    or the ``.frames`` sidecar is populated by the fetch script / extract
    endpoint) -- the UI uses it to pre-fill the "动起来帧数" input so
    the operator doesn't have to know the magic number.
    """
    drivers = LIVE_PORTRAIT_DRIVERS_DIR
    if not drivers.is_dir():
        return []
    stems = {
        p.stem
        for p in drivers.iterdir()
        if p.suffix in (".mp4", ".pkl") and p.is_file()
    }
    out: list[dict] = []
    canonical_set = set(_DRIVER_SOURCES)
    for stem in sorted(stems):
        eff = _driver_file(stem)
        if eff is None:
            continue
        filename, size_bytes = eff
        pkl_path = drivers / f"{stem}.pkl"
        # Frame count: prefer the .pkl (always current), then the sidecar
        # (cheap), then None if we don't know yet. The UI is honest about
        # the "we don't know" case -- placeholder reads "运行后自动填入".
        frames: int | None = None
        if pkl_path.exists():
            import pickle

            try:
                with open(pkl_path, "rb") as f:
                    frames = int(pickle.load(f).get("n_frames", 0)) or None
            except (OSError, ValueError, KeyError, TypeError):
                frames = None
        if frames is None:
            frames = _read_frame_count_sidecar(stem)
        gif_path = drivers / f"{stem}.gif"
        out.append(
            {
                "name": stem,
                "filename": filename,
                "kind": "pkl" if filename.endswith(".pkl") else "mp4",
                "size_bytes": size_bytes,
                "cached": pkl_path.exists(),
                "canonical": stem in canonical_set,
                "label_zh": _DRIVER_LABELS_ZH.get(stem, stem),
                "frames": frames,
                "has_preview": gif_path.is_file(),
            }
        )
    # Canonical ids come first in declared order; the rest are appended
    # alphabetically so the dropdown has a stable, predictable layout.
    out.sort(
        key=lambda d: (
            0 if d["name"] in _CANONICAL_DRIVER_ORDER else 1,
            _CANONICAL_DRIVER_ORDER.index(d["name"])
            if d["name"] in _CANONICAL_DRIVER_ORDER
            else d["name"],
        )
    )
    return out


def resolve_driver(name: str) -> Path:
    """Return the on-disk file for a driver name. Raises :class:`AnimatorError`
    if the driver is missing -- the message includes the available list so
    the API can surface it as a 422 field-level error.
    """
    drivers = LIVE_PORTRAIT_DRIVERS_DIR
    pkl = drivers / f"{name}.pkl"
    if pkl.exists():
        return pkl
    mp4 = drivers / f"{name}.mp4"
    if mp4.exists():
        return mp4
    available = ", ".join(d["name"] for d in list_drivers()) or "(none)"
    raise AnimatorError(
        f"driver {name!r} not found under {drivers}; available: {available}"
    )


class LivePortraitAnimator:
    """Portrait retargeting via LivePortrait (KwaiVGI, MIT)."""

    name = "liveportrait"

    # ----- introspection (used by /api/admin/backends) -----

    def meta(self) -> dict:
        from server.avatar.vram import get_vram_gb

        free_gb = get_vram_gb()
        needed = 6
        return {
            "label_zh": "LivePortrait",
            "group_zh": "本地（无需联网）",
            "description": (
                "本地运行的人像动作迁移模型；内置轻微动作驱动视频，输出约 5s 头像动态视频"
            ),
            "vram_hint": f"~{needed} GB",
            "vram_required_gb": needed,
            # ``None`` when ``torch`` isn't installed or the host has no
            # CUDA device -- the dispatcher will surface a clearer error
            # than a silent "off" would.
            "fits_current_gpu": None if free_gb is None else free_gb >= needed,
            # LivePortrait doesn't use a text prompt -- the driving clip
            # already carries the motion. ``steps`` is meaningless to it.
            "accepts_prompt": False,
            "accepts_num_frames": True,
            "accepts_steps": False,
            # Bound the ``num_frames`` input on the form. The bundled driver
            # is ≈125 frames; we let the form go a bit higher so an operator
            # can loop the motion deliberately.
            "num_frames_min": 1,
            "num_frames_max": _MAX_NUM_FRAMES,
        }

    # ----- readiness -----

    @property
    def is_ready(self) -> bool:
        # ``LIVE_PORTRAIT_DRIVERS_DIR`` is the source of truth: every fetched
        # driver clip lives here. ``subtle`` is the bundled default -- a real
        # installation would add more (``smile``, ``look_around``); for v1
        # we ship the one that already feels "subtle and alive".
        drivers = self._drivers_dir()
        if not (drivers.is_dir() and any(drivers.iterdir())):
            return False
        # The pipeline itself is heavy; trust the env's torch stack to be
        # importable on real hosts. Don't actually ``import LivePortrait``
        # here -- it triggers onnxruntime init that takes seconds for
        # nothing -- defer to ``animate()``.
        try:
            import torch  # noqa: F401
        except ImportError:
            return False
        return True

    # ----- validation (called by the dispatcher) -----

    def validate(self, *, prompt: str, num_frames: int, steps: int) -> dict[str, str]:
        # ``prompt`` and ``steps`` are intentionally ignored -- the per-call
        # Web form may still send them after a backend switch, and the
        # dispatcher can't know they're irrelevant to LivePortrait without
        # asking each backend. Returning {} here is the contract for "this
        # knob is unused, accept whatever the page sent".
        del prompt, steps
        if num_frames < 1:
            return {"num_frames": "must be >= 1"}
        if num_frames > _MAX_NUM_FRAMES:
            return {"num_frames": f"must be <= {_MAX_NUM_FRAMES}"}
        return {}

    # ----- inference -----

    def animate(
        self,
        source_path: Path,
        *,
        prompt: str,
        num_frames: int,
        steps: int,
        driver: str = "subtle",
        out_dir: Path,
    ) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        drivers = self._drivers_dir()
        if not (drivers.is_dir() and any(drivers.iterdir())):
            raise AnimatorError(
                f"no driving clip under {drivers}; run scripts/fetch_liveportrait_weights.sh"
            )
        # Resolve the driver name to an on-disk file. ``resolve_driver`` raises
        # ``AnimatorError`` with the available list if the name is unknown,
        # so the API can surface a clean field-level 422 instead of a generic
        # failure. ``.pkl`` is preferred over ``.mp4`` -- upstream's
        # ``is_template`` branch skips the slow crop + feature-extraction.
        driver_clip = resolve_driver(driver)
        # Clamp to driver length. For ``.pkl`` we read the frame count from
        # the pickled dict; for ``.mp4`` we decode and count. This stays
        # light (no full decode -- just enough to know the length).
        n_driver = self._driver_frame_count(driver_clip)
        # ``num_frames <= 0`` means "use the template's full duration" -- the
        # user's setting is an *upper bound* (they can shorten by entering
        # a smaller value), but the template itself is the source of truth
        # for the longest possible output. This matches the UI placeholder
        # "使用模板时长" and prevents the surprise where the default 49-frame
        # value silently truncates a 78-frame template to 49.
        if int(num_frames) <= 0:
            effective = n_driver
        else:
            effective = min(int(num_frames), n_driver)

        try:
            frames = self._run_pipeline(
                source_path,
                driver_frames=None,
                num_frames=effective,
                driver=driver,
                out_dir=out_dir,
            )
        except AnimatorError:
            raise
        except Exception as exc:  # noqa: BLE001 -- surface verbatim to the operator
            raise AnimatorError(f"LivePortrait inference failed: {exc}")

        # ``_run_pipeline`` returns RGB uint8 at the project canvas (720x1280);
        # encode via the shared PyAV helper so the output plays back in <video>.
        from server.avatar.video import _encode_h264
        from server.config import get_settings as _gs

        settings = _gs()
        rgb_stack = np.ascontiguousarray(frames)
        # Squeeze a leading batch dimension if the pipeline returned one.
        if rgb_stack.ndim == 5 and rgb_stack.shape[0] == 1:
            rgb_stack = rgb_stack[0]
        _encode_h264(out_dir / "out.mp4", rgb_stack, fps=settings.fps)

    # ----- pipeline injection point (overridden by tests) -----

    def _run_pipeline(
        self,
        source_path: Path,
        driver_frames: list[np.ndarray] | None,
        *,
        num_frames: int,
        driver: str = "subtle",
        out_dir: Path,
    ) -> np.ndarray:
        """Run the real LivePortrait inference.

        Upstream's actual API (verified against KwaiVGI/LivePortrait @ the
        SHA pinned in ``scripts/setup_liveportrait.sh``):

        * ``LivePortraitPipeline.__init__`` takes
          ``InferenceConfig`` and ``CropConfig`` dataclasses.
        * The work is done by ``pipe.execute(args: ArgumentConfig)``,
          which takes FILE PATHS for ``source``/``driving``/``output_dir``
          (not pre-decoded frame arrays), writes the result to
          ``args.output_dir``, and returns ``(wfp, wfp_concat)``.
        * ``driver_frames`` is kept in the signature for compatibility with
          tests that still pass it; we ignore it here because upstream
          re-reads the driver from disk (it does its own
          ``cv2.VideoCapture`` or ``pickle.load`` on ``args.driving``).
        * The driver name is resolved via :func:`resolve_driver` -- it
          prefers ``<name>.pkl`` (pre-extracted template, fastest) over
          ``<name>.mp4`` (raw video; upstream will crop + extract).
        * Upstream's default config paths point at the vendored
          ``vendor/LivePortrait-upstream/pretrained_weights/...``; we
          override them so the project's managed tree
          (``data/models/liveportrait/...``) is the source of truth.

        Tests stub this method via ``monkeypatch`` so CI does not need any
        of it; the wrapper still encodes the returned ndarray to mp4 via
        :func:`server.avatar.video._encode_h264`.
        """
        try:
            from LivePortrait.live_portrait_pipeline import LivePortraitPipeline
            from src.config.argument_config import ArgumentConfig  # type: ignore[import-not-found]
            from src.config.inference_config import InferenceConfig  # type: ignore[import-not-found]
            from src.config.crop_config import CropConfig  # type: ignore[import-not-found]
        except ImportError as exc:
            # ``LivePortrait`` is the in-tree shim re-exporting the upstream
            # KwaiVGI/LivePortrait repo. The shim populates
            # ``sys.modules['LivePortrait.live_portrait_pipeline']`` so the
            # plain import works; this ImportError covers "shim not installed"
            # or "shim installed but upstream's heavy deps missing" -- both
            # route through the same setup script.
            raise AnimatorError(
                f"the LivePortrait runtime is not installed ({exc}); "
                f"run scripts/setup_liveportrait.sh, then scripts/fetch_liveportrait_weights.sh"
            )

        from server.config import get_settings

        settings = get_settings()
        # Project-managed weight tree, populated by ``fetch_liveportrait_weights.sh``.
        # Upstream's InferenceConfig defaults expect the same file layout but
        # under ``vendor/LivePortrait-upstream/pretrained_weights/...``; we
        # override so a fresh ``scripts/setup_liveportrait.sh`` run that
        # forgets to fetch weights still fails fast on a missing file rather
        # than silently using whatever defaults ``make_abs_path`` resolves.
        weights = Path(settings.data_dir) / "models" / "liveportrait"
        inference_cfg = InferenceConfig(
            checkpoint_F=str(weights / "liveportrait" / "base_models" / "appearance_feature_extractor.pth"),
            checkpoint_M=str(weights / "liveportrait" / "base_models" / "motion_extractor.pth"),
            checkpoint_G=str(weights / "liveportrait" / "base_models" / "spade_generator.pth"),
            checkpoint_W=str(weights / "liveportrait" / "base_models" / "warping_module.pth"),
            checkpoint_S=str(weights / "liveportrait" / "retargeting_models" / "stitching_retargeting_module.pth"),
        )
        crop_cfg = CropConfig(
            insightface_root=str(weights / "insightface"),
            landmark_ckpt_path=str(weights / "liveportrait" / "landmark.onnx"),
        )

        # Resolve driver name -> file (pre-extracted .pkl preferred over .mp4).
        # ``resolve_driver`` already raises AnimatorError with the available
        # list if the name is unknown, so we don't need a separate existence
        # check here.
        driver_clip = resolve_driver(driver)

        pipe = LivePortraitPipeline(inference_cfg=inference_cfg, crop_cfg=crop_cfg)
        # ``ArgumentConfig`` defaults use ``make_abs_path`` which joins
        # against the upstream config dir. Passing absolute paths makes the
        # join a no-op (``os.path.join`` discards earlier segments when a
        # later one is absolute) so the project paths win.
        args = ArgumentConfig(
            source=str(source_path.absolute()),
            driving=str(driver_clip.absolute()),
            output_dir=str(out_dir.absolute()),
        )

        wfp, _wfp_concat = pipe.execute(args)

        # Upstream returns an mp4 file path; the surrounding wrapper expects
        # an ndarray so it can h264-encode. Decode back to the project canvas.
        from server.avatar.video import decode_frames

        arr, _src_fps = decode_frames(
            Path(wfp),
            target_fps=settings.fps,
            max_frames=num_frames,
            canvas=(settings.frame_width, settings.frame_height),
        )
        return np.ascontiguousarray(arr)

    # ----- helpers -----

    @staticmethod
    def _drivers_dir() -> Path:
        # ``LIVE_PORTRAIT_DRIVERS_DIR`` is set at import time from
        # ``get_settings().data_dir``; the test fixture monkey-patches this
        # to point at ``tmp_path`` so it can drop a fake clip there.
        return LIVE_PORTRAIT_DRIVERS_DIR

    @staticmethod
    def _driver_frame_count(clip: Path) -> int:
        """Return the frame count for a driver clip without decoding the
        full RGB payload.

        For ``.pkl`` (pre-extracted motion template) we read the
        ``n_frames`` key the pipeline wrote on extraction -- a single
        ``pickle.load`` is far cheaper than a ``cv2.VideoCapture`` pass.
        For ``.mp4`` we still need ``cv2`` because there is no
        header-level frame count we can trust without a probe.
        """
        if clip.suffix == ".pkl":
            import pickle

            with open(clip, "rb") as f:
                return int(pickle.load(f)["n_frames"])
        from server.avatar.video import decode_frames

        arr, _ = decode_frames(
            clip,
            target_fps=25,
            max_frames=_MAX_NUM_FRAMES,
            canvas=(720, 1280),
        )
        return len(arr)


# Side-effect: install this backend into the registry as soon as the
# module is imported. The admin endpoint and any ``/api/admin/backends``
# call hits ``ANIMATORS`` lazily, so a missing registration is observable
# to tests (and to operators who forgot the install step).
ANIMATORS[LivePortraitAnimator.name] = LivePortraitAnimator()
