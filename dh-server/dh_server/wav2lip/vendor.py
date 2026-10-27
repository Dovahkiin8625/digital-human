"""Where the Wav2Lip checkout and its weights live.

The upstream is a git checkout of `Rudrabha/Wav2Lip` at a pinned commit,
placed in `vendor/Wav2Lip` by `scripts/setup_wav2lip.sh`. It is not part of
this repository's tree: `vendor/` is gitignored, exactly like `vendor/MuseTalk`
and `vendor/LivePortrait`, because the upstream code is MIT and carries its own
history.

Two things about the upstream layout matter here:

* The repository is a **flat** tree -- `models/`, `face_detection/`, `audio.py`,
  `hparams.py` at the root. Putting its root on `sys.path` would publish
  `models` and `audio` to the whole process, where they would shadow anything
  else with those names. The checkout is therefore mounted as a *private*
  package (see :func:`import_vendor`), so only the modules this server asks
  for are reachable, and nothing else in the process can import them by
  accident.
* The weights are **not** in the checkout. They live under
  `<data_dir>/models/wav2lip/`, fetched by `scripts/fetch_wav2lip_weights.sh`.
"""
from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

#: The upstream commit this server is written against. The engine reaches into
#: ``models.wav2lip.Wav2Lip`` and the pipeline into
#: ``face_detection.detection.sfd``, so an unpinned upstream could break those
#: imports at any time.
WAV2LIP_SHA = "bac9a81e63ecc153202353372e5724b83d9e6322"

#: The lip-sync generator: the trained network itself (36.3 M parameters).
GENERATOR = "wav2lip_gan.pth"

#: The S3FD face detector. The upstream crop geometry is defined against
#: S3FD's boxes, so the detector is part of the model, not an accessory.
S3FD = "s3fd.pth"

#: Every weight the model needs, by filename.
WEIGHT_FILES = (GENERATOR, S3FD)

#: The private package name the vendored modules are mounted under.
_VENDOR_PACKAGE = "_wav2lip_vendor"


def vendor_dir() -> Path:
    """The Wav2Lip checkout."""
    return Path(__file__).resolve().parents[2] / "vendor" / "Wav2Lip"


def weights_dir() -> Path:
    """Where the weights live."""
    from dh_server.config import get_settings

    return Path(get_settings().data_dir) / "models" / "wav2lip"


def weight_path(name: str) -> Path:
    return weights_dir() / name


def missing_weights() -> list[str]:
    """The weight files that are not on disk (empty when ready)."""
    missing = []
    for name in WEIGHT_FILES:
        path = weight_path(name)
        if not path.exists() or path.stat().st_size == 0:
            missing.append(name)
    return missing


def import_vendor(module: str):
    """Import a module from the vendored checkout, under a private name.

    ``module`` is a dotted path relative to the checkout root, e.g.
    ``"models.wav2lip"`` or ``"face_detection"``. The checkout is mounted as
    the package ``_wav2lip_vendor`` rather than added to ``sys.path``: the
    upstream root is flat, and a ``sys.path`` entry would publish ``models``
    and ``audio`` process-wide.

    Raises ``FileNotFoundError`` when the checkout is absent -- the caller
    turns that into a message that names `scripts/setup_wav2lip.sh`.
    """
    root = vendor_dir()
    if not (root / "models" / "wav2lip.py").exists():
        raise FileNotFoundError(
            f"Wav2Lip is not vendored at {root}; "
            "run scripts/setup_wav2lip.sh to check it out"
        )
    package = sys.modules.get(_VENDOR_PACKAGE)
    if package is None:
        package = types.ModuleType(_VENDOR_PACKAGE)
        package.__path__ = [str(root)]
        sys.modules[_VENDOR_PACKAGE] = package
    return importlib.import_module(f"{_VENDOR_PACKAGE}.{module}")
