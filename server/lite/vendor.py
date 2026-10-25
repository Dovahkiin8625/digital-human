"""Where the FeatherTalk checkout and its weights live.

The upstream is a git checkout of `anliyuan/FeatherTalk` at a pinned commit,
placed in `vendor/FeatherTalk` by `scripts/setup_lite.sh`. It is not part of
this repository's tree: `vendor/` is gitignored, exactly like `vendor/MuseTalk`
and `vendor/LivePortrait`, because the upstream code is Apache-2.0/MIT and
carries its own history.

Two details of the upstream layout matter here:

* Three sys.path entries are needed, not one. Upstream's `model.py` does
  `from data_utils.base_module import MobileOneBlock` and its dataset does
  `from face_utils import ...` -- both relative to the repository root. Its
  PFLD backbone goes the other way:
  `data_utils/pfld_mobileone.py` does `from base_module import ...`, i.e.
  relative to `data_utils/` itself. And the audio encoder lives one level
  deeper still: `data_utils/feather_hubert/feather_hubert.py`. So the
  checkout root, `data_utils/` and `data_utils/feather_hubert/` all go on
  the path -- the deepest first, so `import feather_hubert` finds the module
  and not the directory that holds it.
* The weights are **not** in the checkout. They live under
  `<data_dir>/models/lite/`, fetched by `scripts/fetch_lite_weights.sh`.
"""
from __future__ import annotations

import sys
from pathlib import Path

#: The upstream commit this server is written against. The engine reaches into
#: `model.Model` and `feather_hubert.FeatherHuBERTEncoder`, so an unpinned
#: upstream could break the imports at any time.
FEATHERTALK_SHA = "7a8bb0439eb5da5e10601168df329a7052b7edf3"

#: The generic audio encoder. Shared across identities -- it is not part of
#: the personalised pack.
FEATHER_HUBERT = "feather_hubert.pth"

#: SCRFD 2.5G face detector with five keypoints (ONNX).
SCRFD = "scrfd_2.5g_kps.onnx"

#: PFLD-MobileOne backbone: 110 landmarks per face. Upstream ships it inside
#: a training checkpoint (``pfld_backbone`` key); we keep the file as it is
#: rather than re-saving it under a friendlier name.
PFLD = "checkpoint_epoch_335.pth.tar"

#: Every weight the model needs, by filename.
WEIGHT_FILES = (FEATHER_HUBERT, SCRFD, PFLD)

_path_added = False


def vendor_dir() -> Path:
    """The FeatherTalk checkout."""
    return Path(__file__).resolve().parents[2] / "vendor" / "FeatherTalk"


def weights_dir() -> Path:
    """Where the weights live."""
    from server.config import get_settings

    return Path(get_settings().data_dir) / "models" / "lite"


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


def ensure_on_path() -> Path:
    """Put the vendored checkout on ``sys.path`` and return its root.

    Idempotent: the entries are added once per process. Raises
    ``FileNotFoundError`` when the checkout is absent -- the caller turns that
    into a message that names `scripts/setup_lite.sh`.
    """
    global _path_added
    root = vendor_dir()
    if not (root / "model.py").exists():
        raise FileNotFoundError(
            f"FeatherTalk is not vendored at {root}; "
            "run scripts/setup_lite.sh to check it out"
        )
    if not _path_added:
        for entry in (
            root / "data_utils" / "feather_hubert",
            root,
            root / "data_utils",
        ):
            text = str(entry)
            if text not in sys.path:
                sys.path.insert(0, text)
        _path_added = True
    return root
