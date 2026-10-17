#!/usr/bin/env bash
# Set up the LivePortrait runtime inside ``.venv-musetalk``.
#
# Mirrors ``scripts/setup_musetalk.sh``: clones the upstream repo at a
# pinned commit under ``vendor/``, writes a thin shim package under
# ``vendor/LivePortrait/`` that re-exports the upstream pipeline under the
# ``LivePortrait`` namespace our adapter expects, then ``uv pip install -e``s
# it into the ``.venv-musetalk`` interpreter. The heavy runtime deps
# (``insightface``, ``onnxruntime-gpu``, ``tyro``, ...) come in the same
# step so a subsequent ``from LivePortrait.live_portrait_pipeline import
# LivePortraitPipeline`` actually succeeds.
#
# Why a shim and not just ``pip install LivePortrait``:
#   * No PyPI package by that name has ever existed.
#   * Upstream exposes the pipeline at ``src.live_portrait_pipeline``; the
#     adapter imports ``LivePortrait.live_portrait_pipeline``. The shim is
#     a tiny bridge so neither side has to change.
#   * The shim populates ``sys.modules['LivePortrait.live_portrait_pipeline']``
#     eagerly, but a sentinel class is installed if the heavy deps are
#     missing -- so just ``import LivePortrait`` stays cheap and the rest
#     of the avatar engine isn't blocked.
#
# Usage:
#   bash scripts/setup_liveportrait.sh

set -euo pipefail

# Upstream commit. ``main`` is the only branch upstream ships and there are
# no releases; pin to whatever ``main`` was when this script was last
# regenerated. Re-running the script picks up the new SHA from upstream.
LIVEPORTRAIT_SHA="9b294b3d0536135442ea73cb01e6cb3ca7029dd3"

# Same mirrors as ``setup_musetalk.sh``'s MUSETALK_REMOTES. ``github.com``
# is unreliable here (resolves, returns a few KB, then drops); the proxies
# carry the same bytes reliably. Direct is the last-resort fallback.
LIVEPORTRAIT_REMOTES=(
  "https://ghfast.top/https://github.com/KwaiVGI/LivePortrait.git"
  "https://gh-proxy.com/https://github.com/KwaiVGI/LivePortrait.git"
  "https://ghproxy.net/https://github.com/KwaiVGI/LivePortrait.git"
  "https://github.com/KwaiVGI/LivePortrait.git"
)

VENV=".venv-musetalk"
VENDOR_UPSTREAM="vendor/LivePortrait-upstream"
SHIM_DIR="vendor/LivePortrait"
SHIM_PKG_DIR="$SHIM_DIR/LivePortrait"
SHIM_INIT="$SHIM_PKG_DIR/__init__.py"
SHIM_PYPROJECT="$SHIM_DIR/pyproject.toml"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Refuse to clobber a hand-placed tree at the upstream path -- same
# pre-flight check setup_musetalk.sh runs against vendor/MuseTalk.
if [ -e "$VENDOR_UPSTREAM" ] && [ ! -d "$VENDOR_UPSTREAM/.git" ]; then
  echo "ERROR: $VENDOR_UPSTREAM exists but is not a git clone." >&2
  echo "       move it aside, or check it out at $LIVEPORTRAIT_SHA yourself." >&2
  exit 1
fi

step() { printf '\n=== %s ===\n' "$*"; }

command -v uv >/dev/null || { echo "uv not found: https://docs.astral.sh/uv/" >&2; exit 1; }
command -v git >/dev/null || { echo "git not found" >&2; exit 1; }

# ----------------------------------------------------------------- 1. upstream
step "1/4  upstream at $VENDOR_UPSTREAM"
ensure_upstream() {
  local url scratch
  # Same stall guard as setup_musetalk.sh: a black-holed HTTPS otherwise
  # hangs for as long as the OS keeps the socket open.
  local -a git_slow=(-c http.lowSpeedLimit=1024 -c http.lowSpeedTime=30)
  if [ -d "$VENDOR_UPSTREAM/.git" ] \
     && [ "$(git -C "$VENDOR_UPSTREAM" rev-parse HEAD)" = "$LIVEPORTRAIT_SHA" ]; then
    echo "skip (at $LIVEPORTRAIT_SHA): $VENDOR_UPSTREAM"
    return 0
  fi
  for url in "${LIVEPORTRAIT_REMOTES[@]}"; do
    printf 'try: %s\n' "$url"
    if [ -d "$VENDOR_UPSTREAM/.git" ]; then
      # Existing checkout is at the wrong SHA: fetch the new one and reset.
      # Fetch into the existing clone rather than re-cloning; any local
      # build products inside vendor/LivePortrait-upstream/ survive this.
      if git "${git_slow[@]}" -C "$VENDOR_UPSTREAM" fetch --tags "$url" "$LIVEPORTRAIT_SHA" >/dev/null 2>&1 \
         && git -C "$VENDOR_UPSTREAM" checkout --detach "$LIVEPORTRAIT_SHA" >/dev/null 2>&1; then
        git -C "$VENDOR_UPSTREAM" remote set-url origin "$url"
        return 0
      fi
    else
      # Clone to a scratch path and move it into place only on success.
      # A clone that dies mid-transfer would otherwise leave a partial
      # directory that the next attempt would have to delete.
      scratch="$VENDOR_UPSTREAM.clone.$$"
      rm -rf "$scratch"
      mkdir -p "$(dirname "$VENDOR_UPSTREAM")"
      if git "${git_slow[@]}" clone "$url" "$scratch" >/dev/null 2>&1 \
         && git -C "$scratch" checkout --detach "$LIVEPORTRAIT_SHA" >/dev/null 2>&1; then
        mv "$scratch" "$VENDOR_UPSTREAM"
        return 0
      fi
      rm -rf "$scratch"
    fi
    printf '  failed\n'
  done
  return 1
}
if ! ensure_upstream; then
  echo "ERROR: could not fetch LivePortrait from any of: ${LIVEPORTRAIT_REMOTES[*]}" >&2
  echo "       clone it manually into $VENDOR_UPSTREAM at $LIVEPORTRAIT_SHA" >&2
  exit 1
fi
echo "upstream at $(git -C "$VENDOR_UPSTREAM" rev-parse --short HEAD)"

# ------------------------------------------------------------------ 2. shim
step "2/4  shim package at $SHIM_DIR"
# The shim's source is embedded in this script on purpose: ``vendor/`` is
# gitignored, so the files cannot live in the repo. Any edit here is the
# new canonical version; re-running the script regenerates them.
mkdir -p "$SHIM_PKG_DIR"

cat > "$SHIM_INIT" <<'PY'
"""Shim package: re-exports kwaiVGI/LivePortrait under the ``LivePortrait`` namespace.

The upstream ``KwaiVGI/LivePortrait`` exposes its pipeline at
``src.live_portrait_pipeline``; our adapter in
``server.avatar.liveportrait`` imports ``LivePortrait.live_portrait_pipeline``.
This shim bridges the gap so the adapter's import keeps working unchanged.

Import-resolution notes (this matters for the lazy-looking API):

* PEP 562 ``__getattr__`` would handle ``LivePortrait.live_portrait_pipeline``
  attribute access, but the import machinery for
  ``from LivePortrait.live_portrait_pipeline import X`` goes through
  ``importlib.import_module`` which does NOT consult PEP 562. The only
  reliable way to make that import form work is to populate
  ``sys.modules['LivePortrait.live_portrait_pipeline']`` ourselves.
* Upstream pulls heavy deps (``tyro``, ``pyyaml``, ``insightface``,
  ``onnxruntime``, ...) at module load. We wrap the eager import in a
  try/except so just ``import LivePortrait`` stays cheap; if a runtime
  dep is missing we install a sentinel class that raises a clear
  ``ImportError`` on *use* -- only on attribute access, never on import.
"""
from __future__ import annotations

import os as _os
import sys as _sys

__all__ = ["__version__", "LivePortraitPipeline"]
__version__ = "0.1.0"

# Path to the vendored upstream checkout. The shim lives at
# ``vendor/LivePortrait/LivePortrait/__init__.py``; the upstream checkout
# is at ``vendor/LivePortrait-upstream/`` (kept one level up so a clean
# delete of either does not take the other with it). The PARENT must be
# on ``sys.path`` -- not ``src/`` itself -- so Python's namespace-package
# discovery can resolve ``import src.live_portrait_pipeline``.
_UPSTREAM_PARENT = _os.path.normpath(
    _os.path.join(
        _os.path.dirname(_os.path.abspath(__file__)),
        "..",
        "..",
        "LivePortrait-upstream",
    )
)


class _PipelineUnavailable:
    """Sentinel installed when upstream's heavy deps are missing.

    ``import LivePortrait`` succeeds even without ``insightface`` /
    ``onnxruntime`` / ``tyro`` installed -- only attribute access on the
    pipeline raises, with a hint pointing at ``scripts/setup_liveportrait.sh``.
    """

    __slots__ = ("_exc",)

    def __init__(self, exc: ImportError) -> None:
        self._exc = exc

    def __getattr__(self, name: str):
        raise ImportError(
            "LivePortrait runtime is not installed ({}); "
            "run scripts/setup_liveportrait.sh, then "
            "scripts/fetch_liveportrait_weights.sh".format(self._exc)
        ) from self._exc


try:
    if _UPSTREAM_PARENT not in _sys.path:
        _sys.path.insert(0, _UPSTREAM_PARENT)
    # The real import. This triggers upstream's full module body (torch,
    # cv2, insightface, ...) and therefore requires the heavy deps to be
    # installed. ``setup_liveportrait.sh`` installs them in step 4.
    from src.live_portrait_pipeline import (  # type: ignore[import-not-found]
        LivePortraitPipeline as _LivePortraitPipeline,
    )
    # Register under the ``LivePortrait.live_portrait_pipeline`` name so
    # ``from LivePortrait.live_portrait_pipeline import X`` resolves via
    # the regular import machinery (PEP 562 alone does not).
    _sys.modules[__name__ + ".live_portrait_pipeline"] = _sys.modules["src.live_portrait_pipeline"]
    LivePortraitPipeline = _LivePortraitPipeline
except ImportError as _exc:
    LivePortraitPipeline = _PipelineUnavailable(_exc)
PY

cat > "$SHIM_PYPROJECT" <<'TOML'
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "LivePortrait"
version = "0.1.0"
description = "Shim that re-exports KwaiVGI/LivePortrait under the LivePortrait namespace"
requires-python = ">=3.11"
# Runtime deps (insightface, onnxruntime-gpu, tyro, pyyaml, pykalman,
# ffmpeg-python, lmdb, tqdm) are installed by scripts/setup_liveportrait.sh
# so this shim is cheap to install even when the operator doesn't plan
# to *use* LivePortrait.

[tool.setuptools.packages.find]
where = ["."]
include = ["LivePortrait*"]
TOML

echo "shim written: $SHIM_INIT + $SHIM_PYPROJECT"

# ------------------------------------------------------- 3. pip install shim
step "3/4  pip install -e $SHIM_DIR (editable) into $VENV"
if [ ! -x "$VENV/Scripts/python.exe" ]; then
  echo "ERROR: $VENV/Scripts/python.exe missing; run scripts/setup_musetalk.sh first." >&2
  exit 1
fi
uv pip install --python "$VENV" -e "$SHIM_DIR"

# ----------------------------------------------------------- 4. runtime deps
step "4/4  runtime deps into $VENV"
# Subset of upstream's requirements_base.txt -- the inference path only.
# Demo/training deps (gradio, albumentations, matplotlib) are deliberately
# skipped. ``imageio[ffmpeg]`` looks optional but IS required: upstream's
# ``images2video`` writes the result mp4 via imageio's ffmpeg plugin and
# raises ``The FFMPEG plugin is not installed`` without it. ``insightface``
# requires ``onnxruntime`` at runtime; install onnxruntime-gpu first so
# insightface's resolver sees it (it picks up whatever's on the path).
uv pip install --python "$VENV" \
    "onnxruntime-gpu>=1.18,<2" \
    "insightface>=0.7" \
    "tyro>=0.8" \
    "pyyaml>=6.0" \
    "pykalman>=0.9" \
    "ffmpeg-python>=0.2" \
    "tqdm>=4.66" \
    "lmdb>=1.4" \
    "imageio[ffmpeg]>=2.34"

cat <<EOF

LivePortrait ready.

  interpreter : $VENV/Scripts/python.exe
  smoke test  : $VENV/Scripts/python.exe -c \\
                  "from LivePortrait.live_portrait_pipeline import LivePortraitPipeline; \\
                   print(LivePortraitPipeline)"
  run server  : DH_DRIVE_ENGINE=liveportrait DH_AVATAR_PIPELINE=liveportrait \\
                $VENV/Scripts/python.exe -m uvicorn --factory server.main:create_app

EOF
