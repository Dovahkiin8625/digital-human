#!/usr/bin/env bash
# Reproduce the MuseTalk runtime environment from nothing but this checkout.
#
# Creates .venv-musetalk (Python 3.11 + torch cu128), clones MuseTalk at a
# pinned commit, installs the runtime dependency subset, and fetches the
# weights. Every step is idempotent -- re-running skips what is already there.
#
# Why a second venv and not the project's .venv:
#   * .venv is Python 3.14, and there is no cp314 torch wheel on PyPI.
#   * Keeping torch out of .venv is what lets the default `pytest` stay
#     torch-free and under ten seconds.
#   * This venv is ~6 GB with the weights; the other one is a few tens of MB.
#
# Usage:
#   bash scripts/setup_musetalk.sh            # everything
#   bash scripts/setup_musetalk.sh --no-weights   # skip the 4.4 GB download
#
# Requirements: uv (https://docs.astral.sh/uv/), git, curl. Python 3.11 is
# fetched by uv itself if it is not already on the machine.
set -euo pipefail

# MuseTalk commit this project is written against. The engine reaches into
# `musetalk.models.vae.VAE` and `musetalk.utils.blending`, so an unpinned
# upstream could break the imports at any time.
MUSETALK_SHA="0a89dec45a0192b824e3cf4daf96c239440c5ed8"

# github.com is unreliable from some networks (it resolves and answers curl, then
# drops the git HTTPS connection). The proxies serve the same repository and the
# same commit, so try the real host first and fall through. All four were
# verified to answer `git ls-remote` with MUSETALK_SHA on their default branch.
MUSETALK_REMOTES=(
  "https://github.com/TMElyralab/MuseTalk.git"
  "https://ghproxy.net/https://github.com/TMElyralab/MuseTalk.git"
  "https://gh-proxy.com/https://github.com/TMElyralab/MuseTalk.git"
  "https://ghfast.top/https://github.com/TMElyralab/MuseTalk.git"
)

# torch must come from the cu128 index: no cu126-or-earlier wheel carries
# sm_120 kernels, so on an RTX 50-series card the default wheel fails at the
# first kernel launch with "no kernel image is available for execution".
TORCH_INDEX="https://download.pytorch.org/whl/cu128"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV=".venv-musetalk"
VENDOR="vendor/MuseTalk"
WEIGHTS="data/models/musetalk"
FETCH_WEIGHTS=1
# `$#` guard: bare `"$@"` under `set -u` is an error on bash 3.2, which is
# still what macOS ships.
if [ "$#" -gt 0 ]; then
  for arg in "$@"; do
    case "$arg" in
      --no-weights) FETCH_WEIGHTS=0 ;;
      *) echo "unknown argument: $arg" >&2; exit 2 ;;
    esac
  done
fi

step() { printf '\n=== %s ===\n' "$*"; }

command -v uv  >/dev/null || { echo "uv not found: https://docs.astral.sh/uv/" >&2; exit 1; }
command -v git >/dev/null || { echo "git not found" >&2; exit 1; }
if [ "$FETCH_WEIGHTS" = 1 ]; then
  command -v curl >/dev/null || { echo "curl not found (or pass --no-weights)" >&2; exit 1; }
fi

# A directory that is not a git clone at $VENDOR would be destroyed by the
# clone below, and could be a hand-placed source tree. Refuse rather than guess.
if [ -e "$VENDOR" ] && [ ! -d "$VENDOR/.git" ]; then
  echo "ERROR: $VENDOR exists but is not a git clone." >&2
  echo "       move it aside, or check it out at $MUSETALK_SHA yourself." >&2
  exit 1
fi

# ------------------------------------------------------------------ 1. venv
step "1/5  Python 3.11 venv at $VENV"
if [ -x "$VENV/Scripts/python.exe" ] || [ -x "$VENV/bin/python" ]; then
  echo "skip (exists): $VENV"
else
  uv venv --python 3.11 "$VENV"
fi

# ----------------------------------------------------------------- 2. torch
step "2/5  torch + torchvision from $TORCH_INDEX"
# Installed before the project extra so that the resolver sees cu128 wheels
# already satisfied and never reaches for the CPU/default build on PyPI.
uv pip install --python "$VENV" torch torchvision --index-url "$TORCH_INDEX"

# --------------------------------------------------------- 3. runtime deps
step "3/5  project + musetalk runtime subset"
# NOT MuseTalk's own requirements.txt: that file pins numpy==1.23.5 and pulls
# tensorflow, gradio, and tensorboard, none of which the inference path imports.
# The subset is declared in pyproject.toml under [project.optional-dependencies].
uv pip install --python "$VENV" -e ".[musetalk,dev]"

# ------------------------------------------------------------------ 4. source
step "4/5  MuseTalk source at $MUSETALK_SHA"

# Leaves $VENDOR checked out at MUSETALK_SHA, from whichever remote answers
# first. Returns non-zero only when every remote failed.
ensure_source() {
  local url scratch
  # A black-holed remote (packets dropped, no RST) would otherwise hang the
  # clone for as long as the OS keeps the connection open. Abort a transfer that
  # has stalled under 1 KB/s for 30 s, but let a slow one that is progressing
  # run to completion.
  local -a git_slow=(-c http.lowSpeedLimit=1024 -c http.lowSpeedTime=30)
  if [ -d "$VENDOR/.git" ] && [ "$(git -C "$VENDOR" rev-parse HEAD)" = "$MUSETALK_SHA" ]; then
    echo "skip (at $MUSETALK_SHA): $VENDOR"
    return 0
  fi

  for url in "${MUSETALK_REMOTES[@]}"; do
    printf 'try: %s\n' "$url"
    if [ -d "$VENDOR/.git" ]; then
      # Fetch into the existing clone rather than re-cloning; the weights and
      # any local build products inside vendor/ survive a re-run this way.
      if git "${git_slow[@]}" -C "$VENDOR" fetch --tags "$url" >/dev/null 2>&1 &&
         git -C "$VENDOR" checkout --detach "$MUSETALK_SHA" >/dev/null 2>&1; then
        git -C "$VENDOR" remote set-url origin "$url"
        return 0
      fi
    else
      # Clone to a scratch path and move it into place only on success. A clone
      # that dies mid-transfer otherwise leaves a directory that the next
      # attempt would have to delete, and a dropped connection is exactly the
      # failure mode the remote list exists to handle.
      scratch="$VENDOR.clone.$$"
      rm -rf "$scratch"
      mkdir -p "$(dirname "$VENDOR")"
      if git "${git_slow[@]}" clone "$url" "$scratch" >/dev/null 2>&1 &&
         git -C "$scratch" checkout --detach "$MUSETALK_SHA" >/dev/null 2>&1; then
        mv "$scratch" "$VENDOR"
        return 0
      fi
      rm -rf "$scratch"
    fi
    printf '  failed\n'
  done
  return 1
}

if ! ensure_source; then
  echo "ERROR: could not fetch MuseTalk from any of: ${MUSETALK_REMOTES[*]}" >&2
  echo "       clone it manually into $VENDOR at $MUSETALK_SHA" >&2
  exit 1
fi
echo "source at $(git -C "$VENDOR" rev-parse --short HEAD)"

# ---------------------------------------------------------------- 5. weights
step "5/5  weights -> $WEIGHTS"
if [ "$FETCH_WEIGHTS" = 1 ]; then
  bash scripts/fetch_musetalk_weights.sh "$WEIGHTS"
else
  echo "skipped (--no-weights)"
fi

# ------------------------------------------------------------------- verify
step "verify"
# Resolve once: a `a || b` with a heredoc on each side gives bash two
# here-documents on one line, and the second swallows the rest of the file.
if [ -x "$VENV/Scripts/python.exe" ]; then
  PYBIN="$VENV/Scripts/python.exe"
else
  PYBIN="$VENV/bin/python"
fi

"$PYBIN" - <<'PY'
import sys

try:
    import torch
except ImportError as exc:
    sys.exit(f"FAIL: torch missing ({exc})")

print("python     ", sys.version.split()[0])
print("torch      ", torch.__version__)
print("cuda build ", torch.version.cuda)
print("available  ", torch.cuda.is_available())

if not torch.cuda.is_available():
    sys.exit("FAIL: no CUDA device visible -- the engine will fall back to CPU and be far too slow")

cap = torch.cuda.get_device_capability()
print("device     ", torch.cuda.get_device_name(0), f"sm_{cap[0]}{cap[1]}")

# The real test of the wheel: a kernel launch on this specific architecture.
# A cu126-or-earlier wheel imports and reports the device fine, then fails here.
try:
    a = torch.randn(1024, 1024, device="cuda")
    (a @ a).sum().item()
except RuntimeError as exc:
    sys.exit(f"FAIL: matmul on cuda:0 raised {exc!r} -- wrong torch build for this GPU")
print("matmul     ok")
PY

cat <<EOF

Environment ready.

  interpreter : $PYBIN
  run tests   : $PYBIN -m pytest -q -m 'gpu or musetalk'
  run server  : DH_DRIVE_ENGINE=musetalk DH_AVATAR_PIPELINE=musetalk \\
                $PYBIN -m uvicorn --factory server.main:create_app
EOF
