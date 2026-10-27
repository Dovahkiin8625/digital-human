#!/usr/bin/env bash
# Reproduce the `wav2lip` digital-human model from nothing.
#
# Checks out the Wav2Lip source at a pinned commit into dh-server/vendor/ and
# fetches its weights into data/models/wav2lip/. Every step is idempotent --
# re-running skips what is already there.
#
# The weights are large (526 MB) and the source is a checkout, so unlike
# MuseTalk this needs no second virtualenv: the runtime dependencies
# (torch, opencv, librosa, scipy) are the ones .venv-musetalk already has.
#
# Usage:
#   bash scripts/setup_wav2lip.sh              # source + weights
#   bash scripts/setup_wav2lip.sh --no-weights # source only
set -euo pipefail

# The commit this server is written against. The engine reaches into
# ``models.wav2lip.Wav2Lip`` and the pipeline into ``face_detection`` -- an
# unpinned upstream could break those imports at any time.
WAV2LIP_SHA="bac9a81e63ecc153202353372e5724b83d9e6322"

# github.com is unreliable from some networks (it answers curl and then
# drops the git HTTPS connection). The proxies serve the same repository and
# the same commit, so try the real host first and fall through.
WAV2LIP_REMOTES=(
  "https://github.com/Rudrabha/Wav2Lip.git"
  "https://ghproxy.net/https://github.com/Rudrabha/Wav2Lip.git"
  "https://gh-proxy.com/https://github.com/Rudrabha/Wav2Lip.git"
  "https://ghfast.top/https://github.com/Rudrabha/Wav2Lip.git"
)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

VENDOR="dh-server/vendor/Wav2Lip"
WEIGHTS="data/models/wav2lip"
FETCH_WEIGHTS=1
if [ "$#" -gt 0 ]; then
  for arg in "$@"; do
    case "$arg" in
      --no-weights) FETCH_WEIGHTS=0 ;;
      *) echo "unknown argument: $arg" >&2; exit 2 ;;
    esac
  done
fi

step() { printf '\n=== %s ===\n' "$*"; }

command -v git >/dev/null || { echo "git not found" >&2; exit 1; }
if [ "$FETCH_WEIGHTS" = 1 ]; then
  command -v curl >/dev/null || { echo "curl not found (or pass --no-weights)" >&2; exit 1; }
fi

# A directory that is not a git clone would be destroyed by the clone below
# and could be a hand-placed source tree. Refuse rather than guess -- unless
# it already holds the files this server imports, in which case there is
# nothing to fetch and nothing to destroy.
if [ -e "$VENDOR" ] && [ ! -d "$VENDOR/.git" ]; then
  if [ -f "$VENDOR/models/wav2lip.py" ]; then
    echo "note: $VENDOR is already vendored (not a git clone); leaving it alone"
  else
    echo "ERROR: $VENDOR exists but is not a git clone." >&2
    echo "       move it aside, or check it out at $WAV2LIP_SHA yourself." >&2
    exit 1
  fi
fi

step "1/2  Wav2Lip source at $WAV2LIP_SHA"

ensure_source() {
  local url scratch
  local -a git_slow=(-c http.lowSpeedLimit=1024 -c http.lowSpeedTime=30)
  if [ -d "$VENDOR/.git" ] && [ "$(git -C "$VENDOR" rev-parse HEAD)" = "$WAV2LIP_SHA" ]; then
    echo "skip (at $WAV2LIP_SHA): $VENDOR"
    return 0
  fi
  if [ ! -d "$VENDOR/.git" ] && [ -f "$VENDOR/models/wav2lip.py" ]; then
    echo "skip (already vendored): $VENDOR"
    return 0
  fi

  for url in "${WAV2LIP_REMOTES[@]}"; do
    printf 'try: %s\n' "$url"
    if [ -d "$VENDOR/.git" ]; then
      if git "${git_slow[@]}" -C "$VENDOR" fetch --tags "$url" >/dev/null 2>&1 &&
         git -C "$VENDOR" checkout --detach "$WAV2LIP_SHA" >/dev/null 2>&1; then
        git -C "$VENDOR" remote set-url origin "$url"
        return 0
      fi
    else
      scratch="$VENDOR.clone.$$"
      rm -rf "$scratch"
      mkdir -p "$(dirname "$VENDOR")"
      if git "${git_slow[@]}" clone "$url" "$scratch" >/dev/null 2>&1 &&
         git -C "$scratch" checkout --detach "$WAV2LIP_SHA" >/dev/null 2>&1; then
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
  echo "ERROR: could not fetch Wav2Lip from any of: ${WAV2LIP_REMOTES[*]}" >&2
  echo "       clone it manually into $VENDOR at $WAV2LIP_SHA" >&2
  exit 1
fi
echo "source at $(git -C "$VENDOR" rev-parse --short HEAD)"

step "2/2  weights -> $WEIGHTS"
if [ "$FETCH_WEIGHTS" = 1 ]; then
  bash scripts/fetch_wav2lip_weights.sh "$WEIGHTS"
else
  echo "skipped (--no-weights)"
fi

cat <<EOF

The wav2lip model is ready.

  source  : $VENDOR
  weights : $WEIGHTS

Its runtime needs torch, opencv, librosa and scipy -- the ones .venv-musetalk
already has. The engine itself runs on the CPU; no GPU is required to serve it.
EOF
