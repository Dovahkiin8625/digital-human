#!/usr/bin/env bash
# Reproduce the `lite` (FeatherTalk) digital-human model from nothing.
#
# Checks out the FeatherTalk source at a pinned commit into vendor/ and
# fetches its weights into data/models/lite/. Every step is idempotent --
# re-running skips what is already there.
#
# The weights are small (22 MB) and the source is a checkout, so unlike
# MuseTalk this needs no second virtualenv: the runtime dependencies
# (torch, opencv, scipy) are the ones .venv-musetalk already has.
#
# Usage:
#   bash scripts/setup_lite.sh              # source + weights
#   bash scripts/setup_lite.sh --no-weights # source only
set -euo pipefail

# The commit this server is written against. The engine reaches into
# ``model.Model`` and ``feather_hubert.FeatherHuBERTEncoder``, and the
# training path into ``train.py`` and ``dataset_mouth_roi.py`` -- an
# unpinned upstream could break those imports at any time.
FEATHERTALK_SHA="7a8bb0439eb5da5e10601168df329a7052b7edf3"

# github.com is unreliable from some networks (it answers curl and then
# drops the git HTTPS connection). The proxies serve the same repository and
# the same commit, so try the real host first and fall through.
FEATHERTALK_REMOTES=(
  "https://github.com/anliyuan/FeatherTalk.git"
  "https://ghproxy.net/https://github.com/anliyuan/FeatherTalk.git"
  "https://gh-proxy.com/https://github.com/anliyuan/FeatherTalk.git"
  "https://ghfast.top/https://github.com/anliyuan/FeatherTalk.git"
)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENDOR="vendor/FeatherTalk"
WEIGHTS="data/models/lite"
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
# and could be a hand-placed source tree. Refuse rather than guess.
if [ -e "$VENDOR" ] && [ ! -d "$VENDOR/.git" ]; then
  echo "ERROR: $VENDOR exists but is not a git clone." >&2
  echo "       move it aside, or check it out at $FEATHERTALK_SHA yourself." >&2
  exit 1
fi

step "1/2  FeatherTalk source at $FEATHERTALK_SHA"

ensure_source() {
  local url scratch
  local -a git_slow=(-c http.lowSpeedLimit=1024 -c http.lowSpeedTime=30)
  if [ -d "$VENDOR/.git" ] && [ "$(git -C "$VENDOR" rev-parse HEAD)" = "$FEATHERTALK_SHA" ]; then
    echo "skip (at $FEATHERTALK_SHA): $VENDOR"
    return 0
  fi

  for url in "${FEATHERTALK_REMOTES[@]}"; do
    printf 'try: %s\n' "$url"
    if [ -d "$VENDOR/.git" ]; then
      if git "${git_slow[@]}" -C "$VENDOR" fetch --tags "$url" >/dev/null 2>&1 &&
         git "${git_slow[@]}" -C "$VENDOR" checkout --detach "$FEATHERTALK_SHA" >/dev/null 2>&1; then
        git -C "$VENDOR" remote set-url origin "$url"
        return 0
      fi
    else
      scratch="$VENDOR.clone.$$"
      rm -rf "$scratch"
      mkdir -p "$(dirname "$VENDOR")"
      if git "${git_slow[@]}" clone "$url" "$scratch" >/dev/null 2>&1 &&
         git -C "$scratch" checkout --detach "$FEATHERTALK_SHA" >/dev/null 2>&1; then
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
  echo "ERROR: could not fetch FeatherTalk from any of: ${FEATHERTALK_REMOTES[*]}" >&2
  echo "       clone it manually into $VENDOR at $FEATHERTALK_SHA" >&2
  exit 1
fi
echo "source at $(git -C "$VENDOR" rev-parse --short HEAD)"

step "2/2  weights -> $WEIGHTS"
if [ "$FETCH_WEIGHTS" = 1 ]; then
  bash scripts/fetch_lite_weights.sh "$WEIGHTS"
else
  echo "skipped (--no-weights)"
fi

cat <<EOF

The lite model is ready.

  source  : $VENDOR
  weights : $WEIGHTS

Its runtime needs torch and opencv -- the ones .venv-musetalk already has.
The engine itself runs on the CPU; no GPU is required to serve it.
EOF
