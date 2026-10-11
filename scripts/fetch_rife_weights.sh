#!/usr/bin/env bash
# Fetch the RIFE 4.9 flownet used to remap sub-25 fps driver videos onto the
# 25 fps timeline (server/avatar/interpolation.py).
#
# RIFE 4.9 (the v4.7-4.9 timestep-capable architecture: encode stem + four
# IF blocks, not the midpoint-only HDv3 network shipped in the ModelScope
# "cv_rife_video-frame-interpolation" repo) supports arbitrary timesteps,
# which a 24 -> 25 remap needs (positions like 0.96, 0.92, ...).
#
# huggingface.co is unreachable from this network; hf-mirror.com serves the
# same LFS objects and is reachable. The bytes are pinned by sha256 -- the
# checkpoint is a third-party download and torch.load is run with
# weights_only=True, but a hash pin is what makes a swapped mirror response
# fail loudly.
set -euo pipefail

DEST="${1:-data/models/rife/flownet.pkl}"
WANT_SHA="e55fd00f3cc184e3c65961f4bb827a9da022e78eed36b055242c0ac30000d533"
SIZE=21345274

SOURCES=(
  "https://hf-mirror.com/VMTamashii/rife49/resolve/main/rife49.pth"
  "https://huggingface.co/VMTamashii/rife49/resolve/main/rife49.pth"
)

mkdir -p "$(dirname "$DEST")"
if [ -s "$DEST" ]; then
  got="$(sha256sum "$DEST" | cut -d' ' -f1)"
  if [ "$got" = "$WANT_SHA" ]; then
    echo "skip (already present): $DEST"
    exit 0
  fi
  echo "warning: $DEST exists with an unexpected hash; re-downloading"
fi

tmp="$DEST.tmp"
ok=0
for url in "${SOURCES[@]}"; do
  echo "fetch: $url"
  if curl -fL --connect-timeout 15 --max-time 600 -o "$tmp" "$url"; then
    got="$(sha256sum "$tmp" | cut -d' ' -f1)"
    if [ "$got" = "$WANT_SHA" ]; then
      ok=1
      break
    fi
    echo "hash mismatch from $url"
    echo "  want $WANT_SHA"
    echo "  got  $got"
    rm -f "$tmp"
  fi
done

if [ "$ok" -ne 1 ]; then
  cat >&2 <<EOF

ERROR: could not fetch RIFE weights automatically.

Manual install:
  1. Download rife49.pth (RIFE 4.9, $SIZE bytes, sha256 $WANT_SHA):
       https://huggingface.co/VMTamashii/rife49/resolve/main/rife49.pth
     (or the same file from hzwer's Practical-RIFE v4.9 distribution)
  2. Place it at:
       $DEST
  3. Re-run this script to verify the hash.

Without the file, sub-25 fps uploads still work but fall back to repeated
frames, and the admin console shows a Chinese advisory on the avatar.
EOF
  exit 1
fi

mv "$tmp" "$DEST"
echo "ok: $DEST"
