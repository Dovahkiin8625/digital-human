#!/usr/bin/env bash
# Fetch the weights the `wav2lip` digital-human model needs.
#
# Two files, 526 MB in total:
#
#   wav2lip_gan.pth   435.8 MB  the lip-sync generator (36.3 M parameters)
#   s3fd.pth           89.8 MB  the S3FD face detector the crop geometry uses
#
# The bytes are pinned by sha256: these are third-party downloads and
# torch.load is run on them, so a swapped mirror response has to fail loudly
# rather than quietly.
#
# huggingface.co and Google Drive are unreachable from this network, so the
# sources are the hf-mirror proxy and the detector's own host.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST="${1:-$ROOT/data/models/wav2lip}"

mkdir -p "$DEST"

# name:sha256:size:url...
FILES=(
  "wav2lip_gan.pth:ca9ab7b7b812c0e80a6e70a5977c545a1e8a365a6c49d5e533023c034d7ac3d8:435801865:https://hf-mirror.com/camenduru/Wav2Lip/resolve/main/checkpoints/wav2lip_gan.pth|https://huggingface.co/camenduru/Wav2Lip/resolve/main/checkpoints/wav2lip_gan.pth"
  "s3fd.pth:619a31681264d3f7f7fc7a16a42cbbe8b23f31a256f75a366e5a1bcd59b33543:89843225:https://hf-mirror.com/camenduru/Wav2Lip/resolve/main/checkpoints/s3fd-619a316812.pth|https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth"
)

status=0
for entry in "${FILES[@]}"; do
  IFS=":" read -r name want size urls <<<"$entry"
  target="$DEST/$name"
  if [ -s "$target" ]; then
    got="$(sha256sum "$target" | cut -d' ' -f1)"
    if [ "$got" = "$want" ]; then
      echo "skip (already present): $target"
      continue
    fi
    echo "warning: $target exists with an unexpected hash; re-downloading"
  fi

  tmp="$target.tmp"
  ok=0
  IFS="|" read -r -a mirrors <<<"$urls"
  for url in "${mirrors[@]}"; do
    echo "fetch: $url"
    if curl -fL --connect-timeout 15 --max-time 3600 -C - -o "$tmp" "$url"; then
      got="$(sha256sum "$tmp" | cut -d' ' -f1)"
      if [ "$got" = "$want" ]; then
        ok=1
        break
      fi
      echo "  hash mismatch"
      echo "    want $want"
      echo "    got  $got"
      rm -f "$tmp"
    fi
  done

  if [ "$ok" != 1 ]; then
    echo "ERROR: could not fetch $name" >&2
    status=1
    continue
  fi
  mv "$tmp" "$target"
  echo "ok: $target ($size bytes)"
done

if [ "$status" != 0 ]; then
  cat >&2 <<EOF

Manual install:

  wav2lip_gan.pth   <- https://hf-mirror.com/camenduru/Wav2Lip/resolve/main/checkpoints/wav2lip_gan.pth
  s3fd.pth           <- https://hf-mirror.com/camenduru/Wav2Lip/resolve/main/checkpoints/s3fd-619a316812.pth
                      (or https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth)

Place them in:
  $DEST

Without them the wav2lip engine cannot be loaded; the other models are
unaffected.
EOF
  exit 1
fi

echo "wav2lip weights ready in $DEST"
