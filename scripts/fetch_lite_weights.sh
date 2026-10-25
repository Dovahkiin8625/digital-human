#!/usr/bin/env bash
# Fetch the weights the `lite` (FeatherTalk) digital-human model needs.
#
# Three files, 22 MB in total:
#
#   feather_hubert.pth        13.5 MB  the generic audio encoder (3.4 M params)
#   scrfd_2.5g_kps.onnx        3.3 MB  face detector with five keypoints
#   checkpoint_epoch_335.pth.tar  5.0 MB  PFLD-MobileOne backbone (110 landmarks)
#
# All three come from the FeatherTalk repository itself, which is the
# upstream this server vendors (see scripts/setup_lite.sh). The bytes are
# pinned by sha256: these are third-party downloads and torch.load is run on
# them, so a swapped mirror response has to fail loudly rather than quietly.
#
# huggingface.co is unreachable from this network; these files are not on
# Hugging Face at all, so the GitHub raw host and its proxies are the
# sources.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-$ROOT/data/models/lite}"
BASE="https://raw.githubusercontent.com/anliyuan/FeatherTalk/7a8bb0439eb5da5e10601168df329a7052b7edf3"

mkdir -p "$DEST"

# name:path:sha256:size
FILES=(
  "feather_hubert.pth:feather_hubert.pth:3b147e66502132bac3ff76b4730c2a83c341c75251d22a28182a5a1b689cfc46:13480913"
  "scrfd_2.5g_kps.onnx:data_utils/scrfd_2.5g_kps.onnx:32d20c77b9e2dc1d07e94c2ab9d25bdd5cd05eddbe0b46e7b38e7a1eca22e99a:3291017"
  "checkpoint_epoch_335.pth.tar:data_utils/checkpoint_epoch_335.pth.tar:bada866661ad5fa1080a085f51fe9c016c69958c406951afa4afc7840f856de0:5039598"
)

status=0
for entry in "${FILES[@]}"; do
  IFS=":" read -r name path want size <<<"$entry"
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
  for base in "$BASE" \
              "https://ghproxy.net/https://raw.githubusercontent.com/anliyuan/FeatherTalk/7a8bb0439eb5da5e10601168df329a7052b7edf3" \
              "https://ghfast.top/https://raw.githubusercontent.com/anliyuan/FeatherTalk/7a8bb0439eb5da5e10601168df329a7052b7edf3"; do
    url="$base/$path"
    echo "fetch: $url"
    if curl -fL --connect-timeout 15 --max-time 600 -o "$tmp" "$url"; then
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

Manual install (all three files are in the FeatherTalk repository at
commit 7a8bb0439eb5da5e10601168df329a7052b7edf3):

  feather_hubert.pth            <- repository root
  scrfd_2.5g_kps.onnx          <- data_utils/
  checkpoint_epoch_335.pth.tar  <- data_utils/

Place them in:
  $DEST

Without them the lite engine cannot be loaded; the other models are
unaffected.
EOF
  exit 1
fi

echo "lite weights ready in $DEST"
