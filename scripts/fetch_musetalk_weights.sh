#!/usr/bin/env bash
# Fetch the MuseTalk 1.5 runtime weights.
#
# Source of truth is huggingface.co/TMElyralab/MuseTalk, but that host is
# unreachable from this network. ModelScope mirrors the same repos and is
# fast here, so we pull from there. Sizes are verified against the HF repo's
# published file listing (see README).
#
# Skipped deliberately:
#   syncnet/latentsync_syncnet.pt  1.49 GB  -- training-time perceptual loss only
#   dwpose/dw-ll_ucoco_384.pth     407 MB   -- OpenMMLab DWPose; we detect faces
#                                              with a pip-installable detector instead
#   musetalk/pytorch_model.bin     3.17 GB  -- MuseTalk v1.0; we run v1.5
set -euo pipefail

ROOT="${1:-data/models/musetalk}"
MS="https://modelscope.cn/api/v1/models"

# fetch <repo-id> <path-in-repo> <dest>
fetch() {
  local repo="$1" path="$2" dest="$3"
  mkdir -p "$(dirname "$dest")"
  if [ -s "$dest" ]; then
    echo "skip (exists): $dest"
    return 0
  fi
  echo "fetch: $repo/$path -> $dest"
  # Explicit test rather than relying on `set -e`: this function is invoked
  # from a `||` list at one call site, which disables errexit inside it, so a
  # failed curl would otherwise fall through and promote a truncated file.
  if ! curl -sL --fail --retry 5 --retry-delay 3 --retry-all-errors \
    -C - -o "$dest.part" \
    "$MS/$repo/repo?Revision=master&FilePath=$path"; then
    echo "ERROR: fetch failed: $repo/$path" >&2
    return 1
  fi
  mv "$dest.part" "$dest"
}

# --- MuseTalk v1.5 UNet (3.17 GiB) ---
fetch AI-ModelScope/MuseTalk musetalkV15/unet.pth    "$ROOT/musetalkV15/unet.pth"
fetch AI-ModelScope/MuseTalk musetalkV15/musetalk.json "$ROOT/musetalkV15/musetalk.json"

# --- SD VAE (ft-mse), 319 MiB ---
fetch stabilityai/sd-vae-ft-mse config.json                  "$ROOT/sd-vae/config.json"
fetch stabilityai/sd-vae-ft-mse diffusion_pytorch_model.bin  "$ROOT/sd-vae/diffusion_pytorch_model.bin"

# --- Whisper tiny, 144 MiB ---
fetch openai-mirror/whisper-tiny config.json               "$ROOT/whisper/config.json"
fetch openai-mirror/whisper-tiny pytorch_model.bin         "$ROOT/whisper/pytorch_model.bin"
fetch openai-mirror/whisper-tiny preprocessor_config.json  "$ROOT/whisper/preprocessor_config.json"
fetch openai-mirror/whisper-tiny vocab.json                "$ROOT/whisper/vocab.json"
fetch openai-mirror/whisper-tiny merges.txt                "$ROOT/whisper/merges.txt"
fetch openai-mirror/whisper-tiny tokenizer_config.json     "$ROOT/whisper/tokenizer_config.json"
fetch openai-mirror/whisper-tiny normalizer.json           "$ROOT/whisper/normalizer.json"
fetch openai-mirror/whisper-tiny special_tokens_map.json   "$ROOT/whisper/special_tokens_map.json"
fetch openai-mirror/whisper-tiny added_tokens.json         "$ROOT/whisper/added_tokens.json"

# --- face-parse-bisent (biSeNet face parsing, pure PyTorch) ---
fetch AI-ModelScope/face-parse-bisent 79999_iter.pth "$ROOT/face-parse-bisent/79999_iter.pth" || \
  echo "WARN: 79999_iter.pth not on ModelScope under that id; see README fallback"

if [ ! -s "$ROOT/face-parse-bisent/resnet18-5c106cde.pth" ]; then
  curl -sL --fail --retry 5 -o "$ROOT/face-parse-bisent/resnet18-5c106cde.pth" \
    https://download.pytorch.org/models/resnet18-5c106cde.pth
fi

echo "done -> $ROOT"
du -sh "$ROOT"
