#!/usr/bin/env bash
# Download Wan 2.2 TI2V-5B weights for the 形象活化 tab's OSS-i2v slot.
#
# HuggingFace is unreachable on this host; we use the ModelScope mirror of
# ``Wan-AI/Wan2.2-TI2V-5B-Diffusers`` as the primary path. The weights land
# under ``data_dir/models/Wan2.2-TI2V-5B-Diffusers``, matching what
# ``server.avatar.wan22`` expects.
#
# NOTE: a future fp8 backend is intentionally NOT scripted here. Wan-AI has
# not released an officially-quantised FP8 variant of TI2V-5B (only
# third-party Comfy-Org / bullerwins quants exist, which aren't drop-in
# for diffusers). When Wan-AI ships an FP8 release, add it to VARIANTS and
# append a matching row to ``VARIANTS`` in ``server/avatar/wan22.py``.

set -euo pipefail

DATA_DIR="${DATA_DIR:-${DH_DATA_DIR:-$(pwd)/data}}"
MODELS_DIR="$DATA_DIR/models"
mkdir -p "$MODELS_DIR"

VARIANTS=(
    "Wan2.2-TI2V-5B-Diffusers"
)

if command -v modelscope >/dev/null 2>&1; then
    for repo in "${VARIANTS[@]}"; do
        target="$MODELS_DIR/$repo"
        if [ -d "$target" ] && [ -n "$(ls -A "$target" 2>/dev/null)" ]; then
            echo "[fetch_wan22] $repo already present; skipping"
            continue
        fi
        echo "[fetch_wan22] downloading $repo via modelscope ..."
        modelscope download "Wan-AI/$repo" --local_dir "$target"
    done
elif command -v python >/dev/null 2>&1 && python -c "import modelscope" 2>/dev/null; then
    for repo in "${VARIANTS[@]}"; do
        target="$MODELS_DIR/$repo"
        if [ -d "$target" ] && [ -n "$(ls -A "$target" 2>/dev/null)" ]; then
            echo "[fetch_wan22] $repo already present; skipping"
            continue
        fi
        echo "[fetch_wan22] downloading $repo via modelscope SDK ..."
        python -m modelscope download "Wan-AI/$repo" --local_dir "$target"
    done
elif command -v hf >/dev/null 2>&1; then
    # Fallback to ``hf-mirror.com`` which is reachable when huggingface.co is not.
    for repo in "${VARIANTS[@]}"; do
        target="$MODELS_DIR/$repo"
        if [ -d "$target" ] && [ -n "$(ls -A "$target" 2>/dev/null)" ]; then
            echo "[fetch_wan22] $repo already present; skipping"
            continue
        fi
        echo "[fetch_wan22] downloading $repo via hf-mirror.com ..."
        # HF_ENDPOINT only redirects the HF API; hf_xet's CAS endpoint is
        # hardcoded to cas-server.xethub.hf.co and 401s from this host, so
        # disable xet to force regular HTTP downloads that *do* go through
        # the mirror.
        HF_ENDPOINT="https://hf-mirror.com" HF_HUB_DISABLE_XET=1 \
            hf download "Wan-AI/$repo" --local-dir "$target"
    done
else
    echo "[fetch_wan22] no downloader found (need modelscope or hf-mirror);" \
        "pip install modelscope to enable."
fi

echo "[fetch_wan22] done. Models: $MODELS_DIR"
