#!/usr/bin/env bash
# Download LivePortrait weights + the bundled subtle driving clip.
#
# Defaults to DH_DATA_DIR (set by the server) but accepts DATA_DIR= to override.
# HuggingFace is unreachable on this host, so we go through ModelScope first
# and fall back to ``hf-mirror.com``.
#
# Layout:
#   $DATA_DIR/models/liveportrait/   -- the model weights
#   $DATA_DIR/liveportrait_drivers/  -- one or more driving clips; subtle.mp4 is
#                                       the default bundled with the server

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATA_DIR="${DATA_DIR:-${DH_DATA_DIR:-$(pwd)/data}}"
MODELS_DIR="$DATA_DIR/models/liveportrait"
DRIVERS_DIR="$DATA_DIR/liveportrait_drivers"
mkdir -p "$MODELS_DIR" "$DRIVERS_DIR"

# Modelscope mirror is the primary path on this host. The repo is the official
# snapshot of KwaiVGI/LivePortrait at the time of the script's last edit.
REPO="KwaiVGI/LivePortrait"

if command -v modelscope >/dev/null 2>&1; then
    echo "[fetch_liveportrait] using modelscope to download $REPO ..."
    modelscope download "$REPO" --local_dir "$MODELS_DIR"
elif command -v python >/dev/null 2>&1 && python -c "import modelscope" 2>/dev/null; then
    echo "[fetch_liveportrait] using modelscope SDK ..."
    python -m modelscope download "$REPO" --local_dir "$MODELS_DIR"
elif command -v hf >/dev/null 2>&1; then
    # Fallback: ``hf-mirror.com`` is reachable when huggingface.co is not.
    echo "[fetch_liveportrait] modelscope not found; falling back to hf-mirror.com"
    # HF_ENDPOINT only redirects the HF API; hf_xet's CAS endpoint is
    # hardcoded to cas-server.xethub.hf.co and 401s from this host, so
    # disable xet to force regular HTTP downloads that *do* go through
    # the mirror.
    HF_ENDPOINT="https://hf-mirror.com" HF_HUB_DISABLE_XET=1 \
        hf download "$REPO" --local-dir "$MODELS_DIR"
else
    echo "[fetch_liveportrait] no downloader found (need modelscope or hf-mirror);" \
        "skipping weights. pip install modelscope to enable."
fi

# Pull the canonical driving clips and pre-extracted motion templates from
# the LivePortrait repo's example assets. We do not bundle a fresh clip
# in-tree because every operator should be able to swap in their own driver
# (smile / look-around / etc) without forking the server.
#
# Source of truth is the upstream checkout at ``vendor/LivePortrait-upstream/``
# -- populated by ``scripts/setup_liveportrait.sh``. Earlier revisions of
# this script pulled from GitHub raw (which is unreliable on this host:
# resolve, return a few KB, then stall -- see the xet + github notes in
# ``model-weights-mirror.md``); the local-vendor path is faster, doesn't
# hit the network, and lets us ship both raw ``.mp4`` clips and the
# pre-extracted ``.pkl`` motion templates that upstream ships alongside
# them in ``assets/examples/driving/``.
#
# The mapping is in ``server/avatar/liveportrait.py`` -- keep them in
# sync so the API and the fetch script don't drift.
#
# The pattern mirrors ``fetch_musetalk_weights.sh``: write the final
# destination only after a sanity check (the file must be at least 1 KB
# so a half-written copy never satisfies the existence check on the
# next run).

# Vendor upstream assets directory. ``scripts/setup_liveportrait.sh``
# clones to ``vendor/LivePortrait-upstream``; reject if not present so
# we fail with a clear hint instead of pretending the copy succeeded.
VENDOR_UPSTREAM="${ROOT:-$(pwd)}/vendor/LivePortrait-upstream"
VENDOR_DRIVING="$VENDOR_UPSTREAM/assets/examples/driving"

# Mirror of the ``_DRIVER_SOURCES`` table in server/avatar/liveportrait.py.
# Each row is "<source-filename>:<destination-stem>". ``.pkl`` rows skip
# extraction; ``.mp4`` rows are copied as-is. The same stems land in
# ``$DRIVERS_DIR`` and the API speaks them.
DRIVERS=(
    "d0.mp4:subtle"
    "d3.mp4:motion_a"
    "d6.mp4:motion_b"
    "d9.mp4:motion_c"
    "d10.mp4:motion_d"
    "d13.mp4:motion_e"
)

if [ ! -d "$VENDOR_DRIVING" ]; then
    echo "[fetch_liveportrait] upstream checkout missing at $VENDOR_DRIVING;" \
        "run scripts/setup_liveportrait.sh first." >&2
    exit 1
fi

copy_driver() {
    local src_name="$1" stem="$2"
    local src="$VENDOR_DRIVING/$src_name"
    local dst="$DRIVERS_DIR/$stem.${src_name##*.}"  # keep the original extension
    if [ ! -f "$src" ]; then
        echo "[fetch_liveportrait]   WARN: $src_name missing in upstream checkout; skipping $stem"
        return 1
    fi
    if [ -f "$dst" ] && [ "$(wc -c < "$dst")" -ge 1000 ]; then
        echo "[fetch_liveportrait]   skip (present): $dst"
    else
        cp "$src" "$dst"
        echo "[fetch_liveportrait]   copied: $src_name -> $dst ($(wc -c < "$dst") bytes)"
    fi
    # Cache the frame count in a sidecar file so the API can surface it
    # without re-decoding each mp4 on every listing. ``ffprobe`` reads the
    # container's stream metadata -- milliseconds-fast, no decode.
    if [ -f "$dst" ] && [ ! -f "$DRIVERS_DIR/$stem.frames" ] && command -v ffprobe >/dev/null 2>&1; then
        local n_frames
        n_frames=$(ffprobe -v error -select_streams v:0 \
            -show_entries stream=nb_frames -of csv=p=0 "$dst" 2>/dev/null \
            | head -1 | tr -dc '0-9')
        if [ -n "$n_frames" ] && [ "$n_frames" -gt 0 ]; then
            echo "$n_frames" > "$DRIVERS_DIR/$stem.frames"
            echo "[fetch_liveportrait]   frames sidecar: $stem.frames = $n_frames"
        fi
    fi
    # Render a short GIF preview so the system menu and the 形象活化 page
    # can show the motion at a glance. The extract endpoint does the same
    # for user uploads; this keeps the bundled drivers consistent. Skip
    # when ffmpeg is missing (it's a preview, not a hard requirement).
    if [ -f "$dst" ] && [ ! -f "$DRIVERS_DIR/$stem.gif" ] && command -v ffmpeg >/dev/null 2>&1; then
        if [ "${src_name##*.}" = "mp4" ]; then
            local palette="$dst.palette.png"
            ffmpeg -y -i "$dst" -t 5 \
                -vf "fps=12,scale=240:-1:flags=lanczos,palettegen" \
                "$palette" >/dev/null 2>&1 && \
            ffmpeg -y -i "$dst" -t 5 -i "$palette" \
                -filter_complex "fps=12,scale=240:-1:flags=lanczos[x];[x][1:v]paletteuse" \
                -loop 0 "$DRIVERS_DIR/$stem.gif" >/dev/null 2>&1
            rm -f "$palette"
            if [ -f "$DRIVERS_DIR/$stem.gif" ]; then
                echo "[fetch_liveportrait]   preview gif: $stem.gif ($(wc -c < "$DRIVERS_DIR/$stem.gif") bytes)"
            fi
        elif [ "${src_name##*.}" = "pkl" ]; then
            # The .pkl is a keypoint dict, not a video. We need the
            # matching mp4 to render a preview; upstream's readme doesn't
            # ship one for pre-extracted motion templates, so we skip
            # the preview here -- the operator can extract a video
            # variant later if they want a preview.
            :
        fi
    fi
}

for row in "${DRIVERS[@]}"; do
    src_name="${row%%:*}"
    stem="${row##*:}"
    copy_driver "$src_name" "$stem"
done

# Pre-extracted motion templates are tiny (a few KB) so a same-size
# sanity check is enough; we don't bother with the .part + atomic-rename
# dance because the upstream file is local and ``cp`` is atomic on the
# same filesystem.

echo "[fetch_liveportrait] done. Models: $MODELS_DIR ; Drivers: $DRIVERS_DIR"
