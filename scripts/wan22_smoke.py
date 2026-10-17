"""Minimal smoke test: load Wan 2.2 TI2V-5B and run a tiny I2V inference.

Not part of the test suite -- hand-run via:
    .venv-musetalk/Scripts/python.exe /tmp/wan22_smoke.py

Tries to load with bf16 + sequential_cpu_offload and generate 17 frames
@ 720x1280 with 10 inference steps. Reports progress to stdout + writes
``/tmp/wan22_smoke_frames.npy`` on success.
"""
import io
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

WEIGHTS = Path("E:/project/digital-human/data/models/Wan2.2-TI2V-5B-Diffusers")
OUT = Path("E:/project/digital-human/data/_smoke_out/wan22_smoke_frames.npy")
OUT.parent.mkdir(parents=True, exist_ok=True)


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (640, 800), (180, 150, 130)).save(buf, format="PNG")
    return buf.getvalue()


def main():
    print(f"[smoke] weights: {WEIGHTS}", flush=True)
    if not WEIGHTS.is_dir():
        sys.exit(f"[smoke] FAIL: weights dir missing -- run scripts/fetch_wan22_weights.sh")

    t0 = time.time()
    print("[smoke] importing torch + diffusers ...", flush=True)
    import torch
    from diffusers import WanImageToVideoPipeline

    print(f"[smoke]   ... done in {time.time() - t0:.1f}s", flush=True)
    print(f"[smoke] CUDA: available={torch.cuda.is_available()}, "
          f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}",
          flush=True)
    print(f"[smoke] VRAM free: {torch.cuda.mem_get_info()[0] / 1024**3:.1f} GB", flush=True)

    t0 = time.time()
    print("[smoke] loading WanImageToVideoPipeline (bf16 + sequential_cpu_offload) ...",
          flush=True)
    pipe = WanImageToVideoPipeline.from_pretrained(
        str(WEIGHTS),
        torch_dtype=torch.bfloat16,
    )
    print(f"[smoke]   ... from_pretrained done in {time.time() - t0:.1f}s", flush=True)
    try:
        pipe.enable_sequential_cpu_offload()
        print("[smoke] sequential_cpu_offload ENABLED", flush=True)
    except (AttributeError, ValueError) as e:
        print(f"[smoke] sequential_cpu_offload FAILED ({e}); falling back to .to(cuda)",
              flush=True)
        pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[smoke] VRAM after load+offload: "
          f"{torch.cuda.mem_get_info()[0] / 1024**3:.1f} GB free",
          flush=True)

    # 1280x720 portrait source from the test PNG; diffusers expects PIL.
    src = Image.open(io.BytesIO(_png())).convert("RGB")
    print(f"[smoke] source image: {src.size}", flush=True)

    t0 = time.time()
    print("[smoke] running inference: 17 frames @ 704x1280, 10 steps, guidance 5.0 ...",
          flush=True)
    output = pipe(
        image=src,
        prompt="a person turning to face the camera",
        negative_prompt="",
        height=704,
        width=1280,
        num_inference_steps=10,
        num_frames=17,
        guidance_scale=5.0,
        generator=torch.Generator(device="cpu").manual_seed(0),
    ).frames[0]
    print(f"[smoke]   ... inference done in {time.time() - t0:.1f}s", flush=True)
    print(f"[smoke] output: {len(output)} frames, "
          f"shape per frame={np.asarray(output[0]).shape}", flush=True)

    frames = np.stack([np.ascontiguousarray(np.asarray(f)) for f in output])
    print(f"[smoke] stacked frames: {frames.shape}, dtype={frames.dtype}", flush=True)
    np.save(OUT, frames)
    print(f"[smoke] saved -> {OUT}", flush=True)
    print("[smoke] OK", flush=True)


if __name__ == "__main__":
    main()
