"""Real-portrait I2V test: load a real face photo, run I2V, save first/middle/last."""
import io
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

WEIGHTS = Path("E:/project/digital-human/data/models/Wan2.2-TI2V-5B-Diffusers")
REAL_FACE = Path("E:/project/digital-human/vendor/LivePortrait-upstream/assets/examples/source/s2.jpg")
OUT = Path("E:/project/digital-human/data/_smoke_out/real_face")
OUT.mkdir(parents=True, exist_ok=True)


def main():
    src = Image.open(REAL_FACE).convert("RGB")
    print(f"input: {REAL_FACE}  size={src.size}", flush=True)
    # save the input as a side-by-side reference
    src.save(OUT / "00_input_s2.jpg")

    from diffusers import WanImageToVideoPipeline

    pipe = WanImageToVideoPipeline.from_pretrained(
        str(WEIGHTS), torch_dtype=torch.bfloat16,
    )
    try:
        pipe.enable_sequential_cpu_offload()
    except Exception:
        pipe.to("cuda")

    t0 = time.time()
    out = pipe(
        image=src,
        prompt="a person facing the camera, slight smile, eyes blinking slowly",
        negative_prompt="",
        height=480, width=832,
        num_inference_steps=50, num_frames=17,
        guidance_scale=5.0,
        generator=torch.Generator(device="cpu").manual_seed(0),
    ).frames[0]
    print(f"inference: {time.time()-t0:.0f}s, frames={len(out)}", flush=True)

    arr = np.stack([np.asarray(f) for f in out])
    u8 = (arr * 255).clip(0, 255).astype(np.uint8)
    for i in [0, 4, 8, 12, 16]:
        Image.fromarray(u8[i]).save(OUT / f"frame_{i:02d}.png")
        print(f"  saved frame_{i:02d}.png", flush=True)
    print(f"output value range: [{u8.min()}, {u8.max()}] mean={u8.mean():.1f}", flush=True)


if __name__ == "__main__":
    main()
