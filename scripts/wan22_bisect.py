"""Bisect: which knob turns zero into non-zero? H/W or steps?"""
import io
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

WEIGHTS = Path("E:/project/digital-human/data/models/Wan2.2-TI2V-5B-Diffusers")


def _png():
    buf = io.BytesIO()
    Image.new("RGB", (640, 800), (180, 150, 130)).save(buf, format="PNG")
    return buf.getvalue()


def main():
    from diffusers import WanImageToVideoPipeline

    pipe = WanImageToVideoPipeline.from_pretrained(
        str(WEIGHTS), torch_dtype=torch.bfloat16,
    )
    try:
        pipe.enable_sequential_cpu_offload()
    except Exception:
        pipe.to("cuda")
    src = Image.open(io.BytesIO(_png())).convert("RGB")

    cases = [
        # (label, height, width, steps, expected_to_pass)
        ("A 480x832 / 10 steps",   480,  832, 10),
        ("B 480x832 / 50 steps",   480,  832, 50),
        ("C 704x1280 / 50 steps",  704, 1280, 50),
        ("D 704x1280 / 20 steps",  704, 1280, 20),
    ]

    for label, h, w, steps in cases:
        t0 = time.time()
        out = pipe(
            image=src,
            prompt="a person facing the camera, slight smile",
            negative_prompt="",
            height=h, width=w,
            num_inference_steps=steps, num_frames=17,
            guidance_scale=5.0,
            generator=torch.Generator(device="cpu").manual_seed(0),
        )
        arr = np.stack([np.asarray(f) for f in out.frames[0]])
        elapsed = time.time() - t0
        print(f"  {label}: range=[{arr.min():.3f}, {arr.max():.3f}], "
              f"mean={arr.mean():.3f}, elapsed={elapsed:.0f}s", flush=True)
        torch.cuda.empty_cache()

    return 0


if __name__ == "__main__":
    sys.exit(main())
