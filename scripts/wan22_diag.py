"""Diagnose why the I2V output is all zeros on Wan2.2-TI2V-5B.

Try several variations to find the smallest one that yields non-zero frames.
"""
import io
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

WEIGHTS = Path("E:/project/digital-human/data/models/Wan2.2-TI2V-5B-Diffusers")


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (640, 800), (180, 150, 130)).save(buf, format="PNG")
    return buf.getvalue()


def check(name, frames):
    arr = np.stack([np.asarray(f) for f in frames])
    print(f"  {name}: shape={arr.shape}, range=[{arr.min()}, {arr.max()}], "
          f"mean={arr.mean():.1f}", flush=True)


def main():
    print("[diag] loading pipeline ...", flush=True)
    from diffusers import WanImageToVideoPipeline

    t0 = time.time()
    pipe = WanImageToVideoPipeline.from_pretrained(
        str(WEIGHTS), torch_dtype=torch.bfloat16,
    )
    print(f"[diag]   loaded in {time.time()-t0:.1f}s, enabling offload", flush=True)
    try:
        pipe.enable_sequential_cpu_offload()
    except Exception:
        pipe.to("cuda")
    print(f"[diag] VRAM free: {torch.cuda.mem_get_info()[0]/1024**3:.1f} GB", flush=True)

    src = Image.open(io.BytesIO(_png())).convert("RGB")
    print(f"[diag] source PIL: {src.size}", flush=True)

    print(f"\n[diag] RUN A: image=PIL, h=480, w=832, 50 steps, gs=5.0", flush=True)
    out = pipe(
        image=src,
        prompt="a person facing the camera, slight smile",
        negative_prompt="",
        height=480, width=832,
        num_inference_steps=50, num_frames=17,
        guidance_scale=5.0,
        generator=torch.Generator(device="cpu").manual_seed(0),
    )
    check("WanImageToVideo @ 480x832 / 50 steps", out.frames[0])
    if np.asarray(out.frames[0][0]).max() > 0:
        np.save("E:/project/digital-human/data/_smoke_out/diag_a_frames.npy",
                np.stack([np.asarray(f) for f in out.frames[0]]))
        print(f"[diag]   saved (non-zero)", flush=True)
        return 0

    print(f"\n[diag] RUN B: image=ndarray, 480x832, 50 steps, gs=5.0", flush=True)
    out = pipe(
        image=np.asarray(src),
        prompt="a person facing the camera, slight smile",
        negative_prompt="",
        height=480, width=832,
        num_inference_steps=50, num_frames=17,
        guidance_scale=5.0,
        generator=torch.Generator(device="cpu").manual_seed(0),
    )
    check("image=ndarray @ 480x832 / 50 steps", out.frames[0])
    if np.asarray(out.frames[0][0]).max() > 0:
        np.save("E:/project/digital-human/data/_smoke_out/diag_b_frames.npy",
                np.stack([np.asarray(f) for f in out.frames[0]]))
        print(f"[diag]   saved (non-zero)", flush=True)
        return 0

    print(f"\n[diag] RUN C: 704x1280, 50 steps", flush=True)
    out = pipe(
        image=src,
        prompt="a person facing the camera, slight smile",
        negative_prompt="",
        height=704, width=1280,
        num_inference_steps=50, num_frames=17,
        guidance_scale=5.0,
        generator=torch.Generator(device="cpu").manual_seed(0),
    )
    check("704x1280 / 50 steps", out.frames[0])
    if np.asarray(out.frames[0][0]).max() > 0:
        np.save("E:/project/digital-human/data/_smoke_out/diag_c_frames.npy",
                np.stack([np.asarray(f) for f in out.frames[0]]))
        print(f"[diag]   saved (non-zero)", flush=True)
        return 0

    print(f"\n[diag] RUN D: 480x832, 50 steps, gs=1.0 (no CFG)", flush=True)
    out = pipe(
        image=src,
        prompt="a person facing the camera, slight smile",
        negative_prompt="",
        height=480, width=832,
        num_inference_steps=50, num_frames=17,
        guidance_scale=1.0,
        generator=torch.Generator(device="cpu").manual_seed(0),
    )
    check("gs=1.0 @ 480x832", out.frames[0])

    print("[diag] ALL RUNS produced zeros -- model/loader issue, not config", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
