"""Measure peak VRAM during Wan 2.2 TI2V-5B inference on 16GB card.

Runs a 17-frame / 20-step inference and tracks the peak torch.cuda memory
allocated across the run. The result tells us the realistic minimum VRAM
budget (with sequential_cpu_offload) so we can pick a vram_min_gb that
admits 16GB hosts without lying.
"""
import io
import time
from pathlib import Path

import torch
from PIL import Image

WEIGHTS = Path("E:/project/digital-human/data/models/Wan2.2-TI2V-5B-Diffusers")
SRC = Path("E:/project/digital-human/vendor/LivePortrait-upstream/assets/examples/source/s2.jpg")


def main():
    from diffusers import WanImageToVideoPipeline

    src = Image.open(SRC).convert("RGB")
    pipe = WanImageToVideoPipeline.from_pretrained(str(WEIGHTS), torch_dtype=torch.bfloat16)
    try:
        pipe.enable_sequential_cpu_offload()
    except Exception:
        pipe.to("cuda")

    free0, total = torch.cuda.mem_get_info()
    free0_gb = free0 / 1024 ** 3
    total_gb = total / 1024 ** 3
    print(f"[probe] total VRAM: {total_gb:.2f} GB, free before run: {free0_gb:.2f} GB", flush=True)

    torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    out = pipe(
        image=src,
        prompt="a person facing the camera, slight smile, eyes blinking slowly",
        negative_prompt="",
        height=704, width=1280,
        num_inference_steps=20, num_frames=17,
        guidance_scale=5.0,
        generator=torch.Generator(device="cpu").manual_seed(0),
    ).frames[0]
    elapsed = time.time() - t0

    peak_gb = torch.cuda.max_memory_allocated() / 1024 ** 3
    print(f"[probe] inference: {elapsed:.0f}s for {len(out)} frames @ 704x1280 / 20 steps",
          flush=True)
    print(f"[probe] peak VRAM allocated: {peak_gb:.2f} GB "
          f"(out of {total_gb:.2f} GB total, {total_gb - peak_gb:.2f} GB headroom)",
          flush=True)
    print(f"[probe] recommended vram_min_gb: {int(peak_gb) + 2}", flush=True)


if __name__ == "__main__":
    main()
