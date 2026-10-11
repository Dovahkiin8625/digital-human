"""Inspect a RIFE checkpoint's architecture (key names/shapes)."""
import re
import sys

import torch

path = sys.argv[1] if len(sys.argv) > 1 else "scratch/rife49.pth"
# weights_only: never unpickle arbitrary code from a third-party download --
# checkpoints here are plain tensor dicts, so the restricted loader suffices.
sd = torch.load(path, map_location="cpu", weights_only=True)
keys = list(sd.keys())
print("n keys:", len(keys))
for k in keys[:10]:
    print(k, tuple(sd[k].shape))
print("...")
for k in keys[-4:]:
    print(k, tuple(sd[k].shape))
blocks = sorted({m.group(1) for k in keys for m in [re.match(r"(?:module\.)?(block\d+)", k)] if m})
print("blocks:", blocks)
print("has convblock0:", any("convblock0" in k for k in keys))
print("has resnet:", any("resnet" in k for k in keys))
