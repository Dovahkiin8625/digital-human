"""Dump every key/shape of a RIFE checkpoint (safe tensor-only load)."""
import sys

import torch

path = sys.argv[1]
sd = torch.load(path, map_location="cpu", weights_only=True)
for k, v in sd.items():
    print(k, tuple(v.shape))
