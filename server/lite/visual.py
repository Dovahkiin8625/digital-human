"""The visual half of the `lite` model: FeatherTalk's face UNet.

5.46 M parameters, 144x144 in, 144x144 out. It takes the reference face and
the masked face (6 channels) plus 40 audio tokens, and paints the face patch
that gets pasted back into the source frame.

The checkpoint is **personalised**: one per person, trained from a few
minutes of that person speaking. `load_visual_model` in the vendored module
tells the two architectures apart (the current `feathertalk` model and the
older `mobileone` one) from the weights themselves, so the pack does not have
to record which one it holds.

Torch is imported lazily, so this module imports in the torch-free `.venv`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from server.lite import vendor
from server.lite.audio import AUDIO_DIM, AUDIO_TOKENS

#: The square the network works on.
IMAGE_SIZE = 144


class VisualModel:
    """A loaded FeatherTalk visual model, ready to run on CPU."""

    def __init__(self, model, device: str = "cpu"):
        self._model = model
        self._device = device
        self.model_type = getattr(model, "model_type", "feathertalk")
        self.image_size = int(getattr(model, "image_size", IMAGE_SIZE))
        self.audio_half_window = int(getattr(model, "audio_half_window", 10))
        self.raw_hubert_audio = bool(getattr(model, "raw_hubert_audio", True))

    @classmethod
    def load(cls, checkpoint: Path, device: str = "cpu") -> "VisualModel":
        """Load a personalised checkpoint."""
        import torch

        vendor.ensure_on_path()
        from model import load_visual_model

        path = Path(checkpoint)
        if not path.exists():
            raise FileNotFoundError(f"visual checkpoint not found: {path}")
        model = load_visual_model(str(path), torch.device(device), "auto", reparameterize=True)
        return cls(model, device)

    def predict(self, image: np.ndarray, audio: np.ndarray) -> np.ndarray:
        """Paint one face patch.

        ``image`` is ``[6, H, W]`` float32 in ``[0, 1]`` (reference + masked);
        ``audio`` is ``[40, 1024]`` float32. The result is ``[H, W, 3]`` uint8.
        """
        return self.predict_batch(image[None], audio[None])[0]

    def predict_batch(
        self, images: np.ndarray, audios: np.ndarray
    ) -> list[np.ndarray]:
        """Paint several face patches in one forward pass.

        A 100 ms block is two or three frames; batching them is worth about
        15 % of the model's time, and the engine needs them together anyway
        to paste them back in order.
        """
        import torch

        if images.ndim != 4 or images.shape[1] != 6:
            raise ValueError(f"images must have shape [N, 6, H, W], got {images.shape}")
        if audios.shape != (images.shape[0], AUDIO_TOKENS, AUDIO_DIM):
            raise ValueError(
                f"audios must have shape {(images.shape[0], AUDIO_TOKENS, AUDIO_DIM)}, "
                f"got {audios.shape}"
            )
        with torch.no_grad():
            out = self._model(
                torch.from_numpy(np.ascontiguousarray(images)).to(self._device),
                torch.from_numpy(np.ascontiguousarray(audios)).to(self._device),
            )
        frames = (out.cpu().numpy().transpose(0, 2, 3, 1) * 255.0).clip(0, 255)
        return [frame.astype(np.uint8) for frame in frames]
