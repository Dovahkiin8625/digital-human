"""The Wav2Lip generator: the trained lip-sync network.

One checkpoint drives every avatar -- Wav2Lip is a *generic* model, not a
personalised one. The weights are the upstream release; the network itself
comes from the vendored checkout, so the released checkpoint and the code that
reads it cannot drift apart.

    36.3 M parameters, one 96x96 face crop and one 16-frame mel window in,
    one 96x96 face out.

Torch is imported lazily, so this module imports in the torch-free `.venv`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from dh_server.wav2lip.audio import MEL_WINDOW, NUM_MELS
from dh_server.wav2lip.geometry import FACE_SIZE, build_model_input


def pick_device() -> str:
    """Where the model runs: the GPU when the box has one, else the CPU.

    Wav2Lip is small enough to serve on a CPU, but a GPU makes the build
    (a face detection per source frame) and the drive (one forward pass per
    output frame) an order of magnitude quicker. The frames themselves stay
    on the CPU either way: they are decoded, cropped and pasted there, and
    only the network runs on the device.
    """
    from dh_server.engine.base import gpu_available

    return "cuda" if gpu_available() else "cpu"


class Wav2LipModel:
    """The upstream generator, loaded from the released checkpoint."""

    def __init__(self, checkpoint: Path, device: str = "cpu"):
        import torch

        from dh_server.wav2lip import vendor

        module = vendor.import_vendor("models.wav2lip")
        self._torch = torch
        net = module.Wav2Lip()
        # The checkpoint is a plain dict of tensors; `weights_only=True` is
        # the safe way to read a file from outside this repository.
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"]
        net.load_state_dict({k.replace("module.", ""): v for k, v in state.items()})
        self._net = net.to(device).eval()
        self._device = device
        # One throwaway forward pass. On a GPU this builds the device context
        # and the kernels, which takes seconds; paying it here means the first
        # session's first frame is not the one that waits for it.
        with torch.no_grad():
            self._net(
                torch.zeros(1, 1, NUM_MELS, MEL_WINDOW, device=device),
                torch.zeros(1, 6, FACE_SIZE, FACE_SIZE, device=device),
            )

    @property
    def device(self) -> str:
        return self._device

    def predict(self, faces: np.ndarray, mels: np.ndarray) -> np.ndarray:
        """Render a batch of faces.

        ``faces`` is ``[B, 96, 96, 3]`` uint8 (BGR) and ``mels`` is
        ``[B, 16, 80]`` float32. The result is ``[B, 96, 96, 3]`` float32 in
        ``[0, 1]``, in the same channel order as the input.
        """
        torch = self._torch
        check_shapes(faces, mels)
        if faces.shape[0] != mels.shape[0]:
            raise ValueError(
                f"faces and mels disagree: {faces.shape[0]} faces, {mels.shape[0]} mels"
            )
        if faces.shape[0] == 0:
            return np.zeros((0, FACE_SIZE, FACE_SIZE, 3), dtype=np.float32)

        inputs = np.stack([build_model_input(face) for face in faces])
        face_tensor = torch.from_numpy(inputs).to(self._device)
        # The network wants the mel as [B, 1, bins, time].
        mel_tensor = (
            torch.from_numpy(mels.astype(np.float32))
            .permute(0, 2, 1)[:, None]
            .to(self._device)
        )
        with torch.no_grad():
            out = self._net(mel_tensor, face_tensor)
        return out.permute(0, 2, 3, 1).contiguous().cpu().numpy()


def check_shapes(faces: np.ndarray, mels: np.ndarray) -> None:
    """Raise when a batch does not have the shape the network expects.

    A pure check, so the contract can be asserted in the torch-free suite.
    """
    if faces.ndim != 4 or faces.shape[1:] != (FACE_SIZE, FACE_SIZE, 3):
        raise ValueError(
            f"faces must be [B, {FACE_SIZE}, {FACE_SIZE}, 3], got {faces.shape}"
        )
    if mels.ndim != 3 or mels.shape[1:] != (MEL_WINDOW, NUM_MELS):
        raise ValueError(
            f"mels must be [B, {MEL_WINDOW}, {NUM_MELS}], got {mels.shape}"
        )
