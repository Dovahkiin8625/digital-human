"""The causal dataset: the upstream temporal dataset with a causal window.

Kept in its own module because it has to be **picklable**: PyTorch's
DataLoader spawns worker processes on Windows, and a class defined inside a
function cannot be pickled back into the child.

Importing this module requires the vendored FeatherTalk checkout on
``sys.path`` (``server.lite.vendor.ensure_on_path()``), and it imports torch.
So it is imported only when a training run actually starts.
"""
from __future__ import annotations

from dataset_mouth_roi import TemporalMouthRoiDataset  # noqa: F401 -- needs the vendor

from server.lite.audio import AUDIO_TOKENS, CURRENT_TOKEN_START, FUTURE_TOKEN_START


def causal_fill(window):
    """Fill a window's future slots with the current frame's tokens.

    ``window`` is ``[40, 1024]`` (the raw-HuBERT layout). The current frame's
    two tokens sit at ``[20:22]``; the nine frames after it become copies of
    them, in the same interleaved order the serving path produces -- token
    *pairs*, not one token repeated nine times over.
    """
    import torch

    out = window.clone()
    current = out[..., CURRENT_TOKEN_START : CURRENT_TOKEN_START + 2, :]
    repeats = (AUDIO_TOKENS - FUTURE_TOKEN_START) // current.shape[0]
    out[..., FUTURE_TOKEN_START:, :] = current.repeat((repeats, 1))
    return out


class CausalTemporalMouthRoiDataset(TemporalMouthRoiDataset):
    """The upstream dataset, with the audio window made causal.

    Upstream's window spans ten video frames either side of the current one,
    so it reaches 360 ms into the future: rendering a live stream with it
    would show a mouth for audio that has not been spoken yet. Here the
    future slots carry the current frame's tokens -- exactly what
    :mod:`server.lite.audio` feeds the model at serving time, so training
    and inference see the same distribution.
    """

    def _build_frame(self, idx, ref_T=None):
        images, target, audio, masks = super()._build_frame(idx, ref_T)
        if self.raw_hubert_audio and audio.shape[-2] == AUDIO_TOKENS:
            audio = causal_fill(audio)
        return images, target, audio, masks
