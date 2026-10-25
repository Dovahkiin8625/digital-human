"""The `lite` digital-human model: FeatherTalk, driven on a CPU.

FeatherTalk (https://github.com/anliyuan/FeatherTalk, Apache-2.0) is a
*personalised* audio-driven talking head: a small audio encoder turns speech
into features, a 5.5 M-parameter UNet repaints the face patch, and the patch
is pasted back into the source frame. The two halves together are ~9 M
parameters and ~19 MB of weights, against MuseTalk's 947 M -- which is what
makes a CPU-only serving box possible.

The pieces live here rather than in `server/engine/lite_engine.py` so that the
engine file stays a thin adapter over the `DriveEngine` contract:

    vendor.py     where the vendored checkout and the weights are
    geometry.py   face bbox / crop / paste-back  (numpy + cv2, no torch)
    landmarks.py  SCRFD + PFLD landmarks          (torch)
    audio.py      resample + FeatherHuBERT + the causal audio window
    visual.py     the visual UNet                 (torch)
    dataset.py    the training dataset and entry point
    pack.py       the on-disk pack layout

Nothing in this package imports torch at module level, so the torch-free
`.venv` can still import it (and the test suite still runs without a GPU).
"""
