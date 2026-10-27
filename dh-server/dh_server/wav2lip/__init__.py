"""The `wav2lip` digital-human model: Wav2Lip, driven on a CPU.

Wav2Lip (https://github.com/Rudrabha/Wav2Lip, MIT) is a *generic* lip-sync
model: a convolutional network reads a 96x96 crop of the face and a 16-frame
mel spectrogram, and paints the mouth onto the face. One checkpoint drives
every avatar -- nothing is trained per person, and nothing is uploaded but a
video of the person whose face the avatar wears.

The pieces live here rather than in `dh_server/engine/wav2lip_engine.py` so that
the engine file stays a thin adapter over the `DriveEngine` contract:

    vendor.py     where the vendored checkout and the weights are
    audio.py      resample + the streaming mel spectrogram
    geometry.py   face crop / model input / paste-back  (numpy + cv2, no torch)
    frames.py     decoding and letterboxing             (numpy + cv2)
    faces.py      the S3FD detector and the box geometry (torch)
    model.py      the generator                          (torch)
    idle.py       the closed-mouth idle track            (numpy + cv2)
    pack.py       the on-disk pack layout

Nothing in this package imports torch at module level, so the torch-free
`.venv` can still import it (and the test suite still runs without a GPU).
"""
