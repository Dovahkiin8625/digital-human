"""The closed-mouth idle track.

A `wav2lip` pack's frames come from a video of the person *speaking* --
that video is the motion the avatar plays back. Replaying it verbatim while
the session is silent makes the avatar talk to nobody, which is what this
track exists to prevent.

So the build paints a second set of frames: the same head and body motion,
repainted by the model with a **silent** mel window. The model closes the
mouth when it hears nothing, so the idle avatar keeps moving without moving
its lips -- and because it is the same frames in the same order, the mouth
opening later continues from the very same position, with no jump at either
boundary.

The track is baked at build time, on the machine that holds the weights: the
runtime only decodes a JPEG, like any other frame, so an idle session costs
nothing beyond the decode.

Nothing here imports torch at module level: the torch-free ``.venv`` must
still be able to import this module.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from dh_server.wav2lip import frames as wav2lip_frames
from dh_server.wav2lip import geometry
from dh_server.wav2lip import pack as wav2lip_pack
from dh_server.wav2lip.audio import MEL_WINDOW, NUM_MELS, SILENCE


def silence_window() -> np.ndarray:
    """The ``[16, 80]`` mel window a silent stream produces.

    Silence has no past and no future, so every slot of the window carries
    the same value -- the mel of silence, which is also what the engine
    feeds the model when the session has nothing to say.
    """
    return np.full((MEL_WINDOW, NUM_MELS), SILENCE, dtype=np.float32)


def bake_idle_frames(
    root: Path,
    coords: list[dict],
    model,
    *,
    batch_size: int = 8,
    progress=None,
) -> int:
    """Paint the idle track and write it into the pack.

    ``root`` is the avatar directory, ``coords`` the pack's face boxes (the
    same boxes the motion frames use), ``model`` the loaded generator and
    ``window`` the silent mel window. Returns the number of frames written.
    """
    import cv2

    names = sorted(wav2lip_pack.frames_dir(root).glob("*.jpg"))
    if not names:
        raise ValueError(f"wav2lip pack {root} has no frames to bake an idle track from")
    if len(coords) < len(names):
        raise ValueError(
            f"wav2lip pack {root} is inconsistent: {len(names)} frames but "
            f"{len(coords)} face boxes"
        )
    wav2lip_pack.idle_frames_dir(root).mkdir(parents=True, exist_ok=True)
    window = silence_window()

    written = 0
    for start in range(0, len(names), batch_size):
        batch = names[start : start + batch_size]
        sources: list[np.ndarray] = []
        crops: list[np.ndarray] = []
        boxes: list[tuple[int, int, int, int]] = []
        for offset, name in enumerate(batch):
            index = start + offset
            source = cv2.imread(str(name))
            if source is None:
                raise ValueError(f"wav2lip frame {name} could not be decoded")
            box = tuple(int(v) for v in coords[index]["bbox"])
            crop = geometry.crop_face(source, box)
            sources.append(source)
            crops.append(crop)
            boxes.append(box)

        predictions = model.predict(
            np.stack(crops), np.stack([window] * len(batch))
        )
        for offset in range(len(batch)):
            frame = sources[offset].copy()
            patch = (predictions[offset] * 255.0).round().clip(0, 255).astype(np.uint8)
            geometry.paste_prediction(frame, patch, boxes[offset])
            wav2lip_frames.write_jpeg(
                wav2lip_pack.idle_frame_path(root, start + offset), frame
            )
            written += 1
        if progress is not None:
            progress(written, len(names))
    return written
