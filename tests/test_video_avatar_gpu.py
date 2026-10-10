"""End-to-end video avatar against the real weights.

Needs the .venv-musetalk interpreter and a CUDA device. The driver video is
built from MuseTalk's own sample clip so it contains a detectable face.
"""
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
cv2 = pytest.importorskip("cv2")

from server.avatar.musetalk_pipeline import MuseTalkAvatarPipeline  # noqa: E402
from server.engine.musetalk_engine import MuseTalkEngine  # noqa: E402

pytestmark = [pytest.mark.gpu, pytest.mark.musetalk]

SAMPLE = Path(__file__).resolve().parents[1] / "vendor" / "MuseTalk" / "data" / "video" / "sun.mp4"


def _short_video(dst: Path, frames: int = 40) -> Path:
    cap = cv2.VideoCapture(str(SAMPLE))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))
    n = 0
    while n < frames:
        ok, f = cap.read()
        if not ok:
            break
        out.write(f)
        n += 1
    cap.release()
    out.release()
    return dst


@pytest.mark.skipif(not SAMPLE.exists(), reason="MuseTalk sample video not vendored")
def test_video_avatar_prepares_and_streams(tmp_path):
    # DH_DATA_DIR is deliberately NOT redirected: both the pipeline and the
    # engine resolve their weights from it, and only the output goes to tmp_path.
    out = tmp_path / "out"
    # The clip goes inside the pack root because the engine re-decodes
    # root/prep["source"] when loading a video avatar -- production's
    # AvatarStore.save_source places the upload in the avatar dir for the same
    # reason. The pipeline itself never copies the source.
    out.mkdir(parents=True, exist_ok=True)
    src = _short_video(out / "clip.mp4")
    cfg = MuseTalkAvatarPipeline().process("v1", src, out, lambda s, p: None)

    assert cfg["musetalk"]["mode"] == "video"
    assert cfg["musetalk"]["frame_count"] > 1
    assert (out / "musetalk" / "coords.json").exists()

    from server.engine.base import AvatarPack

    pack = AvatarPack(
        avatar_id="v1",
        root=out,
        source_image_path=out / "normalized.png",
        config=cfg,
        prep=cfg["musetalk"],
    )
    eng = MuseTalkEngine()
    import asyncio

    asyncio.run(eng.prepare(pack))
    state = eng._avatars["v1"]
    assert state.cycle_len == cfg["musetalk"]["frame_count"]
    if np.array_equal(state.frames[0], state.frames[1]):
        pytest.skip("sample clip's first two frames are identical")

    ctx = eng.new_context(pack)
    # One production-sized block: AudioIngest uses block_ms=100, i.e. 4800
    # samples at 48 kHz. The vendored get_whisper_chunk's padding arithmetic
    # cannot build a 10-wide whisper clip from a shorter block, and on that
    # assertion failure it calls exit() in the worker thread; a 50 ms block
    # is therefore not a legal drive call.
    frames = asyncio.run(eng.drive(pack, np.zeros(4800, np.float32), 48000, context=ctx))
    assert frames, "video avatar produced no frames"
    assert frames[0].image.shape == (1280, 720, 3)


@pytest.mark.skipif(not SAMPLE.exists(), reason="MuseTalk sample video not vendored")
def test_short_audio_session_does_not_kill_the_worker(tmp_path):
    out = tmp_path / "out"
    out.mkdir(parents=True, exist_ok=True)
    src = _short_video(out / "clip.mp4")
    cfg = MuseTalkAvatarPipeline().process("v2", src, out, lambda s, p: None)

    from server.engine.base import AvatarPack

    pack = AvatarPack(
        avatar_id="v2",
        root=out,
        source_image_path=out / "normalized.png",
        config=cfg,
        prep=cfg["musetalk"],
    )
    eng = MuseTalkEngine()
    import asyncio

    asyncio.run(eng.prepare(pack))
    ctx = eng.new_context(pack)
    # 50 ms: shorter than the vendor pad's full width. Before the fix
    # get_whisper_chunk exit()ed, the worker died, and this drive hung.
    frames = asyncio.run(eng.drive(pack, np.zeros(2400, np.float32), 48000, context=ctx))
    assert len(frames) == 1
    # The worker must still be alive: a following full block drives normally.
    more = asyncio.run(eng.drive(pack, np.zeros(4800, np.float32), 48000, context=ctx))
    assert len(more) == 2


@pytest.mark.skipif(not SAMPLE.exists(), reason="MuseTalk sample video not vendored")
def test_idle_phase_carries_seamlessly_into_speech(tmp_path):
    out = tmp_path / "out"
    out.mkdir(parents=True, exist_ok=True)
    src = _short_video(out / "clip.mp4")
    cfg = MuseTalkAvatarPipeline().process("v3", src, out, lambda s, p: None)

    from server.engine.base import AvatarPack

    pack = AvatarPack(
        avatar_id="v3",
        root=out,
        source_image_path=out / "normalized.png",
        config=cfg,
        prep=cfg["musetalk"],
    )
    eng = MuseTalkEngine()
    import asyncio

    asyncio.run(eng.prepare(pack))
    n = int(cfg["musetalk"]["frame_count"])

    # Simulate an idle track that reached position p (= the position the next
    # idle frame would use), as RenderSession does.
    p = 3
    idle_here = asyncio.run(eng.idle_frame(pack, None, p))
    ctx = eng.new_context(pack)
    ctx.phase = p
    frames = asyncio.run(eng.drive(pack, np.zeros(4800, np.float32), 48000, context=ctx))
    assert len(frames) >= 2
    # The first speech frame uses the same base frame the idle track would have
    # shown next: cycle position p. (MuseTalk edits only the mouth region.)
    from server.engine.musetalk_engine import cycle_index

    assert cycle_index(p, n) == cycle_index(ctx.phase + 0, n)
    # Outside the edited mouth box, idle-at-p and first speech frame agree.
    bbox = [int(v) for v in cfg["musetalk"]["bbox"]]
    x1, y1, x2, y2 = bbox
    rest = np.delete(np.delete(idle_here, slice(y1, y2), axis=0), slice(x1, x2), axis=1)
    got = np.delete(np.delete(frames[0].image, slice(y1, y2), axis=0), slice(x1, x2), axis=1)
    assert np.array_equal(rest, got)
    # The engine reports the position after the emitted speech frames.
    assert eng.idle_position(ctx) == ctx.phase + ctx.emitted
