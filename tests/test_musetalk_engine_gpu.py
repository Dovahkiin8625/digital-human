"""Device-side checks for the MuseTalk engine.

`blend_face` (numpy) is the reference -- it is verified against PIL's masked
paste in `test_musetalk_engine.py` without a GPU. `_blend_face_gpu` is what
`_render` actually runs, so it has to agree with that reference; the two are the
same arithmetic in different venues, and nothing else pins them together.

Skipped wherever torch is missing, which is the default `.venv`.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from server.engine import musetalk_engine as me  # noqa: E402


def _to_chw(a: np.ndarray):
    return torch.from_numpy(a).permute(2, 0, 1).contiguous()


@pytest.mark.parametrize("fill", [0, 1, 128, 254, 255])
def test_gpu_blend_matches_the_numpy_reference(fill):
    """A uniform mask isolates the lerp from the mask's own quantisation."""
    rng = np.random.default_rng(0)
    base = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    face = rng.integers(0, 256, (20, 30, 3), dtype=np.uint8)
    mask = np.full((40, 40), fill, np.uint8)
    bbox = [10, 20, 40, 40]
    crop_box = [5, 15, 45, 55]

    want = me.blend_face(base, face, bbox, mask, crop_box)
    got = (
        me._blend_face_gpu(
            _to_chw(base),
            _to_chw(face),
            bbox,
            torch.from_numpy(mask).to(torch.float32).div_(255.0).unsqueeze(0),
            crop_box,
        )
        .permute(1, 2, 0)
        .numpy()
    )

    assert got.shape == base.shape
    assert got.dtype == np.uint8
    assert np.abs(got.astype(np.int16) - want.astype(np.int16)).max() <= 1


def test_gpu_blend_leaves_the_rest_of_the_canvas_untouched():
    base = np.full((64, 64, 3), 40, np.uint8)
    face = np.full((20, 30, 3), 200, np.uint8)
    got = (
        me._blend_face_gpu(
            _to_chw(base),
            _to_chw(face),
            [10, 20, 40, 40],
            torch.zeros(1, 40, 40),
            [5, 15, 45, 55],
        )
        .permute(1, 2, 0)
        .numpy()
    )
    assert np.array_equal(got, base)  # a zero mask means no paste at all


def test_gpu_blend_does_not_write_through_to_its_inputs():
    """`_render` hands it the avatar's resident canvas, frame after frame."""
    base = np.full((32, 32, 3), 7, np.uint8)
    face = np.full((10, 10, 3), 200, np.uint8)
    base_t = _to_chw(base)
    before = base_t.clone()
    me._blend_face_gpu(
        base_t, _to_chw(face), [4, 4, 14, 14], torch.ones(1, 20, 20), [0, 0, 20, 20]
    )
    assert torch.equal(base_t, before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_gpu_blend_runs_on_the_device():
    base_t = _to_chw(np.full((32, 32, 3), 7, np.uint8)).cuda()
    face_t = _to_chw(np.full((10, 10, 3), 200, np.uint8)).cuda()
    mask_t = torch.ones(1, 20, 20, device="cuda")
    out = me._blend_face_gpu(base_t, face_t, [4, 4, 14, 14], mask_t, [0, 0, 20, 20])
    assert out.device == base_t.device
    assert out[0, 5, 5].item() == 200
