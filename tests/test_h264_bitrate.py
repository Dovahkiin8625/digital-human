"""aiortc ships OpenH264 with a 1 Mbps default and a hard 3 Mbps ceiling.

At 720x1280@25 that turns the mouth region -- the part of the frame that moves
most -- into blocky H.264 noise, so the startup wiring raises both. aiortc has
no per-sender bitrate parameter: the H264Encoder reads module globals at
construction and the REMB setter clamps to them at runtime, so patching the
module constants is the only seam.
"""
import pytest


def test_configure_h264_bitrate_raises_default_and_ceiling(monkeypatch):
    import aiortc.codecs.h264 as h264

    monkeypatch.setattr(h264, "DEFAULT_BITRATE", h264.DEFAULT_BITRATE)
    monkeypatch.setattr(h264, "MAX_BITRATE", h264.MAX_BITRATE)

    from server.core.webrtc import configure_h264_bitrate

    configure_h264_bitrate(3_500_000, 4_000_000)

    assert h264.DEFAULT_BITRATE == 3_500_000
    assert h264.MAX_BITRATE == 4_000_000


def test_configure_h264_bitrate_ceiling_never_below_target(monkeypatch):
    import aiortc.codecs.h264 as h264

    monkeypatch.setattr(h264, "DEFAULT_BITRATE", h264.DEFAULT_BITRATE)
    monkeypatch.setattr(h264, "MAX_BITRATE", h264.MAX_BITRATE)

    from server.core.webrtc import configure_h264_bitrate

    configure_h264_bitrate(4_000_000, 1_000_000)

    assert h264.DEFAULT_BITRATE == 4_000_000
    # A ceiling below the start bitrate would clamp REMB feedback straight back
    # down; the call itself fixes that misconfiguration.
    assert h264.MAX_BITRATE == 4_000_000


def test_settings_default_is_well_above_aiortc_default():
    from server.config import Settings

    s = Settings()
    assert s.video_bitrate >= 3_000_000
    assert s.video_max_bitrate >= s.video_bitrate
