"""Real WebRTC loopback for the audio-return path.

The unit tests prove chunking, pts and negotiation in isolation; this one
connects two aiortc peer connections over localhost UDP (signaling wired the
same way handle_signal_websocket does it) and proves the driving audio actually
crosses the same peer connection as video, encoded as Opus, with its content
intact -- the end-to-end half of the lip-sync contract.
"""

import asyncio
import io

import numpy as np
from PIL import Image


def _seed(client):
    from dh_server.api.deps import hash_api_key
    from dh_server.db.models import Tenant
    from dh_server.db.session import get_engine
    from sqlalchemy.orm import sessionmaker

    db = sessionmaker(bind=get_engine())()
    db.add(Tenant(name="t1", api_key_hash=hash_api_key("key-1")))
    db.commit()
    db.close()
    buf = io.BytesIO()
    Image.new("RGB", (640, 800), (1, 2, 3)).save(buf, format="PNG")
    avatar_id = client.post(
        "/api/v1/avatars",
        headers={"X-API-Key": "key-1"},
        data={"name": "a"},
        files={"file": ("a.png", buf.getvalue(), "image/png")},
    ).json()["avatar_id"]
    return client.post(
        "/api/v1/sessions",
        headers={"X-API-Key": "key-1"},
        json={"avatar_id": avatar_id},
    ).json()["session_id"]


async def _drain(track, bucket):
    while True:
        bucket.append(await track.recv())


async def _wait_for(predicate, timeout=10.0):
    deadline = asyncio.get_event_loop().time() + timeout
    while not predicate():
        if asyncio.get_event_loop().time() > deadline:
            raise AssertionError("timed out waiting for loopback media")
        await asyncio.sleep(0.05)


def test_driving_audio_returns_over_the_webrtc_peer_connection(client):
    sid = _seed(client)

    async def scenario():
        from aiortc import RTCPeerConnection, RTCSessionDescription
        from dh_server.core.stream_sink import NullSink
        from dh_server.core.webrtc import EngineAudioTrack, EngineVideoTrack, WebRTCSink

        manager = client.app.state.session_manager
        render = manager.get_render(sid)

        video_track, audio_track = EngineVideoTrack(), EngineAudioTrack()
        sink = WebRTCSink(video_track, audio_track)
        manager.attach_sink(sid, sink)

        server_pc, client_pc = RTCPeerConnection(), RTCPeerConnection()
        server_pc.addTrack(video_track)
        server_pc.addTrack(audio_track)
        client_pc.addTransceiver("video", direction="recvonly")
        client_pc.addTransceiver("audio", direction="recvonly")

        got_tracks: dict[str, object] = {}

        @client_pc.on("track")
        def _on(track):
            got_tracks[track.kind] = track

        audio_frames: list = []
        video_frames: list = []
        drains: list = []
        try:
            offer = await client_pc.createOffer()
            await client_pc.setLocalDescription(offer)
            await server_pc.setRemoteDescription(
                RTCSessionDescription(sdp=offer.sdp, type="offer")
            )
            answer = await server_pc.createAnswer()
            await server_pc.setLocalDescription(answer)
            await client_pc.setRemoteDescription(
                RTCSessionDescription(sdp=answer.sdp, type="answer")
            )
            # Mirror the signaling handler's direction gate.
            for t in server_pc.getTransceivers():
                if t.kind == "audio":
                    sink.wants_audio = t.currentDirection in ("sendonly", "sendrecv")
            assert sink.wants_audio is True

            await _wait_for(lambda: client_pc.iceConnectionState in ("completed", "connected"))
            await _wait_for(lambda: {"audio", "video"} <= got_tracks.keys())

            drains.append(asyncio.ensure_future(_drain(got_tracks["audio"], audio_frames)))
            drains.append(asyncio.ensure_future(_drain(got_tracks["video"], video_frames)))

            # 1 s of loud 300 Hz sine, fed through the same path the audio WS uses.
            n = 48000
            sine = (0.5 * np.sin(2 * np.pi * 300 * np.arange(n) / n) * 32767).astype(np.int16)
            await render.push_pcm(sine.tobytes())
            await render.flush()

            # 50 Opus frames = 1 s of returned audio; RTP must actually carry it.
            await _wait_for(lambda: len(audio_frames) >= 50)
            assert len(video_frames) > 0
            pcm = np.concatenate([f.to_ndarray().reshape(-1) for f in audio_frames[:50]])
            # Decoder is s16 stereo; mean |x| of a 0.5-amplitude sine is ~10400.
            assert np.abs(pcm).mean() > 5000
        finally:
            for task in drains:
                task.cancel()
            await client_pc.close()
            await server_pc.close()
            await sink.close()
            manager.attach_sink(sid, NullSink())

    client.portal.call(scenario)
