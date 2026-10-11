import asyncio
import json
from fractions import Fraction

import numpy as np
from aiortc import (
    AudioStreamTrack,
    MediaStreamTrack,
    RTCPeerConnection,
    RTCSessionDescription,
    VideoStreamTrack,
)
from aiortc.mediastreams import VIDEO_CLOCK_RATE
from aiortc.sdp import candidate_from_sdp
from av import AudioFrame, VideoFrame
from av.audio.resampler import AudioResampler
from starlette.websockets import WebSocket, WebSocketDisconnect

from server.core.errors import AppError
from server.core.stream_sink import NullSink, SinkClosed, StreamSink


class EngineVideoTrack(VideoStreamTrack):
    def __init__(self, maxsize: int = 8):
        super().__init__()
        self._queue: asyncio.Queue[tuple[float, np.ndarray]] = asyncio.Queue(maxsize=maxsize)
        self.overflows = 0

    async def push(self, image: np.ndarray, pts: float) -> None:
        try:
            self._queue.put_nowait((pts, image))
        except asyncio.QueueFull:
            self.overflows += 1
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            self._queue.put_nowait((pts, image))

    async def recv(self) -> VideoFrame:
        pts, image = await self._queue.get()
        # Hand out the oldest surviving frame, not the freshest. push() already
        # bounds the queue and drops the oldest on overflow, so staleness is
        # capped by maxsize; draining the backlog here as well meant a consumer
        # that was keeping up still lost every frame but the last of each burst.
        # The engine emits a block's 2-3 frames at once, so that turned a 25 fps
        # session into ~8 fps of delivered video.
        frame = VideoFrame.from_ndarray(image, format="rgb24")
        frame.pts = int(pts * VIDEO_CLOCK_RATE)
        frame.time_base = Fraction(1, VIDEO_CLOCK_RATE)
        return frame


#: Opus at 48 kHz ships 20 ms / 960-sample frames; the encoder resamples to
#: that size (opus.py SAMPLES_PER_FRAME), so feed exactly that.
AUDIO_CLOCK_RATE = 48000
AUDIO_FRAME_SAMPLES = 960


class EngineAudioTrack(AudioStreamTrack):
    """Outbound audio mirror of EngineVideoTrack.

    Queued chunks carry pts on the session timeline (the same seconds-axis the
    video frames use, expressed in 48 kHz samples), so RTP timestamps and RTCP
    sender reports line up with video and the browser plays both as one
    lip-synced MediaStream.
    """

    def __init__(self, maxsize: int = 3000):
        super().__init__()
        self._queue: asyncio.Queue[tuple[int, np.ndarray]] = asyncio.Queue(
            maxsize=maxsize
        )
        self.overflows = 0

    async def push(self, pcm: np.ndarray, pts: int) -> None:
        try:
            self._queue.put_nowait((pts, pcm))
        except asyncio.QueueFull:
            self.overflows += 1
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            self._queue.put_nowait((pts, pcm))

    async def recv(self) -> AudioFrame:
        pts, pcm = await self._queue.get()
        frame = AudioFrame.from_ndarray(pcm[None, :], format="s16", layout="mono")
        frame.sample_rate = AUDIO_CLOCK_RATE
        frame.pts = pts
        frame.time_base = Fraction(1, AUDIO_CLOCK_RATE)
        return frame


class WebRTCSink(StreamSink):
    keepalive = True

    def __init__(self, video_track: EngineVideoTrack, audio_track: EngineAudioTrack):
        self.video_track = video_track
        self.audio_track = audio_track
        # Flipped off by the signaling handler when the answer negotiated no
        # outbound audio (an old sendonly console): nothing would drain the
        # audio queue then, so it must not be fed.
        self.wants_audio = True
        self.closed = False

    @property
    def overflows(self) -> int:
        return self.video_track.overflows + self.audio_track.overflows

    async def push(self, image: np.ndarray, pts: float) -> None:
        if self.closed:
            raise SinkClosed("webrtc sink closed")
        await self.video_track.push(image, pts)

    async def push_audio(
        self, samples: np.ndarray, pts: float, sample_rate: int
    ) -> None:
        if self.closed:
            raise SinkClosed("webrtc sink closed")
        if not self.wants_audio or sample_rate != AUDIO_CLOCK_RATE:
            return
        pcm = samples
        if pcm.dtype != np.int16:
            pcm = (np.clip(pcm.astype(np.float32), -1.0, 1.0) * 32767).astype(np.int16)
        start = int(round(pts * AUDIO_CLOCK_RATE))
        for off in range(0, len(pcm), AUDIO_FRAME_SAMPLES):
            chunk = pcm[off : off + AUDIO_FRAME_SAMPLES]
            if len(chunk) < AUDIO_FRAME_SAMPLES:
                # Flush tail (a short final block): pad with silence so the last
                # words are encoded now instead of waiting for more samples.
                chunk = np.concatenate(
                    (chunk, np.zeros(AUDIO_FRAME_SAMPLES - len(chunk), dtype=np.int16))
                )
            await self.audio_track.push(chunk, start + off)

    async def close(self) -> None:
        self.closed = True


async def consume_audio_track(track: MediaStreamTrack, push_pcm) -> None:
    """Drain an inbound RTP audio track into the session's PCM ingest.

    Resamples non-48k/mono frames to s16 mono 48kHz, then hands raw PCM bytes to
    push_pcm. Any failure (track ended, transport torn down, bad frame) exits
    silently so a dead audio track never crashes its owning task.
    """
    resampler = None
    try:
        while True:
            frame = await track.recv()
            if frame.sample_rate != 48000 or frame.layout.name != "mono":
                if resampler is None:
                    resampler = AudioResampler(format="s16", layout="mono", rate=48000)
                frames = resampler.resample(frame)
            else:
                frames = [frame]
            for f in frames:
                arr = f.to_ndarray()
                if arr.dtype != np.int16:
                    arr = (np.clip(arr.astype(np.float32), -1, 1) * 32767).astype(np.int16)
                await push_pcm(arr.tobytes())
    except Exception:
        return


async def handle_signal_websocket(websocket: WebSocket, manager, session_id: str) -> None:
    await websocket.accept()
    try:
        render = manager.get_render(session_id)
    except Exception:
        await websocket.close(code=4404)
        return

    video_track = EngineVideoTrack()
    audio_track = EngineAudioTrack()
    sink = WebRTCSink(video_track, audio_track)
    manager.attach_sink(session_id, sink)
    pc = RTCPeerConnection()
    pc.addTrack(video_track)
    pc.addTrack(audio_track)

    @pc.on("track")
    def on_track(inbound):
        if inbound.kind == "audio":
            asyncio.ensure_future(consume_audio_track(inbound, render.push_pcm))

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            if msg.get("type") == "offer":
                await pc.setRemoteDescription(RTCSessionDescription(sdp=msg["sdp"], type="offer"))
                answer = await pc.createAnswer()
                await pc.setLocalDescription(answer)
                # Only feed the audio track when the answer actually SENDS audio
                # (console offered audio recvonly/sendrecv). A legacy sendonly
                # console yields a recvonly answer: its sender never runs and
                # would leave every pushed frame queued forever.
                for transceiver in pc.getTransceivers():
                    if transceiver.kind == "audio":
                        sink.wants_audio = transceiver.currentDirection in (
                            "sendonly",
                            "sendrecv",
                        )
                await websocket.send_text(json.dumps({"type": "answer", "sdp": pc.localDescription.sdp}))
            elif msg.get("type") == "candidate":
                cand = msg.get("candidate") or {}
                if cand.get("candidate"):
                    ice = candidate_from_sdp(cand["candidate"])
                    ice.sdpMid = cand.get("sdpMid")
                    ice.sdpMLineIndex = cand.get("sdpMLineIndex")
                    await pc.addIceCandidate(ice)
            elif msg.get("type") == "bye":
                break
    finally:
        # Teardown must be order-independent: the session may have been swept or
        # deleted mid-connection, making attach_sink raise. Never let that skip the
        # pc/websocket close below (an unclosed RTCPeerConnection leaks a UDP socket
        # and internal asyncio tasks).
        try:
            await sink.close()
            try:
                manager.attach_sink(session_id, NullSink())
            except AppError:
                pass
        finally:
            await pc.close()
            # Starlette does not auto-close when the endpoint returns; close explicitly
            # so signaling clients observe a deterministic teardown after "bye". A
            # client that already disconnected (or whose close was already sent) makes
            # this raise WebSocketDisconnect/RuntimeError — that is the normal
            # disconnect path, not an error, so it must not escape as an ASGI traceback.
            try:
                await websocket.close()
            except (WebSocketDisconnect, RuntimeError):
                pass
