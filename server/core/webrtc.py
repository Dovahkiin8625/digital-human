import asyncio
import json
from fractions import Fraction

import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.mediastreams import VIDEO_CLOCK_RATE
from aiortc.sdp import candidate_from_sdp
from av import VideoFrame
from starlette.websockets import WebSocket

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
        # Live track: drop any backlog and emit the freshest frame so a slow
        # consumer never renders stale video (push() already bounds the queue).
        while not self._queue.empty():
            pts, image = self._queue.get_nowait()
        frame = VideoFrame.from_ndarray(image, format="rgb24")
        frame.pts = int(pts * VIDEO_CLOCK_RATE)
        frame.time_base = Fraction(1, VIDEO_CLOCK_RATE)
        return frame


class WebRTCSink(StreamSink):
    def __init__(self, track: EngineVideoTrack):
        self.track = track
        self.closed = False

    async def push(self, image: np.ndarray, pts: float) -> None:
        if self.closed:
            raise SinkClosed("webrtc sink closed")
        await self.track.push(image, pts)

    async def close(self) -> None:
        self.closed = True


async def handle_signal_websocket(websocket: WebSocket, manager, session_id: str) -> None:
    await websocket.accept()
    try:
        render = manager.get_render(session_id)
    except Exception:
        await websocket.close(code=4404)
        return

    track = EngineVideoTrack()
    sink = WebRTCSink(track)
    manager.attach_sink(session_id, sink)
    pc = RTCPeerConnection()
    pc.addTrack(track)

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            if msg.get("type") == "offer":
                await pc.setRemoteDescription(RTCSessionDescription(sdp=msg["sdp"], type="offer"))
                answer = await pc.createAnswer()
                await pc.setLocalDescription(answer)
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
        await sink.close()
        manager.attach_sink(session_id, NullSink())
        await pc.close()
        # Starlette does not auto-close when the endpoint returns; close explicitly
        # so signaling clients observe a deterministic teardown after "bye".
        await websocket.close()
