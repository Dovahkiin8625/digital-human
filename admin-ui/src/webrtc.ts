export type SignalMessage =
  | { type: "offer"; sdp: string }
  | { type: "answer"; sdp: string }
  | { type: "candidate"; candidate: RTCIceCandidateInit | null }
  | { type: "bye" };

/**
 * Drives one render session: a WebRTC peer connection for the video preview plus
 * a raw-PCM audio WebSocket for the driving signal.
 *
 * Audio reaches the render session two ways: over the sendonly audio RTP
 * transceiver (microphone) and over the audio WebSocket (file / sine bytes).
 * Both feed the same server-side ingest.
 */
export class DigitalHumanClient {
  private pc: RTCPeerConnection | null = null;
  private ws: WebSocket | null = null;
  private audioWs: WebSocket | null = null;
  private audioTransceiver: RTCRtpTransceiver | null = null;

  constructor(
    private signalUrl: string,
    private audioUrl: string,
    // Bounded so a handshake the server never completes surfaces as an error
    // instead of an indefinite spinner. A WebSocket stuck in CONNECTING fires
    // no event at all, so nothing else would ever settle the promise.
    private openTimeoutMs = 10_000
  ) {}

  /** Resolve once `ws` is open; reject on error, early close, or timeout. */
  private awaitOpen(ws: WebSocket): Promise<void> {
    return new Promise<void>((resolve, reject) => {
      if (ws.readyState === WebSocket.OPEN) {
        resolve();
        return;
      }
      const timer = setTimeout(
        () => reject(new Error(`signal socket did not open within ${this.openTimeoutMs}ms`)),
        this.openTimeoutMs
      );
      const settle = (finish: () => void) => () => {
        clearTimeout(timer);
        ws.onopen = ws.onerror = ws.onclose = null;
        finish();
      };
      ws.onopen = settle(resolve);
      ws.onerror = settle(() => reject(new Error("signal socket failed to open")));
      // A refused upgrade (an HTTP response instead of a 101) can surface as a
      // close with no error event.
      ws.onclose = settle(() => reject(new Error("signal socket closed before it opened")));
    });
  }

  async connect(video: HTMLVideoElement): Promise<void> {
    const ws = new WebSocket(this.signalUrl);
    this.ws = ws;
    this.audioWs = new WebSocket(this.audioUrl);
    try {
      await this.awaitOpen(ws);
    } catch (err) {
      // The audio socket is opened up front and would otherwise be left behind.
      this.close();
      throw err;
    }

    this.pc = new RTCPeerConnection();
    this.pc.ontrack = (ev) => {
      video.srcObject = ev.streams[0];
    };
    this.pc.onicecandidate = (ev) => {
      ws.send(JSON.stringify({ type: "candidate", candidate: ev.candidate }));
    };
    ws.onmessage = async (ev) => {
      const msg = JSON.parse(ev.data as string) as SignalMessage;
      if (msg.type === "answer") {
        await this.pc!.setRemoteDescription({ type: "answer", sdp: msg.sdp });
      }
    };

    // Transceivers must be added BEFORE createOffer so the offer carries both
    // m-lines (video recvonly, audio sendonly). The server attaches its video
    // track before answering, so the answer always has a video m-line.
    this.pc.addTransceiver("video", { direction: "recvonly" });
    this.audioTransceiver = this.pc.addTransceiver("audio", { direction: "sendonly" });

    const offer = await this.pc.createOffer();
    await this.pc.setLocalDescription(offer);
    ws.send(JSON.stringify({ type: "offer", sdp: offer.sdp }));
  }

  async attachMicrophone(): Promise<void> {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const track = stream.getAudioTracks()[0];
    // Reuse the sendonly audio transceiver from connect(): adding the whole
    // stream with addTrack would negotiate a second, conflicting audio m-line.
    if (this.audioTransceiver && track) {
      await this.audioTransceiver.sender.replaceTrack(track);
    }
  }

  sendAudio(pcm16: ArrayBuffer): void {
    if (this.audioWs && this.audioWs.readyState === WebSocket.OPEN) {
      this.audioWs.send(pcm16);
    }
  }

  async sendAudioFile(file: File): Promise<void> {
    await this.streamAudio(file);
  }

  /** Same path as an uploaded file, for audio produced in-page (e.g. TTS). */
  async sendAudioBlob(blob: Blob): Promise<void> {
    await this.streamAudio(blob);
  }

  private async streamAudio(source: Blob): Promise<void> {
    const ctx = new AudioContext();
    try {
      const buf = await ctx.decodeAudioData(await source.arrayBuffer());
      const pcm = resampleToInt16(buf, 48000);
      const chunk = 4800 * 2;
      for (let i = 0; i < pcm.byteLength; i += chunk) {
        this.sendAudio(pcm.slice(i, i + chunk));
        await new Promise((r) => setTimeout(r, 10));
      }
      this.flush();
    } finally {
      // decodeAudioData owns the context; release it or every upload leaks one.
      await ctx.close();
    }
  }

  sendSine(durationSec = 2, freq = 400): void {
    const n = Math.round(48000 * durationSec);
    const pcm = new Int16Array(n);
    for (let i = 0; i < n; i++) {
      pcm[i] = Math.round(Math.sin((2 * Math.PI * freq * i) / 48000) * 12000);
    }
    const chunk = 4800;
    for (let i = 0; i < n; i += chunk) {
      this.sendAudio(pcm.buffer.slice(i * 2, (i + chunk) * 2) as ArrayBuffer);
    }
    this.flush();
  }

  close(): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: "bye" }));
    }
    this.ws?.close();
    this.audioWs?.close();
    this.pc?.close();
    this.ws = null;
    this.audioWs = null;
    this.pc = null;
    this.audioTransceiver = null;
  }

  private flush(): void {
    if (this.audioWs && this.audioWs.readyState === WebSocket.OPEN) {
      this.audioWs.send(JSON.stringify({ type: "flush" }));
    }
  }
}

function resampleToInt16(audio: AudioBuffer, rate: number): ArrayBuffer {
  const ch = audio.getChannelData(0);
  const ratio = audio.sampleRate / rate;
  const out = new Int16Array(Math.floor(ch.length / ratio));
  for (let i = 0; i < out.length; i++) {
    out[i] = Math.round(Math.max(-1, Math.min(1, ch[Math.floor(i * ratio)])) * 32767);
  }
  return out.buffer as ArrayBuffer;
}
