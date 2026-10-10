import { useEffect, useRef, useState } from "react";
import { api, getToken } from "../api";
import { ensureDemoKey } from "../demoKey";
import type { AvatarInfo, SessionInfo } from "../types";
import { DigitalHumanClient } from "../webrtc";

// MiniMax's Chinese system voices, plus an escape hatch for a cloned one.
const VOICES = [
  { id: "male-qn-qingse", label: "青涩青年（男）" },
  { id: "male-qn-jingying", label: "精英青年（男）" },
  { id: "male-qn-badao", label: "霸道青年（男）" },
  { id: "female-shaonv", label: "少女（女）" },
  { id: "female-yujie", label: "御姐（女）" },
  { id: "female-chengshu", label: "成熟女性（女）" },
  { id: "female-tianmei", label: "甜美女性（女）" },
];
const CUSTOM_VOICE = "__custom__";

export default function Debug() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const clientRef = useRef<DigitalHumanClient | null>(null);
  // Hold the key issued by ensureDemoKey() so the metrics poller and DELETE use
  // the exact key that created the session (never a raw localStorage read).
  const keyRef = useRef("");
  const audioRef = useRef<HTMLAudioElement>(null);
  const audioUrlRef = useRef("");
  const [avatars, setAvatars] = useState<AvatarInfo[]>([]);
  const [avatarId, setAvatarId] = useState("");
  const [sessionId, setSessionId] = useState("");
  const [metrics, setMetrics] = useState<SessionInfo["metrics"]>({});
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState("");
  const [text, setText] = useState("");
  const [voice, setVoice] = useState(VOICES[0].id);
  const [customVoice, setCustomVoice] = useState("");
  const [ttsBusy, setTtsBusy] = useState(false);

  useEffect(
    () => () => {
      if (audioUrlRef.current) URL.revokeObjectURL(audioUrlRef.current);
    },
    []
  );

  useEffect(() => {
    api<AvatarInfo[]>("/api/admin/avatars")
      .then((rows) => setAvatars(rows.filter((a) => a.status === "ready")))
      .catch((err) => setError((err as Error).message));
  }, []);

  useEffect(() => {
    if (!sessionId) return;
    const timer = setInterval(async () => {
      try {
        const resp = await fetch(`/api/v1/sessions/${sessionId}`, {
          headers: { "X-API-Key": keyRef.current },
        });
        const data = (await resp.json()) as SessionInfo;
        setMetrics(data.metrics);
        setStatus(data.status);
      } catch {
        /* transient poll failure: keep the previous reading */
      }
    }, 2000);
    return () => clearInterval(timer);
  }, [sessionId]);

  async function start() {
    try {
      setError("");
      const key = await ensureDemoKey();
      keyRef.current = key;
      const resp = await fetch("/api/v1/sessions", {
        method: "POST",
        headers: { "X-API-Key": key, "Content-Type": "application/json" },
        body: JSON.stringify({ avatar_id: avatarId }),
      });
      const body = await resp.json();
      if (!resp.ok) throw new Error(body?.error?.message ?? "create session failed");
      const wsBase = location.origin.replace(/^http/, "ws");
      const client = new DigitalHumanClient(
        `${wsBase}${body.signal_url}?api_key=${key}`,
        `${wsBase}${body.audio_url}?api_key=${key}`
      );
      await client.connect(videoRef.current!);
      clientRef.current = client;
      setSessionId(body.session_id);
    } catch (err) {
      setError((err as Error).message);
    }
  }

  async function stop() {
    clientRef.current?.close();
    clientRef.current = null;
    if (sessionId) {
      await fetch(`/api/v1/sessions/${sessionId}`, {
        method: "DELETE",
        headers: { "X-API-Key": keyRef.current },
      });
    }
    setSessionId("");
    setStatus("closed");
  }

  async function speak() {
    const client = clientRef.current;
    if (!client || !text.trim()) return;
    setError("");
    setTtsBusy(true);
    try {
      const voiceId = voice === CUSTOM_VOICE ? customVoice.trim() : voice;
      const resp = await fetch("/api/admin/tts", {
        method: "POST",
        headers: { "X-Admin-Token": getToken() ?? "", "Content-Type": "application/json" },
        body: JSON.stringify({ text, ...(voiceId ? { voice_id: voiceId } : {}) }),
      });
      if (!resp.ok) {
        let message = resp.statusText;
        try {
          message = (await resp.json())?.error?.message ?? message;
        } catch {
          /* non-json error */
        }
        throw new Error(message);
      }
      const blob = await resp.blob();
      // The WebRTC stream carries video only, so play the clip here too;
      // otherwise you watch the mouth move in silence.
      if (audioUrlRef.current) URL.revokeObjectURL(audioUrlRef.current);
      const url = URL.createObjectURL(blob);
      audioUrlRef.current = url;
      if (audioRef.current) {
        audioRef.current.src = url;
        void audioRef.current.play().catch(() => {});
      }
      await client.sendAudioBlob(blob);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setTtsBusy(false);
    }
  }

  return (
    <div>
      <h2>会话调试</h2>
      <div className="card">
        <select value={avatarId} onChange={(e) => setAvatarId(e.target.value)}>
          <option value="">选择形象…</option>
          {avatars.map((a) => (
            <option key={a.avatar_id} value={a.avatar_id}>
              {a.name}
            </option>
          ))}
        </select>
        <button onClick={start} disabled={!avatarId || !!sessionId}>
          开始调试
        </button>
        <button onClick={stop} disabled={!sessionId}>
          停止
        </button>
        {error && <p className="error">{error}</p>}
      </div>
      <video ref={videoRef} autoPlay playsInline style={{ width: 270, background: "#000" }} />
      <div className="card">
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="输入要让数字人说的话…"
          rows={3}
          style={{ width: "100%" }}
        />
        <div>
          <select value={voice} onChange={(e) => setVoice(e.target.value)}>
            {VOICES.map((v) => (
              <option key={v.id} value={v.id}>
                {v.label}
              </option>
            ))}
            <option value={CUSTOM_VOICE}>自定义 voice_id…</option>
          </select>
          {voice === CUSTOM_VOICE && (
            <input
              value={customVoice}
              onChange={(e) => setCustomVoice(e.target.value)}
              placeholder="MiniMax voice_id"
            />
          )}
          <button onClick={speak} disabled={!sessionId || ttsBusy || !text.trim()}>
            {ttsBusy ? "合成中…" : "合成并驱动"}
          </button>
        </div>
        <audio ref={audioRef} controls />
      </div>
      <div className="card">
        <button onClick={() => clientRef.current?.sendSine()} disabled={!sessionId}>
          假数据驱动
        </button>
        <button onClick={() => clientRef.current?.attachMicrophone()} disabled={!sessionId}>
          麦克风
        </button>
        <input
          type="file"
          accept="audio/*"
          disabled={!sessionId}
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) clientRef.current?.sendAudioFile(f);
          }}
        />
      </div>
      <table>
        <tbody>
          <tr>
            <td>状态</td>
            <td>{status}</td>
          </tr>
          <tr>
            <td>输出帧</td>
            <td>{metrics.frames_out ?? 0}</td>
          </tr>
          <tr>
            <td>丢帧</td>
            <td>{metrics.dropped_frames ?? 0}</td>
          </tr>
          <tr>
            <td>FPS</td>
            <td>{metrics.fps ?? 0}</td>
          </tr>
          <tr>
            <td>平均延迟 ms</td>
            <td>{(metrics.avg_latency_ms ?? 0).toFixed(1)}</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}
