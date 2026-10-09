import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import { ensureDemoKey } from "../demoKey";
import type { AvatarInfo, SessionInfo } from "../types";
import { DigitalHumanClient } from "../webrtc";

export default function Debug() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const clientRef = useRef<DigitalHumanClient | null>(null);
  // Hold the key issued by ensureDemoKey() so the metrics poller and DELETE use
  // the exact key that created the session (never a raw localStorage read).
  const keyRef = useRef("");
  const [avatars, setAvatars] = useState<AvatarInfo[]>([]);
  const [avatarId, setAvatarId] = useState("");
  const [sessionId, setSessionId] = useState("");
  const [metrics, setMetrics] = useState<SessionInfo["metrics"]>({});
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState("");

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
