import { useEffect, useRef, useState } from "react";
import {
  Alert,
  App as AntApp,
  Button,
  Card,
  Col,
  Input,
  Row,
  Select,
  Space,
  Statistic,
  Typography,
} from "antd";
import {
  AudioOutlined,
  CaretRightOutlined,
  ExperimentOutlined,
  PauseOutlined,
  SoundOutlined,
  UploadOutlined,
} from "@ant-design/icons";
import { api, getToken } from "../api";
import StatusTag from "../components/StatusTag";
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

function Metric({ title, value, suffix }: { title: string; value: number | string; suffix?: string }) {
  return <Statistic title={title} value={value} suffix={suffix} />;
}

export default function Debug() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const clientRef = useRef<DigitalHumanClient | null>(null);
  // Hold the key issued by ensureDemoKey() so the metrics poller and DELETE use
  // the exact key that created the session (never a raw localStorage read).
  const keyRef = useRef("");
  // Microphone recording lives in refs: the recorder's onstop closure is
  // registered once and must not capture stale React state.
  const recorderRef = useRef<MediaRecorder | null>(null);
  const recordedChunksRef = useRef<Blob[]>([]);
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
  const [recording, setRecording] = useState(false);
  const [micBusy, setMicBusy] = useState(false);
  const [starting, setStarting] = useState(false);
  const { message } = AntApp.useApp();

  useEffect(
    () => () => {
      // Stop an in-flight recording and release the mic on unmount.
      recorderRef.current?.stream.getTracks().forEach((t) => t.stop());
      if (recorderRef.current?.state !== "inactive") recorderRef.current?.stop();
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

  function postSession(key: string) {
    return fetch("/api/v1/sessions", {
      method: "POST",
      headers: { "X-API-Key": key, "Content-Type": "application/json" },
      body: JSON.stringify({ avatar_id: avatarId }),
    });
  }

  async function start() {
    setStarting(true);
    try {
      setError("");
      let key = await ensureDemoKey();
      let resp = await postSession(key);
      if (resp.status === 401) {
        // The cached console key went stale (minting one rotates the "default"
        // tenant's key, so any other console invalidates ours). Mint a fresh
        // key and retry once rather than leaving the page stuck on a 401.
        key = await ensureDemoKey(true);
        resp = await postSession(key);
      }
      const body = await resp.json();
      if (!resp.ok) throw new Error(body?.error?.message ?? "create session failed");
      keyRef.current = key;
      const wsBase = location.origin.replace(/^http/, "ws");
      const client = new DigitalHumanClient(
        `${wsBase}${body.signal_url}?api_key=${key}`,
        `${wsBase}${body.audio_url}?api_key=${key}`
      );
      await client.connect(videoRef.current!);
      clientRef.current = client;
      setSessionId(body.session_id);
      setStatus("active");
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setStarting(false);
    }
  }

  async function stop() {
    if (recorderRef.current?.state === "recording") recorderRef.current.stop();
    recorderRef.current = null;
    setRecording(false);
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
        let msg = resp.statusText;
        try {
          msg = (await resp.json())?.error?.message ?? msg;
        } catch {
          /* non-json error */
        }
        throw new Error(msg);
      }
      // The clip is sent as driving PCM; the server returns it as audio on the
      // WebRTC MediaStream, in sync with the generated mouth movement.
      const blob = await resp.blob();
      await client.sendAudioBlob(blob);
      message.success("已合成并发送");
    } catch (err) {
      message.error((err as Error).message);
    } finally {
      setTtsBusy(false);
    }
  }

  async function toggleMic() {
    const client = clientRef.current;
    if (!client) return;
    if (recording) {
      // onstop ships the recording through the same driving path as TTS.
      recorderRef.current?.stop();
      setRecording(false);
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : "";
      const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
      recorderRef.current = recorder;
      recordedChunksRef.current = [];
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) recordedChunksRef.current.push(e.data);
      };
      recorder.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        const blob = new Blob(recordedChunksRef.current, {
          type: recorder.mimeType || "audio/webm",
        });
        if (!clientRef.current) return; // session stopped while recording
        setMicBusy(true);
        try {
          await clientRef.current.sendAudioBlob(blob);
          message.success("录音已发送");
        } catch (err) {
          message.error((err as Error).message);
        } finally {
          setMicBusy(false);
        }
      };
      recorder.start();
      setRecording(true);
    } catch (err) {
      message.error(`无法使用麦克风：${(err as Error).message}`);
    }
  }

  return (
    <Space direction="vertical" size={16} style={{ width: "100%" }}>
      <Typography.Title level={3} style={{ margin: 0 }}>
        会话调试
      </Typography.Title>

      <Card title="会话">
        <Space wrap>
          <Select
            style={{ minWidth: 220 }}
            value={avatarId || undefined}
            placeholder="选择形象…"
            onChange={setAvatarId}
            options={avatars.map((a) => ({ value: a.avatar_id, label: a.name }))}
            notFoundContent="没有已就绪的形象"
          />
          <Button
            type="primary"
            icon={<CaretRightOutlined />}
            onClick={start}
            loading={starting}
            disabled={!avatarId || !!sessionId}
          >
            开始调试
          </Button>
          <Button icon={<PauseOutlined />} onClick={stop} disabled={!sessionId}>
            停止
          </Button>
          <span>
            状态：<StatusTag status={status} />
          </span>
        </Space>
        {error && <Alert type="error" showIcon message={error} style={{ marginTop: 16 }} />}
      </Card>

      <Row gutter={[16, 16]}>
        <Col xs={24} lg={10}>
          <Card title="画面">
            <video
              ref={videoRef}
              controls
              autoPlay
              playsInline
              style={{
                width: "100%",
                aspectRatio: "9 / 16",
                maxHeight: 480,
                background: "#000",
                borderRadius: 8,
                display: "block",
              }}
            />
          </Card>
        </Col>
        <Col xs={24} lg={14}>
          <Card title="实时指标">
            <Row gutter={[16, 16]}>
              <Col span={12}>
                <Metric title="输出帧" value={metrics.frames_out ?? 0} />
              </Col>
              <Col span={12}>
                <Metric title="丢帧" value={metrics.dropped_frames ?? 0} />
              </Col>
              <Col span={12}>
                <Metric title="FPS" value={metrics.fps ?? 0} />
              </Col>
              <Col span={12}>
                <Metric title="平均延迟" value={(metrics.avg_latency_ms ?? 0).toFixed(1)} suffix="ms" />
              </Col>
            </Row>
          </Card>
        </Col>
      </Row>

      <Card title="语音合成">
        <Space direction="vertical" size={12} style={{ width: "100%" }}>
          <Input.TextArea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="输入要让数字人说的话…"
            rows={3}
          />
          <Space wrap>
            <Select
              style={{ minWidth: 200 }}
              value={voice}
              onChange={setVoice}
              options={[
                ...VOICES.map((v) => ({ value: v.id, label: v.label })),
                { value: CUSTOM_VOICE, label: "自定义 voice_id…" },
              ]}
            />
            {voice === CUSTOM_VOICE && (
              <Input
                style={{ width: 220 }}
                value={customVoice}
                onChange={(e) => setCustomVoice(e.target.value)}
                placeholder="MiniMax voice_id"
              />
            )}
            <Button
              type="primary"
              icon={<SoundOutlined />}
              onClick={speak}
              loading={ttsBusy}
              disabled={!sessionId || !text.trim()}
            >
              合成并驱动
            </Button>
          </Space>
        </Space>
      </Card>

      <Card title="驱动输入">
        <Space wrap>
          <Button
            icon={<ExperimentOutlined />}
            onClick={() => clientRef.current?.sendSine()}
            disabled={!sessionId}
          >
            假数据驱动
          </Button>
          <Button
            icon={<AudioOutlined />}
            onClick={toggleMic}
            loading={micBusy}
            danger={recording}
            disabled={!sessionId && !recording}
          >
            {recording ? "停止录制并发送" : "麦克风录制"}
          </Button>
          <label>
            <Button icon={<UploadOutlined />} disabled={!sessionId} style={{ pointerEvents: "none" }}>
              上传音频
            </Button>
            <input
              type="file"
              accept="audio/*"
              disabled={!sessionId}
              style={{ display: "none" }}
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) clientRef.current?.sendAudioFile(f);
                e.target.value = "";
              }}
            />
          </label>
        </Space>
      </Card>
    </Space>
  );
}
