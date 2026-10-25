import { useEffect, useMemo, useRef, useState } from "react";
import {
  Alert,
  App as AntApp,
  Button,
  Card,
  Col,
  Descriptions,
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
import { api, live2dModelUrl } from "../api";
import StatusTag from "../components/StatusTag";
import LatencyHistogram from "../components/LatencyHistogram";
import Live2DStage from "../live2d/Live2DStage";
import { useDriveModels } from "../avatars/driveModels";
import type { AvatarInfo, Live2DModelConfig, SessionInfo, SystemInfo } from "../types";
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
  return <Statistic title={title} value={value} suffix={suffix} />
}

// Look up an avatar display name from a list, falling back to the bare id when
// the session was for an avatar that's been deleted since.
function nameFor(avatars: AvatarInfo[], id?: string): string {
  if (!id) return "—"
  const a = avatars.find((x) => x.avatar_id === id)
  return a ? a.name : id.slice(0, 8)
}

export default function Debug() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const clientRef = useRef<DigitalHumanClient | null>(null)
  // Microphone recording lives in refs: the recorder's onstop closure is
  // registered once and must not capture stale React state.
  const recorderRef = useRef<MediaRecorder | null>(null)
  const recordedChunksRef = useRef<Blob[]>([])
  const driveModels = useDriveModels();
  const [avatars, setAvatars] = useState<AvatarInfo[]>([])
  const [avatarId, setAvatarId] = useState("")
  // Live2D mode. The server never creates a session for a live2d avatar
  // (it has no pack to render), so this page must not try: the branch in
  // start() fires BEFORE the session POST, not after it.
  const [live2d, setLive2d] = useState(false)
  const [live2dAudio, setLive2dAudio] = useState<AudioBuffer | null>(null)
  const actxRef = useRef<AudioContext | null>(null)
  const [sessionId, setSessionId] = useState("")
  const [metrics, setMetrics] = useState<SessionInfo["metrics"]>({})
  const [status, setStatus] = useState("idle")
  const [error, setError] = useState("")
  const [text, setText] = useState("")
  // The debug page picks the voice per session now -- voice_id was
  // removed from the MiniMax settings card. The empty default nudges
  // the operator to make a choice rather than silently picking one;
  // ``合成并驱动`` stays disabled until they do.
  const [voice, setVoice] = useState("")
  const [customVoice, setCustomVoice] = useState("")
  const [ttsBusy, setTtsBusy] = useState(false)
  const [recording, setRecording] = useState(false)
  const [micBusy, setMicBusy] = useState(false)
  const [starting, setStarting] = useState(false)
  const [sys, setSys] = useState<SystemInfo | null>(null)
  const { message } = AntApp.useApp()

  useEffect(
    () => () => {
      // Stop an in-flight recording and release the mic on unmount.
      recorderRef.current?.stream.getTracks().forEach((t) => t.stop())
      if (recorderRef.current?.state !== "inactive") recorderRef.current?.stop()
    },
    []
  )

  useEffect(() => {
    api<AvatarInfo[]>("/api/admin/avatars")
      .then((rows) => setAvatars(rows.filter((a) => a.status === "ready")))
      .catch((err) => setError((err as Error).message))
  }, [])

  // Engine / media card: poll the same /system the header uses so a name
  // change shows up here too without a manual refresh.
  useEffect(() => {
    let alive = true
    async function tick() {
      try {
        const data = await api<SystemInfo>("/api/admin/system")
        if (alive) setSys(data)
      } catch {
        /* header already shows the cached value; skip on transient failure */
      }
    }
    tick()
    const t = window.setInterval(tick, 5000)
    return () => {
      alive = false
      window.clearInterval(t)
    }
  }, [])

  useEffect(() => {
    if (!sessionId) return
    const timer = setInterval(async () => {
      try {
        const data = await api<SessionInfo>(`/api/v1/sessions/${sessionId}`)
        setMetrics(data.metrics)
        setStatus(data.status)
      } catch {
        /* transient poll failure: keep the previous reading */
      }
    }, 2000)
    return () => clearInterval(timer)
  }, [sessionId])

  // For the "current avatar" hint below the metrics: resolve the session's
  // avatar_id by reading the latest /api/v1/sessions/{id} once a session
  // exists; sessions endpoint is the only place the avatar_id ships with
  // metrics.
  const sessionAvatarId = useMemo(() => {
    const ai = (metrics as unknown as { avatar_id?: string }).avatar_id
    return ai ?? ""
  }, [metrics])
  const resolvedAvatarName = useMemo(() => {
    // Prefer the avatar currently selected in the picker (the one driving);
    // fall back to whatever the session metrics hint at.
    return nameFor(avatars, avatarId || sessionAvatarId)
  }, [avatars, avatarId, sessionAvatarId])

  async function start() {
    // A live2d avatar has no pack and no session. Branch HERE, before the
    // POST: branching after it would 409 on the server and show an empty
    // video box instead of a canvas.
    const picked = avatars.find((a) => a.avatar_id === avatarId)
    if (picked?.client_rendered) {
      setLive2d(true)
      setStatus("active")
      setError("")
      return
    }
    setLive2d(false)
    setStarting(true)
    try {
      setError("")
      const resp = await fetch("/api/v1/sessions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ avatar_id: avatarId }),
      })
      const body = await resp.json()
      if (!resp.ok) throw new Error(body?.error?.message ?? "create session failed")
      const wsBase = location.origin.replace(/^http/, "ws")
      const client = new DigitalHumanClient(
        `${wsBase}${body.signal_url}`,
        `${wsBase}${body.audio_url}`
      )
      await client.connect(videoRef.current!)
      clientRef.current = client
      setSessionId(body.session_id)
      setStatus("active")
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setStarting(false)
    }
  }

  async function stop() {
    if (recorderRef.current?.state === "recording") recorderRef.current.stop()
    recorderRef.current = null
    setRecording(false)
    clientRef.current?.close()
    clientRef.current = null
    if (sessionId) {
      await api(`/api/v1/sessions/${sessionId}`, { method: "DELETE" })
    }
    setSessionId("")
    setLive2d(false)
    setLive2dAudio(null)
    setStatus("closed")
  }

  async function speak() {
    const client = clientRef.current
    // Live2D mode has no WebRTC client at all -- the clip plays locally and
    // the same buffer feeds the mouth. Only require a client when we are
    // actually driving a video session.
    if (!text.trim()) return
    if (!live2d && !client) return
    const voiceId = voice === CUSTOM_VOICE ? customVoice.trim() : voice
    // voice_id is now required by the synthesise endpoint (no server-
    // wide default). The button is also disabled while voice is empty,
    // but this guard catches the "custom voice not typed yet" case.
    if (!voiceId) {
      message.warning("请先选择音色")
      return
    }
    setError("")
    setTtsBusy(true)
    try {
      const resp = await fetch("/api/admin/tts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, voice_id: voiceId }),
      })
      if (!resp.ok) {
        let msg = resp.statusText
        try {
          msg = (await resp.json())?.error?.message ?? msg
        } catch {
          /* non-json error */
        }
        throw new Error(msg)
      }
      const blob = await resp.blob()
      if (live2d) {
        // No WS audio channel in this mode: the clip plays locally and the
        // same buffer feeds the mouth envelope.
        await playLive2dAudio(blob)
        message.success("已合成并播放")
      } else {
        // The clip is sent as driving PCM; the server returns it as audio on
        // the WebRTC MediaStream, in sync with the generated mouth movement.
        await client?.sendAudioBlob(blob)
        message.success("已合成并发送")
      }
    } catch (err) {
      message.error((err as Error).message)
    } finally {
      setTtsBusy(false)
    }
  }

  async function toggleMic() {
    const client = clientRef.current
    if (!client) return
    if (recording) {
      // onstop ships the recording through the same driving path as TTS.
      recorderRef.current?.stop()
      setRecording(false)
      return
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : ""
      const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined)
      recorderRef.current = recorder
      recordedChunksRef.current = []
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) recordedChunksRef.current.push(e.data)
      }
      recorder.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop())
        const blob = new Blob(recordedChunksRef.current, {
          type: recorder.mimeType || "audio/webm",
        })
        if (!clientRef.current) return // session stopped while recording
        setMicBusy(true)
        try {
          await clientRef.current.sendAudioBlob(blob)
          message.success("录音已发送")
        } catch (err) {
          message.error((err as Error).message)
        } finally {
          setMicBusy(false)
        }
      }
      recorder.start()
      setRecording(true)
    } catch (err) {
      message.error(`无法使用麦克风：${(err as Error).message}`)
    }
  }

  // Pre-format media values once so the JSX reads cleanly. The fallback
  // empties are chosen so a partial response does not NaN the descriptions.
  const frameW = sys?.frame_width ?? "—"
  const frameH = sys?.frame_height ?? "—"

  // Live2D renders on the client, so the "drive" inputs (TTS, mic, file) go
  // to the speakers and the mouth reads the same buffer -- no WS audio
  // channel exists for this mode.
  async function playLive2dAudio(file: Blob) {
    const actx = actxRef.current ?? new AudioContext()
    actxRef.current = actx
    const buf = await actx.decodeAudioData(await file.arrayBuffer())
    const src = actx.createBufferSource()
    src.buffer = buf
    src.connect(actx.destination)
    // Hand the stage the SAME buffer we are playing. Reading the envelope
    // from the buffer that is actually reaching the speakers is what makes
    // mouth/voice drift impossible.
    setLive2dAudio(buf)
    src.start()
  }

  const selectedAvatar = useMemo(
    () => avatars.find((a) => a.avatar_id === avatarId) ?? null,
    [avatars, avatarId],
  )
  const l2dUrl =
    selectedAvatar?.client_rendered
      ? live2dModelUrl(
          selectedAvatar.avatar_id,
          (selectedAvatar.config as Live2DModelConfig)?.model?.entry ?? "model.model3.json",
        )
      : ""
  const fps = sys?.fps ?? "—"
  const sr = sys?.sample_rate ?? "—"
  const br = sys?.video_bitrate
  const idle = sys?.session_idle_timeout

  return (
    <Space direction="vertical" size={16} style={{ width: "100%" }}>
      <Typography.Title level={3} style={{ margin: 0 }}>
        会话调试
      </Typography.Title>

      <Card title="引擎与媒体">
        {sys ? (
          <Descriptions
            size="small"
            column={{ xs: 2, sm: 3, md: 4 }}
            items={[
              { key: "engine", label: "运行时", children: sys.runtimes.map((r) => r.label_zh).join(" · ") },
              { key: "gpu", label: "GPU", children: sys.gpu ? "可用" : "不可用" },
              { key: "queue", label: "队列深度", children: sys.runtimes.reduce((n, r) => n + r.pending, 0) },
              { key: "cached", label: "已缓存形象", children: sys.runtimes.reduce((n, r) => n + r.avatars_cached, 0) },
              { key: "size", label: "帧尺寸", children: `${frameW} × ${frameH}` },
              { key: "fps", label: "FPS", children: String(fps) },
              { key: "sr", label: "采样率 Hz", children: String(sr) },
              { key: "vbr", label: "视频码率", children: br ? `${(br / 1000).toFixed(0)} kbps` : "—" },
              { key: "idle", label: "会话空闲超时 s", children: idle ?? "—" },
            ]}
          />
        ) : (
          <Typography.Text type="secondary">加载中…</Typography.Text>
        )}
      </Card>

      <Card title="会话">
        <Space wrap>
          <Select
            style={{ minWidth: 220 }}
            value={avatarId || undefined}
            placeholder="选择形象…"
            onChange={setAvatarId}
            options={driveModels
              // One group per drive model: an avatar belongs to the model
              // that drives it, so the groups are the models.
              .map((m) => ({
                label: m.label_zh,
                options: avatars
                  .filter((a) => a.drive_model === m.name)
                  .map((a) => ({ value: a.avatar_id, label: a.name })),
              }))
              // Drop a group that has no avatars rather than rendering an
              // empty heading above the list.
              .filter((g) => g.options.length > 0)}
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
          <Button icon={<PauseOutlined />} onClick={stop} disabled={!sessionId && !live2d}>
            停止
          </Button>
          <span>
            状态：<StatusTag status={status} />
          </span>
          {sessionId && (
            <Typography.Text type="secondary">
              会话 {sessionId.slice(0, 8)} · 形象 {resolvedAvatarName}
            </Typography.Text>
          )}
        </Space>
        {error && <Alert type="error" showIcon message={error} style={{ marginTop: 16 }} />}
      </Card>

      <Row gutter={[16, 16]}>
        <Col xs={24} lg={10}>
          <Card title={live2d ? "画面（Live2D，客户端渲染）" : "画面"}>
            {live2d ? (
              l2dUrl && (
                <Live2DStage
                  modelUrl={l2dUrl}
                  audio={live2dAudio}
                  width={270}
                  height={480}
                  onError={(e) => setError(String(e))}
                />
              )
            ) : (
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
            )}
          </Card>
        </Col>
        <Col xs={24} lg={14}>
          <Card title="实时指标">
            <Row gutter={[16, 16]}>
              <Col span={8}>
                <Metric title="输出帧" value={metrics.frames_out ?? 0} />
              </Col>
              <Col span={8}>
                <Metric title="保活帧" value={metrics.keepalive_frames ?? 0} />
              </Col>
              <Col span={8}>
                <Metric title="丢帧" value={metrics.dropped_frames ?? 0} />
              </Col>
              <Col span={8}>
                <Metric title="FPS" value={metrics.fps ?? 0} />
              </Col>
              <Col span={8}>
                <Metric title="块数" value={metrics.blocks ?? 0} />
              </Col>
              <Col span={8}>
                <Metric title="队列深度" value={metrics.queue_depth ?? 0} />
              </Col>
              <Col span={8}>
                <Metric title="溢出" value={metrics.overflows ?? 0} />
              </Col>
              <Col span={8}>
                <Metric title="平均延迟" value={(metrics.avg_latency_ms ?? 0).toFixed(1)} suffix="ms" />
              </Col>
              <Col span={8}>
                <Metric title="p50 延迟" value={(metrics.p50_latency_ms ?? 0).toFixed(1)} suffix="ms" />
              </Col>
              <Col span={8}>
                <Metric title="p95 延迟" value={(metrics.p95_latency_ms ?? 0).toFixed(1)} suffix="ms" />
              </Col>
              <Col span={8}>
                <Metric title="最大延迟" value={(metrics.max_latency_ms ?? 0).toFixed(1)} suffix="ms" />
              </Col>
            </Row>
            <div style={{ marginTop: 16 }}>
              <Typography.Text strong>延迟分布（ms）</Typography.Text>
              <div style={{ marginTop: 8 }}>
                <LatencyHistogram data={metrics.latency_histogram} />
              </div>
            </div>
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
              value={voice || undefined}
              placeholder="请选择音色"
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
              // A live2d avatar has no session, so gating this on sessionId alone left
              // the only way to make the pet talk permanently disabled in that mode.
              disabled={(sessionId || live2d) === false || !text.trim() || !voice || (voice === CUSTOM_VOICE && !customVoice.trim())}
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
            <Button icon={<UploadOutlined />} disabled={!sessionId && !live2d} style={{ pointerEvents: "none" }}>
              上传音频
            </Button>
            <input
              type="file"
              accept="audio/*"
              disabled={!sessionId && !live2d}
              style={{ display: "none" }}
              onChange={(e) => {
                const f = e.target.files?.[0]
                if (f) {
                  if (live2d) {
                    playLive2dAudio(f).catch((err) => setError((err as Error).message))
                  } else {
                    clientRef.current?.sendAudioFile(f)
                  }
                }
                e.target.value = ""
              }}
            />
          </label>
        </Space>
      </Card>
    </Space>
  )
}
