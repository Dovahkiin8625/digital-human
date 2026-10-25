// 数字人对话页 -- a focused chat-driven driving tab.
//
// Lives as its own top-level nav entry (next to 会话调试). The page
// runs one avatar session for its lifetime: pick an avatar, click 开始,
// type into the chat, and the digital human streams the LLM reply while
// typing a subtitle on the video.
//
// The TTS worker, sentence splitter, and SSE streaming live here in
// the same shape they used to sit inside the Debug page. The only
// thing that's new is the page boundary -- the implementation moves
// verbatim, except for the WebRTC session ownership which now lives
// in this page alone (Debug can run its own session in parallel).

import { useEffect, useMemo, useRef, useState } from "react";
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
  Tag,
  Tooltip,
  Typography,
} from "antd";
import {
  CaretRightOutlined,
  PauseOutlined,
  RobotOutlined,
  StopOutlined,
} from "@ant-design/icons";
import { api, live2dModelUrl } from "../api";
import StatusTag from "../components/StatusTag";
import Live2DStage from "../live2d/Live2DStage";
import { useDriveModels } from "../avatars/driveModels";
import { expressionIndex } from "../live2d/render/expressions";
import type { AvatarInfo, Live2DModelConfig, SessionInfo } from "../types";
import { DigitalHumanClient } from "../webrtc";
import {
  drainSentences,
  makeThinkStripper,
  streamChat,
  stripEmotionTags,
  type ChatMessage,
} from "../llm";

// MiniMax's Chinese system voices, plus a custom escape hatch for a
// cloned voice. The list mirrors what the Debug page offers so a
// voice chosen once works across both tabs.
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

// Avatar display name from the list, falling back to the bare id
// when an avatar has been deleted since the session started.
function nameFor(avatars: AvatarInfo[], id?: string): string {
  if (!id) return "—";
  const a = avatars.find((x) => x.avatar_id === id);
  return a ? a.name : id.slice(0, 8);
}

export default function Chat() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const clientRef = useRef<DigitalHumanClient | null>(null);
  // Chat panel state. ``messages`` is the rendered history; ``input``
  // is the pending textarea; ``busy`` disables the send button while a
  // stream is in flight. ``abortRef`` lets a click on 停止 cancel the
  // in-flight stream.
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const abortRef = useRef<{ abort: () => void } | null>(null);
  // Running text buffer + tail while a stream is being chunked into
  // sentences. ``llmBufferRef`` is also what the chat list shows in
  // the in-flight assistant row. Lives as a ``{ tail }`` object (rather
  // than a bare string) so the ``drainSentences`` contract -- which
  // owns the trim-to-tail half of the split -- can't be bypassed by a
  // careless ``buffer += delta`` call. Tracks the un-flushed remainder;
  // the chat bubble text itself is built up from ``displayText`` on
  // every delta.
  const llmBufferRef = useRef<{ tail: string }>({ tail: "" });
  // One think-stripper instance per chat panel. ``ref`` so its internal
  // state (``inThink``, ``thoughtBuf``, ``thoughts``) survives across
  // re-renders without rebuilds; the stripper is built lazily so we
  // don't pay the cost until the operator actually sends a message.
  const stripperRef = useRef<ReturnType<typeof makeThinkStripper> | null>(null);
  // Voice + sentence-queue state for the TTS worker. ``voice`` is the
  // panel's per-conversation voice; the worker is serial on purpose so
  // the digital human doesn't speak two sentences at once.
  const [voice, setVoice] = useState("");
  const [customVoice, setCustomVoice] = useState("");
  const sentenceQueueRef = useRef<string[]>([]);
  const ttsWorkerBusyRef = useRef<boolean>(false);
  // Subtitle overlay: text being driven right now; cleared once the
  // worker has held it long enough to read.
  const [caption, setCaption] = useState<string>("");
  // Avatar picker + session lifecycle.
  const driveModels = useDriveModels();
  const [avatars, setAvatars] = useState<AvatarInfo[]>([]);
  const [avatarId, setAvatarId] = useState("");
  // Live2D mode. The server never creates a session for a live2d avatar,
  // so start() branches before the POST rather than after it.
  const [live2d, setLive2d] = useState(false);
  const [live2dAudio, setLive2dAudio] = useState<AudioBuffer | null>(null);
  const [exprIndex, setExprIndex] = useState(0);
  const actxRef = useRef<AudioContext | null>(null);
  const [sessionId, setSessionId] = useState("");
  const [status, setStatus] = useState("idle");
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState("");
  const { message: toastMessage } = AntApp.useApp();

  useEffect(() => {
    api<AvatarInfo[]>("/api/admin/avatars")
      .then((rows) => setAvatars(rows.filter((a) => a.status === "ready")))
      .catch((err) => setError((err as Error).message));
  }, []);

  // Poll the session's metrics/state once a session exists, so the
  // status badge at the top stays honest about whether WebRTC is
  // still alive. Silent on transient failures.
  useEffect(() => {
    if (!sessionId) return;
    const timer = window.setInterval(async () => {
      try {
        const data = await api<SessionInfo>(`/api/v1/sessions/${sessionId}`);
        setStatus(data.status);
      } catch {
        /* keep last reading */
      }
    }, 2000);
    return () => window.clearInterval(timer);
  }, [sessionId]);

  async function start() {
    // A live2d avatar has no pack and therefore no session. Branch HERE,
    // before the POST: branching after it would 409 and leave an empty
    // video box where the canvas should be.
    if (selectedAvatar?.client_rendered) {
      setLive2d(true);
      setStatus("active");
      setError("");
      return;
    }
    setLive2d(false);
    setStarting(true);
    try {
      setError("");
      const resp = await fetch("/api/v1/sessions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ avatar_id: avatarId }),
      });
      const body = await resp.json();
      if (!resp.ok) throw new Error(body?.error?.message ?? "create session failed");
      const wsBase = location.origin.replace(/^http/, "ws");
      const client = new DigitalHumanClient(
        `${wsBase}${body.signal_url}`,
        `${wsBase}${body.audio_url}`
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
    clientRef.current?.close();
    clientRef.current = null;
    if (sessionId) {
      await api(`/api/v1/sessions/${sessionId}`, { method: "DELETE" });
    }
    // Drop any pending sentences + clear the caption so a fresh
    // start() doesn't pick up where the last one left off.
    sentenceQueueRef.current = [];
    setCaption("");
    setSessionId("");
    setStatus("closed");
  }

  // ----- TTS worker + sentence splitter ---------------------------------
  //
  // The flow:
  //   1. user types a question and presses send
  //   2. streamChat() opens SSE on /api/admin/llm/chat/stream
  //   3. each delta appends to llmBufferRef and updates the in-flight
  //      assistant bubble (so the operator sees text grow live)
  //   4. splitSentences() flushes complete sentences into a queue
  //   5. the worker pulls one sentence at a time, asks MiniMax for
  //      audio, and feeds it to clientRef -- serial so the avatar
  //      speaks in order, not interleaved
  //   6. the worker also drives the subtitle overlay (text shows while
  //      audio is on screen, clears 1.5 s after driving finishes)

  function resolveVoice(): string {
    return voice === CUSTOM_VOICE ? customVoice.trim() : voice;
  }

  async function drainSentenceQueue() {
    if (ttsWorkerBusyRef.current) return;
    const queue = sentenceQueueRef.current;
    if (queue.length === 0) return;
    const client = clientRef.current;
    if (!client && !live2d) {
      // Session stopped mid-conversation; drop the queue so the next
      // start() doesn't surprise the operator with leftover speech.
      queue.length = 0;
      return;
    }
    const voiceId = resolveVoice();
    ttsWorkerBusyRef.current = true;
    try {
      if (!voiceId) {
        // No voice selected: just flash each sentence as a caption in
        // turn -- no audio, no TTS fetch to pipeline. Old behaviour
        // for an edge case the UI already disables upstream.
        while (queue.length > 0) {
          const s = queue.shift()!;
          setCaption(s);
          await new Promise((r) => setTimeout(r, 1500));
          setCaption("");
        }
        return;
      }

      // One TTS fetch wrapped as a promise that resolves to either a
      // blob (success) or an error message. Keeps the loop below free
      // of inline fetch plumbing.
      const fetchOne = (
        sentence: string,
      ): Promise<
        { sentence: string; blob: Blob; tags: string[] } | { sentence: string; error: string }
      > =>
        (async () => {
          // Strip the emotion markers HERE, before the request: this string
          // is the TTS input, and a marker left in is not a cosmetic bug --
          // the pet says "smile" out loud. Restricted to the tags this
          // avatar actually declares, so a bracketed word the user typed
          // survives.
          const { text: spoken, tags } = stripEmotionTags(sentence, expressionTags);
          try {
            const resp = await fetch("/api/admin/tts", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ text: spoken, voice_id: voiceId }),
            });
            if (!resp.ok) {
              const body = await resp.json().catch(() => null);
              const msg = body?.error?.message ?? resp.statusText;
              return { sentence: spoken, error: msg };
            }
            return { sentence: spoken, blob: await resp.blob(), tags };
          } catch (err) {
            return { sentence: spoken, error: (err as Error).message };
          }
        })();

      // Pipeline TTS: kick off the first fetch now, then before each
      // ``sendAudioBlob`` start fetching the NEXT sentence. The fetch
      // latency (~1 s on a cold MiniMax call) is hidden behind the
      // current sentence's audio playback, so sentences flow back-to-
      // back with no inter-sentence gap and the caption is in sync
      // with the audio (the caption only appears once the blob is in
      // hand, so it lands within one network round-trip of the audio).
      let pendingNext: ReturnType<typeof fetchOne> | null = fetchOne(queue.shift()!);
      while (pendingNext) {
        const result = await pendingNext;
        pendingNext = null;
        // Pre-fetch the next sentence in parallel before we start
        // pushing the current sentence's audio.
        if (queue.length > 0) {
          pendingNext = fetchOne(queue.shift()!);
        }
        if ("error" in result) {
          toastMessage.warning(`TTS 失败：${result.error}（跳过此句）`);
          // Brief pause so a failed sentence doesn't snap straight
          // into the next one -- gives the operator a beat to notice
          // the toast.
          await new Promise((r) => setTimeout(r, 400));
          continue;
        }
        // Caption appears here, in lock-step with the audio that's
        // about to start. Replacing the previous caption (not clearing
        // + setting) avoids the blank moment the old implementation
        // had between sentences.
        setCaption(result.sentence);
        try {
          if (live2d) {
            if (result.tags.length > 0) {
              setExprIndex(expressionIndex(result.tags, expressionMap, 0));
            }
            // Play locally and hand the SAME buffer to the stage. There is
            // no WS audio channel in this mode, and reading the envelope
            // from the buffer actually reaching the speakers is what makes
            // mouth/voice drift impossible.
            const actx = actxRef.current ?? new AudioContext();
            actxRef.current = actx;
            const decoded = await actx.decodeAudioData(await result.blob.arrayBuffer());
            const src = actx.createBufferSource();
            src.buffer = decoded;
            src.connect(actx.destination);
            setLive2dAudio(decoded);
            src.start();
            await new Promise((r) => setTimeout(r, (decoded.duration - 0.05) * 1000));
          } else {
            await client!.sendAudioBlob(result.blob);
          }
        } catch (err) {
          toastMessage.warning(`音频发送失败：${(err as Error).message}（跳过此句）`);
        }
      }
      // Tail hold: WebRTC echoes the audio back with a small pipeline
      // delay. Hold the last caption briefly so it doesn't disappear
      // before the mouth stops moving, then clear.
      await new Promise((r) => setTimeout(r, 800));
      setCaption("");
    } finally {
      ttsWorkerBusyRef.current = false;
    }
  }

  function scheduleSentence(sentence: string) {
    // ``splitSentences`` happily emits whitespace-only or pure-
    // punctuation fragments when the model nests newlines / ``。``
    // around a sentence boundary -- the TTS endpoint rejects those
    // with ``empty_text`` after ``.strip()``, surfacing as a noisy
    // toast per fragment. Drop them here so the dispatch site is the
    // one place that knows what TTS will and won't accept.
    if (!sentence.trim()) return;
    sentenceQueueRef.current.push(sentence);
    void drainSentenceQueue();
  }

  async function sendChat() {
    const client = clientRef.current;
    // Live2D has no WebRTC client at all -- the clip plays locally. Only
    // require one when we are actually driving a video session.
    if (!client && !live2d) {
      toastMessage.warning("请先点击「开始」启动会话");
      return;
    }
    const text0 = input.trim();
    if (!text0 || busy) return;
    const voiceId = resolveVoice();
    if (!voiceId) {
      toastMessage.warning("请先选择音色");
      return;
    }
    llmBufferRef.current = { tail: "" };
    // Fresh stripper per turn: reasoning blocks belong to one assistant
    // reply, not the panel's lifetime. Always cheap to build.
    stripperRef.current = makeThinkStripper();
    const nextMessages: ChatMessage[] = [
      ...messages,
      { role: "user", content: text0 },
      { role: "assistant", content: "" },
    ];
    setMessages(nextMessages);
    setInput("");
    setBusy(true);

    // History sent to the server is everything except the trailing
    // empty assistant slot we just created.
    const history = nextMessages.slice(0, -1);
    const stripper = stripperRef.current;
    abortRef.current = streamChat(history, {
      onDelta: (chunk) => {
        // Run the chunk through the think-stripper first so the avatar
        // never speaks reasoning. ``visibleText`` is what feeds the
        // sentence splitter (-> TTS -> audio) AND the chat bubble text;
        // ``newThoughts`` lands on the trailing assistant bubble as a
        // folded chip below the bubble text.
        const out = stripper.feed(chunk);
        // ``drainSentences`` owns the buffer-trim contract: ``newTail``
        // is the un-flushed remainder to feed back next delta.
        // Pre-fix the chat panel used to mutate a string buffer directly
        // and never trim, so every delta re-split (and re-flushed) the
        // entire accumulated text -- the avatar ended up speaking each
        // sentence multiple times.
        const step = drainSentences(llmBufferRef.current, out.visibleText);
        llmBufferRef.current = { tail: step.newTail };
        setMessages((cur) => {
          if (cur.length === 0) return cur;
          const next = cur.slice();
          const lastIdx = next.length - 1;
          const prev = next[lastIdx];
          // Preserve any thoughts previously captured for this bubble;
          // append the ones that just closed so the chip stays
          // up-to-date while the stream is in flight.
          const mergedThoughts =
            out.newThoughts.length > 0
              ? [...(prev.thoughts ?? []), ...out.newThoughts]
              : prev.thoughts;
          next[lastIdx] = { ...prev, content: step.displayText, thoughts: mergedThoughts };
          return next;
        });
        for (const s of step.flushed) scheduleSentence(s);
      },
      onDone: () => {
        // Anything left in the buffer after the stream closes is one
        // final sentence to ship (drainSentences' length fallback).
        const final = drainSentences(llmBufferRef.current, "");
        llmBufferRef.current = { tail: "" };
        for (const s of final.flushed) scheduleSentence(s);
        setBusy(false);
        abortRef.current = null;
      },
      onError: (code, errMsg) => {
        toastMessage.error(`大模型失败：${errMsg}（${code}）`);
        setBusy(false);
        abortRef.current = null;
      },
    });
  }

  function stopChat() {
    abortRef.current?.abort();
    abortRef.current = null;
    setBusy(false);
  }

  const resolvedAvatarName = useMemo(
    () => nameFor(avatars, avatarId),
    [avatars, avatarId]
  );
  const selectedAvatar = useMemo(
    () => avatars.find((a) => a.avatar_id === avatarId) ?? null,
    [avatars, avatarId],
  );
  // The emotion tags this avatar accepts. Read from its expression map,
  // which registration already validated against the model, so stripping
  // can be restricted to these and leave a bracketed word the user typed.
  const expressionMap = useMemo<Record<string, number>>(() => {
    if (!selectedAvatar?.client_rendered) return {};
    return (selectedAvatar.config as Live2DModelConfig)?.expressions ?? {};
  }, [selectedAvatar]);
  const expressionTags = useMemo(() => Object.keys(expressionMap), [expressionMap]);
  const l2dUrl =
    selectedAvatar?.client_rendered
      ? live2dModelUrl(
          selectedAvatar.avatar_id,
          (selectedAvatar.config as Live2DModelConfig)?.model?.entry ?? "model.model3.json",
        )
      : "";
  const resolvedVoice = resolveVoice();

  return (
    <Space direction="vertical" size={16} style={{ width: "100%" }}>
      <Typography.Title level={3} style={{ margin: 0 }}>
        数字人对话
      </Typography.Title>

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
            disabled={!!sessionId}
          />
          <Button
            type="primary"
            icon={<CaretRightOutlined />}
            onClick={start}
            loading={starting}
            disabled={!avatarId || !!sessionId}
          >
            开始
          </Button>
          <Button icon={<PauseOutlined />} onClick={stop} disabled={!sessionId}>
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
          <Card title="画面">
            {/* Subtitle overlay sits on top of the video. Absolute
                positioning keeps it out of the layout flow; the
                ``pointerEvents: none`` makes sure it doesn't swallow
                clicks meant for the underlying controls. Fades out
                automatically when ``caption`` clears. */}
            <div style={{ position: "relative" }}>
              {live2d && l2dUrl ? (
                <Live2DStage
                  modelUrl={l2dUrl}
                  audio={live2dAudio}
                  expressionIndex={exprIndex}
                  width={320}
                  height={560}
                  onError={(e) => setError(String(e))}
                />
              ) : (
                <video
                  ref={videoRef}
                  controls
                  autoPlay
                  playsInline
                  style={{
                    width: "100%",
                    aspectRatio: "9 / 16",
                    maxHeight: 640,
                    background: "#000",
                    borderRadius: 8,
                    display: "block",
                  }}
                />
              )}
              {caption && (
                // ``key={caption}`` forces a remount on every new
                // sentence so the entrance animation in ``styles.css``
                // (``.dh-caption``) replays. Without it, React would
                // reconcile by mutating the text node and skip the
                // animation on the second sentence.
                <div
                  key={caption}
                  aria-live="polite"
                  className="dh-caption"
                  style={{
                    position: "absolute",
                    left: "12%",
                    right: "12%",
                    bottom: 18,
                    padding: "10px 18px",
                    // Frosted glass: backdrop-filter blurs whatever's
                    // behind the caption, the rgba fallback covers
                    // browsers without backdrop-filter. Subtle top-to-
                    // bottom highlight lifts the bar off the video
                    // frame so it doesn't look pasted on.
                    backgroundColor: "rgba(20, 20, 28, 0.7)",
                    backgroundImage:
                      "linear-gradient(to bottom, rgba(255,255,255,0.06), rgba(0,0,0,0))",
                    backdropFilter: "blur(12px) saturate(140%)",
                    WebkitBackdropFilter: "blur(12px) saturate(140%)",
                    color: "#fff",
                    fontSize: 20,
                    fontWeight: 500,
                    lineHeight: 1.45,
                    letterSpacing: 0.2,
                    textAlign: "center",
                    borderRadius: 10,
                    pointerEvents: "none",
                    maxWidth: "76%",
                    margin: "0 auto",
                    // Drop shadow on the text itself -- keeps the
                    // caption legible even if a frame lands a light
                    // skin tone / white background under it.
                    textShadow: "0 1px 4px rgba(0, 0, 0, 0.55)",
                    // System Chinese first (PingFang on macOS, YaHei
                    // on Windows) so the panel reads native instead of
                    // falling back to the browser's default sans-serif.
                    fontFamily:
                      '-apple-system, BlinkMacSystemFont, "PingFang SC", "Microsoft YaHei", "Hiragino Sans GB", sans-serif',
                    boxShadow: "0 4px 18px rgba(0, 0, 0, 0.25)",
                    wordBreak: "break-word",
                  }}
                >
                  {caption}
                </div>
              )}
            </div>
          </Card>
        </Col>
        <Col xs={24} lg={14}>
          <LlmChatCard
            sessionActive={!!sessionId || live2d}
            avatarName={resolvedAvatarName}
            messages={messages}
            busy={busy}
            input={input}
            onInputChange={setInput}
            voice={voice}
            onVoiceChange={setVoice}
            customVoice={customVoice}
            onCustomVoiceChange={setCustomVoice}
            onSend={sendChat}
            onStop={stopChat}
            resolvedVoice={resolvedVoice}
          />
        </Col>
      </Row>
    </Space>
  );
}

// One chat message bubble. User turns sit right-aligned in a soft
// blue; assistant turns sit left-aligned in a softer grey. The
// assistant bubble can be empty when a stream is in flight -- the
// streaming text lands in it token by token.
//
// In-flight assistant bubbles also show a streaming indicator so the
// operator can tell the difference between "model finished" and
// "model still thinking".
//
// Assistant bubbles may also carry a folded ``thoughts`` chip: the
// capture-and-fold shape was chosen because reasoning models emit
// ``<think>…</think>`` for internal planning; that text should never
// reach TTS or the avatar, but a folded summary chip on the panel
// stops the operator thinking the model produced nothing.
function ChatBubble({
  role,
  content,
  streaming,
  thoughts,
}: {
  role: "user" | "assistant";
  content: string;
  streaming?: boolean;
  thoughts?: string[];
}) {
  const isUser = role === "user";
  return (
    <div
      style={{
        display: "flex",
        justifyContent: isUser ? "flex-end" : "flex-start",
        marginBottom: 8,
      }}
    >
      <div
        style={{
          maxWidth: "85%",
          padding: "8px 12px",
          borderRadius: 8,
          background: isUser ? "#1677ff" : "#f0f0f0",
          color: isUser ? "#fff" : "#1f1f1f",
          whiteSpace: "pre-wrap",
          wordBreak: "break-word",
          fontSize: 14,
          lineHeight: 1.5,
        }}
      >
        {content || (streaming ? "…" : "")}
        {streaming && content && (
          <Tag color="processing" style={{ marginLeft: 8 }}>
            正在输入
          </Tag>
        )}
        {!isUser && thoughts && thoughts.length > 0 && (
          // Native ``<details>`` keeps the chip folded by default --
          // the operator sees the count without the reasoning cluttering
          // the panel. Preview is the first 80 chars of each block
          // (truncated with an ellipsis). DOM-only on purpose so we
          // don't pull in another component dep just for this.
          <details
            style={{
              marginTop: 6,
              fontSize: 12,
              color: "#888",
            }}
          >
            <summary style={{ cursor: "pointer" }}>
              💭 已思考 {thoughts.length} 条
            </summary>
            {thoughts.map((t, i) => (
              <div
                key={i}
                style={{
                  marginTop: 4,
                  paddingLeft: 8,
                  borderLeft: "2px solid #d9d9d9",
                  whiteSpace: "pre-wrap",
                  wordBreak: "break-word",
                }}
              >
                {t.length > 80 ? t.slice(0, 80) + "…" : t}
              </div>
            ))}
          </details>
        )}
      </div>
    </div>
  );
}

// The chat panel. Owns nothing -- state lives in the parent so the
// worker, the subtitle, and the bubble list stay in sync.
function LlmChatCard({
  sessionActive,
  avatarName,
  messages,
  busy,
  input,
  onInputChange,
  voice,
  onVoiceChange,
  customVoice,
  onCustomVoiceChange,
  onSend,
  onStop,
  resolvedVoice,
}: {
  sessionActive: boolean;
  avatarName: string;
  messages: ChatMessage[];
  busy: boolean;
  input: string;
  onInputChange: (v: string) => void;
  voice: string;
  onVoiceChange: (v: string) => void;
  customVoice: string;
  onCustomVoiceChange: (v: string) => void;
  onSend: () => void;
  onStop: () => void;
  resolvedVoice: string;
}) {
  const scrollRef = useRef<HTMLDivElement>(null);
  // Keep the message list scrolled to the bottom while new bubbles land.
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  return (
    <Card
      title={
        <Space>
          <RobotOutlined />
          <span>大模型对话</span>
          <Tag color="default">{avatarName}</Tag>
        </Space>
      }
      size="small"
    >
      <Space direction="vertical" size={12} style={{ width: "100%" }}>
        <Space wrap>
          <Select
            style={{ minWidth: 180 }}
            value={voice || undefined}
            placeholder="请选择音色"
            onChange={onVoiceChange}
            disabled={!sessionActive}
            options={[
              ...VOICES.map((v) => ({ value: v.id, label: v.label })),
              { value: CUSTOM_VOICE, label: "自定义 voice_id…" },
            ]}
          />
          {voice === CUSTOM_VOICE && (
            <Input
              style={{ width: 200 }}
              value={customVoice}
              onChange={(e) => onCustomVoiceChange(e.target.value)}
              placeholder="MiniMax voice_id"
            />
          )}
        </Space>
        <div
          ref={scrollRef}
          style={{
            minHeight: 320,
            maxHeight: 420,
            overflowY: "auto",
            border: "1px solid #f0f0f0",
            borderRadius: 6,
            padding: 8,
            background: "#fafafa",
          }}
        >
          {messages.length === 0 ? (
            <Typography.Text type="secondary">
              输入文本后端会调用大模型；回复会逐句驱动数字人说话并在视频底部显示字幕。
            </Typography.Text>
          ) : (
            messages.map((m, i) => (
              <ChatBubble
                key={i}
                role={m.role === "user" ? "user" : "assistant"}
                content={m.content}
                // Reasoning blocks the model emitted but the avatar
                // didn't speak -- rendered as a folded chip below.
                thoughts={m.thoughts}
                // The trailing assistant bubble is "streaming" while
                // busy; everything else is historical.
                streaming={busy && i === messages.length - 1 && m.role === "assistant"}
              />
            ))
          )}
        </div>
        <Input.TextArea
          value={input}
          onChange={(e) => onInputChange(e.target.value)}
          placeholder={sessionActive ? "输入文本…（Enter 发送，Shift+Enter 换行）" : "请先开始会话"}
          autoSize={{ minRows: 2, maxRows: 4 }}
          disabled={!sessionActive}
          onKeyDown={(e) => {
            // Enter to send (without Shift); Shift+Enter inserts a
            // newline. The standard chat-panel idiom.
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              if (!busy && input.trim() && resolvedVoice) onSend();
            }
          }}
        />
        <Space>
          <Button
            type="primary"
            icon={busy ? <StopOutlined /> : <RobotOutlined />}
            onClick={busy ? onStop : onSend}
            disabled={
              !sessionActive ||
              (!busy && (!input.trim() || !resolvedVoice))
            }
          >
            {busy ? "停止" : "发送"}
          </Button>
          {!resolvedVoice && (
            <Typography.Text type="secondary" style={{ fontSize: 12 }}>
              请先选择音色
            </Typography.Text>
          )}
          {sessionActive && !busy && resolvedVoice && !input.trim() && (
            <Tooltip title="输入文本后即可发送">
              <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                输入文本后即可发送
              </Typography.Text>
            </Tooltip>
          )}
        </Space>
      </Space>
    </Card>
  );
}
