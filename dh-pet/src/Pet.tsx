import { useCallback, useEffect, useMemo, useState } from "react";
import { Alert, Button, Input, Space, Typography } from "antd";
import { SendOutlined } from "@ant-design/icons";
import { listAvatars, live2dHealth, live2dModelUrl } from "./api";
import Live2DStage from "./live2d/Live2DStage";
import { configureRuntime } from "./live2d/cubism/core";
import { useLive2DSpeech } from "./live2d/useLive2DSpeech";
import { gazeTarget, type GazeTarget } from "./live2d/render/gaze";
import { drainSentences, makeThinkStripper, streamChat } from "./llm";
import { DEFAULT_VOICE_ID } from "./voices";
import type { AvatarInfo, Live2DHealth, Live2DModelConfig } from "./types";

/**
 * The desktop pet: a transparent window with a canvas and one input box,
 * and nothing else. It renders no console chrome -- no nav, no tables, no
 * theme surface.
 *
 * Runs against whatever server the shell has configured, which is why every
 * call goes through the configured base URL rather than a same-origin
 * relative path.
 */
interface PetConfig {
  serverUrl: string;
  token: string;
}

/** Where the mouse is, in screen coordinates, and where the pet's window
 *  sits. The shell polls for it: the window ignores mouse input most of
 *  the time, and a window that ignores input hears nothing. */
interface CursorReport {
  cursor: { x: number; y: number };
  origin: { x: number; y: number };
}

declare global {
  interface Window {
    petConfig?: {
      initial?: PetConfig;
      set: (patch: Partial<PetConfig>) => Promise<PetConfig>;
      reportCursor?: () => void;
      onCursor?: (cb: (report: CursorReport) => void) => () => void;
    };
  }
}

const FALLBACK: PetConfig = { serverUrl: "http://127.0.0.1:8000", token: "" };

export default function Pet() {
  const [cfg, setCfg] = useState<PetConfig>(FALLBACK);
  const [avatars, setAvatars] = useState<AvatarInfo[]>([]);
  const [health, setHealth] = useState<Live2DHealth | null>(null);
  const [avatarId, setAvatarId] = useState("");
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [gaze, setGaze] = useState<GazeTarget | null>(null);

  useEffect(() => {
    const initial = window.petConfig?.initial;
    if (initial) setCfg({ ...FALLBACK, ...initial });
  }, []);

  // While the window ignores mouse input, Electron still forwards move
  // events to the renderer. Reporting them is what lets the main process
  // lift click-through when the cursor arrives -- without this the input
  // box can never be focused.
  useEffect(() => {
    const report = () => window.petConfig?.reportCursor?.();
    window.addEventListener("mousemove", report);
    return () => window.removeEventListener("mousemove", report);
  }, []);

  // The eyes follow the mouse. The shell polls the cursor's screen
  // position and sends it here; the pet works out where that is relative
  // to itself.
  useEffect(() => {
    const subscribe = window.petConfig?.onCursor;
    if (!subscribe) return;
    return subscribe(({ cursor, origin }) => {
      // The eyes reach the end of their travel at half the window's long
      // side: a cursor just beside the pet gets a glance, one across the
      // room gets the full sweep.
      const span = Math.max(window.innerWidth, window.innerHeight) * 0.5;
      setGaze(gazeTarget(cursor, origin, span));
    });
  }, []);

  const reload = useCallback(async () => {
    try {
      // Every call carries the token. Without it a server that has one
      // configured would 401 on every model fetch and the pet would sit
      // there blank with no explanation.
      const rows = await listAvatars(cfg.serverUrl, cfg.token);
      setAvatars(rows.filter((a) => a.status === "ready"));
      setAvatarId((prev) => prev || (rows[0]?.avatar_id ?? ""));
    } catch (e) {
      setError((e as Error).message);
    }
    try {
      // The SDK health lives on the admin plane, which a token cannot
      // reach. Its absence is not fatal here -- the runtime route is open --
      // so a failure just leaves the card unrendered.
      const h = await live2dHealth(cfg.serverUrl);
      setHealth(h);
    } catch {
      setHealth(null);
    }
  }, [cfg.serverUrl, cfg.token]);

  useEffect(() => {
    reload();
  }, [reload]);

  const avatar = useMemo(
    () => avatars.find((a) => a.avatar_id === avatarId) ?? null,
    [avatars, avatarId],
  );
  const expressionMap = useMemo<Record<string, number>>(
    () => ((avatar?.config as Live2DModelConfig)?.expressions ?? {}),
    [avatar],
  );
  // The document is a file:// page under Electron, so every Live2D URL has
  // to be absolute AND carry the token -- a bare "/api/v1/..." resolves to
  // file:///api/v1/... and 401s on every request.
  useEffect(() => {
    configureRuntime({ baseUrl: cfg.serverUrl, token: cfg.token });
  }, [cfg.serverUrl, cfg.token]);

  const speech = useLive2DSpeech({
    baseUrl: cfg.serverUrl,
    token: cfg.token,
    expressionMap,
    voiceId: DEFAULT_VOICE_ID,
    onError: (message) => setError(message),
  });

  const modelUrl = avatar
    ? live2dModelUrl(
        avatar.avatar_id,
        (avatar.config as Live2DModelConfig)?.model?.entry ?? "model.model3.json",
      )
    : "";

  async function send() {
    const text = draft.trim();
    if (!text || busy || !avatar) return;
    setBusy(true);
    setError("");
    // The reply is spoken sentence by sentence, so the pet's face changes
    // mid-answer instead of only at the end, and the mouth envelope
    // covers one utterance at a time.
    const stripper = makeThinkStripper();
    let buffer = { tail: "" };
    try {
      await new Promise<void>((resolve, reject) => {
        streamChat(
          [{ role: "user", content: text }],
          {
            onDelta: (chunk) => {
              // Reasoning blocks are never spoken: the pet would read the
              // model's private thinking out loud.
              const out = stripper.feed(chunk);
              const step = drainSentences(buffer, out.visibleText);
              buffer = { tail: step.newTail };
              for (const s of step.flushed) speech.enqueue(s);
            },
            onDone: resolve,
            onError: (_code, message) => reject(new Error(message)),
          },
          { baseUrl: cfg.serverUrl.replace(/\/$/, ""), token: cfg.token },
        );
      });
      // The last sentence has no punctuation after it, so it is still in
      // the tail.
      const final = drainSentences(buffer, "");
      for (const s of final.flushed) speech.enqueue(s);
      speech.flush();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
      setDraft("");
    }
  }

  return (
    <div
      style={{
        // Transparent window: no background, no chrome. The page canvas
        // itself is see-through too (see styles.css) -- what is left is the
        // model and the input box.
        background: "transparent",
        display: "flex",
        flexDirection: "column",
        height: "100vh",
        padding: 8,
        gap: 8,
        boxSizing: "border-box",
      }}
    >
      {error && (
        <Alert type="error" showIcon message={error} style={{ fontSize: 12 }} closable />
      )}

      <div style={{ flex: 1, minHeight: 0, display: "flex", justifyContent: "center" }}>
        {modelUrl ? (
          <Live2DStage
            modelUrl={modelUrl}
            audio={speech.audio}
            expressionIndex={speech.expressionIndex}
            gaze={gaze}
            width={260}
            height={400}
            onError={(e) => setError(String(e))}
          />
        ) : (
          <Typography.Text type="secondary" style={{ fontSize: 12, color: "#bbb" }}>
            {avatars.length === 0 ? "没有可用的 Live2D 形象" : "请选择形象"}
          </Typography.Text>
        )}
      </div>

      {speech.caption && (
        <Typography.Text
          className="dh-pet-caption"
          style={{ fontSize: 12, textAlign: "center" }}
        >
          {speech.caption}
        </Typography.Text>
      )}

      <Space.Compact style={{ width: "100%" }}>
        <Input
          size="small"
          value={draft}
          placeholder={avatar ? "说点什么…" : "先在管理台注册 Live2D 模型"}
          disabled={!avatar || busy}
          onChange={(e) => setDraft(e.target.value)}
          onPressEnter={send}
        />
        <Button
          size="small"
          type="primary"
          icon={<SendOutlined />}
          loading={busy}
          disabled={!avatar || busy}
          onClick={send}
        />
      </Space.Compact>

      {health && !health.core_present && (
        <Typography.Text style={{ fontSize: 10, color: "#f0a" }}>
          服务端未找到 Cubism SDK（{health.sdk_dir}）
        </Typography.Text>
      )}
    </div>
  );
}
