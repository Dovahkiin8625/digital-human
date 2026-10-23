import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Alert, Button, Input, Space, Typography } from "antd";
import { SendOutlined } from "@ant-design/icons";
import { live2dHealth, live2dModelUrl } from "../api";
import Live2DStage from "../live2d/Live2DStage";
import { stripEmotionTags, streamChat } from "../llm";
import type { AvatarInfo, Live2DHealth, Live2DModelConfig } from "../types";

/**
 * The desktop pet's page: a transparent window with a canvas and one input
 * box, and nothing else. It shares Live2DStage with the console, so the
 * rendering code exists once, but it renders no console chrome -- no nav,
 * no tables, no theme surface.
 *
 * Runs against whatever server the tray has configured, which is why every
 * call goes through ``baseUrl`` rather than the console's same-origin
 * relative paths.
 */
export interface PetConfig {
  serverUrl: string;
  token: string;
}

declare global {
  interface Window {
    petConfig?: {
      initial?: PetConfig;
      set: (patch: Partial<PetConfig>) => Promise<PetConfig>;
    };
  }
}

const FALLBACK: PetConfig = { serverUrl: "http://127.0.0.1:8000", token: "" };

export default function Pet() {
  const [cfg, setCfg] = useState<PetConfig>(FALLBACK);
  const [avatars, setAvatars] = useState<AvatarInfo[]>([]);
  const [health, setHealth] = useState<Live2DHealth | null>(null);
  const [avatarId, setAvatarId] = useState("");
  const [audio, setAudio] = useState<AudioBuffer | null>(null);
  const [caption, setCaption] = useState("");
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const actxRef = useRef<AudioContext | null>(null);

  useEffect(() => {
    const initial = window.petConfig?.initial;
    if (initial) setCfg({ ...FALLBACK, ...initial });
  }, []);

  const reload = useCallback(async () => {
    const base = cfg.serverUrl.replace(/\/$/, "");
    // Every call carries the token. Without it a server that has one
    // configured would 401 on every model fetch and the pet would sit
    // there blank with no explanation.
    const headers: Record<string, string> = { "X-V1-Token": cfg.token };
    try {
      const resp = await fetch(`${base}/api/v1/avatars?kind=live2d`, { headers });
      if (!resp.ok) throw new Error(`列出 Live2D 形象失败：${resp.status}`);
      const rows = (await resp.json()) as AvatarInfo[];
      setAvatars(rows.filter((a) => a.status === "ready"));
      setAvatarId((prev) => prev || (rows[0]?.avatar_id ?? ""));
    } catch (e) {
      setError((e as Error).message);
    }
    try {
      // The SDK health lives on the admin plane, which a token cannot
      // reach. Its absence is not fatal here -- the runtime route is open --
      // so a failure just leaves the card unrendered.
      const h = await live2dHealth();
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
  const modelUrl = avatar
    ? live2dModelUrl(
        avatar.avatar_id,
        (avatar.config as Live2DModelConfig)?.model?.entry ?? "model.model3.json",
      )
    : "";

  async function speak(text: string) {
    const base = cfg.serverUrl.replace(/\/$/, "");
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      "X-V1-Token": cfg.token,
    };
    const resp = await fetch(`${base}/api/admin/tts`, {
      method: "POST",
      headers,
      body: JSON.stringify({ text, voice_id: "female-shaonv" }),
    });
    if (!resp.ok) throw new Error(`合成失败：${resp.status}`);
    const actx = actxRef.current ?? new AudioContext();
    actxRef.current = actx;
    const decoded = await actx.decodeAudioData(await resp.arrayBuffer());
    const src = actx.createBufferSource();
    src.buffer = decoded;
    src.connect(actx.destination);
    setAudio(decoded);
    src.start();
    await new Promise((r) => setTimeout(r, decoded.duration * 1000));
  }

  async function send() {
    const text = draft.trim();
    if (!text || busy || !avatar) return;
    setBusy(true);
    setError("");
    try {
      const base = cfg.serverUrl.replace(/\/$/, "");
      let reply = "";
      await new Promise<void>((resolve, reject) => {
        streamChat(
          [{ role: "user", content: text }],
          {
            onDelta: (chunk) => {
              reply += chunk;
            },
            onDone: resolve,
            onError: (_code, message) => reject(new Error(message)),
          },
          { baseUrl: base, token: cfg.token },
        );
      });
      // Strip before TTS: the pet would otherwise read the marker aloud.
      const spoken = stripEmotionTags(reply, Object.keys(expressionMap));
      setCaption(spoken.text);
      await speak(spoken.text);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setCaption("");
      setBusy(false);
      setDraft("");
    }
  }

  return (
    <div
      style={{
        // Transparent window: no background, no chrome. The body styles in
        // styles.css are overridden here because the console's opaque
        // background would show as a grey box on the desktop.
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
            audio={audio}
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

      {caption && (
        <Typography.Text style={{ fontSize: 12, textAlign: "center" }}>{caption}</Typography.Text>
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