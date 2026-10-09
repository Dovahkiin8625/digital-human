import { useEffect, useRef, useState } from "react";
import { useParams } from "react-router-dom";
import { api, getToken } from "../api";
import type { AvatarInfo } from "../types";

async function fetchPreviewUrl(avatarId: string, level: number): Promise<string> {
  // A plain <img> cannot attach X-Admin-Token, so fetch the PNG with the header
  // and hand the browser an object URL of the resulting blob.
  const resp = await fetch(
    `/api/admin/avatars/${avatarId}/preview.png?level=${level}&t=${Date.now()}`,
    { headers: { "X-Admin-Token": getToken() ?? "" } }
  );
  if (!resp.ok) throw new Error("preview unavailable");
  const blob = await resp.blob();
  return URL.createObjectURL(blob);
}

export default function AvatarDetail() {
  const { id } = useParams();
  const [info, setInfo] = useState<AvatarInfo | null>(null);
  const [mouth, setMouth] = useState(1);
  const [expression, setExpression] = useState(1);
  const [level, setLevel] = useState(0.5);
  const [preview, setPreview] = useState("");
  const [error, setError] = useState("");
  const previewRef = useRef("");

  async function refresh() {
    if (!id) return;
    const data = await api<AvatarInfo>(`/api/admin/avatars/${id}`);
    setInfo(data);
    setMouth(data.config.mouth_scale ?? 1);
    setExpression(data.config.expression_intensity ?? 1);
  }

  useEffect(() => {
    refresh().catch((err) => setError((err as Error).message));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id]);

  useEffect(() => {
    if (!id) return;
    let cancelled = false;
    fetchPreviewUrl(id, level)
      .then((url) => {
        if (cancelled) {
          URL.revokeObjectURL(url);
          return;
        }
        // Replace: revoke the object URL we are about to drop so blobs do not leak.
        if (previewRef.current) URL.revokeObjectURL(previewRef.current);
        previewRef.current = url;
        setPreview(url);
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [id, level]);

  // Revoke the last object URL when the page unmounts.
  useEffect(() => {
    return () => {
      if (previewRef.current) URL.revokeObjectURL(previewRef.current);
    };
  }, []);

  async function save() {
    if (!id) return;
    try {
      setError("");
      await api(`/api/admin/avatars/${id}/config`, {
        method: "PATCH",
        body: { mouth_scale: mouth, expression_intensity: expression },
      });
      await refresh();
    } catch (err) {
      setError((err as Error).message);
    }
  }

  async function retry() {
    if (!id) return;
    try {
      setError("");
      await api(`/api/admin/avatars/${id}/retry`, { method: "POST", body: {} });
      await refresh();
    } catch (err) {
      setError((err as Error).message);
    }
  }

  if (!info) return <p>加载中…</p>;

  return (
    <div>
      <h2>{info.name}</h2>
      <p>
        状态：<span className={`badge ${info.status}`}>{info.status}</span>
      </p>
      {info.error && (
        <p className="error">
          {info.error} <button onClick={retry}>重试</button>
        </p>
      )}
      {error && <p className="error">{error}</p>}
      <div className="card">
        <label>口型幅度 {mouth.toFixed(2)}</label>
        <input
          type="range"
          min={0}
          max={2}
          step={0.05}
          value={mouth}
          onChange={(e) => setMouth(+e.target.value)}
        />
        <label>表情强度 {expression.toFixed(2)}</label>
        <input
          type="range"
          min={0}
          max={2}
          step={0.05}
          value={expression}
          onChange={(e) => setExpression(+e.target.value)}
        />
        <button onClick={save}>保存参数</button>
      </div>
      <div className="card">
        <label>驱动电平 {level.toFixed(2)}</label>
        <input
          type="range"
          min={0}
          max={1}
          step={0.05}
          value={level}
          onChange={(e) => setLevel(+e.target.value)}
        />
        {preview && <img src={preview} alt="preview" style={{ width: 240 }} />}
      </div>
    </div>
  );
}
