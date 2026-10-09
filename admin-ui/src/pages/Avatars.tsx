import { FormEvent, useEffect, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api } from "../api";
import type { AvatarInfo } from "../types";

const ACTIVE = new Set(["pending", "processing"]);

export default function Avatars() {
  const [items, setItems] = useState<AvatarInfo[]>([]);
  const [name, setName] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [error, setError] = useState("");
  const timer = useRef<number | null>(null);
  const navigate = useNavigate();

  async function refresh() {
    const rows = await api<AvatarInfo[]>("/api/admin/avatars");
    setItems(rows);
    if (timer.current !== null) window.clearTimeout(timer.current);
    // Poll every 2s only while something is still being produced.
    if (rows.some((a) => ACTIVE.has(a.status))) {
      timer.current = window.setTimeout(() => {
        refresh().catch(() => undefined);
      }, 2000);
    } else {
      timer.current = null;
    }
  }

  useEffect(() => {
    refresh().catch(() => undefined);
    return () => {
      if (timer.current !== null) window.clearTimeout(timer.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function onCreate(e: FormEvent) {
    e.preventDefault();
    if (!file) return;
    try {
      setError("");
      const form = new FormData();
      form.append("name", name);
      form.append("file", file);
      await api("/api/admin/avatars", { method: "POST", form });
      setName("");
      setFile(null);
      await refresh();
    } catch (err) {
      setError((err as Error).message);
    }
  }

  async function onDelete(id: string) {
    if (!confirm("删除该形象？")) return;
    try {
      await api(`/api/admin/avatars/${id}`, { method: "DELETE" });
      await refresh();
    } catch (err) {
      setError((err as Error).message);
    }
  }

  return (
    <div>
      <h2>形象管理</h2>
      <form onSubmit={onCreate} className="card">
        <input value={name} onChange={(e) => setName(e.target.value)} placeholder="名称" />
        <input type="file" accept="image/*" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
        <button type="submit">上传创建</button>
        {error && <p className="error">{error}</p>}
      </form>
      <table>
        <thead>
          <tr>
            <th>名称</th>
            <th>状态</th>
            <th>创建时间</th>
            <th>操作</th>
          </tr>
        </thead>
        <tbody>
          {items.map((a) => (
            <tr key={a.avatar_id} onClick={() => navigate(`/avatars/${a.avatar_id}`)}>
              <td>
                <Link to={`/avatars/${a.avatar_id}`} onClick={(e) => e.stopPropagation()}>
                  {a.name}
                </Link>
              </td>
              <td>
                <span className={`badge ${a.status}`}>{a.status}</span>
              </td>
              <td>{new Date(a.created_at).toLocaleString()}</td>
              <td>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onDelete(a.avatar_id);
                  }}
                >
                  删除
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
