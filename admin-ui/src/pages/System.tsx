import { FormEvent, useEffect, useState } from "react";
import { api } from "../api";
import type { SessionInfo, TenantInfo } from "../types";

interface SystemInfo {
  engine: { name: string; gpu: boolean };
  version: string;
  data_dir: string;
  gpu: boolean;
}

interface JobInfo {
  job_id: string;
  avatar_id: string;
  stage: string;
  progress: number;
  error: string | null;
}

export default function System() {
  const [sys, setSys] = useState<SystemInfo | null>(null);
  const [tenants, setTenants] = useState<TenantInfo[]>([]);
  const [jobs, setJobs] = useState<JobInfo[]>([]);
  const [sessions, setSessions] = useState<SessionInfo[]>([]);
  const [tenantName, setTenantName] = useState("");

  async function refresh() {
    setSys(await api<SystemInfo>("/api/admin/system"));
    setTenants(await api<TenantInfo[]>("/api/admin/tenants"));
    setJobs(await api<JobInfo[]>("/api/admin/jobs"));
    setSessions(await api<SessionInfo[]>("/api/admin/sessions"));
  }

  useEffect(() => {
    refresh().catch(() => undefined);
    const timer = setInterval(() => refresh().catch(() => undefined), 2000);
    return () => clearInterval(timer);
  }, []);

  async function onCreateTenant(e: FormEvent) {
    e.preventDefault();
    const created = await api<{ api_key: string }>("/api/admin/tenants", {
      method: "POST",
      body: { name: tenantName },
    });
    window.alert(`API Key（仅显示一次）：${created.api_key}`);
    setTenantName("");
    refresh();
  }

  return (
    <div>
      <h2>系统</h2>
      {sys && (
        <table>
          <tbody>
            <tr><td>引擎</td><td>{sys.engine.name}</td></tr>
            <tr><td>GPU</td><td>{String(sys.gpu)}</td></tr>
            <tr><td>数据目录</td><td>{sys.data_dir}</td></tr>
            <tr><td>版本</td><td>{sys.version}</td></tr>
          </tbody>
        </table>
      )}
      <h3>租户</h3>
      <form onSubmit={onCreateTenant} className="card">
        <input value={tenantName} onChange={(e) => setTenantName(e.target.value)} placeholder="租户名" />
        <button type="submit">新建租户</button>
      </form>
      <table>
        <thead><tr><th>租户</th><th>创建时间</th></tr></thead>
        <tbody>
          {tenants.map((t) => (
            <tr key={t.tenant_id}><td>{t.name}</td><td>{new Date(t.created_at).toLocaleString()}</td></tr>
          ))}
        </tbody>
      </table>
      <h3>任务</h3>
      <table>
        <thead><tr><th>Job</th><th>阶段</th><th>进度</th><th>错误</th></tr></thead>
        <tbody>
          {jobs.map((j) => (
            <tr key={j.job_id}>
              <td>{j.job_id.slice(0, 8)}</td><td>{j.stage}</td><td>{j.progress}%</td>
              <td className="error">{j.error ?? ""}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <h3>会话</h3>
      <table>
        <thead><tr><th>会话</th><th>状态</th><th>帧数</th><th>延迟 ms</th></tr></thead>
        <tbody>
          {sessions.map((s) => (
            <tr key={s.session_id}>
              <td>{s.session_id.slice(0, 8)}</td><td>{s.status}</td>
              <td>{s.metrics.frames_out ?? 0}</td>
              <td>{(s.metrics.avg_latency_ms ?? 0).toFixed(1)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
