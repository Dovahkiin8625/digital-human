import { useEffect, useState } from "react";
import {
  App as AntApp,
  Alert,
  Button,
  Card,
  Popconfirm,
  Progress,
  Space,
  Table,
  Tag,
  Tooltip,
  Typography,
} from "antd";
import type { ColumnsType } from "antd/es/table";
import {
  ApiOutlined,
  CloudDownloadOutlined,
  PoweroffOutlined,
  ReloadOutlined,
  SoundOutlined,
  ThunderboltFilled,
  UserOutlined,
} from "@ant-design/icons";
import { api } from "../api";
import type { GpuSnapshot, ModelsPayload, ModelInfo, ServiceInfo } from "../types";

// Same value the rest of the System page polls at (概览, 任务, 会话). The
// GPU counter moves on a sub-second timescale, but the operator's eye
// doesn't; 2 s is dense enough to feel live without flooding the server.
const POLL_MS = 2000;

function formatBytes(b: number): string {
  if (b <= 0) return "—";
  if (b < 1024 ** 2) return `${(b / 1024).toFixed(1)} KB`;
  if (b < 1024 ** 3) return `${(b / 1024 ** 2).toFixed(1)} MB`;
  return `${(b / 1024 ** 3).toFixed(2)} GB`;
}

function formatGb(g: number | undefined | null): string {
  if (g == null) return "—";
  if (g >= 1) return `${g.toFixed(1)} GB`;
  // Sub-GB allocations still matter (a freshly-loaded whisper); show MB
  // so the operator sees movement, not "0.0 GB".
  return `${(g * 1024).toFixed(0)} MB`;
}

function formatLoadedAgo(iso: string | null, now: number): string {
  if (!iso) return "—";
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return "—";
  const secs = Math.max(0, Math.round((now - then) / 1000));
  if (secs < 60) return `刚刚 (${secs} 秒前)`;
  if (secs < 3600) return `${Math.floor(secs / 60)} 分钟前`;
  if (secs < 86400) return `${Math.floor(secs / 3600)} 小时前`;
  return new Date(iso).toLocaleString();
}

function GpuCard({
  gpu,
  models,
  activeSessions,
}: {
  gpu: GpuSnapshot | null;
  models: ModelInfo[];
  activeSessions: number;
}) {
  // The card surfaces three numbers the operator cares about: how much
  // VRAM is *used right now* (the Progress), how big the GPU is (the
  // caption), and a textual tally of loaded/unloadable models so the
  // card never feels empty on a mock engine.
  if (!gpu) {
    // CPU-only box: there is no GPU memory to track. Render an honest
    // empty bar with a one-line hint so the operator doesn't think the
    // tab is broken.
    const loadedCount = models.filter((m) => m.loaded).length;
    return (
      <Card title="显存" size="small">
        <Space direction="vertical" size={8} style={{ width: "100%" }}>
          <Tag color="default">CPU</Tag>
          <Typography.Text type="secondary">
            当前主机无 CUDA 设备；以下模型在主机内存中运行：{loadedCount} / {models.length}
          </Typography.Text>
          {activeSessions > 0 && (
            <Tag color="warning">{activeSessions} 个会话正在使用</Tag>
          )}
        </Space>
      </Card>
    );
  }

  const used = gpu.allocated_gb;
  const total = gpu.total_gb;
  const pct = total > 0 ? Math.min(100, Math.round((used / total) * 100)) : 0;
  // Colour the bar at the same thresholds the existing UI uses elsewhere
  // (TTS preview, idle video). A reserved-vs-allocated delta is shown
  // inline because the operator cares about *both* -- reserved is what
  // they can't get back until torch.cuda.empty_cache() runs.
  const reservedExtra = Math.max(0, gpu.reserved_gb - used);

  return (
    <Card
      title={
        <Space>
          显存
          <Tag color="success" style={{ marginLeft: 4 }}>
            GPU
          </Tag>
        </Space>
      }
      size="small"
    >
      <Space direction="vertical" size={6} style={{ width: "100%" }}>
        <Progress
          percent={pct}
          // Stroke at higher utilisation -- the operator's eye reads a
          // long red bar faster than text.
          status={pct >= 85 ? "exception" : pct >= 60 ? "active" : "normal"}
          format={() => ""}
          strokeColor={pct >= 85 ? "#ff4d4f" : undefined}
        />
        <Typography.Text>
          已用 <Typography.Text strong>{formatGb(used)}</Typography.Text>
          {" "}/ 总 <Typography.Text strong>{formatGb(total)}</Typography.Text>
        </Typography.Text>
        <Space wrap size={16}>
          <span>
            <Typography.Text type="secondary">设备：</Typography.Text>
            <Typography.Text code style={{ marginLeft: 4 }}>
              {gpu.device}
            </Typography.Text>
          </span>
          <span>
            <Typography.Text type="secondary">型号：</Typography.Text>
            <Typography.Text>{gpu.name}</Typography.Text>
          </span>
          <span>
            <Typography.Text type="secondary">空闲：</Typography.Text>
            <Typography.Text strong>{formatGb(gpu.free_gb)}</Typography.Text>
          </span>
          <Tooltip title="torch 已申请但尚未使用的缓存">
            <span>
              <Typography.Text type="secondary">预留：</Typography.Text>
              <Typography.Text>{formatGb(reservedExtra)}</Typography.Text>
            </span>
          </Tooltip>
          <Tooltip title="本次进程启动以来的峰值占用">
            <span>
              <Typography.Text type="secondary">峰值：</Typography.Text>
              <Typography.Text>{formatGb(gpu.peak_gb)}</Typography.Text>
            </span>
          </Tooltip>
        </Space>
        <Space wrap>
          {activeSessions > 0 && (
            <Tag color="warning">{activeSessions} 个会话正在使用</Tag>
          )}
          <Typography.Text type="secondary">
            {/* Total of all slot sizes listed below; a sanity check the
                operator can compare against `du -sh models/`. */}
            {models.reduce((s, m) => s + m.estimated_bytes, 0) > 0 && (
              <>磁盘权重合计：{formatBytes(models.reduce((s, m) => s + m.estimated_bytes, 0))}</>
            )}
          </Typography.Text>
        </Space>
      </Space>
    </Card>
  );
}

function typeLabel(t: string): string {
  // Map the slim server-side type to the operator-facing Chinese label.
  // New flavours stay legible without crowding the tab.
  if (t === "drive_runtime") return "驱动运行时";
  return t;
}

function kindLabel(k: string): string {
  // Service-row "类别" column. Keep the strings short so they fit in
  // a Tag next to a longer label_zh on the same row.
  if (k === "portrait_animator") return "形象动画器";
  if (k === "tts_service") return "TTS 服务";
  return k;
}

function kindIcon(k: string) {
  if (k === "portrait_animator") return <UserOutlined />;
  if (k === "tts_service") return <SoundOutlined />;
  return <ApiOutlined />;
}

function StatusBadge({ ok, label }: { ok: boolean; label?: string }) {
  // Shared green/red pill used by the service table's 状态 column. The
  // ``label`` argument lets the row override the binary with a more
  // precise string (e.g. "未配置" vs "未就绪") when the server already
  // computed one.
  return ok ? (
    <Tag color="success">{label || "就绪"}</Tag>
  ) : (
    <Tag color="error">{label || "未就绪"}</Tag>
  );
}

export default function Models() {
  const [data, setData] = useState<ModelsPayload | null>(null);
  const [actingOn, setActingOn] = useState<string | null>(null);
  const [actingServiceOn, setActingServiceOn] = useState<string | null>(null);
  const { message } = AntApp.useApp();
  // A second-since-epoch ticker so "loaded_at" ages smoothly between
  // polls without re-rendering every row.
  const [, setNow] = useState(Date.now);

  async function refresh() {
    try {
      const payload = await api<ModelsPayload>("/api/admin/models");
      setData(payload);
      // Pin "now" so the relative timestamps in the table don't drift.
      setNow(Date.now());
    } catch (err) {
      // Poll failures aren't fatal -- keep the prior frame onscreen and
      // surface once as a toast. Mirrors the rest of the page's "fail
      // silently, don't disrupt the operator" posture.
      message.error(`刷新失败：${(err as Error).message}`);
    }
  }

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, POLL_MS);
    return () => clearInterval(t);
  }, []);

  async function loadModel(id: string) {
    setActingOn(id);
    try {
      await api(`/api/admin/models/${encodeURIComponent(id)}/load`, { method: "POST" });
      message.success(`模型 ${id} 已加载`);
      await refresh();
    } catch (err) {
      const e = err as Error & { detail?: { active_sessions?: number } };
      message.error(`加载失败：${e.message}`);
    } finally {
      setActingOn(null);
    }
  }

  async function unloadModel(id: string) {
    setActingOn(id);
    try {
      await api(`/api/admin/models/${encodeURIComponent(id)}/unload`, { method: "POST" });
      message.success(`模型 ${id} 已卸载`);
      await refresh();
    } catch (err) {
      const e = err as Error & { detail?: { active_sessions?: number } };
      if (e.detail?.active_sessions != null) {
        message.error(`无法卸载：还有 ${e.detail.active_sessions} 个活动会话，请先关闭`);
      } else {
        message.error(`卸载失败：${e.message}`);
      }
    } finally {
      setActingOn(null);
    }
  }

  async function loadService(id: string) {
    setActingServiceOn(id);
    try {
      await api(`/api/admin/services/${encodeURIComponent(id)}/load`, { method: "POST" });
      message.success(`服务 ${id} 已加载`);
      await refresh();
    } catch (err) {
      message.error(`加载失败：${(err as Error).message}`);
    } finally {
      setActingServiceOn(null);
    }
  }

  async function unloadService(id: string) {
    setActingServiceOn(id);
    try {
      await api(`/api/admin/services/${encodeURIComponent(id)}/unload`, { method: "POST" });
      message.success(`服务 ${id} 已卸载`);
      await refresh();
    } catch (err) {
      message.error(`卸载失败：${(err as Error).message}`);
    } finally {
      setActingServiceOn(null);
    }
  }

  const columns: ColumnsType<ModelInfo> = [
    {
      title: "模型",
      dataIndex: "id",
      key: "id",
      width: 200,
      render: (id: string, row) => (
        <Space direction="vertical" size={2} style={{ lineHeight: 1.4 }}>
          <Typography.Text strong>{id}</Typography.Text>
          <Tag>{typeLabel(row.type)}</Tag>
        </Space>
      ),
    },
    {
      title: "组件",
      dataIndex: "components",
      key: "components",
      render: (_: unknown, row) => {
        if (!row.components?.length) return <Typography.Text type="secondary">—</Typography.Text>;
        const visible = row.components.slice(0, 4);
        const extra = row.components.length - visible.length;
        return (
          <Space size={4} wrap>
            {visible.map((c) => (
              <Tag key={c}>{c}</Tag>
            ))}
            {extra > 0 && <Tag>+{extra}</Tag>}
          </Space>
        );
      },
    },
    {
      title: "路径",
      dataIndex: "source",
      key: "source",
      ellipsis: true,
      render: (s: string | null) =>
        s ? (
          <Typography.Text code copyable={{ tooltips: ["复制", "已复制"] }}>
            {s}
          </Typography.Text>
        ) : (
          <Typography.Text type="secondary">—</Typography.Text>
        ),
    },
    {
      title: "磁盘大小",
      dataIndex: "estimated_bytes",
      key: "estimated_bytes",
      width: 120,
      render: (b: number) => formatBytes(b),
    },
    {
      title: "状态",
      key: "status",
      width: 180,
      render: (_: unknown, row) => (
        <Space direction="vertical" size={2} style={{ lineHeight: 1.4 }}>
          {row.loaded ? (
            <Tag color="success">已加载</Tag>
          ) : (
            <Tag color="default">未加载</Tag>
          )}
          <Typography.Text type="secondary" style={{ fontSize: 12 }}>
            {row.loaded ? `${formatLoadedAgo(row.loaded_at, Date.now())}` : "—"}
          </Typography.Text>
        </Space>
      ),
    },
    {
      title: "操作",
      key: "actions",
      width: 180,
      render: (_: unknown, row) => {
        // The action shows only what's actually possible, mirroring the
        // server's gate so the operator never has to read a 409/400
        // error to figure out what works on the current engine.
        if (!row.loaded) {
          // can_load = !loaded, gated server-side as well. We don't
          // render a button for mock engine slots: can_unload is the
          // only flag the API exposes today, so a not-yet-loaded mock
          // row with can_unload=False gets a tip explaining why.
          return (
            <Tooltip
              title={
                row.can_unload
                  ? "加载该模型到显存"
                  : "当前引擎不持有可加载的权重"
              }
            >
              <Button
                type="primary"
                size="small"
                icon={<CloudDownloadOutlined />}
                loading={actingOn === row.id}
                disabled={!row.can_unload}
                onClick={() => loadModel(row.id)}
              >
                加载
              </Button>
            </Tooltip>
          );
        }
        return (
          <Popconfirm
            title="确认卸载此模型？"
            description="将释放显存，正在运行的会话将失败。"
            okText="卸载"
            okButtonProps={{ danger: true }}
            cancelText="取消"
            onConfirm={() => unloadModel(row.id)}
            disabled={!row.can_unload}
          >
            <Tooltip
              title={
                row.can_unload
                  ? "卸载并释放显存"
                  : "当前引擎不持有可卸载的权重"
              }
            >
              <Button
                danger
                size="small"
                icon={<ThunderboltFilled />}
                loading={actingOn === row.id}
                disabled={!row.can_unload}
              >
                卸载
              </Button>
            </Tooltip>
          </Popconfirm>
        );
      },
    },
  ];

  return (
    <Space direction="vertical" size={16} style={{ width: "100%" }}>
      <Space align="center" style={{ width: "100%", justifyContent: "space-between" }}>
        <Typography.Title level={5} style={{ margin: 0 }}>
          模型与显存监控
        </Typography.Title>
        <Button icon={<ReloadOutlined />} onClick={refresh}>
          刷新
        </Button>
      </Space>

      {data && (
        <GpuCard gpu={data.gpu} models={data.models} activeSessions={data.active_sessions} />
      )}

      {data && (
        <Card title="模型" size="small" styles={{ body: { padding: 0 } }}>
          <Table<ModelInfo>
            rowKey="id"
            columns={columns}
            dataSource={data.models}
            pagination={false}
            size="small"
            locale={{ emptyText: "暂无模型" }}
          />
        </Card>
      )}

      {data && (
        <ServicesCard
          services={data.services}
          actingServiceOn={actingServiceOn}
          onLoad={loadService}
          onUnload={unloadService}
        />
      )}
    </Space>
  );
}

function ServicesCard({
  services,
  actingServiceOn,
  onLoad,
  onUnload,
}: {
  services: ServiceInfo[];
  actingServiceOn: string | null;
  onLoad: (id: string) => void;
  onUnload: (id: string) => void;
}) {
  // The wider net: every other model this server uses (portrait animators
  // + TTS). These can't be loaded / unloaded in-place like the drive
  // engine can -- they're either on-disk ready or not -- so the table
  // surfaces configuration + readiness + the operator's next action (a
  // fetch script invocation) and skips load/unload buttons entirely.

  // Sort so the most-frequently-relevant rows surface first: ready
  // backends up top, then by kind, then alphabetically. A freshly
  // running box always presents its ready inventory first.
  const sorted = [...services].sort((a, b) => {
    if (a.configured !== b.configured) return a.configured ? -1 : 1;
    if (a.kind !== b.kind) return a.kind.localeCompare(b.kind);
    return a.label_zh.localeCompare(b.label_zh);
  });

  const serviceColumns: ColumnsType<ServiceInfo> = [
    {
      title: "服务",
      key: "label",
      width: 240,
      render: (_: unknown, row) => (
        <Space direction="vertical" size={2} style={{ lineHeight: 1.4 }}>
          <Space size={6}>
            {kindIcon(row.kind)}
            <Typography.Text strong>{row.label_zh}</Typography.Text>
          </Space>
          <Space size={4} wrap>
            <Tag>{kindLabel(row.kind)}</Tag>
            {row.group_zh && <Tag color="default">{row.group_zh}</Tag>}
          </Space>
        </Space>
      ),
    },
    {
      title: "状态",
      key: "status",
      width: 130,
      render: (_: unknown, row) => (
        <Space direction="vertical" size={2} style={{ lineHeight: 1.4 }}>
          <StatusBadge ok={row.configured} label={row.status} />
          {row.kind === "portrait_animator" && !row.configured && (
            <Typography.Text type="secondary" style={{ fontSize: 12 }}>
              权重未就绪
            </Typography.Text>
          )}
        </Space>
      ),
    },
    {
      title: "显存预算",
      key: "vram",
      width: 140,
      render: (_: unknown, row) => {
        if (row.kind === "tts_service") return <Tag>云端</Tag>;
        if (row.vram_required_gb == null) return <Typography.Text type="secondary">复用</Typography.Text>;
        const fits = row.fits_current_gpu;
        return (
          <Space direction="vertical" size={2}>
            <Typography.Text>{row.vram_hint}</Typography.Text>
            <Tag
              color={
                fits == null ? "default" : fits ? "success" : "warning"
              }
            >
              {fits == null ? "未知" : fits ? "足够" : "显存不足"}
            </Tag>
          </Space>
        );
      },
    },
    {
      title: "路径 / 配置",
      key: "source",
      render: (_: unknown, row) => {
        if (row.kind === "tts_service") {
          // TTS rows show the operator what they've configured so they
          // can verify the voice + model without dropping into a
          // settings tab. Missing API key shows the env var name (the
          // fix) alongside.
          return (
            <Space direction="vertical" size={2} style={{ lineHeight: 1.4 }}>
              {row.voice_id && (
                <Typography.Text>
                  <Typography.Text type="secondary">语音：</Typography.Text>
                  <Typography.Text code style={{ marginLeft: 4 }}>
                    {row.voice_id}
                  </Typography.Text>
                </Typography.Text>
              )}
              <Typography.Text type="secondary">
                模型：{row.model ?? "—"} · 采样率 {row.sample_rate ?? "—"}/{" "}
                {row.sample_rate_cap_hz ?? "—"} Hz
              </Typography.Text>
            </Space>
          );
        }
        // Animator rows: on-disk weights path, copyable. The fetch
        // script hint lives in the action column so a missing backend
        // here doesn't crowd the path text.
        return row.weights_dir ? (
          <Typography.Text code copyable={{ tooltips: ["复制", "已复制"] }} ellipsis>
            {row.weights_dir}
          </Typography.Text>
        ) : (
          <Typography.Text type="secondary">—</Typography.Text>
        );
      },
    },
    {
      title: "操作",
      key: "actions",
      width: 220,
      render: (_: unknown, row) => {
        // Three branches the operator cares about:
        //   1. weights missing (with fetch script)  -> show the script
        //   2. animator fully ready                 -> 加载 / 卸载 button
        //   3. TTS / remote service (no in-process weights) -> info text
        //
        // Animator endpoints always 503 instead of auto-loading when
        // their backend isn't pre-allocated here, so the operator's
        // single source of truth for "is this thing on the GPU right
        // now" sits in this tab.
        if (!row.configured && row.missing_script) {
          return (
            <Tooltip title="在项目根目录执行以下命令以下载该模型的权重">
              <Typography.Text code style={{ fontSize: 12 }}>
                {row.missing_script}
              </Typography.Text>
            </Tooltip>
          );
        }
        if (!row.loaded && row.can_load) {
          return (
            <Tooltip title="立即将模型加载到显存（首次加载可能耗时 30 秒以上）">
              <Button
                type="primary"
                size="small"
                icon={<PoweroffOutlined />}
                loading={actingServiceOn === row.id}
                onClick={() => onLoad(row.id)}
              >
                加载
              </Button>
            </Tooltip>
          );
        }
        if (row.loaded && row.can_unload) {
          return (
            <Popconfirm
              title="确认卸载此服务？"
              description="将释放显存，正在使用此服务的请求将失败。"
              okText="卸载"
              okButtonProps={{ danger: true }}
              cancelText="取消"
              onConfirm={() => onUnload(row.id)}
            >
              <Button
                danger
                size="small"
                icon={<ThunderboltFilled />}
                loading={actingServiceOn === row.id}
              >
                卸载
              </Button>
            </Popconfirm>
          );
        }
        if (row.kind === "tts_service" && row.configured) {
          return (
            <Typography.Text type="secondary">
              在调试页的 TTS 面板中可用
            </Typography.Text>
          );
        }
        return (
          <Typography.Text type="secondary">无可用操作</Typography.Text>
        );
      },
    },
  ];

  // The "ready" alert only appears when there's a non-trivial number
  // of services -- on a freshly-installed box most rows will be red
  // and the alert would just repeat itself. Pick a representative
  // non-ready service to call out so the alert's reason is actionable.
  const unready = sorted.filter((s) => !s.configured);
  return (
    <Card
      title={
        <Space>
          服务
          <Tag color="default">{sorted.length}</Tag>
        </Space>
      }
      size="small"
      styles={{ body: { padding: 0 } }}
      extra={
        sorted.length > 0 ? (
          <Typography.Text type="secondary">
            {sorted.filter((s) => s.configured).length} / {sorted.length} 已配置
          </Typography.Text>
        ) : null
      }
    >
      {unready.length > 0 && (
        // One terse alert summarising the gap; per-row the action column
        // carries the actual fetch command. Keeping it brief prevents the
        // tab from feeling like an error report when the operator just
        // hasn't fetched the weights yet.
        <div style={{ padding: "8px 16px 0" }}>
          <Alert
            type="warning"
            showIcon
            message={
              <span>
                以下模型未就绪：{unready.map((s) => s.label_zh).join("、")}
                ；在项目根目录运行对应脚本以下载权重。
              </span>
            }
          />
        </div>
      )}
      <Table<ServiceInfo>
        rowKey="id"
        columns={serviceColumns}
        dataSource={sorted}
        pagination={false}
        size="small"
        locale={{ emptyText: "暂无服务" }}
      />
    </Card>
  );
}
