import { useEffect, useState } from "react";
import {
  App as AntApp,
  Alert,
  Button,
  Card,
  Form,
  Input,
  Modal,
  Popconfirm,
  Progress,
  Select,
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
  RobotOutlined,
  SettingOutlined,
  SoundOutlined,
  ThunderboltFilled,
  UserOutlined,
} from "@ant-design/icons";
import { api } from "../api";
import type {
  DhModelInfo,
  GpuSnapshot,
  LlmConfigInfo,
  LlmConfigUpdate,
  ModelsPayload,
  ModelInfo,
  ServiceInfo,
  TtsConfigInfo,
  TtsConfigUpdate,
} from "../types";

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
  if (t === "portrait_animator") return "形象动画器";
  return t;
}

function kindLabel(k: string): string {
  // Service-row "类别" column. Keep the strings short so they fit in
  // a Tag next to a longer label_zh on the same row.
  if (k === "portrait_animator") return "形象动画器";
  if (k === "tts_service") return "TTS 服务";
  if (k === "llm_service") return "大模型";
  return k;
}

function kindIcon(k: string) {
  if (k === "portrait_animator") return <UserOutlined />;
  if (k === "tts_service") return <SoundOutlined />;
  if (k === "llm_service") return <RobotOutlined />;
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
  // Which service's settings dialog is open, if any. ``null`` means
  // closed. TTS and LLM live in their own dialogs; the row's id picks
  // which one opens.
  const [openDialog, setOpenDialog] = useState<"tts" | "llm" | null>(null);
  // The digital-human model the operator has picked but not saved yet.
  // ``null`` means "follow the server" -- the poll refreshes it every 2 s,
  // and a pending choice would otherwise be overwritten mid-edit.
  const [pickedModel, setPickedModel] = useState<string | null>(null);
  const [savingModel, setSavingModel] = useState(false);
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

  async function saveActiveModel(name: string) {
    setSavingModel(true);
    try {
      const resp = await api<{ selected_model: string; restart_required: boolean }>(
        "/api/admin/models/active",
        { method: "PUT", body: { name } },
      );
      message.success(
        resp.restart_required
          ? `已选择 ${resp.selected_model}，重启服务后生效`
          : `当前已运行 ${resp.selected_model}`,
      );
      // Re-read first, then drop the pending choice: the Select falls back
      // to the server's value, and that value has to be the new one by the
      // time it does.
      await refresh();
      setPickedModel(null);
    } catch (err) {
      message.error(`保存失败：${(err as Error).message}`);
    } finally {
      setSavingModel(false);
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
          {/* Animator rows have a ``label_zh`` (LivePortrait / Wan 2.2 etc.);
              engine rows use the raw id (e.g. ``musetalk_runtime``). The
              id is shown as the secondary line so the operator can copy it. */}
          <Typography.Text strong>{row.label_zh || id}</Typography.Text>
          <Space size={4}>
            <Tag>{typeLabel(row.type)}</Tag>
            {row.label_zh && (
              <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                {id}
              </Typography.Text>
            )}
          </Space>
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
          // The load button is gated on ``can_load``: True when the row
          // exposes a ``load()`` method AND isn't currently loaded.
          // ``can_unload`` is the wrong flag here -- it stays False after
          // an unload cycle, which left the button permanently disabled.
          return (
            <Tooltip
              title={
                row.can_load
                  ? "加载该模型到显存"
                  : "当前引擎不持有可加载的权重"
              }
            >
              <Button
                type="primary"
                size="small"
                icon={<CloudDownloadOutlined />}
                loading={actingOn === row.id}
                disabled={!row.can_load}
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
        <ActiveModelCard
          activeModel={data.active_model}
          selectedModel={data.selected_model}
          availableModels={data.available_models}
          picked={pickedModel}
          saving={savingModel}
          onPick={setPickedModel}
          onSave={(name) => saveActiveModel(name)}
        />
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
          onOpenSettings={(kind) => setOpenDialog(kind)}
        />
      )}

      <MiniMaxConfigDialog
        open={openDialog === "tts"}
        onClose={() => setOpenDialog(null)}
        onSaved={refresh}
      />
      <LlmConfigDialog
        open={openDialog === "llm"}
        onClose={() => setOpenDialog(null)}
        onSaved={refresh}
      />
    </Space>
  );
}

// ----- the digital-human model selector -------------------------------------

// Which digital-human model this box runs. One model bundles the avatar
// pipeline and the drive engine, so there is a single switch -- the two
// have to change together or a pack written by one model would be driven
// by another.
//
// The engine is a process-wide singleton holding the model's weights in
// VRAM, so a change is applied by restarting the server. The card says so
// rather than pretending the switch took effect immediately.
function ActiveModelCard({
  activeModel,
  selectedModel,
  availableModels,
  picked,
  saving,
  onPick,
  onSave,
}: {
  activeModel: DhModelInfo;
  selectedModel: string;
  availableModels: DhModelInfo[];
  picked: string | null;
  saving: boolean;
  onPick: (name: string) => void;
  onSave: (name: string) => void;
}) {
  const chosen = picked ?? selectedModel;
  const label = (name: string) =>
    availableModels.find((m) => m.name === name)?.label_zh ?? name;
  const pending = chosen !== activeModel.name;

  return (
    <Card title="当前数字人模型" size="small">
      <Space direction="vertical" size={8} style={{ width: "100%" }}>
        <Space wrap size={24}>
          <span>
            运行中：<Typography.Text strong>{activeModel.label_zh}</Typography.Text>
          </span>
          <span>
            已选择：<Typography.Text strong>{label(chosen)}</Typography.Text>
          </span>
        </Space>
        <Space wrap>
          <Select
            style={{ minWidth: 240 }}
            value={chosen}
            onChange={onPick}
            options={availableModels.map((m) => ({ value: m.name, label: m.label_zh }))}
          />
          <Button
            type="primary"
            loading={saving}
            disabled={!pending}
            onClick={() => onSave(chosen)}
          >
            保存
          </Button>
        </Space>
        {pending && (
          <Alert
            type="info"
            showIcon
            message="数字人模型在重启服务后生效：引擎是持有显存的进程级单例，形象制作流水线必须与它一起切换。"
          />
        )}
      </Space>
    </Card>
  );
}

function ServicesCard({
  services,
  actingServiceOn,
  onLoad,
  onUnload,
  onOpenSettings,
}: {
  services: ServiceInfo[];
  actingServiceOn: string | null;
  onLoad: (id: string) => void;
  onUnload: (id: string) => void;
  // Open a settings dialog for the given service kind. Parent owns the
  // dialog lifecycle so the click feels like a one-shot modal.
  onOpenSettings: (kind: "tts" | "llm") => void;
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
          // can verify the model + sample rate without dropping into a
          // settings tab. voice_id is per-session in the debug page now,
          // so it doesn't live here.
          return (
            <Space direction="vertical" size={2} style={{ lineHeight: 1.4 }}>
              <Typography.Text type="secondary">
                模型：{row.model ?? "—"} · 采样率 {row.sample_rate ?? "—"}/{" "}
                {row.sample_rate_cap_hz ?? "—"} Hz
              </Typography.Text>
              <Typography.Text type="secondary">
                语音：调试页选择
              </Typography.Text>
            </Space>
          );
        }
        if (row.kind === "llm_service") {
          // LLM rows: show base_url + model so the operator can verify
          // the upstream provider without dropping into settings.
          return (
            <Space direction="vertical" size={2} style={{ lineHeight: 1.4 }}>
              <Typography.Text>
                <Typography.Text type="secondary">Base URL：</Typography.Text>
                <Typography.Text code style={{ marginLeft: 4 }} ellipsis>
                  {row.base_url || "—"}
                </Typography.Text>
              </Typography.Text>
              <Typography.Text type="secondary">
                模型：{row.model ?? "—"}
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
        // Four branches the operator cares about:
        //   1. weights missing (with fetch script)  -> show the script
        //   2. animator fully ready                 -> 加载 / 卸载 button
        //   3. remote service (TTS / LLM)            -> 设置 button (open
        //      the corresponding settings dialog) once configured, or a
        //      fetch hint if missing required env vars.
        //
        // Animator endpoints always 503 instead of auto-loading when
        // their backend isn't pre-allocated here, so the operator's
        // single source of truth for "is this thing on the GPU right
        // now" sits in this tab.
        if (!row.configured && row.missing_script) {
          return (
            <Tooltip title="在项目根目录执行以下命令以下载该模型的权重 / 设置环境变量">
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
        if (row.kind === "tts_service" || row.kind === "llm_service") {
          // Both cloud services are configured from this dialog, not by a
          // fetch script (``missing_script`` is always None server-side),
          // so the [设置] button is the operator's only entry point --
          // including before the service is configured at all. Hiding it
          // until then would leave an unconfigured service with no way
          // to be configured.
          const target = row.kind === "tts_service" ? "tts" : "llm";
          return (
            <Button
              size="small"
              icon={<SettingOutlined />}
              onClick={() => onOpenSettings(target)}
            >
              设置
            </Button>
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

// ----- MiniMax TTS configuration (now a dialog, not a card) -----------------

const MINIMAX_MODEL_OPTIONS = [
  { value: "speech-02-hd", label: "speech-02-hd (默认)" },
  { value: "speech-02-turbo", label: "speech-02-turbo" },
];
const MINIMAX_SAMPLE_RATE_OPTIONS = [
  { value: 8000, label: "8000 Hz" },
  { value: 16000, label: "16000 Hz" },
  { value: 22050, label: "22050 Hz" },
  { value: 24000, label: "24000 Hz" },
  { value: 32000, label: "32000 Hz" },
  { value: 44100, label: "44100 Hz" },
];

function MiniMaxConfigDialog({
  open,
  onClose,
  onSaved,
}: {
  open: boolean;
  onClose: () => void;
  onSaved: () => Promise<void> | void;
}) {
  // The Models tab opens this from the MiniMax TTS row's "设置" action.
  // It's a thinner form than the old inline card: ``voice_id`` is gone
  // (the operator picks it per session in the debug page), and the
  // remaining fields -- API key, base URL, group ID, model, sample rate
  // -- are the same. "测试" still POSTs to /api/admin/tts to verify the
  // round-trip end to end; it uses a hardcoded preset voice_id because
  // the dialog is config-only and shouldn't require the operator to
  // jump to the debug page to test their save.
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [savedMasked, setSavedMasked] = useState<string>("");
  const [apiKeyInput, setApiKeyInput] = useState("");
  const { message } = AntApp.useApp();

  // Refresh every time the dialog opens, not on mount only, so an
  // operator who edits via /admin/settings then opens this dialog sees
  // their change without a manual reload.
  useEffect(() => {
    if (!open) return;
    (async () => {
      try {
        const cfg = await api<TtsConfigInfo>("/api/admin/tts/config");
        form.setFieldsValue({
          model: cfg.model,
          sample_rate: cfg.sample_rate,
          group_id: cfg.group_id,
          base_url: cfg.base_url,
        });
        setSavedMasked(cfg.api_key_masked);
        setApiKeyInput("");
      } catch {
        /* form submit will surface a clear error */
      }
    })();
  }, [open, form]);

  async function save() {
    setLoading(true);
    try {
      const values = await form.validateFields();
      const payload: TtsConfigUpdate = { ...values };
      // Only include the api_key if the operator actually typed something;
      // leaving the field blank keeps whatever key is currently saved.
      if (apiKeyInput.trim()) {
        payload.api_key = apiKeyInput.trim();
      }
      const resp = await api<TtsConfigInfo & { api_key_masked: string }>(
        "/api/admin/tts/config",
        { method: "PUT", body: payload }
      );
      setSavedMasked(resp.api_key_masked);
      setApiKeyInput("");
      message.success("MiniMax 配置已保存");
      await onSaved();
    } catch (err) {
      message.error(`保存失败：${(err as Error).message}`);
    } finally {
      setLoading(false);
    }
  }

  async function testSynth() {
    try {
      const resp = await fetch("/api/admin/tts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        // The synthesise endpoint requires voice_id now; the dialog
        // doesn't carry one (that's the debug page's job), so the test
        // uses a known preset. Operators can swap to a different preset
        // via the debug page once they confirm the round-trip works.
        body: JSON.stringify({ text: "测试", voice_id: "male-qn-qingse" }),
      });
      if (!resp.ok) {
        const body = await resp.json().catch(() => null);
        const code = body?.error?.code;
        if (code === "tts_not_configured") {
          message.warning("未配置 API key，无法测试");
        } else {
          message.error(`测试失败：${body?.error?.message ?? resp.statusText}`);
        }
        return;
      }
      message.success("测试成功：MiniMax 已返回音频");
    } catch (err) {
      message.error(`测试失败：${(err as Error).message}`);
    }
  }

  return (
    <Modal
      title="MiniMax TTS 配置"
      open={open}
      onCancel={onClose}
      footer={null}
      destroyOnClose
      width={560}
    >
      <Form form={form} layout="vertical" style={{ marginTop: 16 }}>
        <Form.Item
          label="API Key"
          extra={
            savedMasked
              ? `当前保存值：${savedMasked}（留空表示保留原值）`
              : "Bearer token，留空表示保留当前值"
          }
        >
          <Input.Password
            value={apiKeyInput}
            onChange={(e) => setApiKeyInput(e.target.value)}
            placeholder="留空表示保留当前 API Key"
          />
        </Form.Item>
        <Form.Item name="group_id" label="Group ID" extra="可选；新版 MiniMax 账号通常无需">
          <Input placeholder="例如 g-xxx" />
        </Form.Item>
        <Form.Item
          name="base_url"
          label="Base URL"
          extra="默认 https://api.minimaxi.com；自建/代理可改"
          rules={[{ required: true, message: "Base URL 必填" }]}
        >
          <Input placeholder="https://api.minimaxi.com" />
        </Form.Item>
        <Form.Item name="model" label="模型">
          <Select options={MINIMAX_MODEL_OPTIONS} />
        </Form.Item>
        <Form.Item name="sample_rate" label="采样率">
          <Select options={MINIMAX_SAMPLE_RATE_OPTIONS} />
        </Form.Item>
        <Typography.Text type="secondary" style={{ display: "block", marginBottom: 16, fontSize: 12 }}>
          语音 ID 不在这里配置 —— 在会话调试页选择音色。
        </Typography.Text>
        <Form.Item style={{ marginBottom: 0 }}>
          <Space>
            <Button type="primary" icon={<CloudDownloadOutlined />} loading={loading} onClick={save}>
              保存
            </Button>
            <Button icon={<SoundOutlined />} onClick={testSynth}>
              测试
            </Button>
          </Space>
        </Form.Item>
      </Form>
    </Modal>
  );
}

// ----- LLM (OpenAI-compatible chat) configuration dialog --------------------


function LlmConfigDialog({
  open,
  onClose,
  onSaved,
}: {
  open: boolean;
  onClose: () => void;
  onSaved: () => Promise<void> | void;
}) {
  // LLM configuration mirrors the TTS dialog: a Modal opened from the
  // service row's "设置" action, three fields (api_key + base_url + model),
  // save round-trips through ``PUT /api/admin/llm/config``. There's no
  // "测试" button: the LLM only exposes streaming chat, which doesn't
  // fit the same one-shot synthesise probe TTS gets. Operators confirm
  // end-to-end via the chat panel in the debug page.
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [savedMasked, setSavedMasked] = useState<string>("");
  const [apiKeyInput, setApiKeyInput] = useState("");
  const { message } = AntApp.useApp();

  useEffect(() => {
    if (!open) return;
    (async () => {
      try {
        const cfg = await api<LlmConfigInfo>("/api/admin/llm/config");
        form.setFieldsValue({
          base_url: cfg.base_url,
          model: cfg.model,
        });
        setSavedMasked(cfg.api_key_masked);
        setApiKeyInput("");
      } catch {
        /* form submit will surface a clear error */
      }
    })();
  }, [open, form]);

  async function save() {
    setLoading(true);
    try {
      const values = await form.validateFields();
      const payload: LlmConfigUpdate = { ...values };
      if (apiKeyInput.trim()) {
        payload.api_key = apiKeyInput.trim();
      }
      const resp = await api<LlmConfigInfo & { api_key_masked: string; configured: boolean }>(
        "/api/admin/llm/config",
        { method: "PUT", body: payload }
      );
      setSavedMasked(resp.api_key_masked);
      setApiKeyInput("");
      message.success("大模型配置已保存");
      await onSaved();
    } catch (err) {
      message.error(`保存失败：${(err as Error).message}`);
    } finally {
      setLoading(false);
    }
  }

  return (
    <Modal
      title="大模型 (OpenAI 兼容) 配置"
      open={open}
      onCancel={onClose}
      footer={null}
      destroyOnClose
      width={560}
    >
      <Form form={form} layout="vertical" style={{ marginTop: 16 }}>
        <Form.Item
          label="API Key"
          extra={
            savedMasked
              ? `当前保存值：${savedMasked}（留空表示保留原值）`
              : "Bearer token，留空表示保留当前值"
          }
        >
          <Input.Password
            value={apiKeyInput}
            onChange={(e) => setApiKeyInput(e.target.value)}
            placeholder="留空表示保留当前 API Key"
          />
        </Form.Item>
        <Form.Item
          name="base_url"
          label="Base URL"
          extra={
            <span>
              例：<code>https://api.openai.com/v1</code>、<code>http://localhost:11434/v1</code>（Ollama）。
              会话调试页大模型对话的流式请求会拼接 <code>/chat/completions</code>。
            </span>
          }
          rules={[
            { required: true, message: "Base URL 必填" },
            {
              validator: (_, v: string) =>
                !v || /^https?:\/\//.test(v)
                  ? Promise.resolve()
                  : Promise.reject(new Error("必须以 http:// 或 https:// 开头")),
            },
          ]}
        >
          <Input placeholder="https://api.openai.com/v1" />
        </Form.Item>
        <Form.Item
          name="model"
          label="模型"
          extra="上游模型标识，例如 gpt-3.5-turbo、gpt-4o-mini、llama3 等"
          rules={[{ required: true, message: "模型必填" }]}
        >
          <Input placeholder="gpt-3.5-turbo" />
        </Form.Item>
        <Form.Item style={{ marginBottom: 0 }}>
          <Space>
            <Button type="primary" icon={<CloudDownloadOutlined />} loading={loading} onClick={save}>
              保存
            </Button>
          </Space>
        </Form.Item>
      </Form>
    </Modal>
  );
}
