import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  Alert,
  App as AntApp,
  Badge,
  Button,
  Card,
  Drawer,
  Empty,
  Flex,
  Form,
  Image,
  Input,
  InputNumber,
  Popconfirm,
  Progress,
  Select,
  Space,
  Table,
  Tag,
  Typography,
  Upload,
} from "antd";
import type { ColumnsType } from "antd/es/table";
import type { UploadFile } from "antd/es/upload/interface";
import {
  CloudUploadOutlined,
  DeleteOutlined,
  DownloadOutlined,
  EyeOutlined,
  InboxOutlined,
  ReloadOutlined,
  SwapOutlined,
} from "@ant-design/icons";
import { api, downloadWithToken } from "../api";
import type { GenerationTask } from "../types";
import { previewUrl as driverPreviewUrl } from "./LivePortraits";

// These come from GET /api/admin/backends — the server reads its own
// registered portrait animators (see server/avatar/animators.py) so the UI
// stays in lockstep with whatever the operator has installed.
type BackendOption = {
  name: string;
  label_zh: string;
  group_zh: string;
  vram_hint: string;
  description: string;
  accepts_prompt: boolean;
  accepts_num_frames: boolean;
  accepts_steps: boolean;
  num_frames_min?: number;
  num_frames_max?: number;
  steps_min?: number;
  steps_max?: number;
  vram_required_gb?: number | null;
  fits_current_gpu?: boolean | null;
};

type BackendsResponse = {
  backends: BackendOption[];
};

type DriverInfo = {
  name: string;
  filename: string;
  kind: "mp4" | "pkl";
  size_bytes: number;
  cached: boolean;
  canonical: boolean;
  label_zh: string;
  frames?: number;
  has_preview?: boolean;
};

function formatBytes(b: number): string {
  if (b < 1024) return `${b} B`;
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB`;
  return `${(b / 1024 / 1024).toFixed(1)} MB`;
}

async function fetchDrivers(): Promise<DriverInfo[]> {
  try {
    return (await api<{ drivers: DriverInfo[] }>("/api/admin/liveportrait-drivers")).drivers ?? [];
  } catch {
    return [];
  }
}

async function fetchSettings(): Promise<{ default_liveportrait_driver?: string } | null> {
  try {
    return await api<{ default_liveportrait_driver?: string }>("/api/admin/settings");
  } catch {
    return null;
  }
}

async function fetchBackends(): Promise<BackendOption[]> {
  try {
    return (await api<BackendsResponse>("/api/admin/backends")).backends ?? [];
  } catch {
    return [];
  }
}

// Poll cadence for the records list while any task is in flight. One second
// is fast enough that the operator sees meaningful progress without
// hammering the server. Mirrors the cadence the deleted Tasks tab used;
// the lifecycle is the same (the new record appears at the top and the
// row's progress bar climbs every tick).
const POLL_MS = 1000;

async function submitTask(
  backend: string,
  file: File,
  prompt: string,
  numFrames: number,
  steps: number,
  driver: string,
): Promise<string> {
  const body = new FormData();
  body.append("backend", backend);
  body.append("file", file);
  body.append("prompt", prompt);
  body.append("num_frames", String(numFrames));
  body.append("steps", String(steps));
  body.append("driver", driver);
  const resp = await fetch("/api/admin/i2v", {
    method: "POST",
    body,
  });
  if (resp.status !== 202) {
    let msg = resp.statusText;
    try {
      const data = await resp.json();
      const err = data?.error ?? {};
      msg = err?.message ?? msg;
      const detail = err?.detail;
      if (detail && typeof detail === "object") {
        const parts = Object.entries(detail).map(([k, v]) => `${k}: ${v}`);
        if (parts.length) msg = `${msg}（${parts.join("；")}）`;
      }
    } catch {
      /* non-json body -- keep statusText */
    }
    throw new Error(msg);
  }
  const json = (await resp.json()) as { task_id: string };
  return json.task_id;
}

async function fetchTaskBytes(id: string): Promise<Blob> {
  // The task result endpoint streams the bytes the animator wrote. Used
  // both for the drawer's <video> preview and for the "use this video to
  // create an avatar" handoff.
  const resp = await fetch(`/api/admin/tasks/${id}/result.mp4`);
  if (!resp.ok) throw new Error(`result fetch failed: ${resp.status}`);
  return await resp.blob();
}

const FALLBACK_DEFAULTS = {
  // ``0`` (or empty) is interpreted server-side as "use the template's
  // full duration" -- this prevents the previous default-49 behaviour
  // from silently truncating a 354-frame template to 49 frames.
  numFrames: 0,
  steps: 20,
  prompt: "a person speaking naturally with subtle head and mouth motion",
};

// Visual language for the status column. Matches the deleted Tasks tab
// so the operator's colour memory carries over.
function statusColor(s: string): string {
  if (s === "done") return "green";
  if (s === "failed") return "red";
  if (s === "running") return "blue";
  return "default";
}

function statusLabel(s: string): string {
  if (s === "queued") return "排队中";
  if (s === "running") return "运行中";
  if (s === "done") return "已完成";
  if (s === "failed") return "失败";
  return s;
}

export default function ActivatePortrait() {
  const [form] = Form.useForm<{ prompt?: string; numFrames?: number; steps?: number }>();
  const [fileList, setFileList] = useState<UploadFile[]>([]);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");
  const [backends, setBackends] = useState<BackendOption[]>([]);
  const [picked, setPicked] = useState<string | undefined>(undefined);
  const [drivers, setDrivers] = useState<DriverInfo[]>([]);
  const [pickedDriver, setPickedDriver] = useState<string | undefined>(undefined);

  // Records list -- every i2v task the server has retained (newest-first).
  // Replaces the deleted Tasks tab's table; the same /api/admin/tasks
  // endpoint feeds it. The list poll (below) handles progress for in-
  // flight rows, so the row an operator just submitted climbs from queued
  // -> running -> done in place.
  const [items, setItems] = useState<GenerationTask[]>([]);
  const [listLoading, setListLoading] = useState(true);
  // Held inside a ref so the polling effect can re-arm itself with the
  // latest refresh closure without retriggering on every render.
  const refreshItemsRef = useRef<() => Promise<void>>(async () => undefined);
  // Captures the first-load failure so a transient blip during a poll tick
  // doesn't keep flashing an error after the list has already rendered.
  const initialLoadFailedRef = useRef(false);

  // 查看 drawer: a single open record + the blob URL for its preview.
  // The drawer re-renders on every list poll so the open row's progress
  // stays live (same pattern the deleted Tasks drawer used).
  const [drawerTask, setDrawerTask] = useState<GenerationTask | null>(null);
  const [drawerUrl, setDrawerUrl] = useState<string>("");
  const [drawerLoadError, setDrawerLoadError] = useState("");

  const navigate = useNavigate();
  const { message } = AntApp.useApp();

  useEffect(() => {
    void fetchBackends().then((rows) => {
      setBackends(rows);
      setPicked((current) => current ?? rows[0]?.name);
    });
  }, []);

  // Driver list -- only meaningful for the LivePortrait backend, but we fetch
  // it eagerly so the dropdown is populated by the time the user picks that
  // backend.
  useEffect(() => {
    void fetchDrivers().then((rows) => {
      setDrivers(rows);
      void fetchSettings().then((settings) => {
        const preferred = settings?.default_liveportrait_driver;
        if (preferred && rows.some((d) => d.name === preferred)) {
          setPickedDriver((current) => current ?? preferred);
        } else {
          setPickedDriver((current) => current ?? rows[0]?.name);
        }
      });
    });
  }, []);

  // When the backend switches away from LivePortrait, the driver value is
  // ignored; reset it to the first available so we don't leak a stale id
  // into the next LivePortrait run.
  useEffect(() => {
    if (picked !== "liveportrait") return;
    setPickedDriver((current) =>
      drivers.find((d) => d.name === current) ? current : drivers[0]?.name,
    );
  }, [picked, drivers]);

  // numFrames carries different semantics per backend:
  //   - LivePortrait / MuseTalk silent: 0 = "use the template's full length"
  //   - Wan 2.x: must be 16N+1 in [17, 113] (the form InputNumber slider
  //     enforces the upper bound; the lower bound and the off-grid case
  //     are checked here)
  // When the operator switches to Wan with a stale LivePortrait-era 0 in
  // the field, snap to 17 (the smallest valid 16N+1) so the request doesn't
  // 400. We only auto-fix when the current value is invalid for Wan -- an
  // operator who already typed 33 keeps their value.
  useEffect(() => {
    if (!picked || !picked.startsWith("wan")) return;
    const current = form.getFieldValue("numFrames");
    const onGrid = (n: unknown): boolean =>
      typeof n === "number" && n >= 17 && (n - 1) % 16 === 0;
    if (!onGrid(current)) {
      form.setFieldsValue({ numFrames: 17 });
    }
  }, [picked, form]);

  // Revoke blob URLs on unmount so we don't leak the rendered clip.
  useEffect(() => {
    return () => {
      if (drawerUrl) URL.revokeObjectURL(drawerUrl);
    };
  }, [drawerUrl]);

  async function refreshItems(): Promise<void> {
    try {
      const body = await api<{ tasks: GenerationTask[] }>("/api/admin/tasks");
      // Only i2v rows surface here; the table's "type" column would always
      // read "图生视频" today, but filtering keeps the page honest when a
      // future task type is registered.
      setItems(body.tasks.filter((t) => t.type === "i2v"));
    } catch (err) {
      if (!initialLoadFailedRef.current) {
        initialLoadFailedRef.current = true;
        message.error((err as Error).message);
      }
    } finally {
      setListLoading(false);
    }
  }
  refreshItemsRef.current = refreshItems;

  // Initial load.
  useEffect(() => {
    void refreshItems();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Poll while any row is in flight. Stops the moment the list shows only
  // terminal states; a new submission re-arms it because the optimistic
  // insert pushes the row into "queued" state.
  const stillRunning = items.some((t) => t.status === "queued" || t.status === "running");
  useEffect(() => {
    if (!stillRunning) return;
    const t = window.setTimeout(() => {
      void refreshItemsRef.current();
    }, POLL_MS);
    return () => window.clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stillRunning, items]);

  // Keep the drawer's open task in sync with the latest poll so the detail
  // panel updates while it's open (e.g. progress climbs from 30% to 90%).
  useEffect(() => {
    if (drawerTask === null) return;
    const fresh = items.find((t) => t.task_id === drawerTask.task_id);
    if (fresh && fresh !== drawerTask) setDrawerTask(fresh);
  }, [items, drawerTask]);

  // Group the registry's flat list into the OptGroup buckets the UI shows.
  const grouped = useMemo(() => {
    const map = new Map<string, BackendOption[]>();
    for (const b of backends) {
      const key = b.group_zh || "其他";
      const arr = map.get(key) ?? [];
      arr.push(b);
      map.set(key, arr);
    }
    return Array.from(map.entries()).map(([label, options]) => ({ label, options }));
  }, [backends]);

  const current = useMemo(() => backends.find((b) => b.name === picked), [backends, picked]);

  async function onGenerate() {
    if (!picked) {
      message.warning("暂未注册任何后端，无法生成");
      return;
    }
    const file = fileList[0]?.originFileObj;
    if (!file) {
      message.warning("请先选择一张图片");
      return;
    }
    let payload: { prompt?: string; numFrames?: number; steps?: number };
    try {
      payload = await form.validateFields();
    } catch {
      /* Form validation messages render inline -- nothing to do here */
      return;
    }
    const prompt = payload.prompt ?? FALLBACK_DEFAULTS.prompt;
    const numFrames = payload.numFrames ?? FALLBACK_DEFAULTS.numFrames;
    const steps = payload.steps ?? FALLBACK_DEFAULTS.steps;
    const driver = pickedDriver ?? drivers[0]?.name ?? "subtle";

    setError("");
    setSubmitting(true);
    let taskId: string;
    try {
      taskId = await submitTask(picked, file, prompt, numFrames, steps, driver);
    } catch (err) {
      setError((err as Error).message);
      setSubmitting(false);
      return;
    }
    // Optimistic insert: the row lands at the top with status=queued
    // before the first poll replaces it. Without this, the operator sees
    // a 1s blank gap between clicking 生成视频 and the row appearing.
    const placeholder: GenerationTask = {
      task_id: taskId,
      type: "i2v",
      status: "queued",
      stage: "",
      progress: 0,
      params: {
        backend: picked,
        prompt,
        num_frames: numFrames,
        steps,
        driver,
        filename: file.name,
      },
      result: null,
      error: null,
      created_at: new Date().toISOString(),
      updated_at: null,
    };
    setItems((prev) => [placeholder, ...prev]);
    setSubmitting(false);
  }

  async function onDelete(taskId: string) {
    try {
      await api(`/api/admin/tasks/${taskId}`, { method: "DELETE" });
      message.success("已删除");
      setItems((prev) => prev.filter((t) => t.task_id !== taskId));
      if (drawerTask?.task_id === taskId) {
        if (drawerUrl) URL.revokeObjectURL(drawerUrl);
        setDrawerUrl("");
        setDrawerTask(null);
      }
    } catch (err) {
      message.error((err as Error).message);
    }
  }

  async function openDrawer(task: GenerationTask) {
    setDrawerTask(task);
    setDrawerLoadError("");
    if (drawerUrl) {
      URL.revokeObjectURL(drawerUrl);
      setDrawerUrl("");
    }
    if (task.status !== "done") return;
    try {
      const blob = await fetchTaskBytes(task.task_id);
      const url = URL.createObjectURL(blob);
      setDrawerUrl(url);
    } catch (err) {
      setDrawerLoadError((err as Error).message);
    }
  }

  function closeDrawer() {
    setDrawerTask(null);
    if (drawerUrl) {
      URL.revokeObjectURL(drawerUrl);
      setDrawerUrl("");
    }
  }

  async function onDownload() {
    if (!drawerTask) return;
    try {
      await downloadWithToken(`/api/admin/tasks/${drawerTask.task_id}/result.mp4`);
    } catch (err) {
      message.error((err as Error).message);
    }
  }

  // Hand the generated mp4 over to the avatar create form. The file
  // travels through the browser (we re-fetch the bytes from the task
  // result endpoint, wrap them as a File, and pass via router state),
  // avoiding any need for server-side staging of "uploads from the i2v
  // tab".
  async function onUseVideoForAvatar() {
    if (!drawerTask) return;
    try {
      const blob = await fetchTaskBytes(drawerTask.task_id);
      const file = new File([blob], "generated.mp4", { type: "video/mp4" });
      navigate("/avatars", { state: { prefilledFile: file } });
    } catch (err) {
      message.error((err as Error).message);
    }
  }

  // The "disabled" hints on each input react to the picked backend: a backend
  // that doesn't read a knob still receives the value silently (the server's
  // ``validate()`` returns {} for that knob), but the form should make the
  // situation visible so the operator isn't confused why their ``steps=10`` is
  // ignored.
  const acceptsPrompt = current?.accepts_prompt ?? false;
  const acceptsNumFrames = current?.accepts_num_frames ?? false;
  const acceptsSteps = current?.accepts_steps ?? false;

  const noBackends = backends.length === 0;

  const columns: ColumnsType<GenerationTask> = [
    {
      title: "任务 ID",
      dataIndex: "task_id",
      width: 110,
      render: (v: string) => (
        <Typography.Text style={{ fontFamily: "monospace" }}>{v.slice(0, 8)}…</Typography.Text>
      ),
    },
    {
      title: "后端",
      width: 140,
      render: (_, row) => {
        // ``params.backend`` is the wire shape the submit form posts;
        // backends[].label_zh is the human-friendly name from the registry.
        // Falls back to the raw id when the registry no longer lists it
        // (e.g. an animator was uninstalled after the run).
        const params = row.params as { backend?: string } | undefined;
        const name = params?.backend;
        if (!name) return <Typography.Text type="secondary">—</Typography.Text>;
        const label = backends.find((b) => b.name === name)?.label_zh ?? name;
        return <Tag>{label}</Tag>;
      },
    },
    {
      title: "状态",
      dataIndex: "status",
      width: 110,
      render: (v: string) => <Badge color={statusColor(v)} text={statusLabel(v)} />,
    },
    {
      title: "进度",
      dataIndex: "progress",
      width: 220,
      render: (v: number, row) => (
        <Progress
          percent={v}
          size="small"
          status={row.status === "failed" ? "exception" : undefined}
          // A running task with no reported progress yet shows the spinner.
          // Done / failed / queued collapse to the flat bar so the table stays calm.
          showInfo={row.status === "running"}
        />
      ),
    },
    {
      title: "创建时间",
      dataIndex: "created_at",
      width: 180,
      render: (v: string | null) => (v ? new Date(v).toLocaleString() : "—"),
    },
    {
      title: "操作",
      width: 180,
      align: "right",
      // The Popconfirm's popup is a React portal whose events still
      // bubble up the React tree -- so the whole cell has to swallow
      // clicks, not just the button.
      render: (_, row) => (
        <Space size={4} onClick={(e) => e.stopPropagation()}>
          <Button size="small" icon={<EyeOutlined />} onClick={() => void openDrawer(row)}>
            查看
          </Button>
          <Popconfirm
            title="删除该记录？"
            description="删除后不可恢复。"
            okText="删除"
            cancelText="取消"
            okButtonProps={{ danger: true }}
            onConfirm={() => void onDelete(row.task_id)}
          >
            <Button danger size="small" icon={<DeleteOutlined />}>
              删除
            </Button>
          </Popconfirm>
        </Space>
      ),
    },
  ];

  return (
    <Space direction="vertical" size={16} style={{ width: "100%" }}>
      <Typography.Title level={3} style={{ margin: 0 }}>
        图生视频
      </Typography.Title>
      <Typography.Paragraph type="secondary" style={{ marginTop: -8 }}>
        上传一张静态人像照片，按所选驱动方式生成一段短动画视频。生成记录保留在下方表格中，可查看、下载、用此创建形象或删除。
      </Typography.Paragraph>

      {error && (
        <Alert
          type="error"
          showIcon
          message="生成失败"
          description={error}
          closable
          onClose={() => setError("")}
        />
      )}

      <Card title="上传与参数">
        {noBackends ? (
          <Alert
            type="warning"
            showIcon
            message="暂未注册任何驱动后端"
            description={
              <span>
                当前服务器上没有可用的图生视频后端；请确认对应权重已下载并通过对应的
                <code> server/avatar/</code> 模块注册到{" "}
                <code>ANIMATORS</code> 字典。
              </span>
            }
          />
        ) : (
          <Form
            form={form}
            layout="vertical"
            initialValues={FALLBACK_DEFAULTS}
            // Tighten the 24px default Form.Item gap so the form
            // collapses from 7 stacked rows to ~5 visible rows. Keeps the
            // labels readable while reclaiming the vertical space.
          >
            <Form.Item
              label="驱动后端"
              required
              tooltip={current?.description}
              style={{ marginBottom: 12 }}
            >
              <Select<string>
                value={picked}
                onChange={setPicked}
                placeholder="选择驱动后端"
                style={{ width: "100%" }}
                options={grouped.flatMap((g) =>
                  g.options.map((o) => {
                    // The dispatcher still refuses a backend that the
                    // host can't run; greying it out in the dropdown
                    // saves the operator a click + a failure to read.
                    const unavailable =
                      o.fits_current_gpu === false &&
                      o.vram_required_gb !== null;
                    // AntD Select accepts ``disabled`` per option, plus
                    // an optional ``title`` for the hover hint that says
                    // *why* the option is unavailable.
                    const hintSuffix = unavailable
                      ? "（当前主机显存不足）"
                      : "";
                    return {
                      label: `${o.label_zh} · VRAM ${o.vram_hint}${hintSuffix}`,
                      value: o.name,
                      disabled: unavailable,
                      title: unavailable
                        ? `需要约 ${o.vram_required_gb} GB 显存`
                        : undefined,
                    };
                  }),
                )}
              />
            </Form.Item>

            {picked === "liveportrait" && (
              <Form.Item
                label="动作模板"
                tooltip="选不同的动作模板。带『模板』标记的驱动已经预提取过特征，生成会更快。完整模板管理请到系统 → 动作模板 页。"
                required
                style={{ marginBottom: 12 }}
              >
                <Flex gap={12} align="center">
                  <Select<string>
                    value={pickedDriver}
                    onChange={setPickedDriver}
                    placeholder={drivers.length === 0 ? "暂无模板，请先到系统→动作模板页添加" : "选择模板"}
                    style={{ flex: 1, minWidth: 0 }}
                    notFoundContent={
                      drivers.length === 0
                        ? "暂无模板，请先到『系统→动作模板』页运行 fetch_liveportrait_weights.sh"
                        : "无匹配项"
                    }
                    options={drivers.map((d) => ({
                      value: d.name,
                      label: `${d.label_zh || d.name} · ${d.kind === "pkl" ? "模板" : "原始视频"} · ${formatBytes(d.size_bytes)}`,
                    }))}
                  />
                  {pickedDriver && drivers.find((d) => d.name === pickedDriver)?.has_preview && (
                    // 72×72 thumbnail pinned to the right of the dropdown.
                    // Was a 240px-wide Form.Item on its own row; the
                    // Image preview keeps the click-to-enlarge behaviour.
                    <Image
                      src={driverPreviewUrl(pickedDriver)}
                      alt={pickedDriver}
                      width={72}
                      height={72}
                      preview={{ src: driverPreviewUrl(pickedDriver) }}
                      style={{
                        objectFit: "cover",
                        borderRadius: 4,
                        background: "#f5f5f5",
                        flexShrink: 0,
                      }}
                    />
                  )}
                </Flex>
              </Form.Item>
            )}

            <Form.Item
              name="prompt"
              label="动起来提示词"
              tooltip={
                acceptsPrompt
                  ? "驱动图片动起来时附带的动作描述"
                  : "当前后端不使用此提示词，留空或填写会被服务端忽略"
              }
              style={{ marginBottom: 12 }}
            >
              <Input.TextArea
                autoSize={{ minRows: 1, maxRows: 4 }}
                disabled={!acceptsPrompt}
                placeholder={current?.accepts_prompt ? undefined : "本后端不使用提示词"}
              />
            </Form.Item>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
              <Form.Item
                name="numFrames"
                label={
                  current
                    ? `动起来帧数 (num_frames) · ${current.num_frames_min ?? "?"} ~ ${
                        current.num_frames_max ?? "?"
                      }`
                    : "动起来帧数 (num_frames)"
                }
                extra={
                  picked === "liveportrait" && pickedDriver
                    ? (() => {
                        const drv = drivers.find((d) => d.name === pickedDriver);
                        if (drv && drv.frames) {
                          return `当前模板「${drv.label_zh || drv.name}」共 ${drv.frames} 帧。留空或填 0 = 用模板全长；填具体值 = 截断。`;
                        }
                        return "当前模板还没算出帧数；首次跑完之后会自动填入。";
                      })()
                    : picked?.startsWith("wan")
                      ? "Wan 2.2 要求帧数 = 16N+1，范围 17..113；选 17 跑得最快，113 ≈ 5s 视频。"
                      : undefined
                }
                rules={
                  acceptsNumFrames && current?.num_frames_min !== undefined
                    ? [
                        // ``0`` is a valid "use template length" value, so
                        // ``required`` would mislead the operator. We
                        // keep an explicit type:number check only.
                        { type: "number", message: "请填写数字" },
                        {
                          type: "number",
                          min: 0,
                          max: current.num_frames_max,
                          message: `0 = 用模板全长，1 ~ ${current.num_frames_max ?? "?"}`,
                        },
                      ]
                    : []
                }
                style={{ marginBottom: 12 }}
              >
                <InputNumber
                  style={{ width: "100%" }}
                  min={0}
                  max={current?.num_frames_max ?? 9999}
                  step={1}
                  precision={0}
                  disabled={!acceptsNumFrames}
                  placeholder={
                    picked === "liveportrait" && pickedDriver
                      ? (() => {
                          const drv = drivers.find((d) => d.name === pickedDriver);
                          return drv?.frames
                            ? `使用模板时长 (${drv.frames})`
                            : "使用模板时长";
                        })()
                      : "0 = 用模板全长"
                  }
                />
              </Form.Item>

              <Form.Item
                name="steps"
                label={
                  current
                    ? `推演步数 (steps) · ${current.steps_min ?? "?"} ~ ${
                        current.steps_max ?? "?"
                      }`
                    : "推演步数 (steps)"
                }
                rules={
                  acceptsSteps && current?.steps_min !== undefined
                    ? [
                        { type: "number", required: true, message: "请填写" },
                        {
                          type: "number",
                          min: current.steps_min,
                          max: current.steps_max,
                          message: `范围 ${current.steps_min} ~ ${current.steps_max ?? "?"}`,
                        },
                      ]
                    : []
                }
                style={{ marginBottom: 12 }}
              >
                <InputNumber
                  style={{ width: "100%" }}
                  min={current?.steps_min ?? 1}
                  max={current?.steps_max ?? 9999}
                  step={1}
                  precision={0}
                  disabled={!acceptsSteps}
                />
              </Form.Item>
            </div>

            <Form.Item label="驱动图片" required style={{ marginBottom: 12 }}>
              <Upload.Dragger
                accept="image/*"
                maxCount={1}
                listType="picture"
                beforeUpload={() => false}
                fileList={fileList}
                onChange={({ fileList: next }) => setFileList(next)}
              >
                <p className="ant-upload-drag-icon">
                  <InboxOutlined />
                </p>
                <p className="ant-upload-drag-text">点击或拖拽图片到此处</p>
                <p className="ant-upload-hint">支持 JPG / PNG建议清晰正脸半身</p>
              </Upload.Dragger>
            </Form.Item>

            <Space>
              <Button
                type="primary"
                icon={<CloudUploadOutlined />}
                loading={submitting}
                onClick={() => void onGenerate()}
              >
                生成视频
              </Button>
              <Button
                onClick={() => {
                  form.setFieldsValue(FALLBACK_DEFAULTS);
                  setFileList([]);
                }}
                disabled={submitting}
              >
                恢复默认
              </Button>
            </Space>
          </Form>
        )}
      </Card>

      <Card
        title="生成记录"
        extra={
          <Button
            icon={<ReloadOutlined />}
            onClick={() => void refreshItems()}
          >
            刷新
          </Button>
        }
      >
        <Table<GenerationTask>
          rowKey="task_id"
          columns={columns}
          dataSource={items}
          loading={listLoading}
          pagination={{ pageSize: 10, hideOnSinglePage: true }}
          locale={{ emptyText: <Empty description="暂无生成记录" /> }}
        />
      </Card>

      <Drawer
        open={drawerTask !== null}
        onClose={closeDrawer}
        title={drawerTask ? `记录详情 ${drawerTask.task_id.slice(0, 8)}…` : ""}
        width={520}
      >
        {drawerTask && (
          <Space direction="vertical" size={16} style={{ width: "100%" }}>
            <div>
              <Typography.Text type="secondary">状态</Typography.Text>
              <div>
                <Badge color={statusColor(drawerTask.status)} text={statusLabel(drawerTask.status)} />
              </div>
            </div>
            <div>
              <Typography.Text type="secondary">进度</Typography.Text>
              <Progress
                percent={drawerTask.progress}
                status={drawerTask.status === "failed" ? "exception" : undefined}
              />
            </div>
            {drawerTask.status !== "done" && (
              <div>
                <Typography.Text type="secondary">阶段</Typography.Text>
                <div>{drawerTask.stage || "排队中…"}</div>
              </div>
            )}
            {drawerTask.status === "done" && (
              <>
                {drawerLoadError ? (
                  <Alert type="error" message={drawerLoadError} showIcon />
                ) : drawerUrl ? (
                  <video
                    key={drawerUrl}
                    src={drawerUrl}
                    controls
                    muted
                    playsInline
                    style={{ width: "100%", background: "#000" }}
                  />
                ) : (
                  <Typography.Text type="secondary">载入预览…</Typography.Text>
                )}
              </>
            )}
            <div>
              <Typography.Text type="secondary">参数</Typography.Text>
              <pre
                style={{
                  background: "#f5f5f5",
                  padding: 12,
                  borderRadius: 6,
                  margin: 0,
                  overflow: "auto",
                  fontSize: 12,
                }}
              >
                {JSON.stringify(drawerTask.params, null, 2)}
              </pre>
            </div>
            {drawerTask.error && (
              <Alert type="error" message={drawerTask.error} showIcon />
            )}
            {drawerTask.status === "done" && (
              <Space wrap>
                <Button icon={<DownloadOutlined />} onClick={() => void onDownload()}>
                  下载 mp4
                </Button>
                <Button
                  type="primary"
                  icon={<SwapOutlined />}
                  onClick={() => void onUseVideoForAvatar()}
                >
                  用此视频创建形象
                </Button>
              </Space>
            )}
          </Space>
        )}
      </Drawer>
    </Space>
  );
}
