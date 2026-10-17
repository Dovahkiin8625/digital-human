import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import {
  Alert,
  App as AntApp,
  Badge,
  Button,
  Card,
  Collapse,
  Empty,
  Form,
  Input,
  InputNumber,
  Popconfirm,
  Select,
  Space,
  Table,
  Tag,
  Tooltip,
  Typography,
  Upload,
} from "antd";
import type { ColumnsType } from "antd/es/table";
import type { RcFile, UploadFile } from "antd/es/upload/interface";
import { InboxOutlined, PlusOutlined, ReloadOutlined, SearchOutlined } from "@ant-design/icons";
import { api } from "../api";
import StatusTag from "../components/StatusTag";
import type { AvatarInfo, TunableParams } from "../types";
import { createReentryGuard } from "./reentryGuard";

const ACTIVE = new Set(["pending", "processing"]);

// Live defaults mirrored from server/avatar/tunables.py. The create form
// pre-fills these values into the IdleVideo fields so the operator can see
// what they're getting; on submit the (possibly edited) values travel with
// the FormData and the server stores them as per-avatar overrides.
const IDLE_CRF_DEFAULT = 18;
const IDLE_PRESET_DEFAULT = "veryfast";
const X264_PRESETS = [
  "ultrafast",
  "superfast",
  "veryfast",
  "faster",
  "fast",
  "medium",
  "slow",
  "slower",
  "veryslow",
] as const;

function videoSeconds(file: File): Promise<number> {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const el = document.createElement("video");
    el.preload = "metadata";
    el.onloadedmetadata = () => {
      URL.revokeObjectURL(url);
      resolve(el.duration);
    };
    el.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("cannot read video"));
    };
    el.src = url;
  });
}

function CopyableId({ id }: { id: string }) {
  // The avatar id is 32 hex chars; only the first 8 are useful in a table.
  // The whole id is what copy() needs to ship to a paste target.
  return (
    <Tooltip title="点击复制">
      <Typography.Text
        copyable={{ text: id, tooltips: false }}
        style={{ fontFamily: "monospace", cursor: "pointer" }}
      >
        {id.slice(0, 8)}…
      </Typography.Text>
    </Tooltip>
  );
}

// Per-knob ranges mirror server/avatar/tunables.py ParamSpec entries. The
// server is the source of truth (it validates the PATCH range), but matching
// it here keeps the form honest and lets the live validate-on-type prevent
// obvious typos.
const IDLE_CRF_MIN = 0;
const IDLE_CRF_MAX = 51;

export default function Avatars() {
  // ``useLocation`` is read once on mount: the 图生视频 tab pushes a
  // ``prefilledFile`` (the mp4 it just generated) via router state, and
  // we seed the upload list with it. We deliberately don't keep watching
  // for state changes -- a subsequent navigation that *doesn't* include
  // the file should leave the previous selection intact. Both ``location``
  // and ``initialPrefilledRef`` must be declared BEFORE the fileList
  // ``useState`` below: React invokes the initializer on first render,
  // which is BEFORE the line of code that declares a ``const`` would
  // execute; reading the const then throws ReferenceError (TDZ).
  const location = useLocation() as { state?: { prefilledFile?: File } | null };
  const initialPrefilledRef = useRef<File | null>(location.state?.prefilledFile ?? null);

  const [items, setItems] = useState<AvatarInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);
  // A stable guard instance for the component's lifetime; `run` acquires
  // synchronously before its task's first await.
  const creatingGuard = useRef(createReentryGuard()).current;
  const [fileList, setFileList] = useState<UploadFile[]>(() => {
    // Seed the upload list from a prefilled file passed via router state
    // (the 图生视频 tab hands the operator the mp4 it just generated so
    // they don't have to re-download + drag it onto the form). We wrap the
    // raw ``File`` in an ``UploadFile`` with ``status='done'`` so AntD shows
    // it like any other accepted upload.
    const seeded = initialPrefilledRef.current;
    if (!seeded) return [];
    // AntD's UploadFile requires ``originFileObj`` to be an ``RcFile`` --
    // i.e. a real ``File`` plus a ``uid: string``. ``File.prototype`` still
    // exposes the legacy ``lastModifiedDate`` getter, but writing to it is
    // forbidden on modern browsers (``TypeError: Cannot set property
    // lastModifiedDate``); we deliberately do NOT try to set it:
    // rc-upload v4 only reads ``uid``/``name``/``type``/``size``/
    // ``lastModified``, and AntD v5's runtime never touches the legacy
    // date. A previous attempt wrapped the File via ``Object.create`` --
    // that satisfied the RcFile shape but stripped the ``[[Blob]]`` internal
    // slot, so ``URL.createObjectURL`` and friends later threw
    // ``TypeError: Illegal invocation`` on the proxy. Mutating the seed
    // File is safe because the i2v drawer hands us a brand-new ``File``
    // in the same closure and never reuses it.
    const rcFile = seeded as RcFile;
    rcFile.uid = "-1";
    return [
      {
        uid: "-1",
        name: seeded.name,
        status: "done",
        originFileObj: rcFile,
      },
    ];
  });
  const [error, setError] = useState("");
  const [form] = Form.useForm<{
    name: string;
    idle_crf?: number;
    idle_preset?: string;
  }>();
  const [nameQuery, setNameQuery] = useState("");
  // Live-effective values for the idle knobs -- used to default the form so
  // an operator who doesn't touch the Advanced section still gets the
  // currently-tuned x264 settings.
  const [idleDefaults, setIdleDefaults] = useState<{
    idle_crf: number;
    idle_preset: string;
  } | null>(null);
  const timer = useRef<number | null>(null);
  const navigate = useNavigate();
  const { message } = AntApp.useApp();

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

  async function loadIdleDefaults() {
    // GET /api/admin/settings returns the full TUNABLES dict; we only need
    // the two knobs the create form exposes. A failure falls back to the
    // module defaults -- the upload itself doesn't need this round trip.
    try {
      const t = await api<TunableParams>("/api/admin/settings");
      setIdleDefaults({
        idle_crf: typeof t.idle_crf === "number" ? t.idle_crf : IDLE_CRF_DEFAULT,
        idle_preset: typeof t.idle_preset === "string" ? t.idle_preset : IDLE_PRESET_DEFAULT,
      });
    } catch {
      setIdleDefaults({ idle_crf: IDLE_CRF_DEFAULT, idle_preset: IDLE_PRESET_DEFAULT });
    }
  }

  useEffect(() => {
    refresh()
      .catch((err) => setError((err as Error).message))
      .finally(() => setLoading(false));
    void loadIdleDefaults();
    // If we got here with a prefilled file, leave a hint for the operator
    // so they don't wonder why their upload card already shows a clip.
    if (initialPrefilledRef.current) {
      message.info(`已载入「图生视频」生成的视频：${initialPrefilledRef.current.name}`);
    }
    return () => {
      if (timer.current !== null) window.clearTimeout(timer.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const visibleItems = useMemo(() => {
    const q = nameQuery.trim().toLowerCase();
    if (!q) return items;
    return items.filter((a) => a.name.toLowerCase().includes(q));
  }, [items, nameQuery]);

  async function onCreate(values: {
    name: string;
    idle_crf?: number;
    idle_preset?: string;
  }) {
    const { name, idle_crf, idle_preset } = values;
    const file = fileList[0]?.originFileObj;
    if (!file) {
      message.warning("请先选择图片或视频");
      return;
    }
    const isVideo = file.type.startsWith("video/");
    // Synchronous re-entry guard: `creating` state flips too late to stop a
    // second click -- while video metadata is awaited (can be tens/hundreds
    // of ms on a slow disk) or in the tick before React re-renders -- and two
    // POSTs would land, with two GPU prepare jobs behind them. `run` takes
    // the lock synchronously before its task reaches its first await, so the
    // probe-await window is covered; the over-15s early return and probe
    // failure settle the task and release the lock via `finally`.
    const { ran } = await creatingGuard.run(async () => {
      setCreating(true);
      try {
        if (isVideo) {
          try {
            const seconds = await videoSeconds(file);
            if (seconds > 15) {
              message.error(`视频 ${seconds.toFixed(1)} 秒，超过 15 秒上限，请先剪短`);
              return;
            }
          } catch {
            // Duration is only a convenience; the server is authoritative.
          }
        }
        const body = new FormData();
        body.append("name", name);
        body.append("file", file);
        // Per-avatar idle-video overrides. Empty fields fall through to the
        // server-side defaults (the column is JSON NOT NULL DEFAULT '{}');
        // the server validates against tunables.ParamSpec bounds.
        if (typeof idle_crf === "number") body.append("idle_crf", String(idle_crf));
        if (typeof idle_preset === "string" && idle_preset !== "") {
          body.append("idle_preset", idle_preset);
        }
        // No per-knob override / animate flag here: image-to-video generation
        // now lives on its own tab ("/video-gen") instead of being inlined in
        // this upload form. The endpoint accepts only name + file + (server
        // side) animate; the latter defaults to false and the operator opts
        // in by visiting the new tab.
        await api("/api/admin/avatars", { method: "POST", form: body });
        message.success("已提交，正在生成");
        form.resetFields();
        setFileList([]);
        await refresh();
      } catch (err) {
        message.error((err as Error).message);
      } finally {
        setCreating(false);
      }
    });
    if (!ran) return;
  }

  async function onDelete(id: string) {
    try {
      await api(`/api/admin/avatars/${id}`, { method: "DELETE" });
      message.success("已删除");
      await refresh();
    } catch (err) {
      message.error((err as Error).message);
    }
  }

  const columns: ColumnsType<AvatarInfo> = [
    {
      title: "形象 ID",
      dataIndex: "avatar_id",
      width: 110,
      render: (v: string) => <CopyableId id={v} />,
    },
    {
      title: "名称",
      dataIndex: "name",
      render: (_, a) => (
        <Link to={`/avatars/${a.avatar_id}`} onClick={(e) => e.stopPropagation()}>
          {a.name}
        </Link>
      ),
    },
    {
      title: "驱动方式",
      width: 110,
      render: (_, a) => {
        // `prep.mode` is only known after the build writes pack.json. Until
        // then fall back to the file-suffix heuristic kept here as a hint.
        const cfg = a.config as Record<string, unknown>;
        const prep = (cfg.musetalk as Record<string, unknown> | undefined) ?? {};
        const mode = (prep.mode as string | undefined) ?? null;
        if (mode === "video") return <Tag color="cyan">视频驱动</Tag>;
        if (mode === "image") return <Tag color="geekblue">图片驱动</Tag>;
        return <Tag>—</Tag>;
      },
    },
    {
      title: "帧数",
      width: 90,
      render: (_, a) => {
        const cfg = a.config as Record<string, unknown>;
        const prep = (cfg.musetalk as Record<string, unknown> | undefined) ?? {};
        const v = prep.frame_count;
        return typeof v === "number" ? v : "—";
      },
    },
    {
      title: "警告",
      width: 90,
      render: (_, a) => {
        const warnings = a.config.warnings ?? [];
        if (warnings.length === 0) return <Typography.Text type="secondary">无</Typography.Text>;
        return (
          <Tooltip title={warnings.join("\n")}>
            <Badge count={warnings.length} color="orange" />
          </Tooltip>
        );
      },
    },
    {
      title: "状态",
      dataIndex: "status",
      width: 110,
      render: (status: string) => <StatusTag status={status} />,
    },
    {
      title: "创建时间",
      dataIndex: "created_at",
      width: 200,
      render: (v: string) => new Date(v).toLocaleString(),
    },
    {
      title: "操作",
      width: 100,
      align: "right",
      render: (_, a) => (
        // The row itself navigates on click, and the Popconfirm's popup is a
        // React portal whose events still bubble up the React tree -- so the
        // whole cell has to swallow clicks, not just the button.
        <div onClick={(e) => e.stopPropagation()}>
          <Popconfirm
            title="删除该形象？"
            description="删除后不可恢复。"
            okText="删除"
            cancelText="取消"
            okButtonProps={{ danger: true }}
            onConfirm={() => onDelete(a.avatar_id)}
          >
            <Button danger size="small">
              删除
            </Button>
          </Popconfirm>
        </div>
      ),
    },
  ];

  return (
    <Space direction="vertical" size={16} style={{ width: "100%" }}>
      <Typography.Title level={3} style={{ margin: 0 }}>
        形象管理
      </Typography.Title>

      <Card title="新建形象">
        <Form
          form={form}
          layout="vertical"
          onFinish={onCreate}
          style={{ maxWidth: 560 }}
          // Defaults pulled from /api/admin/settings so the form mirrors the
          // server's effective values; the server's defaults take over if
          // the round trip fails.
          initialValues={{
            idle_crf: idleDefaults?.idle_crf ?? IDLE_CRF_DEFAULT,
            idle_preset: idleDefaults?.idle_preset ?? IDLE_PRESET_DEFAULT,
          }}
        >
          <Form.Item
            name="name"
            label="名称"
            rules={[{ required: true, message: "请输入名称" }]}
          >
            <Input placeholder="例如：小美" />
          </Form.Item>
          <Form.Item label="驱动图片或视频" required>
            <Upload.Dragger
              accept="image/*,video/*"
              maxCount={1}
              listType="picture"
              beforeUpload={() => false}
              fileList={fileList}
              onChange={({ fileList: next }) => setFileList(next)}
            >
              <p className="ant-upload-drag-icon">
                <InboxOutlined />
              </p>
              <p className="ant-upload-drag-text">点击或拖拽图片 / 视频到此处</p>
              <p className="ant-upload-hint">支持 JPG / PNG / MP4 / MOV，视频 ≤15 秒，建议清晰正脸半身；视频中请保持闭嘴、不说话</p>
            </Upload.Dragger>
          </Form.Item>
          {/*
            Idle-video knobs. Collapsed by default so the form stays calm
            for the common "just upload a photo" path; expand to tweak the
            quality / encode-speed trade-off per avatar.
          */}
          <Form.Item label="高级参数（可选）" style={{ marginBottom: 12 }}>
            <Collapse
              ghost
              items={[
                {
                  key: "idle",
                  label: "空闲视频编码",
                  children: (
                    <Space direction="vertical" size={12} style={{ width: "100%" }}>
                      <Form.Item
                        name="idle_crf"
                        label="视频质量 (CRF)"
                        tooltip="0=无损 51=最小体积，建议 16~23；调高可减小文件但降低画质"
                        style={{ marginBottom: 0 }}
                      >
                        <InputNumber
                          min={IDLE_CRF_MIN}
                          max={IDLE_CRF_MAX}
                          step={1}
                          precision={0}
                          style={{ width: 200 }}
                        />
                      </Form.Item>
                      <Form.Item
                        name="idle_preset"
                        label="x264 编码速度"
                        tooltip="veryslow 文件更小但编码更慢；ultrafast 相反"
                        style={{ marginBottom: 0 }}
                      >
                        <Select
                          style={{ width: 200 }}
                          options={X264_PRESETS.map((p) => ({ value: p, label: p }))}
                        />
                      </Form.Item>
                    </Space>
                  ),
                },
              ]}
            />
          </Form.Item>
          <Button type="primary" htmlType="submit" loading={creating} icon={<PlusOutlined />}>
            上传创建
          </Button>
        </Form>
      </Card>

      {error && <Alert type="error" message={error} showIcon />}

      <Card
        title="形象列表"
        extra={
          <Space>
            <Input
              allowClear
              prefix={<SearchOutlined />}
              placeholder="搜索名称"
              style={{ width: 200 }}
              value={nameQuery}
              onChange={(e) => setNameQuery(e.target.value)}
            />
            <Button
              icon={<ReloadOutlined />}
              onClick={() => {
                setError("");
                refresh().catch((err) => setError((err as Error).message));
              }}
            >
              刷新
            </Button>
          </Space>
        }
      >
        <Table
          rowKey="avatar_id"
          columns={columns}
          dataSource={visibleItems}
          loading={loading}
          scroll={{ x: 1000 }}
          pagination={{ pageSize: 10, hideOnSinglePage: true }}
          locale={{ emptyText: <Empty description="暂无形象，先上传一个吧" /> }}
          onRow={(a) => ({
            onClick: () => navigate(`/avatars/${a.avatar_id}`),
            style: { cursor: "pointer" },
          })}
        />
      </Card>
    </Space>
  );
}