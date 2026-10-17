import { useEffect, useState } from "react";
import {
  Alert,
  App as AntApp,
  Button,
  Card,
  Form,
  Image,
  Input,
  Popconfirm,
  Progress,
  Space,
  Table,
  Tag,
  Typography,
  Upload,
} from "antd";
import type { ColumnsType } from "antd/es/table";
import { DeleteOutlined, InboxOutlined, ReloadOutlined, ThunderboltOutlined } from "@ant-design/icons";
import { api } from "../api";

type DriverInfo = {
  name: string;
  filename: string;
  kind: "mp4" | "pkl";
  size_bytes: number;
  cached: boolean;
  canonical: boolean;
  label_zh: string;
  // Frame count of the template: ``null`` until either the .pkl is read
  // (cheap) or the .frames sidecar is populated by the fetch script.
  // The admin UI uses it to surface "this template is 4.7s" previews.
  frames?: number | null;
  // Whether ``<stem>.gif`` exists next to the mp4/pkl. Drives without
  // a preview get a neutral "无预览" badge instead of a thumbnail.
  has_preview?: boolean;
};

function formatBytes(b: number): string {
  if (b < 1024) return `${b} B`;
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB`;
  return `${(b / 1024 / 1024).toFixed(1)} MB`;
}

function formatFrames(frames: number | null | undefined): string {
  if (frames == null || frames <= 0) return "—";
  // 25 fps is the upstream default; show "~Ns" as a rough preview duration.
  const seconds = frames / 25;
  return `${frames} 帧 · ≈ ${seconds.toFixed(1)}s`;
}

export function previewUrl(name: string): string {
  return `/api/admin/liveportrait-drivers/${encodeURIComponent(name)}/preview`;
}

async function fetchDrivers(): Promise<DriverInfo[]> {
  return (await api<{ drivers: DriverInfo[] }>("/api/admin/liveportrait-drivers")).drivers ?? [];
}

async function deleteDriver(name: string): Promise<void> {
  await api(`/api/admin/liveportrait-drivers/${encodeURIComponent(name)}`, { method: "DELETE" });
}

async function extractTemplate(
  name: string,
  file: File,
  maxSeconds: number,
  onProgress: (pct: number) => void,
): Promise<{ name: string; frames: number; size_bytes: number }> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const form = new FormData();
    form.append("name", name);
    form.append("video", file);
    form.append("max_seconds", String(maxSeconds));
    xhr.open("POST", "/api/admin/liveportrait-drivers/extract");
    xhr.upload.onprogress = (ev) => {
      if (ev.lengthComputable) onProgress(Math.round((ev.loaded / ev.total) * 100));
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText));
        } catch (e) {
          reject(new Error(`malformed response: ${String(e)}`));
        }
      } else {
        let msg = xhr.statusText;
        try {
          const data = JSON.parse(xhr.responseText);
          msg = data?.error?.message ?? msg;
        } catch {
          /* fall through */
        }
        reject(new Error(msg));
      }
    };
    xhr.onerror = () => reject(new Error("network error"));
    xhr.send(form);
  });
}

const kindColor: Record<DriverInfo["kind"], string> = {
  pkl: "success",
  mp4: "processing",
};
const kindLabel: Record<DriverInfo["kind"], string> = {
  pkl: "模板（已提取）",
  mp4: "原始视频",
};

const NAME_PATTERN = /^[A-Za-z0-9_一-鿿㐀-䶿\-.][A-Za-z0-9_一-鿿㐀-䶿\-. ]*$/;

export default function LivePortraits() {
  const [rows, setRows] = useState<DriverInfo[]>([]);
  const [loading, setLoading] = useState(false);
  const [extracting, setExtracting] = useState(false);
  const [progress, setProgress] = useState<number | null>(null);
  const [error, setError] = useState<string>("");
  const [form] = Form.useForm<{ name: string; maxSeconds?: number }>();
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const { message } = AntApp.useApp();

  async function refresh() {
    setLoading(true);
    setError("");
    try {
      setRows(await fetchDrivers());
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    refresh().catch(() => undefined);
  }, []);

  async function onDelete(row: DriverInfo) {
    try {
      await deleteDriver(row.name);
      message.success(`已删除 ${row.name}`);
      await refresh();
    } catch (e) {
      message.error(`删除失败：${(e as Error).message}`);
    }
  }

  async function onExtract() {
    if (!videoFile) {
      message.warning("请先选择一个驱动视频");
      return;
    }
    let payload: { name?: string; maxSeconds?: number };
    try {
      payload = await form.validateFields();
    } catch {
      return;
    }
    const name = (payload.name ?? "").trim();
    if (!name) return;
    const maxSeconds = payload.maxSeconds && payload.maxSeconds > 0 ? payload.maxSeconds : 0;
    setError("");
    setExtracting(true);
    setProgress(0);
    try {
      const result = await extractTemplate(name, videoFile, maxSeconds, setProgress);
      const dur = maxSeconds > 0 ? `（截取到 ${maxSeconds}s）` : "";
      message.success(
        `已提取 ${result.name}${dur}：${result.frames} 帧，${formatBytes(result.size_bytes)}`,
      );
      form.resetFields();
      setVideoFile(null);
      await refresh();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setExtracting(false);
      setProgress(null);
    }
  }

  const columns: ColumnsType<DriverInfo> = [
    {
      title: "预览",
      key: "preview",
      width: 90,
      render: (_: unknown, row) =>
        row.has_preview ? (
          <Image
            src={previewUrl(row.name)}
            alt={row.name}
            width={64}
            height={64}
            style={{ objectFit: "cover", borderRadius: 4, background: "#f5f5f5" }}
            preview={{ src: previewUrl(row.name) }}
          />
        ) : (
          <Tag color="default">无预览</Tag>
        ),
    },
    {
      title: "模板名",
      dataIndex: "name",
      key: "name",
      render: (name: string, row) => (
        <Space direction="vertical" size={0}>
          <span key="zh" style={{ fontWeight: 500 }}>{row.label_zh || name}</span>
          <span key="id" style={{ fontSize: 12, color: "#999" }}>{name}</span>
        </Space>
      ),
    },
    {
      title: "类型",
      dataIndex: "kind",
      key: "kind",
      width: 140,
      render: (kind: DriverInfo["kind"], row) => (
        <Space size={4}>
          <Tag key="kind" color={kindColor[kind]}>{kindLabel[kind]}</Tag>
          {row.kind === "mp4" && row.cached && (
            <Tag key="cached" color="success">已提取</Tag>
          )}
        </Space>
      ),
    },
    {
      title: "时长",
      dataIndex: "frames",
      key: "frames",
      width: 120,
      render: (f: number | null | undefined) => formatFrames(f),
    },
    {
      title: "大小",
      dataIndex: "size_bytes",
      key: "size_bytes",
      width: 100,
      render: (b: number) => formatBytes(b),
    },
    {
      title: "来源",
      dataIndex: "canonical",
      key: "canonical",
      width: 90,
      render: (c: boolean) => (c ? <Tag>内置</Tag> : <Tag color="default">自定义</Tag>),
    },
    {
      title: "文件",
      dataIndex: "filename",
      key: "filename",
      width: 140,
      render: (f: string) => <code style={{ fontSize: 12 }}>{f}</code>,
    },
    {
      title: "操作",
      key: "actions",
      width: 100,
      render: (_: unknown, row) =>
        row.canonical ? (
          <Tag color="default" style={{ margin: 0 }}>内置不可删</Tag>
        ) : (
          <Popconfirm
            title={`删除模板 ${row.name}？`}
            description="mp4 / pkl / gif / 帧数侧车 都会被移除。"
            okText="删除"
            cancelText="取消"
            onConfirm={() => onDelete(row)}
          >
            <Button type="link" size="small" danger icon={<DeleteOutlined />}>
              删除
            </Button>
          </Popconfirm>
        ),
    },
  ];

  return (
    <Space direction="vertical" size={16} style={{ width: "100%" }}>
      <Typography.Paragraph type="secondary" style={{ marginTop: 0, marginBottom: 0 }}>
        管理 LivePortrait 的动作模板。提取后特征被预缓存为 <code>.pkl</code>，
        后续生成时跳过裁剪 + 关键点提取；预渲染的 GIF 缩略图会出现在「形象活化」和
        「参数设置」里供选择时预览。
      </Typography.Paragraph>

      {error && (
        <Alert
          type="error"
          showIcon
          closable
          message="操作失败"
          description={error}
          onClose={() => setError("")}
        />
      )}

      <Card
        title="上传驱动并提取模板"
        extra={
          <Button
            icon={<ReloadOutlined />}
            onClick={() => refresh()}
            loading={loading}
          >
            刷新
          </Button>
        }
      >
        <Form form={form} layout="vertical" style={{ maxWidth: 720 }}>
          <Form.Item
            name="name"
            label="模板名（字母数字下划线或中文，可含空格）"
            tooltip="用这个名字来引用这个模板，例如 smile、head_left_30、微笑。fetch_liveportrait_weights.sh 里的内置模板名是受保护的，不能复用。"
            rules={[
              { required: true, message: "请填写模板名" },
              {
                pattern: NAME_PATTERN,
                message: "只允许字母数字下划线、中文、空格和连字符",
              },
            ]}
          >
            <Input placeholder="例如 smile / 微笑模板" style={{ maxWidth: 320 }} />
          </Form.Item>
          <Form.Item
            name="maxSeconds"
            label="截取时长（秒，留空=整段视频）"
            tooltip="只对前 N 秒做特征提取。例如一段 30 秒视频只想要 5 秒的嘴部动作，填 5。"
          >
            <Input type="number" min={0} step={0.5} placeholder="例如 5" style={{ maxWidth: 160 }} />
          </Form.Item>
          <Form.Item label="驱动视频（mp4）" required>
            <Upload.Dragger
              accept="video/mp4,video/*"
              maxCount={1}
              beforeUpload={(file) => {
                setVideoFile(file);
                return false;
              }}
              onRemove={() => {
                setVideoFile(null);
                return true;
              }}
              fileList={videoFile ? [{ uid: "-1", name: videoFile.name, status: "done" }] : []}
              disabled={extracting}
            >
              <p className="ant-upload-drag-icon">
                <InboxOutlined />
              </p>
              <p className="ant-upload-drag-text">点击或拖拽视频到此处</p>
              <p className="ant-upload-hint">建议清晰人脸正面片段，5–30 秒</p>
            </Upload.Dragger>
          </Form.Item>
          {progress !== null && (
            <Progress percent={progress} status={extracting ? "active" : "success"} />
          )}
          <Form.Item>
            <Button
              type="primary"
              icon={<ThunderboltOutlined />}
              onClick={onExtract}
              loading={extracting}
              disabled={!videoFile}
            >
              提取特征并保存
            </Button>
          </Form.Item>
        </Form>
      </Card>

      <Card title="已注册的模板">
        <Table<DriverInfo>
          rowKey="name"
          columns={columns}
          dataSource={rows}
          loading={loading}
          pagination={false}
          size="middle"
        />
      </Card>
    </Space>
  );
}
