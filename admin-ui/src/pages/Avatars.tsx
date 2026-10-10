import { useEffect, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import {
  Alert,
  App as AntApp,
  Button,
  Card,
  Empty,
  Form,
  Input,
  Popconfirm,
  Space,
  Table,
  Typography,
  Upload,
} from "antd";
import type { ColumnsType } from "antd/es/table";
import type { UploadFile } from "antd/es/upload/interface";
import { InboxOutlined, PlusOutlined, ReloadOutlined } from "@ant-design/icons";
import { api } from "../api";
import StatusTag from "../components/StatusTag";
import type { AvatarInfo } from "../types";
import { createReentryGuard } from "./reentryGuard";

const ACTIVE = new Set(["pending", "processing"]);

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

export default function Avatars() {
  const [items, setItems] = useState<AvatarInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);
  // A stable guard instance for the component's lifetime; `run` acquires
  // synchronously before its task's first await.
  const creatingGuard = useRef(createReentryGuard()).current;
  const [fileList, setFileList] = useState<UploadFile[]>([]);
  const [error, setError] = useState("");
  const [form] = Form.useForm<{ name: string }>();
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

  useEffect(() => {
    refresh()
      .catch((err) => setError((err as Error).message))
      .finally(() => setLoading(false));
    return () => {
      if (timer.current !== null) window.clearTimeout(timer.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function onCreate({ name }: { name: string }) {
    const file = fileList[0]?.originFileObj;
    if (!file) {
      message.warning("请先选择图片或视频");
      return;
    }
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
        if (file.type.startsWith("video/")) {
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
      title: "名称",
      dataIndex: "name",
      render: (_, a) => (
        <Link to={`/avatars/${a.avatar_id}`} onClick={(e) => e.stopPropagation()}>
          {a.name}
        </Link>
      ),
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
        // React portal whose events still bubble up the React tree — so the
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
        <Form form={form} layout="vertical" onFinish={onCreate} style={{ maxWidth: 480 }}>
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
              <p className="ant-upload-text">点击或拖拽图片 / 视频到此处</p>
              <p className="ant-upload-hint">支持 JPG / PNG / MP4 / MOV，视频 ≤15 秒，建议清晰正脸半身；视频中请保持闭嘴、不说话</p>
            </Upload.Dragger>
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
          <Button
            icon={<ReloadOutlined />}
            onClick={() => {
              setError("");
              refresh().catch((err) => setError((err as Error).message));
            }}
          >
            刷新
          </Button>
        }
      >
        <Table
          rowKey="avatar_id"
          columns={columns}
          dataSource={items}
          loading={loading}
          scroll={{ x: 720 }}
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
