import { useEffect, useState } from "react";
import {
  Alert,
  App as AntApp,
  Button,
  Card,
  Form,
  Input,
  Modal,
  Progress,
  Space,
  Table,
  Tag,
  Typography,
} from "antd";
import type { ColumnsType } from "antd/es/table";
import { PlusOutlined, ReloadOutlined } from "@ant-design/icons";
import { api } from "../api";
import StatusTag from "../components/StatusTag";
import type { SessionInfo, TenantInfo } from "../types";

interface EngineInfo {
  name: string;
  gpu: boolean;
  // Only the real engine reports these; the mock has no weights to load.
  models_dir?: string;
  weights_present?: boolean;
  loaded?: boolean;
  avatars_cached?: number;
  pending?: number;
}

interface SystemInfo {
  engine: EngineInfo;
  pipeline: string;
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

const boolTag = (v: boolean, yes = "是", no = "否") => (
  <Tag color={v ? "success" : "error"}>{v ? yes : no}</Tag>
);

export default function System() {
  const [sys, setSys] = useState<SystemInfo | null>(null);
  const [tenants, setTenants] = useState<TenantInfo[]>([]);
  const [jobs, setJobs] = useState<JobInfo[]>([]);
  const [sessions, setSessions] = useState<SessionInfo[]>([]);
  const [creating, setCreating] = useState(false);
  const [newKey, setNewKey] = useState("");
  const [form] = Form.useForm<{ name: string }>();
  const { message } = AntApp.useApp();

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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function onCreateTenant({ name }: { name: string }) {
    setCreating(true);
    try {
      const created = await api<{ api_key: string }>("/api/admin/tenants", {
        method: "POST",
        body: { name },
      });
      setNewKey(created.api_key);
      form.resetFields();
      await refresh();
    } catch (err) {
      message.error((err as Error).message);
    } finally {
      setCreating(false);
    }
  }

  const jobColumns: ColumnsType<JobInfo> = [
    { title: "Job", dataIndex: "job_id", width: 100, render: (v: string) => v.slice(0, 8) },
    { title: "阶段", dataIndex: "stage", width: 120, render: (s: string) => <StatusTag status={s} /> },
    {
      title: "进度",
      dataIndex: "progress",
      width: 220,
      render: (p: number) => <Progress percent={p} size="small" status={p >= 100 ? "success" : "active"} />,
    },
    {
      title: "错误",
      dataIndex: "error",
      render: (e: string | null) => (e ? <Typography.Text type="danger">{e}</Typography.Text> : "—"),
    },
  ];

  const sessionColumns: ColumnsType<SessionInfo> = [
    { title: "会话", dataIndex: "session_id", width: 120, render: (v: string) => v.slice(0, 8) },
    { title: "状态", dataIndex: "status", width: 120, render: (s: string) => <StatusTag status={s} /> },
    { title: "帧数", width: 100, render: (_, s) => s.metrics.frames_out ?? 0 },
    { title: "延迟 ms", width: 120, render: (_, s) => (s.metrics.avg_latency_ms ?? 0).toFixed(1) },
  ];

  return (
    <Space direction="vertical" size={16} style={{ width: "100%" }}>
      <Space align="center" style={{ width: "100%", justifyContent: "space-between" }}>
        <Typography.Title level={3} style={{ margin: 0 }}>
          系统
        </Typography.Title>
        <Button icon={<ReloadOutlined />} onClick={() => refresh().catch(() => undefined)}>
          刷新
        </Button>
      </Space>

      {sys && (
        <Card title="引擎信息">
          <Space direction="vertical" size={8} style={{ width: "100%" }}>
            <Space wrap size={24}>
              <span>
                引擎：<Typography.Text strong>{sys.engine.name}</Typography.Text>
              </span>
              <span>
                制作流水线：<Typography.Text strong>{sys.pipeline}</Typography.Text>
              </span>
              <span>GPU：{boolTag(sys.gpu, "可用", "不可用")}</span>
              {sys.engine.weights_present !== undefined && (
                <span>权重：{boolTag(sys.engine.weights_present, "就绪", "缺失")}</span>
              )}
              {sys.engine.loaded !== undefined && (
                <span>已加载：{boolTag(sys.engine.loaded)}</span>
              )}
            </Space>
            <Typography.Text type="secondary">
              数据目录：{sys.data_dir} · 版本：{sys.version}
            </Typography.Text>
          </Space>
        </Card>
      )}

      <Card title="租户">
        <Form form={form} layout="inline" onFinish={onCreateTenant} style={{ marginBottom: 16 }}>
          <Form.Item name="name" rules={[{ required: true, message: "请输入租户名" }]}>
            <Input placeholder="租户名" style={{ width: 220 }} />
          </Form.Item>
          <Form.Item>
            <Button type="primary" htmlType="submit" loading={creating} icon={<PlusOutlined />}>
              新建租户
            </Button>
          </Form.Item>
        </Form>
        <Table
          rowKey="tenant_id"
          size="small"
          pagination={{ pageSize: 10, hideOnSinglePage: true }}
          columns={[
            { title: "租户", dataIndex: "name" },
            {
              title: "创建时间",
              dataIndex: "created_at",
              render: (v: string) => new Date(v).toLocaleString(),
            },
          ]}
          dataSource={tenants}
        />
      </Card>

      <Card title="任务">
        <Table
          rowKey="job_id"
          size="small"
          scroll={{ x: 720 }}
          pagination={{ pageSize: 10, hideOnSinglePage: true }}
          columns={jobColumns}
          dataSource={jobs}
        />
      </Card>

      <Card title="会话">
        <Table
          rowKey="session_id"
          size="small"
          scroll={{ x: 600 }}
          pagination={{ pageSize: 10, hideOnSinglePage: true }}
          columns={sessionColumns}
          dataSource={sessions}
        />
      </Card>

      <Modal
        title="租户创建成功"
        open={!!newKey}
        onOk={() => setNewKey("")}
        onCancel={() => setNewKey("")}
        okText="我已保存"
        cancelButtonProps={{ style: { display: "none" } }}
      >
        <Alert
          type="warning"
          showIcon
          message="API Key 仅显示一次，请立即保存。"
          style={{ marginBottom: 12 }}
        />
        <Typography.Paragraph copyable={{ text: newKey }} code style={{ wordBreak: "break-all" }}>
          {newKey}
        </Typography.Paragraph>
      </Modal>
    </Space>
  );
}
