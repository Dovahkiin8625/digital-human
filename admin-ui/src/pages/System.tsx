import { useEffect, useState } from "react";
import {
  App as AntApp,
  Button,
  Card,
  Popconfirm,
  Progress,
  Space,
  Table,
  Tabs,
  Tag,
  Typography,
} from "antd";
import type { ColumnsType } from "antd/es/table";
import { DeleteOutlined, ReloadOutlined, SyncOutlined } from "@ant-design/icons";
import { api } from "../api";
import LivePortraits from "./LivePortraits";
import StatusTag from "../components/StatusTag";
import type { SessionInfo } from "../types";

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
  const [jobs, setJobs] = useState<JobInfo[]>([]);
  const [sessions, setSessions] = useState<SessionInfo[]>([]);
  const [sweeping, setSweeping] = useState(false);
  const { message } = AntApp.useApp();

  async function refresh() {
    const [s, j, sess] = await Promise.all([
      api<SystemInfo>("/api/admin/system"),
      api<JobInfo[]>("/api/admin/jobs"),
      api<SessionInfo[]>("/api/admin/sessions"),
    ]);
    setSys(s);
    setJobs(j);
    setSessions(sess);
  }

  useEffect(() => {
    refresh().catch(() => undefined);
  }, []);

  async function onSweep() {
    setSweeping(true);
    try {
      await api("/api/admin/sessions/sweep", { method: "POST" });
      await refresh();
    } catch (err) {
      message.error(`清理失败：${(err as Error).message}`);
    } finally {
      setSweeping(false);
    }
  }

  const jobColumns: ColumnsType<JobInfo> = [
    { title: "Job ID", dataIndex: "job_id", key: "job_id", width: 220 },
    { title: "形象", dataIndex: "avatar_id", key: "avatar_id", width: 220 },
    {
      title: "阶段",
      dataIndex: "stage",
      key: "stage",
      width: 140,
      render: (s: string) => <StatusTag status={s} />,
    },
    {
      title: "进度",
      dataIndex: "progress",
      key: "progress",
      width: 140,
      render: (p: number) => <Progress percent={Math.round(p * 100)} size="small" />,
    },
    {
      title: "错误",
      dataIndex: "error",
      key: "error",
      render: (e: string | null) => (e ? <Typography.Text type="danger">{e}</Typography.Text> : "—"),
    },
  ];

  const sessionColumns: ColumnsType<SessionInfo> = [
    { title: "Session", dataIndex: "session_id", key: "session_id", width: 220 },
    { title: "形象", dataIndex: "avatar_id", key: "avatar_id", width: 220 },
    {
      title: "状态",
      dataIndex: "status",
      key: "status",
      width: 100,
      render: (s: string) => <StatusTag status={s} />,
    },
    {
      title: "最近活跃",
      dataIndex: "last_active_at",
      key: "last_active_at",
      width: 180,
      render: (v: string | null) => (v ? new Date(v).toLocaleString() : "—"),
    },
    {
      title: "操作",
      key: "actions",
      width: 100,
      render: (_: unknown, row) => (
        <Popconfirm title="关闭此会话？" okText="关闭" cancelText="取消" onConfirm={async () => {
          await api(`/api/admin/sessions/${row.session_id}`, { method: "DELETE" });
          await refresh();
        }}>
          <Button danger size="small" icon={<DeleteOutlined />}>关闭</Button>
        </Popconfirm>
      ),
    },
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

      <Tabs
        defaultActiveKey="overview"
        items={[
          {
            key: "overview",
            label: "概览",
            children: (
              <Space direction="vertical" size={16} style={{ width: "100%" }}>
                {sys && (
                  <Card title="引擎信息">
                    <Space direction="vertical" size={8} style={{ width: "100%" }}>
                      <Space wrap size={24}>
                        <span>
                          引擎：<Typography.Text strong>{sys.engine.name}</Typography.Text>
                        </span>
                        <span>
                          流水线：<Typography.Text strong>{sys.pipeline}</Typography.Text>
                        </span>
                        <span>
                          设备：{boolTag(sys.gpu, "GPU", "CPU")}
                        </span>
                        <span>
                          权重：{boolTag(sys.engine.weights_present ?? false, "已加载", "缺失")}
                        </span>
                      </Space>
                      <Typography.Text type="secondary">
                        数据目录：<code>{sys.data_dir || "—"}</code>　·　版本：{sys.version}
                      </Typography.Text>
                    </Space>
                  </Card>
                )}

                <Card
                  title="任务"
                  extra={
                    <Button onClick={refresh} icon={<ReloadOutlined />}>
                      刷新
                    </Button>
                  }
                >
                  <Table<JobInfo>
                    rowKey="job_id"
                    columns={jobColumns}
                    dataSource={jobs}
                    pagination={false}
                    size="small"
                  />
                </Card>

                <Card
                  title="会话"
                  extra={
                    <Button
                      icon={<SyncOutlined />}
                      onClick={onSweep}
                      loading={sweeping}
                    >
                      清理空闲会话
                    </Button>
                  }
                >
                  <Table<SessionInfo>
                    rowKey="session_id"
                    columns={sessionColumns}
                    dataSource={sessions}
                    pagination={false}
                    size="small"
                  />
                </Card>
              </Space>
            ),
          },
          {
            key: "liveportrait",
            label: "动作模板",
            children: <LivePortraits />,
          },
        ]}
      />
    </Space>
  );
}