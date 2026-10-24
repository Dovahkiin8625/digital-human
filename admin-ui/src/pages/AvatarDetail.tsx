import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import {
  Alert,
  App as AntApp,
  Button,
  Card,
  Col,
  Collapse,
  Descriptions,
  Empty,
  Progress,
  Row,
  Space,
  Spin,
  Steps,
  Table,
  Tooltip,
  Typography,
} from "antd";
import type { ColumnsType } from "antd/es/table";
import {
  ArrowLeftOutlined,
  DownloadOutlined,
  FileOutlined,
  FileImageOutlined,
  FileTextOutlined,
  PlayCircleOutlined,
  ReloadOutlined,
} from "@ant-design/icons";
import { api, downloadWithToken } from "../api";
import StatusTag from "../components/StatusTag";
import KindTag from "../avatars/KindTag";
import Live2dAvatarDetail from "./Live2dAvatarDetail";
import type { ArtifactInfo, AvatarInfo, JobInfo, PipelineTimeline } from "../types";

type PreviewKind = "video" | "image";

async function fetchObjectUrl(
  path: string
): Promise<{ url: string }> {
  // No auth/token needed after the no-tenant refactor: a plain <video>/<img>
  // can fetch directly. The server returns the raw bytes either way; we
  // wrap them in a blob URL so the element sees a same-origin source.
  const resp = await fetch(path);
  if (!resp.ok) {
    const err = new Error(`request failed: ${resp.status}`) as Error & {
      status: number;
    };
    err.status = resp.status;
    throw err;
  }
  return { url: URL.createObjectURL(await resp.blob()) };
}

// Tooltip blurb for each quality threshold so the warning <Alert> can show why
// it fired without crowding the table.
const THRESHOLD_HINTS: Record<string, string> = {
  width: "人脸在画面中的水平宽度阈值（face width ≥ 160px）",
  pose: "鼻梁相对两颊的对称度阈值，反映正脸程度（nose ratio ≥ 0.55）",
  sharpness: "面部区域拉普拉斯方差，反映对焦与压缩程度（sharpness ≥ 60）",
  occupancy: "主体在画布中的占比阈值，对应 9:16 竖版构图（occupancy ≥ 0.92）",
};

function KindIcon({ kind }: { kind: string }) {
  if (kind === "image") return <FileImageOutlined />;
  if (kind === "json") return <FileTextOutlined />;
  if (kind === "video") return <PlayCircleOutlined />;
  return <FileOutlined />;
}

function classifyWarning(text: string): keyof typeof THRESHOLD_HINTS {
  if (text.includes("宽") || text.includes("像素")) return "width";
  if (text.includes("偏转") || text.includes("正脸")) return "pose";
  if (text.includes("清晰") || text.includes("压缩")) return "sharpness";
  if (text.includes("构图") || text.includes("黑边") || text.includes("占比")) return "occupancy";
  // Default to width's hint; it's the most common source-quality concern.
  return "width";
}

export function formatBytes(b: number): string {
  if (b < 1024) return `${b} B`;
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB`;
  return `${(b / 1024 / 1024).toFixed(1)} MB`;
}

const STAGES = ["detect_face", "align", "normalize", "build_pack"] as const;
type Stage = (typeof STAGES)[number];

function stageToIndex(s: string | null | undefined): number {
  if (!s) return -1;
  if (s === "queued") return 0;
  const idx = STAGES.indexOf(s as Stage);
  if (idx >= 0) return idx;
  if (s === "done") return STAGES.length;
  return -1;
}

export default function AvatarDetail() {
  const { id } = useParams();
  const [info, setInfo] = useState<AvatarInfo | null>(null);
  const [preview, setPreview] = useState("");
  const [previewKind, setPreviewKind] = useState<PreviewKind>("video");
  const [previewError, setPreviewError] = useState("");
  const [previewLoading, setPreviewLoading] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [artifacts, setArtifacts] = useState<ArtifactInfo[]>([]);
  const [timeline, setTimeline] = useState<PipelineTimeline | null>(null);
  const [latestJob, setLatestJob] = useState<JobInfo | null>(null);
  const previewRef = useRef("");
  const { message } = AntApp.useApp();

  async function refresh() {
    if (!id) return null;
    const data = await api<AvatarInfo>(`/api/admin/avatars/${id}`);
    setInfo(data);
    return data;
  }

  async function loadArtifacts() {
    if (!id) return;
    try {
      const rows = await api<ArtifactInfo[]>(`/api/admin/avatars/${id}/artifacts`);
      setArtifacts(rows);
    } catch {
      /* pipeline may not have written anything yet; empty list is fine */
    }
  }

  async function loadTimeline() {
    if (!id) return;
    try {
      const t = await api<PipelineTimeline>(`/api/admin/avatars/${id}/timeline`);
      setTimeline(t);
    } catch {
      /* fallback uses info.config.warnings below; nothing else needs this */
    }
  }

  async function loadJob() {
    // Look up the latest avatar build job to show "current progress" inline.
    // The /jobs endpoint can return 500+ rows; we only care about this avatar's.
    try {
      const rows = await api<JobInfo[]>("/api/admin/jobs");
      const job = rows.find((j) => j.avatar_id === id) ?? null;
      setLatestJob(job);
    } catch {
      /* session without a job history yet; nothing to render */
    }
  }

  useEffect(() => {
    const signal = { ignored: false };
    refresh()
      .then((data) => {
        if (signal.ignored) return;
        // A Live2D avatar has no build pipeline, so there is no timeline to
        // read and no job to poll -- both requests would only 404. The
        // artifact listing IS shared and is fetched for both kinds.
        if (data?.kind === "live2d") return;
        void loadTimeline();
        void loadJob();
      })
      .catch((err) => {
        if (!signal.ignored) setError((err as Error).message);
      })
      .finally(() => {
        if (!signal.ignored) setLoading(false);
      });
    void loadArtifacts();
    return () => {
      signal.ignored = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id]);

  async function loadPreview(signal?: { ignored: boolean }) {
    if (!id) return;
    setPreviewLoading(true);
    setPreviewError("");
    let url = "";
    let kind: PreviewKind = "video";
    try {
      try {
        ({ url } = await fetchObjectUrl(
          `/api/admin/avatars/${id}/idle.mp4?t=${Date.now()}`
        ));
      } catch (err) {
        // 404 = no idle clip (photo avatar): fall back to the resting image.
        // Any other status is a real failure shown below with a retry.
        if ((err as { status?: number }).status !== 404) {
          throw err;
        }
        ({ url } = await fetchObjectUrl(
          `/api/admin/avatars/${id}/normalized.png?t=${Date.now()}`
        ));
        kind = "image";
      }
      // The avatar may have changed while these fetches were in flight: this
      // effect run is then stale. Do not touch the new run's state or ref, and
      // revoke the blob nobody will render.
      if (signal?.ignored) {
        URL.revokeObjectURL(url);
        return;
      }
      // Replace and revoke the previous object URL so blobs do not leak.
      if (previewRef.current) URL.revokeObjectURL(previewRef.current);
      previewRef.current = url;
      setPreviewKind(kind);
      setPreview(url);
    } catch (err) {
      if (!signal?.ignored) setPreviewError((err as Error).message);
    } finally {
      if (!signal?.ignored) setPreviewLoading(false);
    }
  }

  useEffect(() => {
    // Waits for `info` so the kind is known. A Live2D avatar has no
    // idle.mp4 and no normalized.png -- Live2dAvatarDetail mounts its own
    // canvas instead -- and fetching those two would just 404 twice.
    if (!info || info.kind === "live2d") return;
    const signal = { ignored: false };
    void loadPreview(signal);
    return () => {
      signal.ignored = true;
      if (previewRef.current) URL.revokeObjectURL(previewRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id, info?.kind]);

  async function retry() {
    if (!id) return;
    try {
      setError("");
      await api(`/api/admin/avatars/${id}/retry`, {
        method: "POST",
        body: {},
      });
      await refresh();
      void loadArtifacts();
      void loadTimeline();
      void loadJob();
    } catch (err) {
      message.error((err as Error).message);
    }
  }

  // Effective params could come from pack.json's _effective_params (set by the
  // pipeline at build time) or, when absent, fall back to /api/admin/settings.
  // Memo avoids re-reading in render cycles that don't change the underlying
  // avatar.
  const effectiveParams = useMemo(() => {
    return (info?.config as { _effective_params?: Record<string, unknown> })
      ?._effective_params;
  }, [info]);

  const warningList = useMemo(() => {
    return timeline?.warnings ?? info?.config.warnings ?? [];
  }, [timeline, info]);

  if (loading) {
    return (
      <div style={{ display: "grid", placeItems: "center", padding: 64 }}>
        <Spin tip="加载中…">
          <div style={{ width: 120, height: 60 }} />
        </Spin>
      </div>
    );
  }

  // The fetch failed or the avatar is gone -- show why instead of spinning forever.
  if (!info) {
    return (
      <Space direction="vertical" size={16}>
        <Alert
          type="error"
          showIcon
          message="无法加载该形象"
          description={error || "形象不存在或已被删除"}
        />
        <Link to="/avatars">
          <Button icon={<ArrowLeftOutlined />}>返回列表</Button>
        </Link>
      </Space>
    );
  }

  // The two avatar kinds share this page: the header, the artifact list,
  // the error handling. What differs is the body -- a photoreal avatar
  // has a build pipeline and an idle clip, a Live2D one has a canvas.
  const isLive2d = info.kind === "live2d";

  const artifactColumns: ColumnsType<ArtifactInfo> = [
    {
      title: "路径",
      dataIndex: "path",
      ellipsis: true,
      render: (p: string) => (
        <Tooltip title={p}>
          <Typography.Text code style={{ fontSize: 12 }}>{p}</Typography.Text>
        </Tooltip>
      ),
    },
    { title: "大小", dataIndex: "size", width: 110, render: (b: number) => formatBytes(b) },
    {
      title: "类型",
      dataIndex: "kind",
      width: 100,
      render: (k: string) => (
        <Space size={4}>
          <KindIcon kind={k} />
          <span>{k}</span>
        </Space>
      ),
    },
    {
      title: "操作",
      width: 110,
      render: (_, a) => (
        <Button
          size="small"
          icon={<DownloadOutlined />}
          onClick={() => {
            if (!id) return;
            void downloadWithToken(`/api/admin/avatars/${id}/artifact/${a.path}`).catch((err) =>
              message.error((err as Error).message)
            );
          }}
        >
          下载
        </Button>
      ),
    },
  ];

  // Build a stage list that reacts to the latest avatar-job stage. Steps
  // render below 100% width so the description (frame count, fps, ...) fits.
  const stageCurrent = stageToIndex(latestJob?.stage ?? info.status);
  const stepStatus = (idx: number): "wait" | "process" | "finish" | "error" => {
    if (info.status === "failed") return idx === stageCurrent ? "error" : "wait";
    if (idx < stageCurrent) return "finish";
    if (idx === stageCurrent) return "process";
    return "wait";
  };

  // Stable key list so React keeps the steps mounted across re-renders.
  const musetalk = (info.config.musetalk as Record<string, unknown> | undefined) ?? {};
  const baseImage = (info.config as Record<string, unknown>).base_image as string | undefined;

  return (
    <Space direction="vertical" size={16} style={{ width: "100%" }}>
      <Space align="center" wrap>
        <Link to="/avatars">
          <Button icon={<ArrowLeftOutlined />} type="text">
            返回
          </Button>
        </Link>
        <Typography.Title level={3} style={{ margin: 0 }}>
          {info.name}
        </Typography.Title>
        <StatusTag status={info.status} />
        <KindTag kind={info.kind} />
        {/* A Live2D avatar has no build job to re-run: the model is served
            as-is and there is no pipeline to repeat. */}
        {!isLive2d && (
          <Button icon={<ReloadOutlined />} onClick={() => void retry()} type="primary" ghost>
            按当前设置重建
          </Button>
        )}
      </Space>

      <Descriptions
        column={{ xs: 1, sm: 2, md: 3 }}
        bordered
        size="small"
        items={[
          { key: "id", label: "形象 ID", children: info.avatar_id },
          { key: "kind", label: "类型", children: <KindTag kind={info.kind} /> },
          {
            key: "created",
            label: "创建时间",
            children: new Date(info.created_at).toLocaleString(),
          },
        ]}
      />

      {info.error && (
        <Alert
          type="error"
          showIcon
          message="生成失败"
          description={info.error}
          action={
            <Button size="small" danger onClick={retry}>
              重试
            </Button>
          }
        />
      )}
      {error && <Alert type="error" showIcon message={error} />}

      {isLive2d ? (
        <Live2dAvatarDetail info={info} artifacts={artifacts} />
      ) : (
        <>
          {warningList.length > 0 && (
            <Alert
              type="warning"
              showIcon
              message="素材质量评估（形象已生成，更换更合适的素材可进一步提升效果）"
              description={
                <Space direction="vertical" size={6} style={{ width: "100%" }}>
                  {warningList.map((w, i) => {
                    const kind = classifyWarning(w);
                    const hint = THRESHOLD_HINTS[kind] ?? "";
                    return (
                      <Tooltip key={i} title={hint}>
                        <span style={{ cursor: "help" }}>· {w}</span>
                      </Tooltip>
                    );
                  })}
                </Space>
              }
            />
          )}

          <Row gutter={[16, 16]}>
            <Col xs={24} md={12}>
              <Card title="预览">
                {previewError ? (
                  <Empty description={`预览不可用（${previewError}）`}>
                    <Button
                      icon={<ReloadOutlined />}
                      onClick={() => {
                        void loadPreview();
                      }}
                    >
                      重试
                    </Button>
                  </Empty>
                ) : (
                  <Spin spinning={previewLoading}>
                    <div style={{ minHeight: 120 }}>
                      {preview && previewKind === "video" ? (
                        <video
                          src={preview}
                          controls
                          muted
                          autoPlay
                          playsInline
                          loop
                          style={{ width: 240 }}
                          onError={() =>
                            // A successful fetch only proves the bytes arrived: an
                            // unsupported codec still leaves a black player. Show
                            // the retry UI instead of a silent black box.
                            setPreviewError("视频无法解码播放")
                          }
                        />
                      ) : preview ? (
                        <img
                          src={preview}
                          alt="preview"
                          width={240}
                          onError={() => setPreviewError("图片无法解码显示")}
                        />
                      ) : null}
                    </div>
                  </Spin>
                )}
              </Card>
            </Col>

            <Col xs={24} md={12}>
              <Card title="构建过程">
                <Steps
                  size="small"
                  current={Math.max(0, stageCurrent)}
                  status={info.status === "failed" ? "error" : undefined}
                  items={STAGES.map((s, i) => ({
                    title: <StatusTag status={s} />,
                    status: stepStatus(i),
                  }))}
                />
                <Descriptions
                  size="small"
                  column={2}
                  style={{ marginTop: 12 }}
                  items={[
                    {
                      key: "mode",
                      label: "模式",
                      children: timeline?.mode === "video" ? "视频驱动" : timeline?.mode === "image" ? "图片驱动" : "—",
                    },
                    {
                      key: "frames",
                      label: "帧数",
                      children: timeline?.frame_count ?? "—",
                    },
                    {
                      key: "kept",
                      label: "保留帧",
                      children: timeline?.kept_count ?? "—",
                    },
                    {
                      key: "fps",
                      label: "输出 FPS",
                      children: timeline?.fps ?? "—",
                    },
                    {
                      key: "srcfps",
                      label: "源 FPS",
                      children: timeline?.source_fps ?? "—",
                    },
                    {
                      key: "detector",
                      label: "检测器",
                      children: timeline?.detector ?? "—",
                    },
                    {
                      key: "version",
                      label: "版本",
                      children: timeline?.version ?? "—",
                    },
                  ]}
                />
                {latestJob && ACTIVE_JOB_STAGES.has(latestJob.stage) && (
                  <div style={{ marginTop: 12 }}>
                    <Typography.Text type="secondary">当前会话进度</Typography.Text>
                    <div style={{ marginTop: 4 }}>
                      <Space>
                        <StatusTag status={latestJob.stage} />
                        <Progress
                          percent={latestJob.progress}
                          size="small"
                          style={{ width: 200 }}
                          status={latestJob.progress >= 100 ? "success" : "active"}
                        />
                      </Space>
                    </div>
                  </div>
                )}
              </Card>
            </Col>
          </Row>
        </>
      )}

      <Card title={isLive2d ? "模型文件" : "素材与产物"}>
        {artifacts.length === 0 ? (
          <Empty description={isLive2d ? "暂无模型文件" : "暂无产物；可在构建完成后下载"} />
        ) : (
          <Table
            rowKey="path"
            size="small"
            columns={artifactColumns}
            dataSource={artifacts}
            pagination={{ pageSize: 10, hideOnSinglePage: true }}
            scroll={{ x: 600 }}
          />
        )}
      </Card>

      {!isLive2d && (
        <Card title="底层配置">
          <Descriptions
            size="small"
            column={2}
            bordered
            items={[
              { key: "bbox", label: "bbox", children: JSON.stringify(musetalk.bbox ?? "—") },
              { key: "crop", label: "crop", children: JSON.stringify(musetalk.crop ?? "—") },
              { key: "detector", label: "detector", children: String(musetalk.detector ?? "—") },
              { key: "version", label: "version", children: String(musetalk.version ?? timeline?.version ?? "—") },
              { key: "frames", label: "frame_count", children: String(musetalk.frame_count ?? timeline?.frame_count ?? "—") },
              { key: "fps", label: "fps", children: String(musetalk.fps ?? timeline?.fps ?? "—") },
              { key: "srcfps", label: "source_fps", children: String(musetalk.source_fps ?? timeline?.source_fps ?? "—") },
              { key: "mode", label: "mode", children: String(musetalk.mode ?? timeline?.mode ?? "—") },
              { key: "base", label: "base_image", children: baseImage ?? "—" },
            ]}
          />
          <Collapse
            ghost
            style={{ marginTop: 12 }}
            items={[
              {
                key: "eff",
                label: "实际生效参数（_effective_params）",
                children:
                  effectiveParams && Object.keys(effectiveParams).length > 0 ? (
                    <Descriptions
                      size="small"
                      column={2}
                      bordered
                      items={Object.entries(effectiveParams).map(([k, v]) => ({
                        key: k,
                        label: k,
                        children: JSON.stringify(v),
                      }))}
                    />
                  ) : (
                    <Empty description="此形象尚未记录生效参数" />
                  ),
              },
            ]}
          />
          <Typography.Text type="secondary" style={{ display: "block", marginTop: 8 }}>
            调整「参数设置」中的全局值后，下次重建即生效。
          </Typography.Text>
        </Card>
      )}
    </Space>
  );
}

// The avatar-job stages that should show a live progress bar in the detail
// page. done/failed/queued reflect terminal-ish states and render without one.
const ACTIVE_JOB_STAGES = new Set(["detect_face", "align", "normalize", "build_pack"]);
