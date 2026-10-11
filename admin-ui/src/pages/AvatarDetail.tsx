import { useEffect, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import {
  Alert,
  App as AntApp,
  Button,
  Card,
  Col,
  Descriptions,
  Empty,
  Row,
  Space,
  Spin,
  Typography,
} from "antd";
import { ArrowLeftOutlined, ReloadOutlined } from "@ant-design/icons";
import { api, getToken } from "../api";
import StatusTag from "../components/StatusTag";
import type { AvatarInfo } from "../types";

type PreviewKind = "video" | "image";

async function fetchObjectUrl(
  path: string
): Promise<{ url: string }> {
  // A plain <video>/<img> cannot attach X-Admin-Token; fetch with the header
  // and hand the element an object URL of the blob.
  const resp = await fetch(path, {
    headers: { "X-Admin-Token": getToken() ?? "" },
  });
  if (!resp.ok) {
    const err = new Error(`request failed: ${resp.status}`) as Error & {
      status: number;
    };
    err.status = resp.status;
    throw err;
  }
  return { url: URL.createObjectURL(await resp.blob()) };
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
  const previewRef = useRef("");
  const { message } = AntApp.useApp();

  async function refresh() {
    if (!id) return;
    const data = await api<AvatarInfo>(`/api/admin/avatars/${id}`);
    setInfo(data);
  }

  useEffect(() => {
    refresh()
      .catch((err) => setError((err as Error).message))
      .finally(() => setLoading(false));
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
    const signal = { ignored: false };
    void loadPreview(signal);
    return () => {
      signal.ignored = true;
      if (previewRef.current) URL.revokeObjectURL(previewRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id]);

  async function retry() {
    if (!id) return;
    try {
      setError("");
      await api(`/api/admin/avatars/${id}/retry`, {
        method: "POST",
        body: {},
      });
      await refresh();
    } catch (err) {
      message.error((err as Error).message);
    }
  }

  if (loading) {
    return (
      <div style={{ display: "grid", placeItems: "center", padding: 64 }}>
        <Spin tip="加载中…">
          <div style={{ width: 120, height: 60 }} />
        </Spin>
      </div>
    );
  }

  // The fetch failed or the avatar is gone — show why instead of spinning forever.
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
      </Space>

      <Descriptions
        column={{ xs: 1, sm: 2, md: 3 }}
        bordered
        size="small"
        items={[
          { key: "id", label: "形象 ID", children: info.avatar_id },
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
      {(info.config.warnings ?? []).length > 0 && (
        <Alert
          type="warning"
          showIcon
          message="素材质量提示（形象已生成，更换更合适的素材可进一步提升效果）"
          description={
            <ul style={{ margin: 0, paddingInlineStart: 18 }}>
              {info.config.warnings?.map((w, i) => <li key={i}>{w}</li>)}
            </ul>
          }
        />
      )}

      <Row gutter={[16, 16]}>
        <Col xs={24}>
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
      </Row>
    </Space>
  );
}
