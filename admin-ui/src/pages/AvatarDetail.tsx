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
  Image,
  Row,
  Slider,
  Space,
  Spin,
  Typography,
} from "antd";
import { ArrowLeftOutlined, ReloadOutlined } from "@ant-design/icons";
import { api, getToken } from "../api";
import StatusTag from "../components/StatusTag";
import type { AvatarInfo } from "../types";

async function fetchPreviewUrl(avatarId: string, level: number): Promise<string> {
  // A plain <img> cannot attach X-Admin-Token, so fetch the PNG with the header
  // and hand the browser an object URL of the resulting blob.
  const resp = await fetch(
    `/api/admin/avatars/${avatarId}/preview.png?level=${level}&t=${Date.now()}`,
    { headers: { "X-Admin-Token": getToken() ?? "" } }
  );
  if (!resp.ok) throw new Error("preview unavailable");
  const blob = await resp.blob();
  return URL.createObjectURL(blob);
}

export default function AvatarDetail() {
  const { id } = useParams();
  const [info, setInfo] = useState<AvatarInfo | null>(null);
  const [mouth, setMouth] = useState(1);
  const [expression, setExpression] = useState(1);
  const [level, setLevel] = useState(0.5);
  // The level the preview was last fetched for. Kept separate from `level` so
  // dragging the slider updates the label live while the (expensive) preview
  // request only fires once the user lets go.
  const [fetchLevel, setFetchLevel] = useState(0.5);
  const [preview, setPreview] = useState("");
  const [previewError, setPreviewError] = useState("");
  const [previewLoading, setPreviewLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const previewRef = useRef("");
  const { message } = AntApp.useApp();

  async function refresh() {
    if (!id) return;
    const data = await api<AvatarInfo>(`/api/admin/avatars/${id}`);
    setInfo(data);
    setMouth(data.config.mouth_scale ?? 1);
    setExpression(data.config.expression_intensity ?? 1);
  }

  useEffect(() => {
    refresh()
      .catch((err) => setError((err as Error).message))
      .finally(() => setLoading(false));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id]);

  useEffect(() => {
    if (!id) return;
    let cancelled = false;
    setPreviewLoading(true);
    setPreviewError("");
    fetchPreviewUrl(id, fetchLevel)
      .then((url) => {
        if (cancelled) {
          URL.revokeObjectURL(url);
          return;
        }
        // Replace: revoke the object URL we are about to drop so blobs do not leak.
        if (previewRef.current) URL.revokeObjectURL(previewRef.current);
        previewRef.current = url;
        setPreview(url);
      })
      .catch((err) => {
        if (!cancelled) setPreviewError((err as Error).message);
      })
      .finally(() => {
        if (!cancelled) setPreviewLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [id, fetchLevel]);

  // Revoke the last object URL when the page unmounts.
  useEffect(() => {
    return () => {
      if (previewRef.current) URL.revokeObjectURL(previewRef.current);
    };
  }, []);

  async function save() {
    if (!id) return;
    setSaving(true);
    try {
      setError("");
      await api(`/api/admin/avatars/${id}/config`, {
        method: "PATCH",
        body: { mouth_scale: mouth, expression_intensity: expression },
      });
      message.success("参数已保存");
      await refresh();
    } catch (err) {
      message.error((err as Error).message);
    } finally {
      setSaving(false);
    }
  }

  async function retry() {
    if (!id) return;
    try {
      setError("");
      await api(`/api/admin/avatars/${id}/retry`, { method: "POST", body: {} });
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
        <Alert type="error" showIcon message="无法加载该形象" description={error || "形象不存在或已被删除"} />
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
          { key: "created", label: "创建时间", children: new Date(info.created_at).toLocaleString() },
          {
            key: "config",
            label: "当前参数",
            children: `口型 ${(info.config.mouth_scale ?? 1).toFixed(2)} · 表情 ${(
              info.config.expression_intensity ?? 1
            ).toFixed(2)}`,
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

      <Row gutter={[16, 16]}>
        <Col xs={24} lg={10}>
          <Card title="生成参数">
            <Typography.Text>口型幅度 {mouth.toFixed(2)}</Typography.Text>
            <Slider min={0} max={2} step={0.05} value={mouth} onChange={setMouth} />
            <Typography.Text>表情强度 {expression.toFixed(2)}</Typography.Text>
            <Slider min={0} max={2} step={0.05} value={expression} onChange={setExpression} />
            <Button type="primary" onClick={save} loading={saving}>
              保存参数
            </Button>
          </Card>
        </Col>
        <Col xs={24} lg={14}>
          <Card title="预览">
            <Typography.Text>驱动电平 {level.toFixed(2)}</Typography.Text>
            <Slider
              min={0}
              max={1}
              step={0.05}
              value={level}
              onChange={setLevel}
              onChangeComplete={setFetchLevel}
            />
            {previewError ? (
              <Empty description={previewError}>
                <Button
                  icon={<ReloadOutlined />}
                  onClick={() => {
                    setPreviewError("");
                    setPreviewLoading(true);
                    fetchPreviewUrl(id!, fetchLevel)
                      .then((url) => {
                        if (previewRef.current) URL.revokeObjectURL(previewRef.current);
                        previewRef.current = url;
                        setPreview(url);
                      })
                      .catch((err) => setPreviewError((err as Error).message))
                      .finally(() => setPreviewLoading(false));
                  }}
                >
                  重试
                </Button>
              </Empty>
            ) : (
              <Spin spinning={previewLoading}>
                <div style={{ minHeight: 120 }}>
                  {preview && <Image src={preview} alt="preview" width={240} />}
                </div>
              </Spin>
            )}
          </Card>
        </Col>
      </Row>
    </Space>
  );
}
