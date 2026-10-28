import { useEffect, useMemo, useState } from "react";
import {
  AutoComplete,
  Button,
  Card,
  Col,
  Descriptions,
  Empty,
  Popconfirm,
  Row,
  Select,
  Space,
  Tag,
  Typography,
  App as AntApp,
} from "antd";
import { DeleteOutlined, PlusOutlined, ReloadOutlined } from "@ant-design/icons";
import Live2DStage from "../live2d/Live2DStage";
import { live2dModelInventory, live2dModelUrl, updateLive2DConfig } from "../api";
import { EMOTION_TAGS, NO_EXPRESSION } from "../live2d/render/expressions";
import { formatBytes } from "../format";
import type {
  ArtifactInfo,
  AvatarInfo,
  Live2DModelConfig,
  Live2DModelInventory,
} from "../types";

/**
 * The Live2D half of the avatar detail page.
 *
 * Everything here is a different *kind* of the same thing, not a
 * different feature: the model files come from the same artifact listing
 * a photoreal avatar uses, and the preview is the same <Live2DStage>
 * component the debug page and the chat page mount.
 * What a Live2D avatar does not have is a build pipeline -- so there is
 * no idle clip to fetch, no timeline, and nothing to retry.
 */
export default function Live2dAvatarDetail({
  info,
  artifacts,
  onSaved,
}: {
  info: AvatarInfo;
  artifacts: ArtifactInfo[];
  onSaved?: () => void;
}) {
  const config = (info.config ?? {}) as Live2DModelConfig;
  const entry = config.model?.entry ?? "";
  // An avatar registered without an emotion map is normal, not broken --
  // render nothing rather than iterating undefined.
  const tags = Object.entries(config.expressions ?? {});
  // A canvas cannot raise this itself: a failed model load is a blank
  // rectangle with no error anywhere. The stage hands us the exception,
  // so the page has to be the one that shows it.
  const [stageError, setStageError] = useState("");
  // Which expression the preview is wearing. -1 is the model's own rest
  // face, which is what the pet wears between replies.
  const [preview, setPreview] = useState(NO_EXPRESSION);
  const [inventory, setInventory] = useState<Live2DModelInventory | null>(null);

  useEffect(() => {
    let alive = true;
    live2dModelInventory(info.avatar_id)
      .then((inv) => alive && setInventory(inv))
      .catch(() => {
        /* the editor below reports its own failure */
      });
    return () => {
      alive = false;
    };
  }, [info.avatar_id]);

  const totalBytes = useMemo(
    () => artifacts.reduce((n, a) => n + a.size, 0),
    [artifacts],
  );

  return (
    <Space direction="vertical" size={16} style={{ width: "100%" }}>
      <Row gutter={[16, 16]}>
        <Col xs={24} md={12}>
          <Card title="预览">
            {stageError ? (
              <Empty description={`模型加载失败（${stageError}）`}>
                <Button
                  icon={<ReloadOutlined />}
                  onClick={() => window.location.reload()}
                >
                  重试
                </Button>
              </Empty>
            ) : entry ? (
              // audio=null: this is the idle layer only. The mouth is driven
              // by the AudioBuffer actually playing, and nothing is playing
              // here, so the model breathes and blinks without a mouth.
              <Live2DStage
                modelUrl={live2dModelUrl(info.avatar_id, entry)}
                audio={null}
                expressionIndex={preview}
                width={240}
                height={360}
                onError={(e) => setStageError(e instanceof Error ? e.message : String(e))}
              />
            ) : (
              <Empty description="该形象没有入口文件，无法预览" />
            )}
            {/* The model's own expressions, so an operator can see what
                each one looks like before deciding which emotion wears
                it. */}
            {inventory && inventory.expressions.length > 0 && (
              <Space size={4} wrap style={{ marginTop: 12 }}>
                <Button
                  size="small"
                  type={preview === NO_EXPRESSION ? "primary" : "default"}
                  onClick={() => setPreview(NO_EXPRESSION)}
                >
                  默认表情
                </Button>
                {inventory.expressions.map((e) => (
                  <Button
                    key={e.index}
                    size="small"
                    type={preview === e.index ? "primary" : "default"}
                    onClick={() => setPreview(e.index)}
                  >
                    {e.name}
                  </Button>
                ))}
              </Space>
            )}
          </Card>
        </Col>

        <Col xs={24} md={12}>
          <Card title="模型信息">
            <Descriptions
              size="small"
              column={1}
              bordered
              items={[
                { key: "entry", label: "入口文件", children: entry || "—" },
                {
                  key: "expressions",
                  label: "表情映射",
                  children:
                    tags.length === 0 ? (
                      <Typography.Text type="secondary">无</Typography.Text>
                    ) : (
                      <Space size={4} wrap>
                        {tags.map(([tag, idx]) => (
                          <Tag key={tag}>
                            {tag} → {idx}
                          </Tag>
                        ))}
                      </Space>
                    ),
                },
                { key: "files", label: "文件数", children: artifacts.length },
                { key: "size", label: "总大小", children: formatBytes(totalBytes) },
              ]}
            />
            <Typography.Paragraph type="secondary" style={{ marginTop: 12, marginBottom: 0 }}>
              Live2D 形象在客户端用 WebGL 渲染，服务端不产出任何画面，因此没有构建过程、也没有可重建的产物。
            </Typography.Paragraph>
          </Card>
        </Col>
      </Row>

      <ExpressionEditor info={info} inventory={inventory} onSaved={onSaved} />
    </Space>
  );
}

/**
 * The emotion map editor.
 *
 * The map is written in terms of POSITIONS in the model's own expression
 * list, so the choices here come from the project itself rather than from
 * a hand-typed index -- the names behind those positions are what the
 * operator is actually choosing between, and a number would mean nothing
 * to them.
 */
function ExpressionEditor({
  info,
  inventory,
  onSaved,
}: {
  info: AvatarInfo;
  inventory: Live2DModelInventory | null;
  onSaved?: () => void;
}) {
  const { message } = AntApp.useApp();
  const [rows, setRows] = useState<Array<{ tag: string; index: number }>>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  // The stored map seeds the rows. It changes when the avatar is saved,
  // so the editor follows it rather than keeping a copy of its own.
  useEffect(() => {
    const current = (info.config as Live2DModelConfig)?.expressions ?? {};
    setRows(Object.entries(current).map(([tag, index]) => ({ tag, index })));
  }, [info.config]);

  const tagOptions = useMemo(() => {
    const names = new Set<string>([...EMOTION_TAGS, ...rows.map((r) => r.tag)]);
    return Array.from(names).map((t) => ({ value: t }));
  }, [rows]);

  const expressionOptions = useMemo(
    () =>
      (inventory?.expressions ?? []).map((e) => ({
        value: e.index,
        label: `${e.index} · ${e.name}`,
      })),
    [inventory],
  );

  const setRow = (i: number, patch: Partial<{ tag: string; index: number }>) => {
    setRows((cur) => cur.map((row, n) => (n === i ? { ...row, ...patch } : row)));
  };

  const save = async () => {
    setBusy(true);
    setError("");
    try {
      const expressions: Record<string, number> = {};
      for (const row of rows) {
        const tag = row.tag.trim();
        // A row with no tag is a row the operator has not filled in yet.
        if (!tag) continue;
        expressions[tag] = row.index;
      }
      await updateLive2DConfig(info.avatar_id, { expressions });
      message.success("表情映射已保存");
      onSaved?.();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  if (error && !inventory) {
    return (
      <Card title="表情映射">
        <Typography.Text type="danger">{error}</Typography.Text>
      </Card>
    );
  }

  if (inventory && inventory.expressions.length === 0) {
    return (
      <Card title="表情映射">
        <Empty description="该模型没有表情文件，无法设置情绪映射" />
      </Card>
    );
  }

  return (
    <Card
      title="表情映射"
      extra={
        <Space>
          <Button
            icon={<PlusOutlined />}
            onClick={() =>
              setRows((cur) => [
                ...cur,
                { tag: "", index: inventory?.expressions[0]?.index ?? 0 },
              ])
            }
          >
            添加
          </Button>
          <Button type="primary" loading={busy} onClick={save}>
            保存
          </Button>
        </Space>
      }
    >
      <Typography.Paragraph type="secondary" style={{ marginBottom: 12 }}>
        左边是情绪标签，右边是模型自带表情列表里的下标。模型没有对应表情时留空即可，未识别的情绪会退回中性。
      </Typography.Paragraph>

      {error && (
        <Typography.Text type="danger" style={{ display: "block", marginBottom: 8 }}>
          {error}
        </Typography.Text>
      )}

      <Space direction="vertical" style={{ width: "100%" }}>
        {rows.map((row, i) => (
          <Space key={i}>
            <AutoComplete
              options={tagOptions}
              value={row.tag}
              placeholder="情绪标签，如 smile"
              style={{ width: 200 }}
              onChange={(v) => setRow(i, { tag: v })}
            />
            <Select
              options={expressionOptions}
              value={row.index}
              style={{ width: 260 }}
              onChange={(v) => setRow(i, { index: v })}
            />
            <Popconfirm
              title="删除这一条？"
              onConfirm={() => setRows((cur) => cur.filter((_, n) => n !== i))}
            >
              <Button icon={<DeleteOutlined />} />
            </Popconfirm>
          </Space>
        ))}
        {rows.length === 0 && (
          <Typography.Text type="secondary">
            还没有映射。点「添加」把情绪标签连到模型的表情上。
          </Typography.Text>
        )}
      </Space>

      {inventory && Object.keys(inventory.motions).length > 0 && (
        <Typography.Paragraph type="secondary" style={{ marginTop: 12, marginBottom: 0 }}>
          模型自带动作：
          {Object.entries(inventory.motions)
            .map(([group, n]) => `${group}（${n}）`)
            .join("、")}
        </Typography.Paragraph>
      )}
    </Card>
  );
}
