import { useMemo, useState } from "react";
import { Button, Card, Col, Descriptions, Empty, Row, Space, Tag, Typography } from "antd";
import { ReloadOutlined } from "@ant-design/icons";
import Live2DStage from "../live2d/Live2DStage";
import { live2dModelUrl } from "../api";
import { formatBytes } from "../format";
import type { ArtifactInfo, AvatarInfo, Live2DModelConfig } from "../types";

/**
 * The Live2D half of the avatar detail page.
 *
 * Everything here is a different *kind* of the same thing, not a
 * different feature: the model files come from the same artifact listing
 * a photoreal avatar uses, and the preview is the same <Live2DStage>
 * component the debug page, the chat page and the desktop pet mount.
 * What a Live2D avatar does not have is a build pipeline -- so there is
 * no idle clip to fetch, no timeline, and nothing to retry.
 */
export default function Live2dAvatarDetail({
  info,
  artifacts,
}: {
  info: AvatarInfo;
  artifacts: ArtifactInfo[];
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
                width={240}
                height={360}
                onError={(e) => setStageError(e instanceof Error ? e.message : String(e))}
              />
            ) : (
              <Empty description="该形象没有入口文件，无法预览" />
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
    </Space>
  );
}
