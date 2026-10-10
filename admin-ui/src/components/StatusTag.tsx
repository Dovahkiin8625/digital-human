import { Tag } from "antd";

// One table covering every status-ish string the API hands back: avatar status,
// avatar-job stage, and stream-session status. Keys do not collide across the
// three, so a single lookup keeps the labels consistent everywhere.
const STATUS: Record<string, { color: string; text: string }> = {
  // Avatar lifecycle
  pending: { color: "gold", text: "排队中" },
  processing: { color: "processing", text: "生成中" },
  ready: { color: "success", text: "就绪" },
  failed: { color: "error", text: "失败" },
  // Avatar build job stage
  queued: { color: "default", text: "排队中" },
  detect_face: { color: "processing", text: "人脸检测" },
  align: { color: "processing", text: "对齐" },
  normalize: { color: "processing", text: "归一化" },
  build_pack: { color: "processing", text: "打包" },
  done: { color: "success", text: "完成" },
  // Stream session (server.core.render_session.SessionState)
  idle: { color: "default", text: "空闲" },
  listening: { color: "processing", text: "待命" },
  speaking: { color: "cyan", text: "播报中" },
  closed: { color: "default", text: "已关闭" },
  error: { color: "error", text: "错误" },
};

export default function StatusTag({ status }: { status: string }) {
  const s = STATUS[status] ?? { color: "default", text: status };
  return <Tag color={s.color}>{s.text}</Tag>;
}
