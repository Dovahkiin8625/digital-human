import { Tag } from "antd";
import type { AvatarKind } from "../types";

// One place that decides how a kind reads. The two kinds are the same
// feature -- a selectable, drivable, deletable avatar -- so this tag is
// the only thing telling them apart, and it must not drift between the
// list and the detail header.
const KINDS: Record<AvatarKind, { color: string; text: string }> = {
  photoreal: { color: "blue", text: "真人" },
  live2d: { color: "purple", text: "Live2D" },
};

export default function KindTag({ kind }: { kind: AvatarKind }) {
  const s = KINDS[kind] ?? { color: "default", text: kind };
  return <Tag color={s.color}>{s.text}</Tag>;
}
