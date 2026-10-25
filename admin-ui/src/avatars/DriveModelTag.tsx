import { Tag } from "antd";
import type { DriveModelId, DriveModelInfo } from "../types";

/**
 * One place that decides how a drive model reads.
 *
 * The label comes from the server's registry (the console fetched it), so
 * a newly registered model shows its own name without a frontend change.
 * The colour is a local fallback keyed by id.
 */
const COLORS: Record<string, string> = {
  musetalk: "blue",
  live2d: "purple",
};

export default function DriveModelTag({
  model,
  models,
}: {
  model: DriveModelId;
  models: DriveModelInfo[];
}) {
  const label = models.find((m) => m.name === model)?.label_zh ?? model;
  return <Tag color={COLORS[model] ?? "default"}>{label}</Tag>;
}
