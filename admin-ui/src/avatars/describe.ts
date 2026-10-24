import type { AvatarInfo } from "../types";

/** How one avatar is driven, as the list's 驱动方式 column reads it. */
export type DriveLabel = "图片驱动" | "视频驱动" | "Cubism";

/**
 * The single cell that differs between the two avatar kinds.
 *
 * A Live2D model has no source pack and no drive mode -- it is a Cubism
 * project the browser draws itself -- so it gets the runtime name. A
 * photoreal avatar's mode is only known once the pipeline has written
 * pack.json, so before that this returns null and the table renders a
 * dash rather than guessing from the file suffix.
 */
export function driveLabel(
  avatar: Pick<AvatarInfo, "kind" | "config">
): DriveLabel | null {
  if (avatar.kind === "live2d") return "Cubism";
  const cfg = avatar.config as Record<string, unknown> | null | undefined;
  const prep = (cfg?.musetalk as Record<string, unknown> | undefined) ?? {};
  const mode = prep.mode;
  if (mode === "video") return "视频驱动";
  if (mode === "image") return "图片驱动";
  return null;
}

export type ExpressionsResult =
  | { ok: true; value: Record<string, number> }
  | { ok: false; error: string };

/**
 * Validate the operator's emotion map before it is submitted.
 *
 * The index is checked here rather than left to the server's
 * `validate_model_dir` because a bad value does not fail at registration --
 * it becomes an `expressionIndex` the Cubism stage cannot use and surfaces
 * minutes later as a face that never changes.
 */
export function parseExpressions(raw: string): ExpressionsResult {
  const trimmed = raw.trim();
  if (!trimmed) return { ok: true, value: {} };

  let parsed: unknown;
  try {
    parsed = JSON.parse(trimmed);
  } catch {
    // A syntax error is the operator's typo. The server's 422 would say
    // only that the expression map is wrong, not that the braces are
    // unbalanced, which is the half of the message that helps.
    return { ok: false, error: "表情映射不是合法 JSON" };
  }
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
    return { ok: false, error: '表情映射必须是形如 {"smile": 0} 的对象' };
  }

  const value: Record<string, number> = {};
  for (const [tag, idx] of Object.entries(parsed as Record<string, unknown>)) {
    if (typeof idx !== "number" || !Number.isInteger(idx) || idx < 0) {
      return { ok: false, error: `表情映射「${tag}」的取值必须是非负整数` };
    }
    value[tag] = idx;
  }
  return { ok: true, value };
}
