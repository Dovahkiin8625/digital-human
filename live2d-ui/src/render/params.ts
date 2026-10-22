/** Parameter range resolution.
 *
 * Cubism models do not share a parameter convention. ParamMouthOpenY can
 * be 0..1, -1..1, or 0..30 depending on the rig, and some models spell it
 * differently or omit it entirely. Assuming 0..1 pins the mouth shut on
 * most real models and inverts it on the signed ones.
 *
 * The ranges come off the moc3 itself (the framework reads
 * `_model.parameters.minimumValues`), so they are available as soon as the
 * model is loaded and do not depend on which motions have been read.
 */

export interface ParamIndex {
  getParameterIndex(name: string): number
  getParameterMinimumValue(i: number): number
  getParameterMaximumValue(i: number): number
}

/** The model's real mouth-openness range, or null if it has no such
 *  parameter -- in which case the caller renders a still mouth rather than
 *  failing. */
export function resolveMouthRange(model: ParamIndex): { min: number; max: number } | null {
  const i = model.getParameterIndex("ParamMouthOpenY")
  if (i < 0) return null
  return { min: model.getParameterMinimumValue(i), max: model.getParameterMaximumValue(i) }
}

/** Map a normalized 0..1 value onto the model's own range. */
export function mapToRange(value01: number, range: { min: number; max: number }): number {
  const v = Math.min(1, Math.max(0, value01))
  return range.min + v * (range.max - range.min)
}