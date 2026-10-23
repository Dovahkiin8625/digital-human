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

/** The model's real range for any named parameter, or null if the rig does
 *  not define it. Every driven parameter goes through this: the same model
 *  ships ParamAngleX in -30..30 and ParamMouthOpenY in 0..1, so an absolute
 *  constant that looks right for one is wrong for the others. */
export function resolveRange(
  model: ParamIndex,
  name: string,
): { min: number; max: number } | null {
  const i = model.getParameterIndex(name)
  if (i < 0) return null
  return { min: model.getParameterMinimumValue(i), max: model.getParameterMaximumValue(i) }
}

/** Convenience for the mouth, which is the parameter with the most
 *  obvious failure mode when its range is assumed rather than read. */
export function resolveMouthRange(model: ParamIndex): { min: number; max: number } | null {
  return resolveRange(model, "ParamMouthOpenY")
}

/** Map a normalized 0..1 value onto the model's own range. */
export function mapToRange(value01: number, range: { min: number; max: number }): number {
  const v = Math.min(1, Math.max(0, value01))
  return range.min + v * (range.max - range.min)
}

/** Map a signed normalized -1..1 value onto a range, centred on it. */
export function mapSigned(value: number, range: { min: number; max: number }): number {
  return mapToRange(0.5 + Math.min(1, Math.max(-1, value)) / 2, range);
}