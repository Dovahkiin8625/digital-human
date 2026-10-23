import { mapSigned, mapToRange, resolveRange } from "./params"
import type { IdlePose } from "./idle"

/**
 * One frame, in the order Cubism requires.
 *
 * The ordering is the whole point of this file. Cubism's expression and
 * motion managers lay down their own parameter values when they update;
 * anything we write before them can be overwritten. So they go first, we
 * write after, and our values win -- which is how the mouth stays ours
 * even when an expression also touches the mouth region.
 */
export interface FrameInputs {
  model: {
    setParameterValueByIndex(i: number, v: number): void
    getParameterIndex(name: string): number
    getParameterMinimumValue(i: number): number
    getParameterMaximumValue(i: number): number
  }
  expression: {
    setExpression(index: number): void
    update(): void
  }
  motion: { update(): void }
  physics: { update(): void }
  model3: { update(): void }
  expressionIndex: number
  open: number
  form: number
  idle: IdlePose
  breathScale: number
}

export function applyFrame(f: FrameInputs): void {
  // 1-2. Cubism lays down expression and motion first.
  f.expression.setExpression(f.expressionIndex)
  f.expression.update()
  f.motion.update()

  // 3. We write afterwards, so we win any parameter an expression also
  //    touches. Writing before the managers run is the single most common
  //    cause of mouth-versus-expression flicker.
  const mouthIdx = f.model.getParameterIndex("ParamMouthOpenY")
  if (mouthIdx >= 0) {
    const range = resolveRange(f.model, "ParamMouthOpenY")
    if (range) {
      f.model.setParameterValueByIndex(mouthIdx, mapToRange(f.open, range))
    }
  }

  // Every remaining parameter goes through the model's OWN range too. The
  // same rig ships ParamAngleX in -30..30 and ParamMouthOpenY in 0..1, so
  // an absolute constant that looks right for one is wrong for the others:
  // writing a head angle of ±2.5 into a -30..30 parameter reaches only ~8%
  // of the model's travel and the pet reads as frozen.
  const put = (name: string, value: number, signed: boolean) => {
    const i = f.model.getParameterIndex(name)
    if (i < 0) return
    const range = resolveRange(f.model, name)
    if (!range) return
    f.model.setParameterValueByIndex(i, signed ? mapSigned(value, range) : mapToRange(value, range))
  }
  put("ParamMouthForm", f.form, false)
  put("ParamAngleX", f.idle.angleX * f.breathScale, true)
  put("ParamAngleY", f.idle.angleY * f.breathScale, true)
  put("ParamAngleZ", f.idle.angleZ * f.breathScale, true)
  put("ParamEyeLOpen", f.idle.eyeOpen, false)
  put("ParamEyeBallX", f.idle.eyeBallX, true)
  put("ParamEyeBallY", f.idle.eyeBallY, true)

  // 4-5. Physics derives hair and cloth from the head params we just wrote,
  //      then the pose is baked into vertices.
  f.physics.update()
  f.model3.update()
}