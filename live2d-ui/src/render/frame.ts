import { mapToRange, resolveMouthRange } from "./params"
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

/** Fallback range for the form parameter, which is conventionally signed. */
const MOUTH_FORM_FALLBACK = { min: -1, max: 1 }

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
    const range = resolveMouthRange(f.model)
    if (range) {
      f.model.setParameterValueByIndex(mouthIdx, mapToRange(f.open, range))
    }
  }

  const formIdx = f.model.getParameterIndex("ParamMouthForm")
  if (formIdx >= 0) {
    f.model.setParameterValueByIndex(formIdx, mapToRange(f.form, MOUTH_FORM_FALLBACK))
  }

  const set = (name: string, v: number) => {
    const i = f.model.getParameterIndex(name)
    if (i >= 0) f.model.setParameterValueByIndex(i, v)
  }
  set("ParamAngleX", f.idle.angleX * f.breathScale)
  set("ParamAngleY", f.idle.angleY * f.breathScale)
  set("ParamAngleZ", f.idle.angleZ * f.breathScale)
  set("ParamEyeLOpen", f.idle.eyeOpen)
  set("ParamEyeBallX", f.idle.eyeBallX)
  set("ParamEyeBallY", f.idle.eyeBallY)

  // 4-5. Physics derives hair and cloth from the head params we just wrote,
  //      then the pose is baked into vertices.
  f.physics.update()
  f.model3.update()
}