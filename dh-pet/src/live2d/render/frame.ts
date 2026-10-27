import { clampToRange, mapToRange, offsetInRange, resolveRange } from "./params"
import type { IdlePose } from "./idle"

/**
 * One frame, in the order Cubism requires.
 *
 * The ordering is the whole point of this file. Cubism's motion and
 * expression managers lay down their own parameter values when they
 * update; anything we write before them can be overwritten. So they go
 * first, we write after, and our values win -- which is how the mouth
 * stays ours even when an expression also touches the mouth region.
 *
 * The motion runs before the expression on purpose. The expression is a
 * face; the motion is an animation. Laying the face over the animation
 * is the SDK's own order, and it is the only one where an expression that
 * shares a parameter with a motion survives instead of being wiped.
 */
export interface FrameInputs {
  model: {
    setParameterValueByIndex(i: number, v: number): void
    /** The value currently in the parameter. Our idle layer reads it so
     *  it can add to what Cubism wrote rather than replace it. */
    getParameterValueByIndex(i: number): number
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
  pose: { update(): void }
  model3: { update(): void }
  expressionIndex: number
  open: number
  form: number
  idle: IdlePose
  breathScale: number
}

/**
 * How much of a parameter's own range each driven parameter may use.
 *
 * A fraction, not a number of degrees: rigs disagree wildly on scale --
 * Natori's ParamAngleZ is -30..30, another model's can be -1..1 -- so
 * "three degrees of head tilt" is not a portable quantity. A fraction of
 * the travel is: a tenth of the swing means the same amount of movement
 * on every rig.
 *
 * The eye balls are 1 because their values are already expressed as a
 * fraction of the travel (the saccade picks a spot within it).
 */
const AMPLITUDE = {
  ParamAngleX: 0.12,
  ParamAngleY: 0.1,
  ParamAngleZ: 0.15,
  ParamEyeBallX: 1,
  ParamEyeBallY: 1,
}

export function applyFrame(f: FrameInputs): void {
  // 1-2. Cubism lays down the motion and the expression first.
  f.motion.update()
  f.expression.setExpression(f.expressionIndex)
  f.expression.update()

  // 3. We write afterwards, so we win any parameter an expression also
  //    touches. Writing before the managers run is the single most common
  //    cause of mouth-versus-expression flicker.
  //
  //    The mouth is written ABSOLUTELY: it is ours, and an expression
  //    that carries mouth parameters must not be able to take it.
  const mouthIdx = f.model.getParameterIndex("ParamMouthOpenY")
  if (mouthIdx >= 0) {
    const range = resolveRange(f.model, "ParamMouthOpenY")
    if (range) {
      f.model.setParameterValueByIndex(mouthIdx, mapToRange(f.open, range))
    }
  }

  // The mouth's SHAPE rides along with it. This parameter is signed on
  // most rigs (Natori's is -3..1) and means something different on each
  // one, so it is mapped onto the model's own range rather than written
  // as a raw 0..1 -- a raw 0..1 lands inverted on a signed parameter.
  const formIdx = f.model.getParameterIndex("ParamMouthForm")
  if (formIdx >= 0) {
    const range = resolveRange(f.model, "ParamMouthForm")
    if (range) {
      f.model.setParameterValueByIndex(formIdx, mapToRange(f.form, range))
    }
  }

  // Every remaining parameter is an OFFSET layered on what Cubism
  // already wrote, not a replacement for it. The same rig ships
  // ParamAngleX in -30..30 and ParamMouthOpenY in 0..1, so an absolute
  // constant that looks right for one is wrong for the others: writing a
  // head angle of +-2.5 into a -30..30 parameter reaches only ~8% of the
  // model's travel and the pet reads as frozen -- while writing the same
  // value as a fraction of the range reaches a tenth of it on every rig.
  const offset = (name: string, value: number, amplitude: number) => {
    const i = f.model.getParameterIndex(name)
    if (i < 0) return
    const range = resolveRange(f.model, name)
    if (!range) return
    const current = f.model.getParameterValueByIndex(i)
    const next = current + offsetInRange(value, range, amplitude)
    f.model.setParameterValueByIndex(i, clampToRange(next, range))
  }
  offset("ParamAngleX", f.idle.angleX * f.breathScale, AMPLITUDE.ParamAngleX)
  offset("ParamAngleY", f.idle.angleY * f.breathScale, AMPLITUDE.ParamAngleY)
  offset("ParamAngleZ", f.idle.angleZ * f.breathScale, AMPLITUDE.ParamAngleZ)
  offset("ParamEyeBallX", f.idle.eyeBallX, AMPLITUDE.ParamEyeBallX)
  offset("ParamEyeBallY", f.idle.eyeBallY, AMPLITUDE.ParamEyeBallY)

  // Eye openness is MULTIPLIED rather than written outright. An
  // expression that narrows the eyes must still be able to blink, so the
  // blink rides on top of whatever the expression set instead of
  // replacing it. The model's own default is the baseline: Natori's
  // ParamEyeLOpen defaults to 1, and multiplying keeps it there.
  //
  // Both eyes, or the pet winks. A rig with one eye parameter only
  // (the left) is the common case and the right one is simply absent.
  for (const name of ["ParamEyeLOpen", "ParamEyeROpen"]) {
    const i = f.model.getParameterIndex(name)
    if (i < 0) continue
    const range = resolveRange(f.model, name)
    if (!range) continue
    const current = f.model.getParameterValueByIndex(i)
    f.model.setParameterValueByIndex(i, clampToRange(current * f.idle.eyeOpen, range))
  }

  // 4-6. Physics derives hair and cloth from the head params we just
  //      wrote, then the pose adjusts the parts it owns. Both must run
  //      before the bake, or the pose only reaches the vertices a frame
  //      late.
  f.physics.update()
  f.pose.update()
  f.model3.update()
}
