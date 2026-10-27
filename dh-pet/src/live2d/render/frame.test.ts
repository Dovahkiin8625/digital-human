import { describe, expect, it } from "vitest"
import { applyFrame } from "./frame"

function recorder() {
  const calls: string[] = []
  return {
    calls,
    model: {
      setParameterValueByIndex: (_i: number, _v: number) => {
        calls.push("setParam")
      },
      getParameterValueByIndex: () => 0,
      getParameterIndex: () => 0,
      getParameterMinimumValue: () => 0,
      getParameterMaximumValue: () => 1,
    },
    expression: {
      setExpression: () => calls.push("setExpression"),
      update: () => calls.push("expression"),
    },
    motion: { update: () => calls.push("motion") },
    physics: { update: () => calls.push("physics") },
    pose: { update: () => calls.push("pose") },
    model3: { update: () => calls.push("model") },
  }
}

/** A model that records what was written, and what was there before. */
function trackingModel(
  ranges: Record<string, [number, number]>,
  current: Record<string, number> = {},
) {
  const names = Object.keys(ranges)
  const written: Array<[string, number]> = []
  return {
    written,
    model: {
      setParameterValueByIndex: (i: number, v: number) => {
        written.push([names[i], v])
      },
      getParameterValueByIndex: (i: number) => current[names[i]] ?? 0,
      getParameterIndex: (n: string) => names.indexOf(n),
      getParameterMinimumValue: (i: number) => ranges[names[i]][0],
      getParameterMaximumValue: (i: number) => ranges[names[i]][1],
    },
  }
}

const IDLE = { angleX: 0, angleY: 0, angleZ: 0, eyeOpen: 1, eyeBallX: 0, eyeBallY: 0 }

describe("applyFrame", () => {
  // The single most common Cubism integration bug: writing parameters before
  // the expression and motion managers run lets them overwrite the mouth.
  it("writes our params after Cubism's managers so we win the conflict", () => {
    const r = recorder()
    applyFrame({ ...r, expressionIndex: 0, open: 0.5, form: 0.2, idle: IDLE, breathScale: 1 })
    const firstParam = r.calls.indexOf("setParam")
    expect(firstParam).toBeGreaterThan(r.calls.indexOf("expression"))
    expect(firstParam).toBeGreaterThan(r.calls.indexOf("motion"))
  })

  // The expression is a face; the motion is an animation. Laying the face
  // over the animation is the SDK's own order, and the only one where an
  // expression that shares a parameter with a motion survives.
  it("runs the motion before the expression", () => {
    const r = recorder()
    applyFrame({ ...r, expressionIndex: 0, open: 0, form: 0, idle: IDLE, breathScale: 1 })
    expect(r.calls.indexOf("motion")).toBeLessThan(r.calls.indexOf("expression"))
  })

  // Physics derives hair and cloth from the head params we just wrote, so
  // it has to run before the pose, and the pose before the bake -- or the
  // pose only reaches the vertices a frame late.
  it("runs physics, then the pose, then the bake", () => {
    const r = recorder()
    applyFrame({ ...r, expressionIndex: 0, open: 0, form: 0, idle: IDLE, breathScale: 1 })
    expect(r.calls.indexOf("physics")).toBeLessThan(r.calls.indexOf("pose"))
    expect(r.calls.indexOf("pose")).toBeLessThan(r.calls.indexOf("model"))
  })

  it("pushes the expression index down to the manager", () => {
    const r = recorder()
    applyFrame({ ...r, expressionIndex: 7, open: 0, form: 0, idle: IDLE, breathScale: 1 })
    expect(r.calls.indexOf("setExpression")).toBeGreaterThanOrEqual(0)
    expect(r.calls.indexOf("setExpression")).toBeLessThan(r.calls.indexOf("expression"))
  })

  // Review Focus #2: a rig with no ParamMouthOpenY must render a still
  // mouth and keep everything else alive -- not throw and vanish. The fake
  // deliberately still HAS the idle parameters, because "no mouth" and "no
  // parameters at all" are different failure modes and only the first is
  // realistic.
  it("still animates when the model has no mouth parameter", () => {
    const calls: string[] = []
    const names = ["ParamAngleX", "ParamAngleY", "ParamAngleZ", "ParamEyeLOpen"]
    const r = {
      calls,
      model: {
        setParameterValueByIndex: (i: number, _v: number) => {
          calls.push(`set:${names[i]}`)
        },
        getParameterValueByIndex: () => 0,
        getParameterIndex: (n: string) => names.indexOf(n),
        getParameterMinimumValue: () => -30,
        getParameterMaximumValue: () => 30,
      },
      expression: { setExpression: () => {}, update: () => {} },
      motion: { update: () => {} },
      physics: { update: () => {} },
      pose: { update: () => {} },
      model3: { update: () => calls.push("bake") },
    }
    applyFrame({ ...r, expressionIndex: 0, open: 1, form: 1, idle: IDLE, breathScale: 1 })
    expect(calls).toContain("set:ParamAngleZ")
    expect(calls).toContain("set:ParamEyeLOpen")
    expect(calls).not.toContain("set:ParamMouthOpenY")
    expect(calls).toContain("bake")
  })

  // The idle layer emits normalized shapes; every one has to land inside
  // the MODEL's range. Writing a raw +-2.5 into a -30..30 head parameter
  // reaches only ~8% of the model's travel and the pet reads as frozen --
  // while writing the FULL range would swing the head through 30 degrees
  // every breath. The offset is a FRACTION of the range.
  it("writes each driven parameter as a fraction of the model's own range", () => {
    const ranges: Record<string, [number, number]> = {
      ParamMouthOpenY: [0, 1],
      ParamMouthForm: [-3, 1],
      ParamAngleX: [-30, 30],
      ParamAngleY: [-30, 30],
      ParamAngleZ: [-30, 30],
      ParamEyeLOpen: [0, 1.2],
      ParamEyeROpen: [0, 1.2],
      ParamEyeBallX: [-1, 1],
      ParamEyeBallY: [-1, 1],
    }
    const { model, written } = trackingModel(ranges)
    applyFrame({
      model,
      expression: { setExpression: () => {}, update: () => {} },
      motion: { update: () => {} },
      physics: { update: () => {} },
      pose: { update: () => {} },
      model3: { update: () => {} },
      expressionIndex: 0,
      open: 0.5,
      form: 1,
      idle: { angleX: 1, angleY: -1, angleZ: 1, eyeOpen: 0.5, eyeBallX: 1, eyeBallY: -1 },
      breathScale: 1,
    })
    expect(written.length).toBeGreaterThanOrEqual(8)
    for (const [name, value] of written) {
      const [lo, hi] = ranges[name]
      expect(value, `${name} out of range`).toBeGreaterThanOrEqual(lo)
      expect(value, `${name} out of range`).toBeLessThanOrEqual(hi)
    }
    // The head moves -- but a fraction of the travel, not all of it.
    const angleZ = written.find(([n]) => n === "ParamAngleZ")?.[1]
    expect(angleZ).toBeGreaterThan(4)
    expect(angleZ).toBeLessThan(6)
  })

  // The head sway is an OFFSET on top of what Cubism already wrote, so the
  // model's own idle animation stays visible instead of being replaced.
  it("layers the head sway on top of the model's own animation", () => {
    const ranges: Record<string, [number, number]> = {
      ParamAngleZ: [-30, 30],
      ParamEyeLOpen: [0, 1.2],
    }
    const { model, written } = trackingModel(ranges, { ParamAngleZ: 12 })
    applyFrame({
      model,
      expression: { setExpression: () => {}, update: () => {} },
      motion: { update: () => {} },
      physics: { update: () => {} },
      pose: { update: () => {} },
      model3: { update: () => {} },
      expressionIndex: 0,
      open: 0,
      form: 0,
      idle: { angleX: 0, angleY: 0, angleZ: 1, eyeOpen: 1, eyeBallX: 0, eyeBallY: 0 },
      breathScale: 1,
    })
    const angleZ = written.find(([n]) => n === "ParamAngleZ")?.[1]
    // 12 (the animation) plus the sway's fraction of the range.
    expect(angleZ).toBeGreaterThan(12)
    expect(angleZ).toBeLessThan(20)
  })

  // An expression that narrows the eyes must still be able to blink, so
  // the eye openness is scaled rather than replaced.
  it("scales the eye openness instead of replacing it", () => {
    const ranges: Record<string, [number, number]> = {
      ParamEyeLOpen: [0, 1.2],
      ParamEyeROpen: [0, 1.2],
    }
    const { model, written } = trackingModel(ranges, {
      ParamEyeLOpen: 0.6,
      ParamEyeROpen: 0.6,
    })
    applyFrame({
      model,
      expression: { setExpression: () => {}, update: () => {} },
      motion: { update: () => {} },
      physics: { update: () => {} },
      pose: { update: () => {} },
      model3: { update: () => {} },
      expressionIndex: 0,
      open: 0,
      form: 0,
      idle: { angleX: 0, angleY: 0, angleZ: 0, eyeOpen: 0.5, eyeBallX: 0, eyeBallY: 0 },
      breathScale: 1,
    })
    const eyes = written.filter(([n]) => n === "ParamEyeLOpen" || n === "ParamEyeROpen")
    // Both eyes, or the pet winks.
    expect(eyes).toHaveLength(2)
    for (const [, value] of eyes) expect(value).toBeCloseTo(0.3, 6)
  })

  // The mouth is OURS. It is written outright, so an expression that
  // carries mouth parameters cannot take it.
  it("writes the mouth outright", () => {
    const ranges: Record<string, [number, number]> = {
      ParamMouthOpenY: [0, 1],
      ParamMouthForm: [-3, 1],
    }
    const { model, written } = trackingModel(ranges, { ParamMouthOpenY: 0.9 })
    applyFrame({
      model,
      expression: { setExpression: () => {}, update: () => {} },
      motion: { update: () => {} },
      physics: { update: () => {} },
      pose: { update: () => {} },
      model3: { update: () => {} },
      expressionIndex: 0,
      open: 0.25,
      form: 0.5,
      idle: IDLE,
      breathScale: 1,
    })
    const mouth = written.find(([n]) => n === "ParamMouthOpenY")?.[1]
    expect(mouth).toBeCloseTo(0.25, 6)
  })
})
