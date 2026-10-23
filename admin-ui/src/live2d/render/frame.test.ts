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
    model3: { update: () => calls.push("model") },
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

  // Physics derives hair and cloth from the head params we just wrote, so
  // it has to run before the pose is baked into vertices.
  it("runs physics before the pose bake", () => {
    const r = recorder()
    applyFrame({ ...r, expressionIndex: 0, open: 0, form: 0, idle: IDLE, breathScale: 1 })
    expect(r.calls.indexOf("physics")).toBeLessThan(r.calls.indexOf("model"))
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
        getParameterIndex: (n: string) => names.indexOf(n),
        getParameterMinimumValue: () => -30,
        getParameterMaximumValue: () => 30,
      },
      expression: { setExpression: () => {}, update: () => {} },
      motion: { update: () => {} },
      physics: { update: () => {} },
      model3: { update: () => calls.push("bake") },
    }
    applyFrame({ ...r, expressionIndex: 0, open: 1, form: 1, idle: IDLE, breathScale: 1 })
    expect(calls).toContain("set:ParamAngleZ")
    expect(calls).toContain("set:ParamEyeLOpen")
    expect(calls).not.toContain("set:ParamMouthOpenY")
    expect(calls).toContain("bake")
  })

  // The idle layer emits signed normalized values; every one has to land
  // inside the MODEL's range. Writing a raw +-2.5 into a -30..30 head
  // parameter reaches ~8% of the travel and the pet reads as frozen.
  it("maps every driven parameter into the model's own range", () => {
    const written: Array<[string, number]> = []
    const ranges: Record<string, [number, number]> = {
      ParamMouthOpenY: [0, 1],
      ParamMouthForm: [-2, 1],
      ParamAngleX: [-30, 30],
      ParamAngleY: [-30, 30],
      ParamAngleZ: [-30, 30],
      ParamEyeLOpen: [0, 1.2],
      ParamEyeBallX: [-1, 1],
      ParamEyeBallY: [-1, 1],
    }
    const names = Object.keys(ranges)
    const r = {
      calls: [] as string[],
      model: {
        setParameterValueByIndex: (i: number, v: number) => {
          written.push([names[i], v])
        },
        getParameterIndex: (n: string) => names.indexOf(n),
        getParameterMinimumValue: (i: number) => ranges[names[i]][0],
        getParameterMaximumValue: (i: number) => ranges[names[i]][1],
      },
      expression: { setExpression: () => {}, update: () => {} },
      motion: { update: () => {} },
      physics: { update: () => {} },
      model3: { update: () => {} },
    }
    applyFrame({
      ...r,
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
    // And it actually USES the range rather than clustering at the default.
    const angleZ = written.find(([n]) => n === "ParamAngleZ")?.[1]
    expect(angleZ).toBeGreaterThan(25)
  })
})
