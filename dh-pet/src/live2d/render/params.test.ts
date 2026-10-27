import { describe, expect, it } from "vitest"
import { clampToRange, mapToRange, offsetInRange, resolveMouthRange } from "./params"

function fakeModel(specs: Record<string, { min: number; max: number }>) {
  const names = Object.keys(specs)
  return {
    getParameterIndex: (n: string) => names.indexOf(n),
    getParameterMinimumValue: (i: number) => specs[names[i]].min,
    getParameterMaximumValue: (i: number) => specs[names[i]].max,
  }
}

describe("resolveMouthRange", () => {
  // These are real ranges seen across shipped models. Assuming 0..1 pins
  // the mouth shut or inverts it on most of them.
  it.each([
    [0, 1],
    [-1, 1],
    [0, 30],
    [-30, 30],
  ])("reads a %i..%i range off the model", (min, max) => {
    expect(resolveMouthRange(fakeModel({ ParamMouthOpenY: { min, max } }))).toEqual({
      min,
      max,
    })
  })

  // Not every rig spells it ParamMouthOpenY, and some ship no mouth
  // parameter at all. Those must degrade to a still mouth, not throw.
  it("returns null when the model has no mouth parameter", () => {
    expect(resolveMouthRange(fakeModel({ ParamAngleZ: { min: -30, max: 30 } }))).toBeNull()
  })
})

describe("mapToRange", () => {
  it("maps 0 and 1 onto the range endpoints", () => {
    expect(mapToRange(0, { min: 0, max: 30 })).toBe(0)
    expect(mapToRange(1, { min: 0, max: 30 })).toBe(30)
  })

  it("maps the midpoint to the range midpoint", () => {
    expect(mapToRange(0.5, { min: -1, max: 1 })).toBe(0)
  })

  it("clamps out-of-range input", () => {
    expect(mapToRange(1.4, { min: 0, max: 1 })).toBe(1)
    expect(mapToRange(-0.2, { min: 0, max: 1 })).toBe(0)
  })
})

describe("offsetInRange", () => {
  // An offset is a FRACTION of the range, because rigs disagree wildly on
  // scale: ParamAngleZ is -30..30 on one model and -1..1 on another, and
  // "three degrees of head tilt" is not a portable quantity.
  it("reaches a fraction of the range's half-span", () => {
    expect(offsetInRange(1, { min: -30, max: 30 }, 0.15)).toBeCloseTo(4.5, 6)
    expect(offsetInRange(1, { min: -1, max: 1 }, 0.15)).toBeCloseTo(0.15, 6)
  })

  it("is signed", () => {
    expect(offsetInRange(-1, { min: -30, max: 30 }, 0.15)).toBeCloseTo(-4.5, 6)
  })

  it("clamps out-of-range input", () => {
    expect(offsetInRange(4, { min: -30, max: 30 }, 0.15)).toBeCloseTo(4.5, 6)
    expect(offsetInRange(-4, { min: -30, max: 30 }, 0.15)).toBeCloseTo(-4.5, 6)
  })
})

describe("clampToRange", () => {
  it("keeps a value inside the range", () => {
    expect(clampToRange(5, { min: -1, max: 1 })).toBe(1)
    expect(clampToRange(-5, { min: -1, max: 1 })).toBe(-1)
    expect(clampToRange(0.5, { min: -1, max: 1 })).toBe(0.5)
  })
})
