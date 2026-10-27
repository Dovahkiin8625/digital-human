import { describe, expect, it } from "vitest"
import { gazePose, gazeTarget, smoothGaze } from "./gaze"
import type { IdlePose } from "./idle"

const IDLE: IdlePose = {
  angleX: 0,
  angleY: 0,
  angleZ: 0,
  eyeOpen: 1,
  eyeBallX: 0.1,
  eyeBallY: 0.05,
}

describe("gazeTarget", () => {
  const origin = { x: 100, y: 100 }

  it("looks to the right when the cursor is to the right", () => {
    const t = gazeTarget({ x: 200, y: 100 }, origin, 200)
    expect(t.x).toBeGreaterThan(0)
    expect(t.y).toBeCloseTo(0, 6)
  })

  it("looks to the left when the cursor is to the left", () => {
    const t = gazeTarget({ x: 0, y: 100 }, origin, 200)
    expect(t.x).toBeLessThan(0)
  })

  // Screen Y grows downward; the eye-ball parameter grows upward. Without
  // the flip the pet looks at the floor when the cursor is above it.
  it("looks up when the cursor is above it", () => {
    const t = gazeTarget({ x: 100, y: 0 }, origin, 200)
    expect(t.y).toBeGreaterThan(0)
  })

  it("looks down when the cursor is below it", () => {
    const t = gazeTarget({ x: 100, y: 200 }, origin, 200)
    expect(t.y).toBeLessThan(0)
  })

  // The distance decides how far the eyes travel, not the direction
  // alone: a cursor just beside the pet gets a glance, one across the
  // room gets the full sweep.
  it("travels further the further away the cursor is", () => {
    const near = gazeTarget({ x: 120, y: 100 }, origin, 200)
    const far = gazeTarget({ x: 300, y: 100 }, origin, 200)
    expect(Math.abs(far.x)).toBeGreaterThan(Math.abs(near.x))
  })

  it("never leaves the model's range", () => {
    const t = gazeTarget({ x: 99999, y: -99999 }, origin, 200)
    expect(t.x).toBeLessThanOrEqual(1)
    expect(t.y).toBeLessThanOrEqual(1)
    expect(t.x).toBeGreaterThanOrEqual(-1)
    expect(t.y).toBeGreaterThanOrEqual(-1)
  })
})

describe("smoothGaze", () => {
  it("does not snap to the target", () => {
    const out = smoothGaze({ x: 0, y: 0 }, { x: 1, y: 0 }, 1 / 60)
    expect(out.x).toBeGreaterThan(0)
    expect(out.x).toBeLessThan(1)
  })

  it("converges on the target", () => {
    let g = { x: 0, y: 0 }
    for (let i = 0; i < 200; i++) g = smoothGaze(g, { x: 1, y: -1 }, 1 / 60)
    expect(g.x).toBeCloseTo(1, 3)
    expect(g.y).toBeCloseTo(-1, 3)
  })

  // Frame-rate independent: a slow frame and two fast ones must land in
  // the same place, or the eyes move faster on a fast machine.
  it("moves the same distance per unit time regardless of frame rate", () => {
    const slow = smoothGaze({ x: 0, y: 0 }, { x: 1, y: 0 }, 0.1)
    let fast = { x: 0, y: 0 }
    for (let i = 0; i < 10; i++) fast = smoothGaze(fast, { x: 1, y: 0 }, 0.01)
    expect(fast.x).toBeCloseTo(slow.x, 6)
  })
})

describe("gazePose", () => {
  // With nothing tracked the eyes keep their own saccades.
  it("keeps the idle saccades when nothing is tracked", () => {
    const { pose, smoothed } = gazePose(IDLE, null, null, 1 / 60)
    expect(pose.eyeBallX).toBe(IDLE.eyeBallX)
    expect(pose.eyeBallY).toBe(IDLE.eyeBallY)
    expect(smoothed).toBeNull()
  })

  it("moves the eyes toward the target", () => {
    const { pose } = gazePose(IDLE, { x: 1, y: 0 }, null, 1 / 60)
    expect(pose.eyeBallX).toBeGreaterThan(IDLE.eyeBallX)
    expect(pose.eyeBallY).not.toBe(IDLE.eyeBallY)
  })

  // The rest of the idle layer is untouched by the gaze: the pet still
  // breathes and blinks while it looks at you.
  it("leaves the rest of the idle pose alone", () => {
    const { pose } = gazePose(IDLE, { x: 1, y: 1 }, null, 1 / 60)
    expect(pose.eyeOpen).toBe(IDLE.eyeOpen)
    expect(pose.angleZ).toBe(IDLE.angleZ)
  })

  // A cursor that appears must not snap the eyes, and a cursor that
  // vanishes must not either.
  it("does not snap when the target appears", () => {
    const { pose } = gazePose(IDLE, { x: 1, y: 0 }, null, 1 / 60)
    expect(pose.eyeBallX).toBeLessThan(1)
  })

  it("keeps the smoothed state for the next frame", () => {
    const first = gazePose(IDLE, { x: 1, y: 0 }, null, 1 / 60)
    const second = gazePose(IDLE, { x: 1, y: 0 }, first.smoothed, 1 / 60)
    expect(second.smoothed!.x).toBeGreaterThan(first.smoothed!.x)
  })
})
