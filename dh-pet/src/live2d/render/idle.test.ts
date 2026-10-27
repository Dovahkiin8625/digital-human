import { describe, expect, it } from "vitest"
import { idlePose } from "./idle"

describe("idlePose", () => {
  // The whole layer is a pure function of time, with no mutable PRNG state
  // carried through the render loop. That is what makes it replayable --
  // a regression in the blink schedule shows up as a failing assertion
  // rather than as a pet that blinks differently on Tuesdays.
  it("is a pure function of time", () => {
    expect(idlePose(12345, { seed: 7 })).toEqual(idlePose(12345, { seed: 7 }))
  })

  it("breathes continuously with no reset", () => {
    const a = idlePose(10000)
    const b = idlePose(10016)
    expect(Math.abs(a.angleZ - b.angleZ)).toBeLessThan(0.1)
  })

  // A sine sampled just before and just after a zero crossing can differ by
  // the full amplitude, so this is not a phase-reset check by itself -- it
  // is a "no discontinuity" check. The phase-continuity requirement proper
  // is that breathScale scales amplitude and never shifts phase, below.
  it("scales breathing by breathScale without shifting phase", () => {
    const t = 900
    const full = idlePose(t, { breathScale: 1 })
    const half = idlePose(t, { breathScale: 0.5 })
    expect(half.angleZ).toBeCloseTo(full.angleZ * 0.5, 6)
  })

  it("keeps the eyes open most of the time", () => {
    let open = 0
    for (let t = 0; t < 60000; t += 40) if (idlePose(t, { seed: 3 }).eyeOpen > 0.9) open++
    expect(open / 1500).toBeGreaterThan(0.8)
  })

  it("closes the eyes sometimes", () => {
    let closed = 0
    for (let t = 0; t < 60000; t += 40) if (idlePose(t, { seed: 3 }).eyeOpen < 0.3) closed++
    expect(closed).toBeGreaterThan(0)
  })

  // Same seed must not produce the same schedule forever -- a fixed blink
  // cadence reads as mechanical in a way randomness hides.
  it("varies blink timing across seeds", () => {
    const firstClosedAt = (seed: number) => {
      for (let t = 0; t < 20000; t += 40) if (idlePose(t, { seed }).eyeOpen < 0.3) return t
      return -1
    }
    expect(firstClosedAt(1)).not.toBe(firstClosedAt(2))
  })

  it("holds eyeBall within a small range", () => {
    for (let t = 0; t < 30000; t += 200) {
      const p = idlePose(t, { seed: 5 })
      expect(Math.abs(p.eyeBallX)).toBeLessThanOrEqual(0.3)
      expect(Math.abs(p.eyeBallY)).toBeLessThanOrEqual(0.2)
    }
  })
})