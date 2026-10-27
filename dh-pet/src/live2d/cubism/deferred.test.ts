import { describe, expect, it } from "vitest"
import { createDeferred } from "./deferred"

describe("createDeferred", () => {
  // The bug this exists for: the React component sets the expression on
  // its first render, and the model takes seconds to load. A value dropped
  // on the floor there is an expression the pet never wears -- no error,
  // no blank canvas, just a face that does not react.
  it("hands the target the value it missed", () => {
    const d = createDeferred(-1)
    d.set(4)
    const seen: number[] = []
    d.bind((v) => seen.push(v))
    expect(seen).toEqual([4])
  })

  it("applies later values to the bound target", () => {
    const d = createDeferred(-1)
    const seen: number[] = []
    d.bind((v) => seen.push(v))
    d.set(4)
    d.set(2)
    expect(seen).toEqual([-1, 4, 2])
  })

  it("keeps the current value", () => {
    const d = createDeferred(-1)
    d.set(7)
    expect(d.get()).toBe(7)
  })

  // A second model (the user switched avatars) rebinds, and must be handed
  // the current value too -- not the one it had when the first model was
  // bound.
  it("hands a rebound target the current value", () => {
    const d = createDeferred(-1)
    const first: number[] = []
    d.bind((v) => first.push(v))
    d.set(4)
    const second: number[] = []
    d.bind((v) => second.push(v))
    expect(second).toEqual([4])
  })
})
