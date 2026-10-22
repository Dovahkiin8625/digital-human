import { describe, expect, it } from "vitest"
import { expressionIndex } from "./expressions"

describe("expressionIndex", () => {
  it("maps a known tag", () => {
    expect(expressionIndex(["smile"], { smile: 3 })).toBe(3)
  })

  // The last tag wins: it is the one nearest the sentence end, so it is
  // the expression in effect as the line finishes.
  it("uses the last tag when several arrive in one sentence", () => {
    expect(expressionIndex(["smile", "think"], { smile: 3, think: 5 })).toBe(5)
  })

  // A model swapped in at runtime must not be able to blank the pet.
  it("falls back to neutral for an unknown tag", () => {
    expect(expressionIndex(["wink"], { smile: 3 }, 0)).toBe(0)
  })

  it("falls back to neutral for no tag", () => {
    expect(expressionIndex([], { smile: 3 }, 0)).toBe(0)
  })
})