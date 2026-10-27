import { describe, expect, it } from "vitest"
import {
  EMOTION_TAGS,
  expressionIndex,
  neutralIndex,
  NO_EXPRESSION,
  strippableTags,
} from "./expressions"

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
  it("falls back for an unknown tag", () => {
    expect(expressionIndex(["wink"], { smile: 3 }, 0)).toBe(0)
  })

  it("falls back for no tag", () => {
    expect(expressionIndex([], { smile: 3 }, 0)).toBe(0)
  })

  // The default is NO expression, not expression 0. On a real model the
  // first expression is whatever the author happened to put first --
  // Natori's is Angry -- so a reply with no emotion in it must not land
  // on an angry face.
  it("applies no expression by default", () => {
    expect(expressionIndex([], { smile: 3 })).toBe(NO_EXPRESSION)
  })
})

describe("neutralIndex", () => {
  it("uses the operator's own neutral", () => {
    expect(neutralIndex({ neutral: 2, smile: 4 })).toBe(2)
  })

  it("is no expression when the operator mapped none", () => {
    expect(neutralIndex({ smile: 4 })).toBe(NO_EXPRESSION)
  })
})

describe("strippableTags", () => {
  // The vocabulary is always stripped, so a tag the model does not map is
  // still removed instead of being spoken aloud as a word.
  it("includes the fixed vocabulary whatever the model maps", () => {
    expect(strippableTags({ smile: 0 })).toEqual(
      expect.arrayContaining([...EMOTION_TAGS]),
    )
  })

  it("includes the model's own tags", () => {
    expect(strippableTags({ wink: 0 })).toContain("wink")
  })

  it("does not repeat a tag the model also declares", () => {
    const tags = strippableTags({ smile: 0 })
    expect(tags.filter((t) => t === "smile")).toHaveLength(1)
  })
})
