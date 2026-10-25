import { describe, expect, it } from "vitest"
import { driveLabel, parseExpressions } from "./describe"

function avatar(
  config: Record<string, unknown>,
  drive_model = "musetalk",
  client_rendered = false,
) {
  return { drive_model, client_rendered, config } as never
}

describe("driveLabel", () => {
  // The one cell that differs between the drive models, so it is the one
  // worth testing: a wrong label here mislabels every row in the table.
  it("names the client runtime for a client-rendered model", () => {
    expect(driveLabel(avatar({}, "live2d", true))).toBe("客户端渲染")
  })

  it("names the drive mode the pipeline recorded", () => {
    expect(driveLabel(avatar({ musetalk: { mode: "video" } }))).toBe("视频驱动")
    expect(driveLabel(avatar({ musetalk: { mode: "image" } }))).toBe("图片驱动")
  })

  // The pack's config is keyed by the model name, so a model that writes
  // its own block is read from that block.
  it("reads the mode from the avatar's own model block", () => {
    expect(driveLabel(avatar({ other: { mode: "video" } }, "other"))).toBe("视频驱动")
    expect(driveLabel(avatar({ musetalk: { mode: "video" } }, "other"))).toBeNull()
  })

  // pack.json is written by the build; before it lands there is genuinely
  // no mode to report, and an em dash is the honest cell.
  it("returns null before the pack exists", () => {
    expect(driveLabel(avatar({}))).toBeNull()
    expect(driveLabel(avatar({ musetalk: {} }))).toBeNull()
  })

  it("survives a client-rendered row with no config block at all", () => {
    expect(driveLabel({ drive_model: "live2d", client_rendered: true } as never)).toBe(
      "客户端渲染",
    )
  })
})

describe("parseExpressions", () => {
  it("accepts an empty field as an empty map", () => {
    expect(parseExpressions("")).toEqual({ ok: true, value: {} })
    expect(parseExpressions("   ")).toEqual({ ok: true, value: {} })
  })

  it("accepts a well-formed map", () => {
    expect(parseExpressions('{"smile": 0, "think": 2}')).toEqual({
      ok: true,
      value: { smile: 0, think: 2 },
    })
  })

  // Caught here rather than server-side: a JSON syntax error is the
  // operator's typo, and the server's 422 would say only that the map is
  // wrong, not that the braces are unbalanced.
  it("names the syntax error", () => {
    expect(parseExpressions("{not json")).toEqual({
      ok: false,
      error: "表情映射不是合法 JSON",
    })
  })

  it("rejects a non-object", () => {
    expect(parseExpressions("[0, 1]")).toEqual({
      ok: false,
      error: '表情映射必须是形如 {"smile": 0} 的对象',
    })
    expect(parseExpressions('"smile"').ok).toBe(false)
  })

  // A non-integer here becomes an expressionIndex the Cubism stage cannot
  // use, and the failure surfaces as a blank face rather than an error.
  it("rejects a value that is not a non-negative integer", () => {
    expect(parseExpressions('{"smile": -1}').ok).toBe(false)
    expect(parseExpressions('{"smile": 1.5}').ok).toBe(false)
    expect(parseExpressions('{"smile": "0"}').ok).toBe(false)
  })
})
