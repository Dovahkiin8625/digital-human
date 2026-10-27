import { describe, expect, it, vi } from "vitest";
import { mountStage, type StageLike } from "./mountStage";
import type { ParameterReport } from "./cubism/types";

const PARAMS: ParameterReport[] = [{ id: "ParamMouthOpenY", min: 0, max: 1, default: 0 }];

/** Minimal stand-in. The real CubismStage needs WebGL and a canvas, but the
 *  lifecycle this module owns is all that is under test here. */
function fakeStage(overrides: Partial<StageLike> = {}) {
  const calls: string[] = [];
  let failLoop: ((e: unknown) => void) | null = null;
  const stage: StageLike = {
    init: async (modelUrl) => {
      calls.push(`init:${modelUrl}`)
      return PARAMS
    },
    start: (onError) => {
      calls.push("start")
      failLoop = onError
    },
    stop: () => {
      calls.push("stop")
    },
    speak: () => {},
    ...overrides,
  }
  return {
    stage,
    calls,
    fireLoopError: (e: unknown) => failLoop?.(e),
  }
}

describe("mountStage", () => {
  it("reports the parameters once the model is loaded", async () => {
    const f = fakeStage()
    const onReady = vi.fn()
    mountStage(f.stage, "/m/model3.json", { onReady })
    await vi.waitFor(() => expect(onReady).toHaveBeenCalledWith(PARAMS))
    expect(f.calls).toContain("start")
  })

  // Review Focus: a broken model must surface through onError, never as an
  // unhandled rejection that leaves a blank stage with no explanation.
  it("routes an init failure to onError instead of throwing", async () => {
    const f = fakeStage({ init: async () => Promise.reject(new Error("bad moc")) })
    const onError = vi.fn()
    mountStage(f.stage, "/m", { onError })
    await vi.waitFor(() => expect(onError).toHaveBeenCalled())
    expect(f.calls).not.toContain("start")
  })

  // Review Focus: a leaked requestAnimationFrame keeps the previous model
  // holding the GPU after the user switches away.
  it("stops the render loop when the consumer unmounts", async () => {
    const f = fakeStage()
    const handle = mountStage(f.stage, "/m", {})
    await vi.waitFor(() => expect(f.calls).toContain("start"))
    handle.stop()
    expect(f.calls).toContain("stop")
  })

  it("is safe to stop twice", async () => {
    const f = fakeStage()
    const handle = mountStage(f.stage, "/m", {})
    await vi.waitFor(() => expect(f.calls).toContain("start"))
    handle.stop()
    handle.stop()
    expect(f.calls.filter((c) => c === "stop")).toHaveLength(1)
  })

  it("stops the loop when the render loop itself throws", async () => {
    const f = fakeStage()
    const onError = vi.fn()
    mountStage(f.stage, "/m", { onError })
    await vi.waitFor(() => expect(f.calls).toContain("start"))
    f.fireLoopError(new Error("WebGL context lost"))
    expect(onError).toHaveBeenCalled()
    expect(f.calls).toContain("stop")
  })

  // Unmounting before the model finishes loading must not resurrect the
  // stage -- init resolves later and would otherwise start a dead loop.
  it("does not start the loop if stopped before init resolved", async () => {
    let resolveInit: ((p: ParameterReport[]) => void) | null = null
    const f = fakeStage({
      init: () => new Promise<ParameterReport[]>((r) => (resolveInit = r)),
    })
    const handle = mountStage(f.stage, "/m", {})
    handle.stop()
    resolveInit?.(PARAMS)
    await new Promise((r) => setTimeout(r, 10))
    expect(f.calls).not.toContain("start")
  })
})