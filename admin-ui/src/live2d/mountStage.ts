import type { ParameterReport } from "./cubism/types";

/** The part of CubismStage this module drives. CubismStage satisfies it. */
export interface StageLike {
  init(modelUrl: string): Promise<ParameterReport[]>
  start(onError: (e: unknown) => void): void
  stop(): void
  speak(buffer: AudioBuffer): void
}

export interface MountHandlers {
  onReady?: (params: ParameterReport[]) => void
  onError?: (e: unknown) => void
}

export interface MountHandle {
  /** Stop the render loop and release the stage. Safe to call twice. */
  stop(): void
}

/**
 * Own a stage's load-then-render lifecycle.
 *
 * Split out of the React component so it can be tested without a DOM: the
 * component is a thin shell over this, and the behaviour worth pinning --
 * errors routed to a handler instead of escaping, and the loop stopped on
 * unmount -- is all here.
 *
 * The `stopped` flag matters more than it looks. Model loading is async, so
 * unmounting while init is in flight is the normal case when the user
 * switches avatars quickly; without the guard, init resolves afterwards and
 * starts a render loop for a component that no longer exists.
 */
export function mountStage(
  stage: StageLike,
  modelUrl: string,
  handlers: MountHandlers = {},
): MountHandle {
  let stopped = false

  const stop = () => {
    if (stopped) return
    stopped = true
    stage.stop()
  }

  stage
    .init(modelUrl)
    .then((params) => {
      if (stopped) return
      handlers.onReady?.(params)
      stage.start((e) => {
        handlers.onError?.(e)
        stop()
      })
    })
    .catch((e: unknown) => {
      if (stopped) return
      handlers.onError?.(e)
    })

  return { stop }
}