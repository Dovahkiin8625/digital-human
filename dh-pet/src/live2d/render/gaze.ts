/**
 * Where the pet looks.
 *
 * The cursor arrives in screen coordinates, reported by the Electron
 * shell. It has to be polled rather than listened for: the pet's window
 * ignores mouse input most of the time, and a window that ignores input
 * receives no move events at all -- so the pet could never follow a
 * cursor that was somewhere else on the desktop.
 */

import type { IdlePose } from "./idle"

export interface GazeTarget {
  x: number
  y: number
}

export interface Point {
  x: number
  y: number
}

function clamp(value: number, low: number, high: number): number {
  return Math.min(high, Math.max(low, value))
}

/**
 * The direction from the pet to the cursor, as a look-at target.
 *
 * The DISTANCE decides how far the eyes travel, not the direction alone:
 * a cursor just beside the pet gets a glance, one across the room gets
 * the full sweep. `span` is the distance at which the eyes reach the end
 * of their travel.
 */
export function gazeTarget(cursor: Point, origin: Point, span: number): GazeTarget {
  const s = span > 0 ? span : 1
  const dx = (cursor.x - origin.x) / s
  // Screen Y grows downward; the eye-ball parameter grows upward.
  const dy = -(cursor.y - origin.y) / s
  return { x: clamp(dx, -1, 1), y: clamp(dy, -1, 1) }
}

/**
 * Frame-rate independent exponential smoothing.
 *
 * A pet whose eyes snap to the cursor looks like a webcam; a pet whose
 * eyes drift over a tenth of a second looks like it is looking.
 */
export function smoothGaze(
  current: GazeTarget,
  target: GazeTarget,
  dtSeconds: number,
  tau = 0.12,
): GazeTarget {
  const k = 1 - Math.exp(-Math.max(0, dtSeconds) / Math.max(0.001, tau))
  return {
    x: current.x + (target.x - current.x) * k,
    y: current.y + (target.y - current.y) * k,
  }
}

/**
 * The idle pose with the gaze folded in.
 *
 * When something is being tracked the eyes go where it says and their own
 * saccades stop; otherwise they keep the saccades. The switch is
 * smoothed, so a cursor that appears or vanishes does not snap the eyes.
 *
 * Returns the smoothed gaze alongside, because it is the state the next
 * frame starts from.
 */
export function gazePose(
  base: IdlePose,
  target: GazeTarget | null,
  current: GazeTarget | null,
  dtSeconds: number,
): { pose: IdlePose; smoothed: GazeTarget | null } {
  if (!target) return { pose: base, smoothed: null }
  // Start from wherever the eyes already are, so the first tracked frame
  // does not jump.
  const from = current ?? { x: base.eyeBallX, y: base.eyeBallY }
  const smoothed = smoothGaze(from, target, dtSeconds)
  return {
    pose: { ...base, eyeBallX: smoothed.x, eyeBallY: smoothed.y },
    smoothed,
  }
}
