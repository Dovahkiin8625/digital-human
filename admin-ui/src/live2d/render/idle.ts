/**
 * The idle "life" layer: breathing, blinking, gaze drift, head sway.
 *
 * Everything is a pure function of (time, seed). No PRNG state is carried
 * through the render loop -- each timing decision is hashed from
 * (seed, bucket). That costs nothing at runtime and buys two things: the
 * layer is exactly replayable in a test, and it survives a dropped frame
 * or a backgrounded window without the pose jumping.
 *
 * Phase is likewise a function of t alone and is never reset per utterance.
 * Restarting the breath phase at every sentence boundary puts a visible
 * discontinuity at the start of each one.
 */

export interface IdlePose {
  angleX: number
  angleY: number
  angleZ: number
  eyeOpen: number
  eyeBallX: number
  eyeBallY: number
}

export interface IdleOptions {
  /** 0..1 multiplier on breathing and sway amplitude. Lowered while speaking
   *  so the motion stops competing with the mouth. */
  breathScale?: number
  /** Seeds the blink and saccade schedule. */
  seed?: number
}

const BREATH_MS = 4000
const BLINK_SLOT_MS = 4000
const BLINK_MS = 150
const SACCADE_SLOT_MS = 1200
const EYE_MIN = 0.05

/** Deterministic 0..1 hash. Lets every timing decision derive from
 *  (seed, bucket) instead of from mutable generator state. */
function hash01(seed: number, i: number): number {
  let t = Math.imul(seed ^ i, 0x9e3779b1) >>> 0
  t = Math.imul(t ^ (t >>> 16), 0x85ebca6b) >>> 0
  t = Math.imul(t ^ (t >>> 13), 0xc2b2ae35) >>> 0
  return ((t ^ (t >>> 16)) >>> 0) / 4294967296
}

export function idlePose(tMs: number, opts: IdleOptions = {}): IdlePose {
  const { breathScale = 1, seed = 1 } = opts

  // Mutually incommensurable periods, so the composite never visibly
  // repeats. Phase derives from t only.
  const s = (2 * Math.PI * tMs) / BREATH_MS
  const angleZ = Math.sin(s) * 2.5 * breathScale
  const angleX = (Math.sin(s / 1.7 + 0.4) * 1.2 + Math.sin(s / 4.1) * 0.6) * breathScale
  const angleY = (Math.sin(s / 2.3 + 1.1) * 1.0 + Math.sin(s / 3.3) * 0.5) * breathScale

  // One candidate blink per slot, taken with probability 0.45, positioned
  // inside the slot by hash. Eyes are open the rest of the time.
  const blinkSlot = Math.floor(tMs / BLINK_SLOT_MS)
  const inSlot = tMs - blinkSlot * BLINK_SLOT_MS
  let eyeOpen = 1
  if (hash01(seed, blinkSlot) < 0.45) {
    const at = hash01(seed, blinkSlot + 1) * (BLINK_SLOT_MS - BLINK_MS * 3)
    const d = inSlot - at
    if (d >= 0 && d < BLINK_MS) {
      eyeOpen = EYE_MIN + (1 - EYE_MIN) * (1 - Math.sin((Math.PI * d) / BLINK_MS))
    }
    // Occasional double blink, on its own hash so it is not universal.
    if (hash01(seed, blinkSlot + 2) < 0.25) {
      const d2 = inSlot - at - 380
      if (d2 >= 0 && d2 < BLINK_MS) {
        eyeOpen = Math.min(
          eyeOpen,
          EYE_MIN + (1 - EYE_MIN) * (1 - Math.sin((Math.PI * d2) / BLINK_MS)),
        )
      }
    }
  }

  // Gaze drifts to a new target per saccade slot. The render loop
  // crossfades between slots; here we only emit the target.
  const sacSlot = Math.floor(tMs / SACCADE_SLOT_MS)
  const eyeBallX = (hash01(seed, sacSlot + 1000) - 0.5) * 0.6
  const eyeBallY = (hash01(seed, sacSlot + 2000) - 0.5) * 0.4

  return { angleX, angleY, angleZ, eyeOpen, eyeBallX, eyeBallY }
}