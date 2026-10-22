/**
 * Turn a decoded audio buffer into per-frame mouth parameters.
 *
 * This runs ONCE per utterance, offline, rather than sampling a live
 * AnalyserNode every frame. Three reasons that matters:
 *
 *   1. Determinism -- the same clip yields the same curve in Electron and in
 *      a browser, and the mouth stays correct while the audio graph is
 *      paused, throttled, or occluded.
 *   2. Normalization -- MiniMax's level varies a lot between sentences.
 *      Per-clip peak normalization keeps a quiet sentence from having a
 *      static mouth.
 *   3. Smoothing quality -- the asymmetric one-pole below is better applied
 *      across the whole clip than approximated frame by frame.
 *
 * The output is indexed by wall-clock time at render time, against the same
 * AudioBuffer feeding the speakers, so mouth and voice cannot drift.
 */

/** Structural subset of AudioBuffer. Taking this shape instead of the DOM
 *  type keeps the math testable under vitest's node environment. */
export interface PcmSource {
  readonly numberOfChannels: number
  readonly sampleRate: number
  readonly length: number
  getChannelData(channel: number): Float32Array
}

export interface MouthAnalysis {
  /** Mouth openness per frame, 0..1. Length = ceil(duration * pointsPerSecond). */
  open: Float32Array
  /** Mouth form per frame, 0..1. 0 = spread, 1 = rounded. */
  form: Float32Array
}

export interface AnalyzeOptions {
  pointsPerSecond?: number
}

const DEFAULT_POINTS_PER_SECOND = 60

/** Below this the clip is silence, not a quiet utterance. */
const SILENCE_RMS = 0.005

/** Asymmetric one-pole coefficients: open fast, close slow. */
const ATTACK = 0.6
const RELEASE = 0.18

/** Below this mouth openness, zero crossings are noise rather than a shape. */
const FORM_FLOOR = 0.1

export function analyzeSpeech(
  src: PcmSource,
  { pointsPerSecond = DEFAULT_POINTS_PER_SECOND }: AnalyzeOptions = {},
): MouthAnalysis {
  const frames = Math.max(1, Math.ceil((src.length / src.sampleRate) * pointsPerSecond))
  const perFrame = src.length / frames

  const channels: Float32Array[] = []
  for (let c = 0; c < src.numberOfChannels; c++) channels.push(src.getChannelData(c))

  const rms = new Float32Array(frames)
  const zcr = new Float32Array(frames)
  for (let f = 0; f < frames; f++) {
    const start = Math.floor(f * perFrame)
    const end = Math.min(src.length, Math.floor((f + 1) * perFrame))
    let sumSq = 0
    let crossings = 0
    let prev = 0
    for (const data of channels) {
      for (let i = start; i < end; i++) {
        const s = data[i]
        sumSq += s * s
        if (i > start && s >= 0 !== prev >= 0) crossings++
        prev = s
      }
    }
    const count = (end - start) * channels.length
    rms[f] = count > 0 ? Math.sqrt(sumSq / count) : 0
    zcr[f] = count > 0 ? crossings / count : 0
  }

  // Gate on absolute level BEFORE normalizing. Peak-normalizing a clip that
  // is only noise would stretch that noise across the full 0..1 range and
  // leave the mouth permanently half open.
  let peak = 0
  for (const v of rms) if (v > peak) peak = v
  const gain = peak > SILENCE_RMS ? 1 / peak : 0

  const open = new Float32Array(frames)
  let env = 0
  for (let f = 0; f < frames; f++) {
    const target = Math.min(1, rms[f] * gain)
    env += (target - env) * (target > env ? ATTACK : RELEASE)
    open[f] = env
  }

  let zPeak = 0
  for (const v of zcr) if (v > zPeak) zPeak = v
  const zGain = zPeak > 0 ? 1 / zPeak : 0
  const form = new Float32Array(frames)
  for (let f = 0; f < frames; f++) {
    form[f] = open[f] > FORM_FLOOR ? Math.min(1, zcr[f] * zGain) : 0
  }

  return { open, form }
}