import { describe, expect, it } from "vitest"
import { analyzeSpeech, type PcmSource } from "./analyze"

// A structural stand-in for AudioBuffer. Taking a plain shape rather than the
// DOM type is what lets these tests run under vitest's default node
// environment with no canvas and no jsdom.
function pcm(channels: Float32Array[], sampleRate = 48000): PcmSource {
  return {
    numberOfChannels: channels.length,
    sampleRate,
    length: channels[0].length,
    getChannelData: (c: number) => channels[c],
  }
}

function tone(seconds: number, sampleRate = 48000, amp = 0.5, hz = 220): Float32Array {
  const n = Math.round(seconds * sampleRate)
  const out = new Float32Array(n)
  for (let i = 0; i < n; i++) out[i] = amp * Math.sin((2 * Math.PI * hz * i) / sampleRate)
  return out
}

describe("analyzeSpeech", () => {
  it("returns all zeros for silence", () => {
    const r = analyzeSpeech(pcm([tone(0.5, 48000, 0)]))
    expect(r.open.every((v) => v === 0)).toBe(true)
    expect(r.form.every((v) => v === 0)).toBe(true)
  })

  // A quiet-but-nonzero clip must read as silence. Peak-normalizing noise
  // would otherwise stretch it across the full range and leave the mouth
  // permanently half open.
  it("returns all zeros when only near-silence is present", () => {
    const noise = new Float32Array(48000)
    for (let i = 0; i < noise.length; i++) noise[i] = (i % 7) * 1e-4
    const r = analyzeSpeech(pcm([noise]))
    expect(Math.max(...r.open)).toBe(0)
  })

  it("peaks at 1.0 regardless of input amplitude", () => {
    const quiet = analyzeSpeech(pcm([tone(0.5, 48000, 0.05)]))
    const loud = analyzeSpeech(pcm([tone(0.5, 48000, 0.9)]))
    expect(Math.max(...quiet.open)).toBeCloseTo(1, 2)
    expect(Math.max(...loud.open)).toBeCloseTo(1, 2)
  })

  it("emits one value per pointsPerSecond frame", () => {
    const r = analyzeSpeech(pcm([tone(1.0, 48000)]), { pointsPerSecond: 60 })
    expect(r.open.length).toBe(60)
  })

  // Fast open, slow close. A symmetric filter reads as a pulsing sine; this
  // asymmetry is what reads as speech.
  it("closes more slowly than it opens", () => {
    const buf = new Float32Array(48000)
    buf.fill(0.5, 0, 24000)
    const r = analyzeSpeech(pcm([buf]), { pointsPerSecond: 60 })
    const peakIdx = r.open.indexOf(Math.max(...r.open))
    const framesToHalf = r.open.findIndex((v, i) => i > peakIdx && v < 0.5)
    expect(framesToHalf).toBeGreaterThan(peakIdx + 1)
  })

  it("averages across channels", () => {
    const left = tone(0.5)
    const right = new Float32Array(left.length)
    const both = analyzeSpeech(pcm([left, right]))
    const leftOnly = analyzeSpeech(pcm([left]))
    expect(Math.max(...both.open)).toBeCloseTo(Math.max(...leftOnly.open), 2)
  })
})