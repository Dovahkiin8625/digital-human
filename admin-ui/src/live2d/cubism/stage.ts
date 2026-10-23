import { analyzeSpeech, type MouthAnalysis } from "../audio/analyze";
import { loadCore } from "./core";
import type { ParameterReport } from "./types";
import type { StageModel } from "./model";

const POINTS_PER_SECOND = 60;

/**
 * Owns the render loop and the mouth envelope.
 *
 * The framework-dependent module is imported dynamically, after the core is
 * on the page: Framework/src/model/cubismmodel.ts dereferences
 * Live2DCubismCore at module-evaluation time, so a static import here would
 * throw before loadCore() ever ran.
 */
export class CubismStage {
  private model: StageModel | null = null;
  private gl: WebGLRenderingContext | null = null;
  private raf = 0;
  private last = 0;
  private analysis: MouthAnalysis | null = null;
  private speakStart = 0;
  private running = false;

  constructor(private canvas: HTMLCanvasElement) {}

  async init(modelUrl: string): Promise<ParameterReport[]> {
    await loadCore();
    const { StageModel: Model } = await import("./model");
    const gl = this.canvas.getContext("webgl2") ?? this.canvas.getContext("webgl");
    if (!gl) throw new Error("WebGL unavailable");
    this.gl = gl;

    const model = new Model();
    this.model = model;
    const slash = modelUrl.lastIndexOf("/");
    await model.loadAssets(
      modelUrl.slice(0, slash + 1),
      modelUrl.slice(slash + 1),
      gl,
      this.canvas.width,
      this.canvas.height,
    );
    return model.reportParameters();
  }

  /** Begin a mouth envelope. Play the same AudioBuffer through the speakers
   *  and the mouth cannot drift from the voice. */
  speak(buffer: AudioBuffer): void {
    this.analysis = analyzeSpeech(buffer, { pointsPerSecond: POINTS_PER_SECOND });
    // Loading the core, the moc3 and every texture takes from hundreds of
    // ms to seconds, and the loop does not run until it finishes. An
    // utterance begun before then would be indexed from a start time the loop
    // never observed -- the mouth would open mid-utterance, or not at all if
    // the load outlasted the clip. Hold the envelope and anchor it to the
    // first frame there is something to draw it on.
    this.speakStart = this.running ? performance.now() : 0;
  }

  /** Set the expression for subsequent frames. Inert for now: Cubism 5
   *  drives expressions through a motion queue rather than an index, so the
   *  adapter in model.ts ignores it. Wired end to end so the plumbing is
   *  in place when that lands. */
  setExpression(index: number): void {
    if (this.model) this.model.expressionIndex = index;
  }

  start(onError: (e: unknown) => void): void {
    if (!this.model) throw new Error("init() must resolve before start()");
    this.running = true;
    this.last = performance.now();
    const loop = () => {
      const now = performance.now();
      const dt = (now - this.last) / 1000;
      this.last = now;
      const { open, form } = this.sample(now);
      try {
        this.model!.renderFrame(
          this.gl!,
          this.canvas.width,
          this.canvas.height,
          { open, form, speaking: this.analysis !== null },
          dt,
        );
      } catch (e) {
        cancelAnimationFrame(this.raf);
        onError(e);
        return;
      }
      this.raf = requestAnimationFrame(loop);
    };
    this.raf = requestAnimationFrame(loop);
  }

  stop(): void {
    cancelAnimationFrame(this.raf);
  }

  private sample(now: number): { open: number; form: number } {
    if (!this.analysis) return { open: 0, form: 0 };
    // Held by speak() while the model was still loading.
    if (this.speakStart === 0) this.speakStart = now;
    const i = Math.floor(((now - this.speakStart) / 1000) * POINTS_PER_SECOND);
    if (i >= this.analysis.open.length) {
      this.analysis = null;
      return { open: 0, form: 0 };
    }
    return { open: this.analysis.open[i], form: this.analysis.form[i] };
  }
}

export { SHADER_PATH } from "./core";
export type { FrameState, ParameterReport } from "./types";