import { analyzeSpeech, type MouthAnalysis } from "../audio/analyze";
import { idlePose, type IdlePose } from "../render/idle";
import { gazePose, type GazeTarget } from "../render/gaze";
import { NO_EXPRESSION } from "../render/expressions";
import { createDeferred } from "./deferred";
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
  /** Where the pet is looking, in the model's own eye-ball units. Null
   *  when nothing is being tracked, and the eyes saccade on their own. */
  private gaze: GazeTarget | null = null;
  /** The smoothed gaze, so the eyes drift toward the cursor rather than
   *  snapping to it. */
  private gazeSmoothed: GazeTarget | null = null;
  /** The expression the caller asked for. Deferred until there is a model
   *  to put it on -- see setExpression. */
  private readonly expression = createDeferred<number>(NO_EXPRESSION);

  constructor(private canvas: HTMLCanvasElement) {}

  async init(modelUrl: string): Promise<ParameterReport[]> {
    await loadCore();
    const { StageModel: Model } = await import("./model");
    const gl = this.canvas.getContext("webgl2") ?? this.canvas.getContext("webgl");
    if (!gl) throw new Error("WebGL unavailable");
    this.gl = gl;

    const model = new Model();
    this.model = model;
    // The expression the caller set before the model existed.
    this.expression.bind((index) => {
      model.expressionIndex = index;
    });
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

  /** Set the expression for subsequent frames.
   *
   *  The value is the position in the model's OWN expression list, which is
   *  what the operator's emotion map is written against. -1 means "no
   *  expression", which leaves the model in its rest pose.
   *
   *  The stage only records it; the model enqueues the motion, and only
   *  when the index actually changes -- see StageModel.setExpression.
   *
   *  It is RECORDED rather than applied, because the caller sets it long
   *  before the model exists: loading the core, the moc3 and every
   *  texture takes seconds, and the React component sets the expression
   *  on its first render. A value dropped on the floor there is an
   *  expression the pet never wears. */
  setExpression(index: number): void {
    this.expression.set(index);
  }

  /** Look at a point, in the model's eye-ball units. Pass null to stop
   *  tracking and let the eyes saccade on their own again. */
  setGaze(target: GazeTarget | null): void {
    this.gaze = target;
    if (!target) this.gazeSmoothed = null;
  }

  /** Play the model's interaction motion, if it has one. */
  tap(): void {
    this.model?.tap();
  }

  /** The names of the model's expressions, for diagnostics. */
  expressions(): string[] {
    return this.model?.getExpressions() ?? [];
  }

  /** The model's motion groups, for diagnostics. */
  motionGroups(): Record<string, number> {
    return this.model?.getMotionGroups() ?? {};
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
      const speaking = this.analysis !== null;
      try {
        // The model's own idle animation. A model with no Idle group simply
        // keeps its procedural animation and no more.
        this.model!.playIdleMotion();
        this.model!.renderFrame(
          this.gl!,
          this.canvas.width,
          this.canvas.height,
          { open, form, speaking, idle: this.idleFor(now, dt, speaking) },
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

  /** The idle pose for this frame.
   *
   *  When something is being tracked the eyes go where it says; otherwise
   *  they keep their own saccades. The switch is smoothed, so a cursor
   *  that appears or vanishes does not snap the eyes. */
  private idleFor(now: number, dt: number, speaking: boolean): IdlePose {
    const base = idlePose(now, { breathScale: speaking ? 0.35 : 1 });
    const { pose, smoothed } = gazePose(base, this.gaze, this.gazeSmoothed, dt);
    this.gazeSmoothed = smoothed;
    return pose;
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
