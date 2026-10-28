import { CubismFramework } from "@framework/live2dcubismframework";
import { CubismMatrix44 } from "@framework/math/cubismmatrix44";
import { CubismModelSettingJson } from "@framework/cubismmodelsettingjson";
import { CubismPhysicsUpdater } from "@framework/motion/cubismphysicsupdater";
import { CubismPoseUpdater } from "@framework/motion/cubismposeupdater";
import { CubismUserModel } from "@framework/model/cubismusermodel";
import type { ICubismModelSetting } from "@framework/icubismmodelsetting";
import type { ACubismMotion } from "@framework/motion/acubismmotion";
import { CubismMotion } from "@framework/motion/cubismmotion";

import { idlePose } from "../render/idle";
import { applyFrame, type FrameInputs } from "../render/frame";
import { fetchAsset, SHADER_PATH } from "./core";
import type { FrameState, ParameterReport } from "./types";

/** The motion group the pet idles in. Cubism projects name their groups
 *  freely, but "Idle" is the SDK's own convention and what the shipped
 *  models use. */
export const IDLE_GROUP = "Idle";

/** The group played when the pet is clicked. */
export const TAP_GROUP = "TapBody";

/** Motion priorities, from the SDK's own demo. */
const PRIORITY_IDLE = 1;
const PRIORITY_FORCE = 3;

/** "No expression has been applied yet" -- distinct from "no expression",
 *  which is index -1. */
const UNSET = -2;

/**
 * A CubismUserModel plus the GL plumbing.
 *
 * Everything here imports the Cubism framework statically, so this module
 * must not be imported until loadCore() has resolved -- see core.ts.
 *
 * Two framework habits are deliberately NOT used, per spec section 5.4: the
 * auto breath / eyeblink / lipsync updaters are never instantiated, and our
 * parameters are written after the managers run. Both sides writing the same
 * parameter is what makes a model judder.
 *
 * Cubism 5 API notes, learned from this SDK (they differ from Cubism 4):
 *   - getParameterIndex takes a CubismIdHandle, resolved from a plain string.
 *   - There is no SetExpression. CubismExpressionMotionManager is a motion
 *     QUEUE, so an expression is applied by enqueueing its motion; the
 *     manager cross-fades it against whatever was there.
 */
export class StageModel extends CubismUserModel {
  physics: CubismPhysicsUpdater | null = null;
  pose: CubismPoseUpdater | null = null;

  /** Expression motions, keyed by their position in the model's own
   *  expression list. The position is what the operator's emotion map
   *  refers to, and it is the numbering the server validates against. */
  private readonly expressions = new Map<number, ACubismMotion>();
  /** The names of those expressions, in the same order. For diagnostics. */
  private readonly expressionNames: string[] = [];
  /** Motion groups, each a list of motions in the model's own order. */
  private readonly motions = new Map<string, CubismMotion[]>();
  /** The expression currently applied.
   *
   *  Tracked so a repeated index does not enqueue the motion again: the
   *  Cubism 5 expression manager is a queue, and every startMotion pushes
   *  an entry, so calling it every frame would grow the queue without
   *  bound. */
  private appliedExpression = UNSET;
  /** The last motion played from each group, so the next one is not the
   *  same one twice in a row. */
  private readonly lastPlayed = new Map<string, number>();

  /** Expression index for the current frame. Set by the stage; -1 means
   *  "apply no expression", which leaves the model in its rest pose. */
  expressionIndex = -1;

  /** CubismModel adapter for applyFrame: parameter names in, index out. */
  private readonly paramModel = {
    setParameterValueByIndex: (i: number, v: number) => this._model.setParameterValueByIndex(i, v),
    getParameterValueByIndex: (i: number) => this._model.getParameterValueByIndex(i),
    getParameterIndex: (name: string) =>
      this._model.getParameterIndex(CubismFramework.getIdManager().getId(name)),
    getParameterMinimumValue: (i: number) => this._model.getParameterMinimumValue(i),
    getParameterMaximumValue: (i: number) => this._model.getParameterMaximumValue(i),
  };

  async loadAssets(
    dir: string,
    fileName: string,
    gl: WebGLRenderingContext,
    width: number,
    height: number,
  ): Promise<void> {
    // Framework global init, in this order. startUp() flips s_isStarted;
    // initialize() allocates the id manager and the rest. Both guard against
    // double-init, but initialize() asserts startUp() already ran --
    // getIdManager() returns null until both have, and every parameter
    // lookup goes through it.
    CubismFramework.startUp();
    CubismFramework.initialize();

    const home = dir.endsWith("/") ? dir : `${dir}/`;
    const settingBuffer = await fetchAsset(`${home}${fileName}`).then((r) => {
      if (!r.ok) throw new Error(`model file ${r.status}: ${r.statusText}`)
      return r.arrayBuffer()
    });
    const setting: ICubismModelSetting = new CubismModelSettingJson(
      settingBuffer,
      settingBuffer.byteLength,
    );

    const moc = await fetchAsset(`${home}${setting.getModelFileName()}`).then((r) => {
      if (!r.ok) throw new Error(`model file ${r.status}: ${r.statusText}`)
      return r.arrayBuffer()
    });
    this.loadModel(moc, false);

    // createRenderer initializes against the just-loaded CubismModel, so it
    // must come after loadModel. getRenderer() is null until it runs.
    this.createRenderer(width, height);
    this.getRenderer().startUp(gl);
    this.getRenderer().loadShaders(SHADER_PATH());

    if (setting.getPhysicsFileName() !== "") {
      const p = await fetchAsset(`${home}${setting.getPhysicsFileName()}`).then((r) => {
      if (!r.ok) throw new Error(`model file ${r.status}: ${r.statusText}`)
      return r.arrayBuffer()
    });
      this.loadPhysics(p, p.byteLength);
      this.physics = new CubismPhysicsUpdater(this._physics);
    }
    if (setting.getPoseFileName() !== "") {
      const p = await fetchAsset(`${home}${setting.getPoseFileName()}`).then((r) => {
      if (!r.ok) throw new Error(`model file ${r.status}: ${r.statusText}`)
      return r.arrayBuffer()
    });
      this.loadPose(p, p.byteLength);
      this.pose = new CubismPoseUpdater(this._pose);
    }

    for (let i = 0; i < setting.getTextureCount(); i++) {
      const name = setting.getTextureFileName(i);
      if (name === "") continue;
      const blob = await fetchAsset(`${home}${name}`).then((r) => r.blob());
      this.getRenderer().bindTexture(i, createTexture(gl, await createImageBitmap(blob)));
    }

    // Expressions and motions are optional. A model with neither is still
    // a perfectly good Live2D model -- it just never changes face and
    // never animates -- so a missing group is not an error.
    await this.loadExpressions(setting, home);
    await this.loadMotions(setting, home);
  }

  /** Load the model's expression files, keyed by their position in the
   *  expression list. */
  private async loadExpressions(setting: ICubismModelSetting, home: string): Promise<void> {
    for (let i = 0; i < setting.getExpressionCount(); i++) {
      const name = setting.getExpressionName(i);
      const file = setting.getExpressionFileName(i);
      this.expressionNames.push(name);
      if (!file) continue;
      try {
        const buffer = await fetchAsset(`${home}${file}`).then((r) => {
          if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
          return r.arrayBuffer();
        });
        const motion = this.loadExpression(buffer, buffer.byteLength, name);
        if (motion) this.expressions.set(i, motion);
      } catch {
        // One unreadable expression must not cost the model its face
        // entirely. The rest of the list still loads.
      }
    }
  }

  /** Load the model's motion groups. */
  private async loadMotions(setting: ICubismModelSetting, home: string): Promise<void> {
    for (let g = 0; g < setting.getMotionGroupCount(); g++) {
      const group = setting.getMotionGroupName(g);
      const count = setting.getMotionCount(group);
      const list: CubismMotion[] = [];
      for (let j = 0; j < count; j++) {
        const file = setting.getMotionFileName(group, j);
        if (!file) continue;
        try {
          const buffer = await fetchAsset(`${home}${file}`).then((r) => {
            if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
            return r.arrayBuffer();
          });
          // The fade times live in the model setting, so they are handed
          // over rather than left at the motion's own defaults.
          const motion = this.loadMotion(
            buffer,
            buffer.byteLength,
            file,
            undefined,
            undefined,
            setting,
            group,
            j,
          );
          if (motion) {
            // The framework dereferences these two lists unguarded while
            // updating a motion, and they start out null -- a motion
            // loaded without them throws on its first frame.
            //
            // Empty on purpose. They are the targets for a motion's
            // EyeBlink and LipSync curves, which are Cubism's OWN
            // automatic blink and lip sync -- and this pet drives both
            // itself (spec 5.4). Filling them in would let a motion fight
            // the idle layer for the eyes and the mouth.
            motion.setEffectIds([], []);
            list.push(motion);
          }
        } catch {
          // Same as above: a motion that will not load is simply absent.
        }
      }
      if (list.length) this.motions.set(group, list);
    }
  }

  /** The names of the model's expressions, in list order. */
  getExpressions(): string[] {
    return this.expressionNames.slice();
  }

  /** The motion groups the model declares, with how many motions each
   *  holds. */
  getMotionGroups(): Record<string, number> {
    const out: Record<string, number> = {};
    for (const [group, list] of this.motions) out[group] = list.length;
    return out;
  }

  /** Apply an expression by its index in the model's own list.
   *
   *  Cubism 5 has no SetExpression: expressions are motions in a queue and
   *  the manager cross-fades them. So applying one means enqueueing it --
   *  and enqueueing is only correct when the expression CHANGES, because
   *  every enqueue pushes a queue entry. */
  setExpression(index: number): void {
    if (index === this.appliedExpression) return;
    this.appliedExpression = index;
    if (index < 0) {
      // No expression: drop whatever is queued so the model falls back to
      // its rest pose.
      this._expressionManager.stopAllMotions();
      return;
    }
    const motion = this.expressions.get(index);
    // A model without that expression keeps its rest face. It is not an
    // error: the operator's map is validated against the model at
    // registration, but a model swapped in at runtime may differ.
    if (!motion) return;
    this._expressionManager.startMotion(motion, false);
  }

  /** Play a motion from a group.
   *
   *  Priority follows the SDK's own scheme: an idle motion will not
   *  interrupt a higher-priority one, and a force motion interrupts
   *  whatever is playing. */
  private startMotion(group: string, index: number, priority: number): boolean {
    const list = this.motions.get(group);
    if (!list || index < 0 || index >= list.length) return false;
    if (priority === PRIORITY_FORCE) {
      this._motionManager.setReservePriority(priority);
    } else if (!this._motionManager.reserveMotion(priority)) {
      return false;
    }
    this._motionManager.startMotionPriority(list[index], false, priority);
    this.lastPlayed.set(group, index);
    return true;
  }

  /** Pick a motion from a group, avoiding the one that played last. */
  private pickNext(group: string): number {
    const list = this.motions.get(group);
    if (!list || list.length === 0) return -1;
    if (list.length === 1) return 0;
    const last = this.lastPlayed.get(group) ?? -1;
    let next = Math.floor(Math.random() * (list.length - 1));
    if (next >= last) next += 1;
    return next;
  }

  /** Start an idle motion, if the model has one and nothing is playing.
   *
   *  Returns false when there was nothing to play -- a model with no Idle
   *  group simply keeps its own procedural animation and no more. */
  playIdleMotion(): boolean {
    if (!this._motionManager.isFinished()) return false;
    const next = this.pickNext(IDLE_GROUP);
    if (next < 0) return false;
    return this.startMotion(IDLE_GROUP, next, PRIORITY_IDLE);
  }

  /** Play the interaction motion, interrupting whatever is playing. */
  tap(): boolean {
    const next = this.pickNext(TAP_GROUP);
    if (next < 0) return false;
    return this.startMotion(TAP_GROUP, next, PRIORITY_FORCE);
  }

  /** Read every parameter's real range off the moc3. These come from the
   *  native model struct rather than from the motions, so they are final as
   *  soon as the model is loaded. */
  reportParameters(): ParameterReport[] {
    const p = this._model.getModel().parameters;
    const out: ParameterReport[] = [];
    for (let i = 0; i < p.count; i++) {
      out.push({
        id: p.ids[i],
        min: p.minimumValues[i],
        max: p.maximumValues[i],
        default: p.defaultValues[i],
      });
    }
    return out;
  }

  renderFrame(gl: WebGLRenderingContext, w: number, h: number, f: FrameState, dt: number): void {
    // Aspect-preserving orthographic scale, the same shape the SDK's Demo
    // uses. The model is already in normalized units around the origin, so
    // scaling by 1/PixelsPerUnit (a natural-looking but wrong guess) shrinks
    // it to an invisible speck.
    const projection = new CubismMatrix44();
    if (w > h) projection.scale(1.0, w / h);
    else projection.scale(h / w, 1.0);

    // The baseline. Every parameter goes back to the model's own default
    // before anything is layered on top of it.
    //
    // This is not optional, and it is the reason the frame looks the way
    // it does. Cubism's expressions are ADDITIVE on most rigs -- every
    // one of Natori's is -- and the framework adds them to whatever the
    // parameter currently holds. Without the reset, the value one frame
    // leaves behind is the value the next frame adds to, and a blush ramps
    // 0 -> 1 -> 2 -> 3 until the face comes apart. The reset is also what
    // makes a motion's fade-out land on the rest pose instead of on
    // wherever the last frame happened to leave the head.
    this._model.loadParameters();

    const inputs: FrameInputs = {
      model: this.paramModel,
      expression: {
        setExpression: (index: number) => this.setExpression(index),
        update: () => this._expressionManager.updateMotion(this._model, dt),
      },
      motion: { update: () => this._motionManager.updateMotion(this._model, dt) },
      physics: { update: () => this.physics?.onLateUpdate(this._model, dt) },
      pose: { update: () => this.pose?.onLateUpdate(this._model, dt) },
      model3: { update: () => this._model.update() },
      expressionIndex: this.expressionIndex,
      open: f.open,
      form: f.form,
      idle: idlePose(performance.now(), { breathScale: f.speaking ? 0.35 : 1 }),
      breathScale: 1,
    };

    this.getRenderer().setMvpMatrix(projection);
    // null = the default framebuffer; the SDK's type does not admit null.
    this.getRenderer().setRenderState(null as unknown as WebGLFramebuffer, [0, 0, w, h]);

    applyFrame(inputs);

    this.getRenderer().drawModel(SHADER_PATH());
    gl.flush();
  }
}

function createTexture(gl: WebGLRenderingContext, bitmap: ImageBitmap): WebGLTexture {
  const tex = gl.createTexture();
  if (!tex) throw new Error("createTexture failed");
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, 1);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, bitmap);
  gl.generateMipmap(gl.TEXTURE_2D);
  gl.bindTexture(gl.TEXTURE_2D, null);
  return tex;
}
