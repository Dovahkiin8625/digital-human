import { CubismFramework } from "@framework/live2dcubismframework";
import { CubismMatrix44 } from "@framework/math/cubismmatrix44";
import { CubismModelSettingJson } from "@framework/cubismmodelsettingjson";
import { CubismPhysicsUpdater } from "@framework/motion/cubismphysicsupdater";
import { CubismPoseUpdater } from "@framework/motion/cubismposeupdater";
import { CubismUserModel } from "@framework/model/cubismusermodel";
import type { ICubismModelSetting } from "@framework/icubismmodelsetting";

import { idlePose } from "../render/idle";
import { applyFrame, type FrameInputs } from "../render/frame";
import { SHADER_PATH } from "./core";
import type { FrameState, ParameterReport } from "./types";

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
 *   - CubismExpressionMotionManager has no SetExpression; expressions are
 *     enqueued motions in Cubism 5. setExpression below is a no-op
 *     placeholder -- see the ledger, the expression feature is NOT wired.
 */
export class StageModel extends CubismUserModel {
  physics: CubismPhysicsUpdater | null = null;
  pose: CubismPoseUpdater | null = null;
  /** Expression index for the current frame. Set by the stage each frame;
   *  the expression adapter below is still a no-op under Cubism 5, which
   *  drives expressions as a motion queue rather than by index. */
  expressionIndex = 0;

  /** CubismModel adapter for applyFrame: parameter names in, index out. */
  private readonly paramModel = {
    setParameterValueByIndex: (i: number, v: number) => this._model.setParameterValueByIndex(i, v),
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
    const settingBuffer = await fetch(`${home}${fileName}`).then((r) => r.arrayBuffer());
    const setting: ICubismModelSetting = new CubismModelSettingJson(
      settingBuffer,
      settingBuffer.byteLength,
    );

    const moc = await fetch(`${home}${setting.getModelFileName()}`).then((r) => r.arrayBuffer());
    this.loadModel(moc, false);

    // createRenderer initializes against the just-loaded CubismModel, so it
    // must come after loadModel. getRenderer() is null until it runs.
    this.createRenderer(width, height);
    this.getRenderer().startUp(gl);
    this.getRenderer().loadShaders(SHADER_PATH);

    if (setting.getPhysicsFileName() !== "") {
      const p = await fetch(`${home}${setting.getPhysicsFileName()}`).then((r) => r.arrayBuffer());
      this.loadPhysics(p, p.byteLength);
      this.physics = new CubismPhysicsUpdater(this._physics);
    }
    if (setting.getPoseFileName() !== "") {
      const p = await fetch(`${home}${setting.getPoseFileName()}`).then((r) => r.arrayBuffer());
      this.loadPose(p, p.byteLength);
      this.pose = new CubismPoseUpdater(this._pose);
    }

    for (let i = 0; i < setting.getTextureCount(); i++) {
      const name = setting.getTextureFileName(i);
      if (name === "") continue;
      const blob = await fetch(`${home}${name}`).then((r) => r.blob());
      this.getRenderer().bindTexture(i, createTexture(gl, await createImageBitmap(blob)));
    }
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

    const inputs: FrameInputs = {
      model: this.paramModel,
      expression: { setExpression: () => {}, update: () => {} },
      motion: { update: () => this._motionManager.updateMotion(this._model, dt) },
      physics: { update: () => this.physics?.onLateUpdate(this._model, dt) },
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
    this.pose?.onLateUpdate(this._model, dt);

    this.getRenderer().drawModel(SHADER_PATH);
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