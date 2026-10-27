import type { IdlePose } from "../render/idle";

export interface ParameterReport {
  id: string;
  min: number;
  max: number;
  default: number;
}

export interface FrameState {
  open: number;
  form: number;
  speaking: boolean;
  /** The idle pose for this frame: breathing, blinking, gaze. Computed by
   *  the stage, which is the only thing that knows where the cursor is. */
  idle: IdlePose;
}
