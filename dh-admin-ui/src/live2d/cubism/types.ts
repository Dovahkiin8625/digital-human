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
}