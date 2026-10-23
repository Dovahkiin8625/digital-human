import { useEffect, useRef } from "react";
import { CubismStage } from "./cubism/stage";
import { mountStage } from "./mountStage";
import type { ParameterReport } from "./cubism/types";

export interface Live2DStageProps {
  /** Model entry URL. Admin-console callers pass the authenticated hosting
   *  endpoint; the debug stage passes the local sample. */
  modelUrl: string
  /** The utterance being spoken. Pass the SAME AudioBuffer the caller plays
   *  -- the mouth reads the audio that is actually reaching the speakers,
   *  which is what makes drift impossible. */
  audio: AudioBuffer | null
  /** Index into the model's own expression list, driven by the emotion tags
   *  in the reply. Wired through but currently inert: Cubism 5 replaced
   *  index-based expressions with a motion queue, so the adapter inside
   *  the stage is a no-op until that is designed. */
  expressionIndex?: number
  width?: number
  height?: number
  onReady?: (params: ParameterReport[]) => void
  onError?: (e: unknown) => void
}

/**
 * A Live2D model on a canvas, driven by the idle layer and the mouth
 * envelope. Shared by the Avatars, Debug and Chat pages, and by the
 * Electron pet's route.
 *
 * Renders no chrome: sizing is the caller's business, so the same component
 * works inside the console's panel and inside a bare transparent window.
 */
export default function Live2DStage({
  modelUrl,
  audio,
  expressionIndex = 0,
  width = 480,
  height = 720,
  onReady,
  onError,
}: Live2DStageProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const stageRef = useRef<CubismStage | null>(null);
  const exprRef = useRef(expressionIndex);
  exprRef.current = expressionIndex;
  // Kept in refs so a caller passing inline arrow functions does not tear the
  // stage down and rebuild the model on every render.
  const onReadyRef = useRef(onReady);
  const onErrorRef = useRef(onError);
  onReadyRef.current = onReady;
  onErrorRef.current = onError;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);

    const stage = new CubismStage(canvas);
    stageRef.current = stage;
    const handle = mountStage(stage, modelUrl, {
      onReady: (p) => onReadyRef.current?.(p),
      onError: (e) => onErrorRef.current?.(e),
    });

    return () => {
      handle.stop();
      stageRef.current = null;
    };
  }, [modelUrl, width, height]);

  useEffect(() => {
    if (audio) stageRef.current?.speak(audio);
  }, [audio]);

  useEffect(() => {
    stageRef.current?.setExpression(expressionIndex);
  }, [expressionIndex]);

  return <canvas ref={canvasRef} style={{ width, height, display: "block" }} />;
}