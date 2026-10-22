import { useEffect, useRef, useState } from "react";
import { CubismStage, SAMPLE_MODEL, type ParameterReport } from "./cubism/stage";

const WIDTH = 480;
const HEIGHT = 720;

/** Debug stage: proves the render loop, the idle layer and the mouth
 *  envelope against a real model. Not the product UI. */
export default function DebugStage() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const stageRef = useRef<CubismStage | null>(null);
  const actxRef = useRef<AudioContext | null>(null);
  const [status, setStatus] = useState("加载中…");
  const [error, setError] = useState<string | null>(null);
  const [params, setParams] = useState<ParameterReport[]>([]);
  const [driven, setDriven] = useState<Record<string, string>>({});

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = WIDTH * dpr;
    canvas.height = HEIGHT * dpr;

    const stage = new CubismStage(canvas);
    stageRef.current = stage;
    let cancelled = false;

    stage
      .init(SAMPLE_MODEL)
      .then((report) => {
        if (cancelled) return;
        setParams(report);
        const picked: Record<string, string> = {};
        for (const id of [
          "ParamMouthOpenY",
          "ParamMouthForm",
          "ParamAngleX",
          "ParamAngleY",
          "ParamAngleZ",
          "ParamEyeLOpen",
          "ParamEyeBallX",
          "ParamEyeBallY",
        ]) {
          const hit = report.find((p) => p.id === id);
          if (hit) picked[id] = `${hit.min} .. ${hit.max}`;
          else picked[id] = "(缺失)";
        }
        setDriven(picked);
        setStatus(`就绪 — ${report.length} 个参数`);
        stage.start((e) => setError(String(e)));
      })
      .catch((e) => {
        if (!cancelled) setError(String(e));
      });

    return () => {
      cancelled = true;
      stage.stop();
    };
  }, []);

  async function onAudio(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const actx = actxRef.current ?? new AudioContext();
      actxRef.current = actx;
      const buffer = await actx.decodeAudioData(await file.arrayBuffer());
      const src = actx.createBufferSource();
      src.buffer = buffer;
      src.connect(actx.destination);
      // The envelope is indexed against the same clock that starts this
      // source, which is why the mouth cannot drift from the voice.
      stageRef.current?.speak(buffer);
      src.start();
    } catch (err) {
      setError(String(err));
    }
  }

  return (
    <div className="debug-stage">
      <div className="canvas-wrap">
        <canvas ref={canvasRef} style={{ width: WIDTH, height: HEIGHT }} />
      </div>
      <div className="panel">
        <p className="status">{error ? `错误：${error}` : status}</p>
        <label className="file">
          加载音频文件
          <input type="file" accept="audio/*" onChange={onAudio} />
        </label>
        <h3>我们驱动的参数（模型真实取值范围）</h3>
        <table>
          <tbody>
            {Object.entries(driven).map(([id, range]) => (
              <tr key={id}>
                <td>{id}</td>
                <td>{range}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <details>
          <summary>全部 {params.length} 个参数</summary>
          <table>
            <tbody>
              {params.map((p) => (
                <tr key={p.id}>
                  <td>{p.id}</td>
                  <td>
                    {p.min} .. {p.max}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </details>
      </div>
    </div>
  );
}