import { useCallback, useEffect, useRef, useState } from "react";
import { stripEmotionTags } from "../llm";
import {
  expressionIndex,
  neutralIndex,
  strippableTags,
} from "./render/expressions";

export interface SpeechOptions {
  /** Absolute server root. Empty for the console (same origin); the
   *  Electron pet points at a deployment. */
  baseUrl: string;
  /** Sent as X-V1-Token. */
  token: string;
  /** Emotion tag -> index into the model's own expression list. */
  expressionMap: Record<string, number>;
  voiceId: string;
  onError?: (message: string) => void;
}

export interface SpeechDriver {
  /** Queue a sentence to be spoken. */
  enqueue(sentence: string): void;
  /** Speak whatever is left of the queue. */
  flush(): void;
  /** Throw away everything queued. */
  clear(): void;
  busy: boolean;
  caption: string;
  /** The expression the pet is wearing. -1 means "the model's rest face". */
  expressionIndex: number;
  /** The utterance being spoken, for the mouth envelope. The SAME buffer
   *  that reaches the speakers. */
  audio: AudioBuffer | null;
}

interface Prepared {
  text: string;
  expression: number;
  blob?: Blob;
  error?: string;
}

/**
 * The Live2D half of "say something": a sentence queue, a TTS call per
 * sentence, and the expression that sentence wears.
 *
 * The queue is serial on purpose. Two sentences spoken at once would
 * overlap their mouth envelopes -- the mouth reads ONE buffer at a time,
 * so the second would cut the first off mid-word.
 *
 * The next sentence is fetched while the current one is still playing, so
 * the pause between sentences is the gap between two requests rather than
 * a round trip.
 */
export function useLive2DSpeech(opts: SpeechOptions): SpeechDriver {
  const [busy, setBusy] = useState(false);
  const [caption, setCaption] = useState("");
  const [expression, setExpression] = useState(neutralIndex(opts.expressionMap));
  const [audio, setAudio] = useState<AudioBuffer | null>(null);

  const queueRef = useRef<string[]>([]);
  const runningRef = useRef(false);
  const actxRef = useRef<AudioContext | null>(null);
  // The options change whenever the page re-renders, but the worker reads
  // them through a ref so a stale closure cannot speak with the wrong map.
  const optsRef = useRef(opts);
  optsRef.current = opts;

  const prepare = useCallback(async (sentence: string): Promise<Prepared | null> => {
    const trimmed = sentence.trim();
    if (!trimmed) return null;
    const { expressionMap, baseUrl, token, voiceId } = optsRef.current;
    // Strip before TTS: the pet would otherwise read the marker aloud.
    // The allowed set is the model's own tags plus the fixed vocabulary,
    // so a tag the model does not map is still removed instead of being
    // spoken as a word.
    const { text: spoken, tags } = stripEmotionTags(trimmed, strippableTags(expressionMap));
    const face = expressionIndex(tags, expressionMap, neutralIndex(expressionMap));
    try {
      const resp = await fetch(`${baseUrl.replace(/\/$/, "")}/api/admin/tts`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(token ? { "X-V1-Token": token } : {}),
        },
        body: JSON.stringify({ text: spoken, voice_id: voiceId }),
      });
      if (!resp.ok) throw new Error(`合成失败：${resp.status}`);
      return { text: spoken, expression: face, blob: await resp.blob() };
    } catch (e) {
      // The sentence still happened. It gets its caption and its
      // expression; only the voice is missing.
      return { text: spoken, expression: face, error: (e as Error).message };
    }
  }, []);

  const deliver = useCallback(async (item: Prepared) => {
    setCaption(item.text);
    setExpression(item.expression);
    if (item.error || !item.blob) {
      optsRef.current.onError?.(item.error ?? "合成失败");
      await new Promise((r) => setTimeout(r, 400));
      return;
    }
    try {
      const actx = actxRef.current ?? new AudioContext();
      actxRef.current = actx;
      const decoded = await actx.decodeAudioData(await item.blob.arrayBuffer());
      const src = actx.createBufferSource();
      src.buffer = decoded;
      src.connect(actx.destination);
      // The same buffer drives the mouth and reaches the speakers, so the
      // two cannot drift.
      setAudio(decoded);
      src.start();
      await new Promise((r) => setTimeout(r, Math.max(0, decoded.duration - 0.05) * 1000));
    } catch (e) {
      optsRef.current.onError?.(`音频播放失败：${(e as Error).message}`);
    }
  }, []);

  const worker = useCallback(async () => {
    runningRef.current = true;
    setBusy(true);
    try {
      let pending: Promise<Prepared | null> | null = null;
      for (;;) {
        if (!pending) {
          const sentence = queueRef.current.shift();
          if (sentence === undefined) {
            // Nothing left. Clear the running flag and re-check in the
            // same tick: a sentence that arrives now finds the worker
            // stopped and starts a new one, and one that arrived a
            // moment ago is still in the queue and is picked up here.
            runningRef.current = false;
            if (queueRef.current.length === 0) break;
            runningRef.current = true;
            continue;
          }
          pending = prepare(sentence);
        }
        const item = await pending;
        pending = null;
        // Start fetching the next sentence while this one is spoken.
        const next = queueRef.current.shift();
        pending = next === undefined ? null : prepare(next);
        if (!item) continue;
        await deliver(item);
      }
    } finally {
      runningRef.current = false;
      setBusy(false);
    }
  }, [prepare, deliver]);

  const workerRef = useRef(worker);
  workerRef.current = worker;

  const enqueue = useCallback((sentence: string) => {
    if (!sentence.trim()) return;
    queueRef.current.push(sentence);
    if (!runningRef.current) void workerRef.current();
  }, []);

  const flush = useCallback(() => {
    if (!runningRef.current) void workerRef.current();
  }, []);

  const clear = useCallback(() => {
    queueRef.current = [];
    setCaption("");
  }, []);

  useEffect(() => {
    // A different model means a different expression map, so the face
    // returns to that model's neutral rather than keeping the last
    // model's expression.
    setExpression(neutralIndex(opts.expressionMap));
  }, [opts.expressionMap]);

  return { enqueue, flush, clear, busy, caption, expressionIndex: expression, audio };
}
