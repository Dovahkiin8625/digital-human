// Browser-side helpers for the LLM streaming chat panel.
//
// Three pieces, each tiny on purpose:
//
// * streamChat -- opens POST /api/admin/llm/chat/stream with the
//   request body, walks the SSE response, and dispatches each frame
//   to caller-supplied callbacks. The function returns once the
//   upstream closes (or errors); the caller never has to await "did
//   the stream end?".
// * splitSentences -- pure function: given the LLM's running text
//   buffer and the next delta, returns {complete, tail} so the caller
//   can flush ``complete`` into the TTS queue while ``tail`` carries
//   the partial sentence forward.
//
// The sentence splitter's job is to ship something that TTS can
// swallow without sounding robotic. We pick the cheapest rule that
// works for Chinese + light English: a hard cut at Chinese
// sentence-final punctuation, with a length-fallback (40 chars) so a
// reply without punctuation still flows. Smarter heuristics (commas,
// dual-language detection) can come later if real replies prove this
// too coarse.

export type Role = "user" | "assistant" | "system";

export interface ChatMessage {
  role: Role;
  content: string;
  // Captured reasoning blocks (``<think>…</think>``) for this turn.
  // Surfaced as a folded chip in the chat panel; never sent to TTS or
  // the avatar. Absent on user / system turns and on legacy replies.
  // Crucially: UI-only state. The wire shape for the LLM endpoint is
  // strictly ``{role, content}`` -- ``thoughts`` would trip the
  // backend's ``extra='forbid'`` Pydantic config and produce a 422.
  thoughts?: string[];
}

/**
 * Project a chat panel message list down to the wire shape the LLM
 * endpoint accepts. ``thoughts`` (and any future UI-only field) is
 * dropped here so the JSON payload survives Pydantic validation.
 */
export function toLlmMessages(
  messages: ChatMessage[],
): Array<{ role: Role; content: string }> {
  return messages.map((m) => ({ role: m.role, content: m.content }));
}

// Frames the chat-stream endpoint emits. Kept loose (``unknown`` is
// allowed) so an upstream with an extra field doesn't 500 the panel
// mid-stream.
export type StreamFrame =
  | { delta: string }
  | { done: true }
  | { error: { code: string; message: string } };

export interface StreamCallbacks {
  onDelta: (chunk: string) => void;
  onDone: () => void;
  onError: (code: string, message: string) => void;
}

/**
 * Open an SSE stream against ``/api/admin/llm/chat/stream`` and pump
 * frames into the caller. Returns a controller with ``abort()`` so the
 * caller can cancel mid-flight.
 */
export function streamChat(
  messages: ChatMessage[],
  cb: StreamCallbacks,
): { abort: () => void } {
  const controller = new AbortController();
  void (async () => {
    try {
      const resp = await fetch("/api/admin/llm/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: toLlmMessages(messages) }),
        signal: controller.signal,
      });
      if (!resp.ok) {
        // Non-200 -- the server-side ``AppError`` envelope reaches the
        // client as JSON. Surface the message so the panel can show it.
        let code = "llm_failed";
        let message = resp.statusText;
        try {
          const body = (await resp.json()) as { error?: { code?: string; message?: string } };
          if (body?.error?.message) message = body.error.message;
          if (body?.error?.code) code = body.error.code;
        } catch {
          /* non-json */
        }
        cb.onError(code, message);
        return;
      }
      if (!resp.body) {
        cb.onError("llm_failed", "response has no body");
        return;
      }
      const reader = resp.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";
      // Read chunks; SSE separates frames with a blank line, so we
      // accumulate into ``buffer`` until we see ``\n\n``.
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        let sep = buffer.indexOf("\n\n");
        while (sep !== -1) {
          const frame = buffer.slice(0, sep);
          buffer = buffer.slice(sep + 2);
          dispatchFrame(frame, cb);
          sep = buffer.indexOf("\n\n");
        }
      }
      // Drain anything left in the buffer (the upstream might omit the
      // final separator; ``dispatchFrame`` is a no-op for blank input).
      if (buffer.trim()) dispatchFrame(buffer, cb);
      cb.onDone();
    } catch (err) {
      if ((err as Error).name === "AbortError") {
        // Caller cancelled; don't surface as an error -- they know.
        return;
      }
      cb.onError("llm_failed", (err as Error).message || "stream failed");
    }
  })();
  return { abort: () => controller.abort() };
}

function dispatchFrame(frame: string, cb: StreamCallbacks): void {
  // Each ``data:`` line carries one JSON object; multiple ``data:``
  // lines per frame is allowed by the spec, but our server only emits
  // one. Lines starting with ``:`` are comments -- skip.
  const lines = frame.split("\n").filter((l) => l && !l.startsWith(":"));
  for (const line of lines) {
    if (!line.startsWith("data:")) continue;
    const payload = line.slice("data:".length).trim();
    if (!payload) continue;
    let parsed: StreamFrame;
    try {
      parsed = JSON.parse(payload);
    } catch {
      continue;
    }
    if ("delta" in parsed) {
      cb.onDelta(parsed.delta);
    } else if ("done" in parsed) {
      // The HTTP response is about to close; treat as terminal.
      cb.onDone();
    } else if ("error" in parsed) {
      cb.onError(parsed.error.code, parsed.error.message);
    }
  }
}

/**
 * Sentence-level chunking for the running LLM reply.
 *
 * Inputs:
 *   ``buffer`` -- the assistant text accumulated so far, possibly
 *     ending mid-word (e.g. ``"你"``).
 *   ``delta``  -- the next token to append (e.g. ``"好"``).
 *
 * Output:
 *   ``complete`` -- zero or more fully-formed sentences to flush to TTS.
 *   ``tail``     -- the leftover partial sentence (may be empty).
 *
 * Rule: split on Chinese sentence-final punctuation (。！？；\n), and
 * when ``buffer + delta`` exceeds ``MAX_BUFFER_CHARS`` without any
 * punctuation, flush the buffer as one chunk so TTS doesn't sit idle
 * waiting for a period.
 */
export interface SentenceSplit {
  complete: string[];
  tail: string;
}

// ``String.prototype.matchAll`` requires the regex to carry the global
// flag -- V8 throws ``String.prototype.matchAll called with a non-global
// RegExp argument`` otherwise. The historic bug shipped without ``g``
// and the first chat reply in the panel surfaced as ``llm_failed``.
const SENTENCE_BOUNDARIES = /([。！？；\n])/g;
// Above this many characters without a boundary, flush anyway.
const MAX_BUFFER_CHARS = 40;

export function splitSentences(buffer: string, delta: string): SentenceSplit {
  const merged = buffer + delta;
  const boundaryMatches = [...merged.matchAll(SENTENCE_BOUNDARIES)];
  // No punctuation found and we haven't hit the length fallback -- keep
  // accumulating so a sentence like "你好" doesn't get split in half.
  if (boundaryMatches.length === 0 && merged.length <= MAX_BUFFER_CHARS) {
    return { complete: [], tail: merged };
  }
  const sentences: string[] = [];
  let cursor = 0;
  let lastBoundaryEnd = 0;
  for (const match of boundaryMatches) {
    const end = match.index! + match[0].length;
    lastBoundaryEnd = end;
    sentences.push(merged.slice(cursor, end));
    cursor = end;
  }
  // Length fallback: if we never hit a boundary and the buffer is now
  // "long enough", flush everything as one chunk.
  if (sentences.length === 0 && merged.length > MAX_BUFFER_CHARS) {
    return { complete: [merged], tail: "" };
  }
  // Otherwise, anything past the last boundary is the new tail.
  return { complete: sentences, tail: merged.slice(lastBoundaryEnd) };
}

// ---------------------------------------------------------------------------
// ``drainSentences`` -- the buffer-management wrapper that lets the chat
// panel split incrementally without re-flushing older sentences.
//
// ``splitSentences`` is pure, but the *running* buffer that callers feed
// it is what was producing the duplicated-TTS bug: the panel used to do
// ``buffer += delta`` followed by ``splitSentences(buffer, "")`` and
// dispatch ``split.complete``, but never trimmed the buffer back to
// ``split.tail``. Every subsequent delta re-split the entire accumulated
// text -- including the sentences just dispatched -- so the avatar ended
// up speaking each sentence multiple times. ``drainSentences`` owns the
// "trim to tail" half of that dance so the caller can't forget it.
//
// Inputs:
//   ``state.tail`` -- the un-flushed remainder carried over from the
//     last call. Pass ``{ tail: "" }`` for the first delta of a turn.
//   ``delta``      -- the new chunk the upstream just delivered.
//
// Outputs:
//   ``flushed``    -- complete sentences ready to feed to TTS / the
//     avatar pipeline. Drop whitespace-only entries before fetching
//     audio; the dispatcher in ``Chat.tsx`` already does this.
//   ``displayText``-- ``flushed + newTail`` rendered as one string;
//     the chat panel uses this for the assistant bubble content.
//   ``newTail``    -- the trailing characters that didn't reach a
//     boundary yet. Caller persists this and feeds it back on the
//     next delta.
// ---------------------------------------------------------------------------

export interface SentenceQueueState {
  tail: string;
}

export interface SentenceQueueStep {
  flushed: string[];
  displayText: string;
  newTail: string;
}

export function drainSentences(
  state: SentenceQueueState,
  delta: string,
): SentenceQueueStep {
  const merged = state.tail + delta;
  const split = splitSentences(merged, "");
  return {
    flushed: split.complete,
    displayText: split.complete.join("") + split.tail,
    newTail: split.tail,
  };
}

// ---------------------------------------------------------------------------
// ``<think>`` stripper -- model reasoning blocks the avatar must not speak.
//
// Recent reasoning models (Qwen3, DeepSeek-R1, etc.) wrap their chain-of-
// thought in ``<think>…</think>``. The avatar pipeline must NEVER ship
// that content to TTS or to the driving client; if it does, the digital
// human blurts out its private reasoning out loud. The stripper below
// runs the LLM stream through itself: ``feed()`` consumes a delta and
// returns the slice that may be passed to ``splitSentences`` / spoken,
// plus any reasoning blocks that were *just* closed by this delta.
//
// A stripper instance owns its state across calls: when ``<think>``
// arrives split across deltas (e.g. ``"<th"`` then ``"ink>…"`` then
// ``"…</think>"``) the stripper holds back chars that could be the
// start of a tag and never lets them leak into the visible buffer or
// the TTS pipeline.
//
// State per stripper:
//
//   * ``inThink`` -- currently between ``<think>`` and ``</think>``;
//     chars land in ``thoughtBuf`` instead of the visible buffer.
//   * ``thoughtBuf`` -- the partial reasoning body for the *open*
//     think block; cleared when ``</think>`` lands and the contents
//     are pushed onto the closed ``thoughts`` array.
//   * ``pending`` -- the trailing substring of ``rest`` that *could*
//     still grow into the next tag, kept across ``feed()`` calls so a
//     tag split between two network reads is still recognised.
// ---------------------------------------------------------------------------

export interface ThinkStripperState {
  thoughts: string[];
}

export interface FeedResult {
  /** Chars safe to ship to TTS / the avatar / the chat bubble text. */
  visibleText: string;
  /** Reasoning blocks that closed during *this* call. */
  newThoughts: string[];
}

const THINK_OPEN = "<think>";
const THINK_CLOSE = "</think>";

function tagPrefixLookahead(rest: string, target: string): string {
  // Longest proper suffix of ``rest`` that is also a prefix of
  // ``target`` -- the chars we have to hold back to avoid mid-tag
  // flushing when ``feed()`` is called with a chunk that ends inside
  // a tag. Empty string means the next char is unconstrained.
  const max = Math.min(rest.length, target.length - 1);
  for (let k = max; k > 0; k--) {
    const suffix = rest.slice(rest.length - k);
    if (target.startsWith(suffix)) return suffix;
  }
  return "";
}

export function makeThinkStripper(): {
  feed: (delta: string) => FeedResult;
  state: () => ThinkStripperState;
} {
  let inThink = false;
  let thoughtBuf = "";
  let pending = "";
  const thoughts: string[] = [];

  return {
    feed(delta: string): FeedResult {
      // Re-attach any partial-tag chars held back from a prior call.
      // ``pending`` is only ever a suffix that *could* grow into a tag
      // boundary -- never real reasoning or visible text.
      let rest = pending + delta;
      pending = "";
      let visible = "";
      const newThoughts: string[] = [];

      while (rest.length > 0) {
        const target = inThink ? THINK_CLOSE : THINK_OPEN;
        const tlen = target.length;
        const idx = rest.indexOf(target);

        if (idx >= 0) {
          // Tag lands at ``idx``. Whatever's before the tag goes into
          // whichever buffer we're currently filling; then we cross the
          // boundary and the loop continues with the rest of the input.
          const before = rest.slice(0, idx);
          if (inThink) {
            thoughtBuf += before;
            newThoughts.push(thoughtBuf);
            thoughtBuf = "";
          } else {
            visible += before;
          }
          inThink = !inThink;
          rest = rest.slice(idx + tlen);
          continue;
        }

        // No tag in this ``rest`` -- the only thing that could still
        // grow into a tag boundary is a trailing ``rest`` suffix that
        // matches a prefix of ``target``. Flush the safe prefix; hold
        // the lookahead for the next call.
        const lookahead = tagPrefixLookahead(rest, target);
        const pre = rest.slice(0, rest.length - lookahead.length);
        if (inThink) {
          thoughtBuf += pre;
        } else {
          visible += pre;
        }
        pending = lookahead;
        rest = "";
      }

      // Anything closed this call gets appended to the lifetime list so
      // ``state()`` reflects every reasoning block the stripper has
      // seen so far, not just the most recent batch.
      for (const t of newThoughts) thoughts.push(t);

      return { visibleText: visible, newThoughts };
    },

    state(): ThinkStripperState {
      // Defensive copy so callers can't mutate our internal accumulator.
      return { thoughts: thoughts.slice() };
    },
  };
}
