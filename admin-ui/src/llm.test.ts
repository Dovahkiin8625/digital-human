// Tests for the pure helpers in ``llm.ts``. ``streamChat`` needs a
// real SSE-speaking endpoint, which is what the integration tests on
// the server side cover; here we focus on the deterministic helpers:
// ``splitSentences`` (sentence-level chunking for TTS) and
// ``makeThinkStripper`` (consumes ``<think>...</think>`` blocks before
// they reach TTS / the avatar).
//
// Run with ``npx vitest run admin-ui/src/llm.test.ts`` once vitest is
// wired up. For now this file is written to vitest's shape so it slots
// in cleanly when the project's test infra is ready.

import { describe, expect, it } from "vitest";
import {
  drainSentences,
  makeThinkStripper,
  splitSentences,
  toLlmMessages,
} from "./llm";

describe("splitSentences", () => {
  it("returns the buffer as tail when nothing crosses a boundary", () => {
    const out = splitSentences("你好", "");
    expect(out).toEqual({ complete: [], tail: "你好" });
  });

  it("flushes a completed sentence on its boundary", () => {
    const out = splitSentences("", "你好世界。");
    expect(out.complete).toEqual(["你好世界。"]);
    expect(out.tail).toBe("");
  });

  it("keeps the unterminated tail when only part of a sentence is in", () => {
    // Two deltas: "你好。" flushes, then "再见" sits in tail.
    const first = splitSentences("", "你好。");
    const second = splitSentences(first.tail, "再见");
    expect(first.complete).toEqual(["你好。"]);
    expect(first.tail).toBe("");
    expect(second.complete).toEqual([]);
    expect(second.tail).toBe("再见");
  });

  it("splits multiple sentences on a single delta", () => {
    const out = splitSentences("", "第一句。第二句！");
    expect(out.complete).toEqual(["第一句。", "第二句！"]);
    expect(out.tail).toBe("");
  });

  it("flushes via the length fallback when no boundary arrives in time", () => {
    // 41 chars without punctuation => flushed as one chunk.
    const long =
      "一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一" +
      "一二三四五六七八九十";
    expect(long.length).toBe(41);
    const out = splitSentences("", long);
    expect(out.complete).toEqual([long]);
    expect(out.tail).toBe("");
  });

  it("does not flush early when under the length cap", () => {
    // 40 chars exactly -- still under the cap, no boundary.
    const len40 = "一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十";
    expect(len40.length).toBe(40);
    const out = splitSentences("", len40);
    expect(out.complete).toEqual([]);
    expect(out.tail).toBe(len40);
  });

  it("treats newline as a boundary", () => {
    const out = splitSentences("", "第一行\n第二行。");
    expect(out.complete).toEqual(["第一行\n", "第二行。"]);
    expect(out.tail).toBe("");
  });
});

describe("makeThinkStripper", () => {
  it("strips a single think block arriving in one delta", () => {
    // ``visibleText`` is what feeds the TTS / avatar path; the captured
    // reasoning rides in ``thoughts`` so the chat panel can render a
    // folded summary chip without ever speaking it.
    const s = makeThinkStripper();
    const out = s.feed("<think>reasoning</think>text");
    expect(out.visibleText).toBe("text");
    expect(out.newThoughts).toEqual(["reasoning"]);
    expect(s.state().thoughts).toEqual(["reasoning"]);
  });

  it("handles ``<think>`` split across multiple deltas", () => {
    // The opening tag can land in one delta, the body in another, and
    // the close in a third. The stripper must buffer the body until
    // ``</think>`` arrives, and only then expose the captured
    // reasoning. Operators see ``real reply`` in the panel; nothing
    // inside the think block ever reaches the visible text.
    const s = makeThinkStripper();
    expect(s.feed("<think>rea").visibleText).toBe("");
    expect(s.feed("soning").visibleText).toBe("");
    const final = s.feed("</think>real reply");
    expect(final.visibleText).toBe("real reply");
    expect(final.newThoughts).toEqual(["reasoning"]);
  });

  it("captures multiple think blocks in one delta", () => {
    // Common pattern for chain-of-thought models: emit ``a``, decide,
    // emit ``c``, reply. Each closure produces one entry in
    // ``thoughts``; the visible buffer is just ``bd``.
    const s = makeThinkStripper();
    const out = s.feed("<think>a</think>b<think>c</think>d");
    expect(out.visibleText).toBe("bd");
    expect(out.newThoughts).toEqual(["a", "c"]);
  });

  it("passes through text with no think blocks", () => {
    const s = makeThinkStripper();
    const out = s.feed("hello world");
    expect(out.visibleText).toBe("hello world");
    expect(out.newThoughts).toEqual([]);
    expect(s.state().thoughts).toEqual([]);
  });

  it("keeps buffering when ``</think>`` has not yet arrived", () => {
    // Critical for stream-safe filtering: if the model thinks for a
    // while, the stripper must NOT leak partial reasoning into the
    // visible buffer. The whole point is that TTS never hears it.
    const s = makeThinkStripper();
    expect(s.feed("<think>still working").visibleText).toBe("");
    expect(s.feed(" on it").visibleText).toBe("");
    expect(s.state().thoughts).toEqual([]);
  });

  it("threads a close split across deltas with the follow-up text", () => {
    // ``</think>`` arrives mid-delta while the closing ``</th`` is in
    // a prior delta -- the stripper must still close on the literal
    // close-tag boundary and resume emitting text after it.
    const s = makeThinkStripper();
    expect(s.feed("<think>reas").visibleText).toBe("");
    expect(s.feed("oning</th").visibleText).toBe("");
    expect(s.feed("ink>real reply").visibleText).toBe("real reply");
    expect(s.state().thoughts).toEqual(["reasoning"]);
  });
});

describe("drainSentences", () => {
  // The bug this helper exists to prevent: an LLM stream that produces
  // two deltas in a row, each containing one or more complete
  // sentences, would re-flush the first delta's sentences a second
  // time on the second delta if the running buffer kept the just-
  // flushed prefix. ``splitSentences`` is pure; the buffer-handling is
  // what was wrong in the chat panel. ``drainSentences`` owns that
  // contract -- the caller stores ``newTail`` and feeds it back in.

  it("flushes one sentence in one delta", () => {
    const step = drainSentences({ tail: "" }, "你好世界。");
    expect(step.flushed).toEqual(["你好世界。"]);
    expect(step.displayText).toBe("你好世界。");
    expect(step.newTail).toBe("");
  });

  it("buffers a partial sentence across deltas", () => {
    const a = drainSentences({ tail: "" }, "你");
    expect(a.flushed).toEqual([]);
    expect(a.newTail).toBe("你");
    const b = drainSentences({ tail: a.newTail }, "好。");
    expect(b.flushed).toEqual(["你好。"]);
    expect(b.newTail).toBe("");
  });

  it("flushes multiple sentences in one delta", () => {
    const step = drainSentences(
      { tail: "" },
      "第一句。第二句！第三句；",
    );
    expect(step.flushed).toEqual(["第一句。", "第二句！", "第三句；"]);
    expect(step.displayText).toBe("第一句。第二句！第三句；");
    expect(step.newTail).toBe("");
  });

  it("does not re-flush a sentence that has already been drained", () => {
    // Regression for the "数字人说两遍" report: ``streamChat`` delivers
    // each chunk as a separate delta call, and the previous code path
    // kept the flushed prefix around in the running buffer. Each
    // subsequent delta would re-flush the same sentence again, so the
    // avatar narrated every sentence multiple times. With the helper
    // owning the trim-to-tail contract, each sentence lands in
    // ``flushed`` exactly once.
    const first = drainSentences({ tail: "" }, "你好。");
    expect(first.flushed).toEqual(["你好。"]);
    const second = drainSentences({ tail: first.newTail }, "下一句。");
    expect(second.flushed).toEqual(["下一句。"]);
    // Display in the second step is just the new tail's piece --
    // the caller maintains its own running display via
    // ``displayText`` accumulation if it wants continuity.
    expect(second.displayText).toBe("下一句。");
  });

  it("flushes via the length fallback when no boundary arrives", () => {
    const long = "一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一";
    const step = drainSentences({ tail: "" }, long);
    expect(step.flushed).toEqual([long]);
    expect(step.newTail).toBe("");
  });

  it("returns an empty flushed list on empty input", () => {
    const step = drainSentences({ tail: "" }, "");
    expect(step.flushed).toEqual([]);
    expect(step.displayText).toBe("");
    expect(step.newTail).toBe("");
  });
});

describe("toLlmMessages", () => {
  // Regression: the chat panel keeps UI-only fields (``thoughts``)
  // on each assistant turn so the chip can render. The LLM endpoint
  // enforces ``extra='forbid'`` on each message dict, so any UI
  // field sent over the wire produces a 422 -- which surfaced as
  // "second message fails". The projection strips UI-only fields and
  // owns the wire-shape contract at the boundary function.

  it("drops ``thoughts`` from each message", () => {
    const out = toLlmMessages([
      { role: "user", content: "你好" },
      { role: "assistant", content: "世界", thoughts: ["some reasoning"] },
    ]);
    expect(out).toEqual([
      { role: "user", content: "你好" },
      { role: "assistant", content: "世界" },
    ]);
    // Belt-and-braces: no ``thoughts`` key anywhere in the projection.
    expect(JSON.stringify(out).includes("thoughts")).toBe(false);
  });

  it("returns an empty array for empty input", () => {
    expect(toLlmMessages([])).toEqual([]);
  });

  it("preserves role and content when ``thoughts`` is absent", () => {
    const out = toLlmMessages([
      { role: "system", content: "be terse" },
      { role: "user", content: "hi" },
    ]);
    expect(out).toEqual([
      { role: "system", content: "be terse" },
      { role: "user", content: "hi" },
    ]);
  });
});
