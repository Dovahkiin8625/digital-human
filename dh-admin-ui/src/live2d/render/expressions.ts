/**
 * The emotion vocabulary, and how a reply's tags become an expression.
 *
 * Two decisions live here, and both exist because a real model punishes
 * the naive version.
 */

/**
 * The tags the reply format defines.
 *
 * These are ALWAYS stripped from a reply, whether or not the model maps
 * them: a tag the pet does not understand must not end up read aloud as
 * a word. A bracket outside this set is left alone, so a bracketed word
 * the user typed survives -- by shape alone ``[note]`` is
 * indistinguishable from ``[smile]``, and eating it corrupts the line.
 */
export const EMOTION_TAGS = ["smile", "sad", "think", "surprise", "neutral"]

/** The index that means "apply no expression at all". */
export const NO_EXPRESSION = -1

/**
 * The tags to strip from a reply: the model's own mapping plus the fixed
 * vocabulary.
 *
 * The union matters. Without the vocabulary a tag the model does not map
 * would be spoken aloud; without the model's own tags a model with an
 * expression the vocabulary does not name could never be reached.
 */
export function strippableTags(map: Record<string, number>): string[] {
  return Array.from(new Set([...Object.keys(map), ...EMOTION_TAGS]))
}

/**
 * The expression to fall back to.
 *
 * NOT "expression 0". On a real model the first expression is whatever
 * the author happened to put first -- Natori's is Angry -- so a reply
 * with no emotion in it would land on an angry face. The operator's own
 * neutral is used when there is one; otherwise no expression is applied
 * and the model keeps its rest pose.
 */
export function neutralIndex(map: Record<string, number>): number {
  const mapped = map["neutral"]
  return mapped === undefined ? NO_EXPRESSION : mapped
}

/** Resolve the emotion tags seen in the current sentence to an expression
 *  index.
 *
 *  Unknown tags resolve to the fallback rather than throwing: the model and
 *  its expression map are operator-configured and can be swapped at
 *  runtime, and an unrecognised tag must not be able to blank the pet. */
export function expressionIndex(
  tags: string[],
  map: Record<string, number>,
  fallback = NO_EXPRESSION,
): number {
  // Last tag wins: it is nearest the end of the line, so it is the
  // expression still in effect as the sentence finishes.
  for (let i = tags.length - 1; i >= 0; i--) {
    const hit = map[tags[i]]
    if (hit !== undefined) return hit
  }
  return fallback
}
