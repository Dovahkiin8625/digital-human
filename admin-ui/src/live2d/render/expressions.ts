/** Resolve the emotion tags seen in the current sentence to an expression
 *  index.
 *
 *  Unknown tags resolve to the fallback rather than throwing: the model and
 *  its expression map are operator-configured and can be swapped at
 *  runtime, and an unrecognised tag must not be able to blank the pet. */
export function expressionIndex(
  tags: string[],
  map: Record<string, number>,
  fallback = 0,
): number {
  // Last tag wins: it is nearest the end of the line, so it is the
  // expression still in effect as the sentence finishes.
  for (let i = tags.length - 1; i >= 0; i--) {
    const hit = map[tags[i]]
    if (hit !== undefined) return hit
  }
  return fallback
}