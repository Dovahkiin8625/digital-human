/**
 * Human-readable byte count.
 *
 * Lives here rather than in a page because two pages that share a
 * component would otherwise have to import it from each other -- and a
 * page importing from a page is a cycle waiting to happen.
 */
export function formatBytes(b: number): string {
  if (b < 1024) return `${b} B`;
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB`;
  return `${(b / 1024 / 1024).toFixed(1)} MB`;
}
