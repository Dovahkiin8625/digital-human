/**
 * A value that may be set before its target exists.
 *
 * The React component sets the expression on its first render, but the
 * model takes seconds to load -- the core, the moc3 and every texture.
 * A plain field would drop the value on the floor, and the symptom is
 * nasty: no error, no blank canvas, just an expression the pet never
 * wears.
 */
export interface Deferred<T> {
  /** Set the value, and apply it if there is a target yet. */
  set(value: T): void;
  /** Attach the target, and hand it the value it missed. */
  bind(apply: (value: T) => void): void;
  /** The current value. */
  get(): T;
}

export function createDeferred<T>(initial: T): Deferred<T> {
  let value = initial;
  let apply: ((value: T) => void) | null = null;
  return {
    set(next: T) {
      value = next;
      apply?.(next);
    },
    bind(fn: (value: T) => void) {
      apply = fn;
      fn(value);
    },
    get() {
      return value;
    },
  };
}
