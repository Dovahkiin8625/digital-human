// A zero-dependency async re-entry lock. React state flips too late to stop a
// second click that lands while an earlier invocation is awaiting something
// (video metadata, the upload itself): the lock is acquired synchronously
// before the task's first await and released in `finally`, so a concurrent
// caller is told it did not run instead of slipping through.
export interface ReentryGuard {
  isHeld(): boolean;
  run<T>(task: () => Promise<T>): Promise<{ ran: boolean; value: T | undefined }>;
}

export function createReentryGuard(): ReentryGuard {
  let held = false;
  return {
    isHeld: () => held,
    async run<T>(task: () => Promise<T>) {
      if (held) return { ran: false, value: undefined };
      held = true;
      try {
        return { ran: true, value: await task() };
      } finally {
        held = false;
      }
    },
  };
}
