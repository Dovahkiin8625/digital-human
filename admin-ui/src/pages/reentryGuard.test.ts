import { expect, it } from "vitest";
import { createReentryGuard } from "./reentryGuard";

function delayed<T>(value: T, ms: number): Promise<T> {
  return new Promise((resolve) => setTimeout(() => resolve(value), ms));
}

it("a second call while the first task awaits never runs its task", async () => {
  const guard = createReentryGuard();
  let posts = 0;
  const probe = Promise.withResolvers<void>();

  const first = guard.run(async () => {
    await probe.promise;
    posts += 1;
  });
  // The duplicate click lands while the first duration probe is still pending.
  const second = guard.run(async () => {
    posts += 1;
  });
  expect(guard.isHeld()).toBe(true);
  probe.resolve();
  const [a, b] = await Promise.all([first, second]);
  expect(a.ran).toBe(true);
  expect(b.ran).toBe(false);
  expect(posts).toBe(1);
});

it("the lock is released after the task settles and reusable afterwards", async () => {
  const guard = createReentryGuard();
  expect(guard.isHeld()).toBe(false);
  const first = await guard.run(() => delayed(undefined, 5));
  expect(first.ran).toBe(true);
  expect(guard.isHeld()).toBe(false);
  const second = await guard.run(async () => 7);
  expect(second.ran).toBe(true);
  expect(second.value).toBe(7);
});

it("an early return or a failing-then-caught probe still releases the lock", async () => {
  const guard = createReentryGuard();
  // Models the over-15s early return: task resolves, lock released.
  await guard.run(async () => {
    await delayed(undefined, 1);
    return;
  });
  expect(guard.isHeld()).toBe(false);
  // Models the catch { proceed to POST } path: a rejected probe swallowed
  // inside the task must not strand the lock.
  let posts = 0;
  const probe = Promise.withResolvers<void>();
  const run = guard.run(async () => {
    try {
      await probe.promise;
    } catch {
      // Duration is only a convenience; the server is authoritative.
    }
    posts += 1;
  });
  probe.reject(new Error("cannot read video"));
  const result = await run;
  expect(result.ran).toBe(true);
  expect(posts).toBe(1);
  expect(guard.isHeld()).toBe(false);
});