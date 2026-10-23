import { test } from "node:test";
import assert from "node:assert/strict";
import { createReentryGuard } from "./reentryGuard.ts";

function delayed<T>(value: T, ms: number): Promise<T> {
  return new Promise((resolve) => setTimeout(() => resolve(value), ms));
}

test("a second call while the first task awaits never runs its task", async () => {
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
  assert.equal(guard.isHeld(), true);
  probe.resolve();
  const [a, b] = await Promise.all([first, second]);
  assert.equal(a.ran, true);
  assert.equal(b.ran, false);
  assert.equal(posts, 1);
});

test("the lock is released after the task settles and reusable afterwards", async () => {
  const guard = createReentryGuard();
  assert.equal(guard.isHeld(), false);
  const first = await guard.run(() => delayed(undefined, 5));
  assert.equal(first.ran, true);
  assert.equal(guard.isHeld(), false);
  const second = await guard.run(async () => 7);
  assert.equal(second.ran, true);
  assert.equal(second.value, 7);
});

test("an early return or a failing-then-caught probe still releases the lock", async () => {
  const guard = createReentryGuard();
  // Models the over-15s early return: task resolves, lock released.
  await guard.run(async () => {
    await delayed(undefined, 1);
    return;
  });
  assert.equal(guard.isHeld(), false);
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
  assert.equal(result.ran, true);
  assert.equal(posts, 1);
  assert.equal(guard.isHeld(), false);
});
