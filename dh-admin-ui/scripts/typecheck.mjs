/**
 * Typecheck our code, without typechecking the Cubism SDK's source.
 *
 * The SDK compiles itself with `noImplicitAny` only -- not `strict`, not
 * `strictNullChecks` -- and its Framework source has ~650 null-assignability
 * errors under those settings. Those are not ours to fix. Relaxing our own
 * strictness to accommodate them would weaken every check we actually want
 * on our own files.
 *
 * So: run tsc across the whole program (the SDK is in it via the @framework
 * path alias), then fail only on diagnostics that originate outside the SDK.
 */
import { spawnSync } from "node:child_process";

const SDK_MARKER = "CubismSdkForWeb";

const proc = spawnSync("npx", ["tsc", "--noEmit", "--pretty", "false"], {
  encoding: "utf8",
  shell: true,
});

if (proc.error) {
  console.error("failed to run tsc:", proc.error.message);
  process.exit(1);
}

const lines = `${proc.stdout ?? ""}`.split("\n");
const errors = lines.filter((l) => /error TS\d+:/.test(l));
const ours = errors.filter((l) => !l.includes(SDK_MARKER));

if (ours.length > 0) {
  console.error(`${ours.length} type error(s) in our code:`);
  for (const l of ours) console.error("  " + l);
  process.exit(1);
}

console.log(`typecheck ok (${errors.length} SDK-internal diagnostics ignored)`);