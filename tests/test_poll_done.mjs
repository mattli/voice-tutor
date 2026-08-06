// Verifies static/study.html's poll termination logic (pollIsDone) against the
// telemetry shapes it must handle — an unsatisfiable done-condition makes every
// session poll to the 60s cap (regression seen with the telemetry-redaction work).
// Extracts the ACTUAL shipped function from study.html (between sentinels) so the
// test can't drift from what runs. Run: node tests/test_poll_done.mjs
import { readFileSync } from 'node:fs';
import assert from 'node:assert';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const html = readFileSync(
  join(dirname(fileURLToPath(import.meta.url)), '..', 'static', 'study.html'),
  'utf8',
);
const m = html.match(/\/\/ POLL_DONE_START([\s\S]*?)\/\/ POLL_DONE_END/);
assert(m, 'pollIsDone sentinel block not found in study.html');
const pollIsDone = new Function(m[1] + '\nreturn pollIsDone;')();

const R = 'recap', C = { total: 1 };

// analyzed (user spoke), matt, everything landed → done
assert.strictEqual(
  pollIsDone({ recap: R, cost: C, has_prompt: true, memory_append: 'm', analysis: 'a', expects_summary: true }, true),
  true, 'matt analyzed complete → done');

// analyzed, matt, analysis not yet written → keep polling
assert.strictEqual(
  pollIsDone({ recap: R, cost: C, has_prompt: true, memory_append: 'm', analysis: null, expects_summary: true }, true),
  false, 'matt analyzed, analysis pending → not done');

// skipped (only kickoff fired), matt → must NOT wait for summary/analysis
assert.strictEqual(
  pollIsDone({ recap: R, cost: C, has_prompt: true, memory_append: null, analysis: null, expects_summary: false }, true),
  true, 'matt skipped → done');

// non-matt analyzed: analysis/has_prompt are redacted (null/false) → must still terminate
assert.strictEqual(
  pollIsDone({ recap: R, cost: C, has_prompt: false, memory_append: 'm', analysis: null, expects_summary: true }, false),
  true, 'tester analyzed → done despite redacted fields');

// non-matt skipped → done
assert.strictEqual(
  pollIsDone({ recap: R, cost: C, has_prompt: false, memory_append: null, analysis: null, expects_summary: false }, false),
  true, 'tester skipped → done');

// recap/cost not yet landed → keep polling
assert.strictEqual(
  pollIsDone({ recap: null, cost: C, expects_summary: false }, false),
  false, 'no recap yet → not done');

// ── coverage: expects_coverage gates the wait, and only when true ──
// A false verdict must leave the client waiting for NOTHING — otherwise every
// session without a claim map polls to the 60s cap for a sidecar that is never
// coming (the failure mode expects_summary already exists to prevent).
assert.strictEqual(
  pollIsDone({ recap: R, cost: C, has_prompt: false, expects_summary: false, expects_coverage: false, coverage: null }, false),
  true, 'no coverage expected → done, never waits');

// A server that predates the field (undefined) behaves exactly as before.
assert.strictEqual(
  pollIsDone({ recap: R, cost: C, has_prompt: false, expects_summary: false }, false),
  true, 'expects_coverage absent → done');

// Expected but not yet judged → keep polling (bounded by MAX_POLL_ATTEMPTS).
assert.strictEqual(
  pollIsDone({ recap: R, cost: C, has_prompt: false, expects_summary: false, expects_coverage: true, coverage: null, coverage_status: 'pending' }, false),
  false, 'coverage expected, sidecar pending → not done');

assert.strictEqual(
  pollIsDone({ recap: R, cost: C, has_prompt: false, expects_summary: false, expects_coverage: true, coverage: { total: null, at_session: null, session: null }, coverage_status: 'ready' }, false),
  true, 'coverage landed → done');

// ── the judge FAILED: a settled outcome, not a slow one ──
// The old condition waited for coverage.session to appear, which a failed judge
// never produces — so every failure ran all 30 polls (60s of requests) after
// the last artifact had already landed. 'failed' must end the poll immediately,
// and the accumulated total is released with it (no delta, but a real number).
assert.strictEqual(
  pollIsDone({ recap: R, cost: C, has_prompt: false, expects_summary: false, expects_coverage: true, coverage: { total: { covered: 16, total: 63 }, at_session: null, session: null }, coverage_status: 'failed' }, false),
  true, 'judge failed → done, never spins to the cap');

// A pending status must hold the poll open even when a total is present —
// during teardown the server withholds `total`, but a stale/odd payload that
// still carries one must not be mistaken for a settled result.
assert.strictEqual(
  pollIsDone({ recap: R, cost: C, has_prompt: false, expects_summary: false, expects_coverage: true, coverage: { total: { covered: 16, total: 63 }, session: null }, coverage_status: 'pending' }, false),
  false, 'pending with a total present → still not done');

// Older server: no coverage_status field at all → falls back to the artifact
// test, i.e. exactly the previous behaviour rather than a premature stop.
assert.strictEqual(
  pollIsDone({ recap: R, cost: C, has_prompt: false, expects_summary: false, expects_coverage: true, coverage: { total: { covered: 16, total: 63 }, session: null } }, false),
  false, 'no coverage_status, delta missing → not done (legacy fallback)');

assert.strictEqual(
  pollIsDone({ recap: R, cost: C, has_prompt: false, expects_summary: false, expects_coverage: true, coverage: { total: { covered: 16, total: 63 }, session: { covered: 7, new_claims: 4 } } }, false),
  true, 'no coverage_status, delta present → done (legacy fallback)');

console.log('poll-done: all 15 cases passed');
