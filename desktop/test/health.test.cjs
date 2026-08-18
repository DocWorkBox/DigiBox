'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');

const {
  probeAvtrService,
  waitForAvtrService,
} = require('../lib/health.cjs');

function response(status, body) {
  return {
    ok: status >= 200 && status < 300,
    status,
    async json() { return body; },
  };
}

test('probeAvtrService rejects an unrelated process on the expected port', async () => {
  const result = await probeAvtrService({
    url: 'http://127.0.0.1:7860/health',
    fetchImpl: async () => response(200, { status: 'ok' }),
  });

  assert.equal(result.healthy, false);
  assert.equal(result.reason, 'identity-mismatch');
});

test('waitForAvtrService retries until the AVTR streamer identity is ready', async () => {
  let calls = 0;
  const result = await waitForAvtrService({
    url: 'http://127.0.0.1:7860/health',
    timeoutMs: 100,
    intervalMs: 1,
    now: (() => { let value = 0; return () => value += 5; })(),
    sleep: async () => {},
    fetchImpl: async () => {
      calls += 1;
      if (calls < 3) throw new Error('ECONNREFUSED');
      return response(200, { service: 'avtr1-streamer', status: 'ready' });
    },
  });

  assert.equal(result.healthy, true);
  assert.equal(calls, 3);
});

test('waitForAvtrService stops promptly when the child exits', async () => {
  await assert.rejects(
    waitForAvtrService({
      url: 'http://127.0.0.1:7860/health',
      timeoutMs: 100,
      intervalMs: 1,
      now: (() => { let value = 0; return () => value += 5; })(),
      sleep: async () => {},
      fetchImpl: async () => { throw new Error('ECONNREFUSED'); },
      processPoll: () => 7,
    }),
    /exited with code 7/,
  );
});
