'use strict';

function timeoutSignal(timeoutMs) {
  if (typeof AbortSignal !== 'undefined' && typeof AbortSignal.timeout === 'function') {
    return AbortSignal.timeout(timeoutMs);
  }
  return undefined;
}

async function probeAvtrService(options = {}) {
  const fetchImpl = options.fetchImpl || globalThis.fetch;
  const url = options.url || 'http://127.0.0.1:7860/health';
  const requestTimeoutMs = options.requestTimeoutMs || 2000;
  try {
    const response = await fetchImpl(url, {
      method: 'GET',
      cache: 'no-store',
      signal: options.signal || timeoutSignal(requestTimeoutMs),
    });
    if (!response.ok) {
      return { healthy: false, reason: `http-${response.status}`, status: response.status };
    }
    const body = await response.json();
    if (!body || body.service !== 'avtr1-streamer') {
      return { healthy: false, reason: 'identity-mismatch', body };
    }
    const healthy = body.status === 'ready' || body.status === 'degraded';
    return {
      healthy,
      reason: healthy ? null : `status-${body.status || 'unknown'}`,
      body,
    };
  } catch (error) {
    return {
      healthy: false,
      reason: 'request-failed',
      error: error instanceof Error ? error.message : String(error),
    };
  }
}

async function waitForAvtrService(options = {}) {
  const timeoutMs = options.timeoutMs || 300000;
  const intervalMs = options.intervalMs || 500;
  const now = options.now || Date.now;
  const sleep = options.sleep || ((duration) => new Promise((resolve) => setTimeout(resolve, duration)));
  const deadline = now() + timeoutMs;
  let lastResult = null;

  while (now() < deadline) {
    if (options.signal?.aborted) {
      throw new Error('AVTR startup was cancelled');
    }
    if (options.processPoll) {
      const returnCode = options.processPoll();
      if (returnCode !== null && returnCode !== undefined) {
        throw new Error(`AVTR backend exited with code ${returnCode}`);
      }
    }
    lastResult = await probeAvtrService(options);
    if (lastResult.healthy) return lastResult;
    await sleep(intervalMs);
  }

  const detail = lastResult?.error || lastResult?.reason || 'no response';
  throw new Error(`AVTR backend did not become ready within ${timeoutMs} ms (${detail})`);
}

module.exports = {
  probeAvtrService,
  waitForAvtrService,
};
