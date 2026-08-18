'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const { EventEmitter } = require('node:events');
const path = require('node:path');

const {
  buildBackendEnvironment,
  childExitMarker,
  ensureBackend,
  isAudioMediaPermissionAllowed,
  isTrustedSplashSender,
  resolveMemoryRoot,
  runTaskkill,
  stopBackend,
} = require('../lib/backend-supervisor.cjs');

test('ensureBackend attaches to a healthy external service without spawning', async () => {
  let spawnCalls = 0;
  const session = await ensureBackend({
    probe: async () => ({ healthy: true }),
    spawnBackend: () => { spawnCalls += 1; },
  });

  assert.equal(session.ownership, 'external');
  assert.equal(spawnCalls, 0);
});

test('ensureBackend starts one orchestrator and waits for identity health', async () => {
  const child = { pid: 4242, poll: () => null };
  let waitChild;
  const session = await ensureBackend({
    probe: async () => ({ healthy: false }),
    spawnBackend: () => child,
    waitForReady: async (received) => {
      waitChild = received;
      return { healthy: true };
    },
  });

  assert.equal(session.ownership, 'desktop');
  assert.equal(session.child, child);
  assert.equal(waitChild, child);
});

test('ensureBackend never spawns after startup cancellation', async () => {
  const controller = new AbortController();
  controller.abort();
  let spawnCalls = 0;

  await assert.rejects(
    ensureBackend({
      signal: controller.signal,
      probe: async () => ({ healthy: false }),
      spawnBackend: () => { spawnCalls += 1; },
    }),
    (error) => error?.name === 'AbortError',
  );

  assert.equal(spawnCalls, 0);
});

test('ensureBackend forwards cancellation and cleans a child spawned during startup', async () => {
  const controller = new AbortController();
  const child = { pid: 4343 };
  let cleanedChild = null;

  await assert.rejects(
    ensureBackend({
      signal: controller.signal,
      probe: async () => ({ healthy: false }),
      spawnBackend: () => child,
      waitForReady: async (received, context) => {
        assert.equal(received, child);
        assert.equal(context.signal, controller.signal);
        controller.abort();
        return { healthy: true };
      },
      onStartupFailure: async (received) => { cleanedChild = received; },
    }),
    (error) => error?.name === 'AbortError',
  );

  assert.equal(cleanedChild, child);
});

test('stopBackend never stops an attached external service', async () => {
  const calls = [];
  const result = await stopBackend(
    { ownership: 'external' },
    {
      requestStop: async () => calls.push('request'),
      waitForExit: async () => { calls.push('wait'); return true; },
      killTree: async () => calls.push('kill'),
    },
  );

  assert.equal(result.stopped, false);
  assert.deepEqual(calls, []);
});

test('stopBackend requests cooperative shutdown before force-killing an owned tree', async () => {
  const calls = [];
  const child = { pid: 777 };
  let waits = 0;
  const result = await stopBackend(
    { ownership: 'desktop', child },
    {
      requestStop: async () => calls.push('request'),
      waitForExit: async () => {
        waits += 1;
        calls.push(waits === 1 ? 'wait' : 'confirm');
        return waits > 1;
      },
      killTree: async (received) => calls.push(`kill:${received.pid}`),
    },
  );

  assert.equal(result.forced, true);
  assert.deepEqual(calls, ['request', 'wait', 'kill:777', 'confirm']);
});

test('stopBackend coalesces concurrent shutdown requests for the same child', async () => {
  const child = { pid: 778 };
  let requestCalls = 0;
  let waitCalls = 0;
  const options = {
    requestStop: async () => { requestCalls += 1; },
    waitForExit: async () => { waitCalls += 1; return true; },
    killTree: async () => assert.fail('force kill should not run'),
  };

  const [first, second] = await Promise.all([
    stopBackend({ ownership: 'desktop', child }, options),
    stopBackend({ ownership: 'desktop', child }, options),
  ]);

  assert.deepEqual(first, second);
  assert.equal(requestCalls, 1);
  assert.equal(waitCalls, 1);
});

test('stopBackend surfaces a failed force kill when the child remains alive', async () => {
  const child = { pid: 779 };
  let waits = 0;

  await assert.rejects(
    stopBackend(
      { ownership: 'desktop', child },
      {
        requestStop: async () => {},
        waitForExit: async () => { waits += 1; return false; },
        killTree: async () => { throw new Error('taskkill exited with code 5: Access is denied'); },
      },
    ),
    /taskkill exited with code 5.*Access is denied/,
  );

  assert.equal(waits, 2);
});

test('childExitMarker notices both exit codes and signal termination', () => {
  assert.equal(childExitMarker({ exitCode: null, signalCode: null }), null);
  assert.equal(childExitMarker({ exitCode: 7, signalCode: null }), 7);
  assert.equal(childExitMarker({ exitCode: null, signalCode: 'SIGTERM' }), 'signal SIGTERM');
});

test('runTaskkill rejects a non-zero taskkill result with stderr', async () => {
  const killer = new EventEmitter();
  killer.stderr = new EventEmitter();
  const spawnProcess = () => killer;
  const pending = runTaskkill(
    { pid: 9090 },
    { spawnProcess, executable: 'taskkill.exe' },
  );
  queueMicrotask(() => {
    killer.stderr.emit('data', Buffer.from('Access is denied', 'utf8'));
    killer.emit('close', 5, null);
  });

  await assert.rejects(pending, /taskkill exited with code 5.*Access is denied/);
});

test('managed runtime environment excludes host Python and virtualenv state', () => {
  const environment = buildBackendEnvironment({
    baseEnvironment: {
      PATH: 'C:\\Windows\\System32',
      PYTHONHOME: 'C:\\HostPython',
      PYTHONPATH: 'C:\\HostModules',
      PYTHONSTARTUP: 'C:\\host-startup.py',
      VIRTUAL_ENV: 'C:\\host-venv',
      CONDA_PREFIX: 'C:\\conda',
      AVTR1_MAIN_PYTHONPATH: 'C:\\stale-main-layer',
      AVTR1_COSYVOICE_PYTHONPATH: 'C:\\stale-cosyvoice-layer',
      AVTR1_FEYNOBG_PYTHONPATH: 'C:\\stale-feynobg-layer',
    },
    runtime: {
      mode: 'managed',
      root: 'D:\\AVTR-Runtime',
      source: 'D:\\AVTR-Runtime\\src',
    },
    stopFile: 'D:\\Temp\\desktop.stop',
    delimiter: ';',
  });

  assert.equal(environment.PATH, 'C:\\Windows\\System32');
  assert.equal(environment.PYTHONPATH, 'D:\\AVTR-Runtime\\src');
  assert.equal(environment.PYTHONHOME, undefined);
  assert.equal(environment.PYTHONSTARTUP, undefined);
  assert.equal(environment.VIRTUAL_ENV, undefined);
  assert.equal(environment.CONDA_PREFIX, undefined);
  assert.equal(environment.AVTR1_MAIN_PYTHONPATH, undefined);
  assert.equal(environment.AVTR1_COSYVOICE_PYTHONPATH, undefined);
  assert.equal(environment.AVTR1_FEYNOBG_PYTHONPATH, undefined);
  assert.equal(environment.AVTR1_RUNTIME_ROOT, 'D:\\AVTR-Runtime');
  assert.equal(environment.AVTR1_DESKTOP_STOP_FILE, 'D:\\Temp\\desktop.stop');
});

test('development runtime may retain an explicitly configured host PYTHONPATH', () => {
  const environment = buildBackendEnvironment({
    baseEnvironment: { PYTHONPATH: 'C:\\DeveloperModules' },
    runtime: {
      mode: 'development',
      root: 'F:\\AVTR-1',
      source: 'F:\\AVTR-1\\src',
    },
    stopFile: 'D:\\Temp\\desktop.stop',
    delimiter: ';',
  });

  assert.equal(environment.PYTHONPATH, 'F:\\AVTR-1\\src;C:\\DeveloperModules');
});

test('portable-v2 environment routes one Python through three isolated package layers', () => {
  const runtime = {
    mode: 'managed',
    layout: 'portable-v2',
    root: 'D:\\DigiBox\\avtr-runtime',
    source: 'D:\\DigiBox\\avtr-runtime\\src',
    pythonPath: {
      main: 'D:\\DigiBox\\avtr-runtime\\packages\\main;D:\\DigiBox\\avtr-runtime\\packages\\shared;D:\\DigiBox\\avtr-runtime\\src',
      cosyvoice: 'D:\\DigiBox\\avtr-runtime\\packages\\cosyvoice;D:\\DigiBox\\avtr-runtime\\packages\\shared;D:\\DigiBox\\avtr-runtime\\src',
      feynobg: 'D:\\DigiBox\\avtr-runtime\\packages\\feynobg;D:\\DigiBox\\avtr-runtime\\packages\\shared;D:\\DigiBox\\avtr-runtime\\src',
    },
  };

  const environment = buildBackendEnvironment({
    baseEnvironment: { PYTHONPATH: 'C:\\HostModules' },
    runtime,
    stopFile: 'D:\\Temp\\desktop.stop',
    delimiter: ';',
  });

  assert.equal(environment.PYTHONPATH, runtime.pythonPath.main);
  assert.equal(environment.AVTR1_MAIN_PYTHONPATH, runtime.pythonPath.main);
  assert.equal(environment.AVTR1_COSYVOICE_PYTHONPATH, runtime.pythonPath.cosyvoice);
  assert.equal(environment.AVTR1_FEYNOBG_PYTHONPATH, runtime.pythonPath.feynobg);
});

test('portable-v1 and portable-v2 keep writable assets and cloned voices under userData', () => {
  const userDataRoot = 'C:\\Users\\Alice\\AppData\\Roaming\\DigiBox';
  const runtimes = [
    {
      mode: 'managed',
      layout: 'portable-v1',
      root: 'D:\\DigiBox-v1\\avtr-runtime',
      source: 'D:\\DigiBox-v1\\avtr-runtime\\src',
    },
    {
      mode: 'managed',
      layout: 'portable-v2',
      root: 'D:\\DigiBox-v2\\avtr-runtime',
      source: 'D:\\DigiBox-v2\\avtr-runtime\\src',
      pythonPath: {
        main: 'D:\\DigiBox-v2\\avtr-runtime\\packages\\main',
        cosyvoice: 'D:\\DigiBox-v2\\avtr-runtime\\packages\\cosyvoice',
        feynobg: 'D:\\DigiBox-v2\\avtr-runtime\\packages\\feynobg',
      },
    },
  ];

  for (const runtime of runtimes) {
    const environment = buildBackendEnvironment({
      baseEnvironment: {
        AVTR1_USER_ASSETS_ROOT: `${runtime.root}\\artifacts\\main\\user_assets`,
        AVTR1_COSYVOICE_SPEAKER_CACHE: `${runtime.root}\\models\\spk2info.pt`,
      },
      runtime,
      stopFile: 'D:\\Temp\\desktop.stop',
      userDataRoot,
    });

    assert.equal(
      environment.AVTR1_USER_ASSETS_ROOT,
      path.win32.join(userDataRoot, 'user_assets'),
    );
    assert.equal(
      environment.AVTR1_COSYVOICE_SPEAKER_CACHE,
      path.win32.join(userDataRoot, 'cosyvoice', 'spk2info.pt'),
    );
    assert.equal(path.win32.isAbsolute(environment.AVTR1_USER_ASSETS_ROOT), true);
    assert.equal(path.win32.isAbsolute(environment.AVTR1_COSYVOICE_SPEAKER_CACHE), true);
    assert.equal(environment.AVTR1_USER_ASSETS_ROOT.startsWith(runtime.root), false);
    assert.equal(environment.AVTR1_COSYVOICE_SPEAKER_CACHE.startsWith(runtime.root), false);
  }
});

test('persistent data routing replaces case-insensitive stale Windows environment keys', () => {
  const environment = buildBackendEnvironment({
    baseEnvironment: {
      avtr1_user_assets_root: 'D:\\Runtime\\user_assets',
      avtr1_cosyvoice_speaker_cache: 'D:\\Runtime\\spk2info.pt',
    },
    runtime: {
      mode: 'managed',
      root: 'D:\\Runtime',
      source: 'D:\\Runtime\\src',
    },
    stopFile: 'D:\\Temp\\desktop.stop',
    userDataRoot: 'C:\\Users\\Alice\\AppData\\Roaming\\DigiBox',
  });

  const keys = Object.keys(environment).map((key) => key.toUpperCase());
  assert.equal(keys.filter((key) => key === 'AVTR1_USER_ASSETS_ROOT').length, 1);
  assert.equal(keys.filter((key) => key === 'AVTR1_COSYVOICE_SPEAKER_CACHE').length, 1);
});

test('memory root is derived only from Windows LocalAppData', () => {
  assert.equal(
    resolveMemoryRoot({ LOCALAPPDATA: 'C:\\Users\\Alice\\AppData\\Local' }),
    'C:\\Users\\Alice\\AppData\\Local\\DigiBox\\memory',
  );
  assert.equal(resolveMemoryRoot({ LOCALAPPDATA: 'relative\\Local' }), null);
  assert.equal(
    resolveMemoryRoot({ APPDATA: 'C:\\Users\\Alice\\AppData\\Roaming' }),
    null,
  );
});

test('unsafe LocalAppData disables memory and removes a stale inherited root', () => {
  const environment = buildBackendEnvironment({
    baseEnvironment: {
      avtr1_memory_root: 'D:\\DigiBox-Runtime\\memory',
    },
    runtime: {
      mode: 'managed',
      root: 'D:\\DigiBox-Runtime',
      source: 'D:\\DigiBox-Runtime\\src',
    },
    stopFile: 'D:\\Temp\\desktop.stop',
    memoryRoot: resolveMemoryRoot({ LOCALAPPDATA: 'relative\\Local' }),
  });

  const memoryKeys = Object.keys(environment)
    .filter((key) => key.toUpperCase() === 'AVTR1_MEMORY_ROOT');
  assert.deepEqual(memoryKeys, []);
});

test('backend environment replaces stale memory roots with the explicit LocalAppData path', () => {
  const memoryRoot = 'C:\\Users\\Alice\\AppData\\Local\\DigiBox\\memory';
  const environment = buildBackendEnvironment({
    baseEnvironment: {
      avtr1_memory_root: 'D:\\DigiBox-Runtime\\memory',
    },
    runtime: {
      mode: 'managed',
      root: 'D:\\DigiBox-Runtime',
      source: 'D:\\DigiBox-Runtime\\src',
    },
    stopFile: 'D:\\Temp\\desktop.stop',
    memoryRoot,
  });

  const memoryKeys = Object.keys(environment)
    .filter((key) => key.toUpperCase() === 'AVTR1_MEMORY_ROOT');
  assert.deepEqual(memoryKeys, ['AVTR1_MEMORY_ROOT']);
  assert.equal(environment.AVTR1_MEMORY_ROOT, memoryRoot);
  assert.equal(environment.AVTR1_MEMORY_ROOT.startsWith('D:\\DigiBox-Runtime'), false);
});

test('media permission only allows microphone audio on the exact app origin', () => {
  const appOrigin = 'http://127.0.0.1:7860';
  assert.equal(isAudioMediaPermissionAllowed(
    'media', `${appOrigin}/`, { mediaTypes: ['audio'] }, appOrigin,
  ), true);
  assert.equal(isAudioMediaPermissionAllowed(
    'media', `${appOrigin}/`, { mediaTypes: ['video'] }, appOrigin,
  ), false);
  assert.equal(isAudioMediaPermissionAllowed(
    'media', `${appOrigin}/`, { mediaTypes: ['audio', 'video'] }, appOrigin,
  ), false);
  assert.equal(isAudioMediaPermissionAllowed(
    'media', 'http://localhost:7860/', { mediaTypes: ['audio'] }, appOrigin,
  ), false);
  assert.equal(isAudioMediaPermissionAllowed(
    'media', `${appOrigin}/`, { mediaType: 'audio' }, appOrigin,
  ), true);
  assert.equal(isAudioMediaPermissionAllowed(
    'media', `${appOrigin}/`, {}, appOrigin,
  ), false);
});

test('privileged IPC requires both the main webContents and the local splash frame', () => {
  const mainWindow = {
    isDestroyed: () => false,
    webContents: { id: 42 },
  };
  const splashUrl = 'file:///F:/AVTR-1/desktop/splash.html';
  const event = {
    sender: { id: 42 },
    senderFrame: { url: splashUrl },
  };

  assert.equal(isTrustedSplashSender(event, mainWindow, splashUrl), true);
  assert.equal(isTrustedSplashSender(
    { ...event, senderFrame: { url: 'http://127.0.0.1:7860/' } },
    mainWindow,
    splashUrl,
  ), false);
  assert.equal(isTrustedSplashSender(
    { ...event, sender: { id: 99 } },
    mainWindow,
    splashUrl,
  ), false);
});
