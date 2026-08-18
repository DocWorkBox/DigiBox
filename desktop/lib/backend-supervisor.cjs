'use strict';

const path = require('node:path');

const ACTIVE_STOPS = new WeakMap();
const MANAGED_PYTHON_ENVIRONMENT_KEYS = new Set([
  'CONDA_DEFAULT_ENV',
  'CONDA_PREFIX',
  'CONDA_PREFIX_1',
  'CONDA_PROMPT_MODIFIER',
  'AVTR1_COSYVOICE_PYTHONPATH',
  'AVTR1_FEYNOBG_PYTHONPATH',
  'AVTR1_MAIN_PYTHONPATH',
  'PYTHONHOME',
  'PYTHONPATH',
  'PYTHONSTARTUP',
  'VIRTUAL_ENV',
  'VIRTUAL_ENV_PROMPT',
]);

function cancellationError(signal) {
  const reason = signal?.reason;
  if (reason instanceof Error && reason.name === 'AbortError') return reason;
  const message = reason instanceof Error
    ? reason.message
    : (typeof reason === 'string' && reason ? reason : 'AVTR startup was cancelled');
  const error = new Error(message, reason instanceof Error ? { cause: reason } : undefined);
  error.name = 'AbortError';
  return error;
}

function throwIfCancelled(signal) {
  if (signal?.aborted) throw cancellationError(signal);
}

async function ensureBackend(options = {}) {
  if (typeof options.probe !== 'function') throw new TypeError('probe is required');
  if (typeof options.spawnBackend !== 'function') throw new TypeError('spawnBackend is required');

  throwIfCancelled(options.signal);
  const existing = await options.probe();
  throwIfCancelled(options.signal);
  if (existing?.healthy) {
    return {
      ownership: 'external',
      child: null,
      health: existing,
    };
  }

  const child = options.spawnBackend();
  if (!child || typeof child.pid !== 'number') {
    throw new Error('Backend launcher did not return a child process');
  }
  if (typeof options.waitForReady !== 'function') {
    throw new TypeError('waitForReady is required when the backend is not already running');
  }
  try {
    const health = await options.waitForReady(child, { signal: options.signal });
    throwIfCancelled(options.signal);
    if (!health?.healthy) throw new Error('Backend readiness check failed');
    return {
      ownership: 'desktop',
      child,
      health,
    };
  } catch (error) {
    if (typeof options.onStartupFailure === 'function') {
      await options.onStartupFailure(child, error);
    }
    throw error;
  }
}

function childExitMarker(child) {
  if (!child) return null;
  if (child.exitCode !== null && child.exitCode !== undefined) return child.exitCode;
  if (child.signalCode !== null && child.signalCode !== undefined) {
    return `signal ${child.signalCode}`;
  }
  return null;
}

function stopBackend(session, options = {}) {
  if (!session || session.ownership !== 'desktop' || !session.child) {
    return Promise.resolve({ stopped: false, forced: false });
  }
  if (typeof options.requestStop !== 'function') throw new TypeError('requestStop is required');
  if (typeof options.waitForExit !== 'function') throw new TypeError('waitForExit is required');
  if (typeof options.killTree !== 'function') throw new TypeError('killTree is required');

  const child = session.child;
  const active = ACTIVE_STOPS.get(child);
  if (active) return active;

  const operation = (async () => {
    if (childExitMarker(child) !== null) return { stopped: true, forced: false };

    await options.requestStop(child);
    const exited = await options.waitForExit(child);
    if (exited) return { stopped: true, forced: false };

    let killError = null;
    try {
      await options.killTree(child);
    } catch (error) {
      killError = error;
    }

    const confirmExit = options.confirmExit || options.waitForExit;
    const forcedExit = await confirmExit(child);
    if (!forcedExit) {
      if (killError) throw killError;
      throw new Error(`Backend process tree ${child.pid} remained alive after taskkill`);
    }
    return { stopped: true, forced: true };
  })();

  ACTIVE_STOPS.set(child, operation);
  const clear = () => {
    if (ACTIVE_STOPS.get(child) === operation) ACTIVE_STOPS.delete(child);
  };
  operation.then(clear, clear);
  return operation;
}

function runTaskkill(child, options = {}) {
  if (!child || typeof child.pid !== 'number') {
    return Promise.reject(new TypeError('A child process with a numeric pid is required'));
  }
  const spawnProcess = options.spawnProcess;
  if (typeof spawnProcess !== 'function') {
    return Promise.reject(new TypeError('spawnProcess is required'));
  }
  const executable = options.executable || 'C:\\Windows\\System32\\taskkill.exe';
  return new Promise((resolve, reject) => {
    let settled = false;
    let stderr = '';
    const finish = (callback, value) => {
      if (settled) return;
      settled = true;
      callback(value);
    };
    let killer;
    try {
      killer = spawnProcess(executable, ['/PID', String(child.pid), '/T', '/F'], {
        windowsHide: true,
        shell: false,
        stdio: ['ignore', 'ignore', 'pipe'],
      });
    } catch (error) {
      reject(error);
      return;
    }
    killer.stderr?.on('data', (chunk) => { stderr += String(chunk); });
    killer.once('error', (error) => finish(reject, error));
    killer.once('close', (code, signalName) => {
      if (code === 0) {
        finish(resolve, { code, signal: signalName || null });
        return;
      }
      const detail = stderr.trim();
      const outcome = code === null ? `signal ${signalName || 'unknown'}` : `code ${code}`;
      finish(
        reject,
        new Error(`taskkill exited with ${outcome}${detail ? `: ${detail}` : ''}`),
      );
    });
  });
}

function setWindowsEnvironmentValue(environment, key, value) {
  for (const existing of Object.keys(environment)) {
    if (existing.toUpperCase() === key) delete environment[existing];
  }
  environment[key] = value;
}

function resolveMemoryRoot(environment = {}) {
  const localAppData = typeof environment.LOCALAPPDATA === 'string'
    ? environment.LOCALAPPDATA.trim()
    : '';
  if (!localAppData || !path.win32.isAbsolute(localAppData)) {
    return null;
  }
  return path.win32.join(path.win32.resolve(localAppData), 'DigiBox', 'memory');
}

function buildBackendEnvironment(options = {}) {
  const runtime = options.runtime;
  if (!runtime?.root || !runtime?.source) throw new TypeError('runtime root and source are required');
  const environment = { ...(options.baseEnvironment || {}) };
  const delimiter = options.delimiter || ';';
  const hostPythonPath = environment.PYTHONPATH;

  if (runtime.mode === 'managed') {
    for (const key of Object.keys(environment)) {
      if (MANAGED_PYTHON_ENVIRONMENT_KEYS.has(key.toUpperCase())) delete environment[key];
    }
  }

  environment.AVTR1_DESKTOP_STOP_FILE = options.stopFile;
  environment.AVTR1_RUNTIME_ROOT = runtime.root;
  environment.AVTR1_APP_ROOT = runtime.root;
  environment.AVTR1_SINGLE_ENV = '1';
  environment.PYTHONNOUSERSITE = '1';
  environment.PYTHONUNBUFFERED = '1';
  environment.PYTHONUTF8 = '1';
  if (options.userDataRoot !== undefined) {
    if (
      typeof options.userDataRoot !== 'string'
      || !path.win32.isAbsolute(options.userDataRoot)
    ) {
      throw new TypeError('userDataRoot must be an absolute Windows path');
    }
    const userDataRoot = path.win32.resolve(options.userDataRoot);
    setWindowsEnvironmentValue(
      environment,
      'AVTR1_USER_ASSETS_ROOT',
      path.win32.join(userDataRoot, 'user_assets'),
    );
    setWindowsEnvironmentValue(
      environment,
      'AVTR1_COSYVOICE_SPEAKER_CACHE',
      path.win32.join(userDataRoot, 'cosyvoice', 'spk2info.pt'),
    );
  }
  for (const existing of Object.keys(environment)) {
    if (existing.toUpperCase() === 'AVTR1_MEMORY_ROOT') delete environment[existing];
  }
  if (options.memoryRoot !== undefined && options.memoryRoot !== null) {
    if (
      typeof options.memoryRoot !== 'string'
      || !path.win32.isAbsolute(options.memoryRoot)
    ) {
      throw new TypeError('memoryRoot must be an absolute Windows path');
    }
    setWindowsEnvironmentValue(
      environment,
      'AVTR1_MEMORY_ROOT',
      path.win32.resolve(options.memoryRoot),
    );
  }
  if (runtime.pythonPath) {
    environment.AVTR1_MAIN_PYTHONPATH = runtime.pythonPath.main;
    environment.AVTR1_COSYVOICE_PYTHONPATH = runtime.pythonPath.cosyvoice;
    environment.AVTR1_FEYNOBG_PYTHONPATH = runtime.pythonPath.feynobg;
    environment.PYTHONPATH = runtime.pythonPath.main;
  } else {
    environment.PYTHONPATH = runtime.mode === 'managed'
      ? runtime.source
      : [runtime.source, hostPythonPath].filter(Boolean).join(delimiter);
  }
  return environment;
}

function isAudioMediaPermissionAllowed(permission, requestingUrl, details, appOrigin) {
  if (permission !== 'media') return false;
  try {
    if (new URL(requestingUrl).origin !== appOrigin) return false;
  } catch {
    return false;
  }
  const mediaTypes = Array.isArray(details?.mediaTypes)
    ? details.mediaTypes
    : (typeof details?.mediaType === 'string' ? [details.mediaType] : []);
  return mediaTypes.length > 0 && mediaTypes.every((mediaType) => mediaType === 'audio');
}

function isTrustedSplashSender(event, mainWindow, splashUrl) {
  try {
    return Boolean(
      mainWindow
      && !mainWindow.isDestroyed()
      && event?.sender?.id === mainWindow.webContents.id
      && event?.senderFrame?.url === splashUrl,
    );
  } catch {
    return false;
  }
}

module.exports = {
  buildBackendEnvironment,
  childExitMarker,
  ensureBackend,
  isAudioMediaPermissionAllowed,
  isTrustedSplashSender,
  resolveMemoryRoot,
  runTaskkill,
  stopBackend,
};
