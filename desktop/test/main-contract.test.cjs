'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const mainSource = fs.readFileSync(path.join(__dirname, '..', 'main.cjs'), 'utf8');

test('Electron main tracks a child before readiness and aborts startup on quit', () => {
  assert.match(mainSource, /let ownedChild = null;/);
  assert.match(mainSource, /let startupAbortController = null;/);
  assert.match(mainSource, /ownedChild = child;/);
  assert.match(mainSource, /startupAbortController\?\.abort\(\)/);
  assert.match(mainSource, /signal:\s*controller\.signal/);
});

test('Electron main cleans an owned backend when loading the application page fails', () => {
  assert.match(
    mainSource,
    /catch \(error\) \{[\s\S]*?await stopOwnedSession\(backendSession\);[\s\S]*?backendSession = null;/,
  );
});

test('Electron main uses exact audio permission details and splash-only privileged IPC', () => {
  assert.match(mainSource, /isAudioMediaPermissionAllowed/);
  assert.match(mainSource, /details\?\.mediaTypes/);
  assert.match(mainSource, /isTrustedSplashSender/);
  assert.match(mainSource, /pathToFileURL\(splashPath\)/);
});

test('Electron main checks signalCode and guards quit cleanup from re-entry', () => {
  assert.match(mainSource, /childExitMarker\(child\)/);
  assert.match(mainSource, /if \(stopping\) return;/);
  assert.match(mainSource, /runTaskkill/);
});

test('Electron main gives both portable-v2 workers the shared Python executable', () => {
  assert.match(
    mainSource,
    /const sharedWorkerPython = runtime\.layout === 'portable-v2' \? runtime\.python : null;/,
  );
  assert.match(mainSource, /sharedWorkerPython \|\| managedWorker\(runtime\.root, 'python-cosyvoice'\)/);
  assert.match(mainSource, /sharedWorkerPython \|\| managedWorker\(runtime\.root, 'python-feynobg'\)/);
});

test('Electron main supplies its persistent userData root to the backend environment', () => {
  assert.match(mainSource, /userDataRoot:\s*app\.getPath\('userData'\)/);
});

test('Electron main supplies an explicit LocalAppData memory root to the backend environment', () => {
  assert.match(mainSource, /resolveMemoryRoot/);
  assert.match(mainSource, /memoryRoot:\s*resolveMemoryRoot\(process\.env\)/);
  assert.doesNotMatch(
    mainSource,
    /memoryRoot:\s*app\.getPath\('userData'\)/,
  );
});
