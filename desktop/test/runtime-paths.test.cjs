'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');

const {
  inspectRuntimeRoot,
  resolveRuntimeRoot,
} = require('../lib/runtime-paths.cjs');

function fakeExists(entries) {
  const normalized = new Set(entries.map((entry) => path.win32.normalize(entry).toLowerCase()));
  return (candidate) => normalized.has(path.win32.normalize(candidate).toLowerCase());
}

function portableV1Manifest() {
  return {
    schemaVersion: 1,
    runtimeId: 'digibox-portable-v1-test',
    python: { main: '3.12.9', cosyvoice: '3.10.17', feynobg: '3.12.10' },
    tensorrt: { engines: [] },
  };
}

function portableV2Manifest() {
  return {
    schemaVersion: 2,
    layout: 'portable-v2',
    runtimeId: 'digibox-portable-v2-test',
    paths: {
      python: 'python/python.exe',
      orchestrator: 'scripts/run_local_stream.py',
      source: 'src',
      artifacts: 'artifacts/main',
      models: 'models',
    },
    python: {
      version: '3.12.9',
      packageLayers: {
        main: ['packages/main', 'packages/shared', 'src'],
        cosyvoice: [
          'packages/cosyvoice',
          'packages/shared',
          'third_party/CosyVoice',
          'third_party/CosyVoice/third_party/Matcha-TTS',
          'src',
        ],
        feynobg: ['packages/feynobg', 'packages/shared', 'src'],
      },
    },
    components: {
      dependenciesIncluded: true,
      modelsIncluded: true,
      frontendVendorIncluded: true,
      tensorRtBuildInputsIncluded: true,
    },
    tensorrt: { engines: [] },
  };
}

test('inspectRuntimeRoot accepts the existing development venv layout', () => {
  const root = 'F:\\AVTR-1';
  const existsSync = fakeExists([
    `${root}\\.venv\\Scripts\\python.exe`,
    `${root}\\scripts\\run_local_stream.py`,
    `${root}\\src`,
    `${root}\\artifacts\\main`,
  ]);

  const result = inspectRuntimeRoot(root, { existsSync, platform: 'win32' });

  assert.equal(result.valid, true);
  assert.equal(result.mode, 'development');
  assert.equal(result.python, path.win32.join(root, '.venv', 'Scripts', 'python.exe'));
  assert.deepEqual(result.missing, []);
});

test('inspectRuntimeRoot accepts a relocatable managed runtime without pyvenv.cfg', () => {
  const root = 'D:\\AVTR-1 Runtime';
  const existsSync = fakeExists([
    `${root}\\python-main\\python.exe`,
    `${root}\\runtime-manifest.json`,
    `${root}\\scripts\\run_local_stream.py`,
    `${root}\\src`,
    `${root}\\artifacts\\main`,
  ]);

  const result = inspectRuntimeRoot(root, {
    existsSync,
    readFileSync: () => JSON.stringify(portableV1Manifest()),
    platform: 'win32',
  });

  assert.equal(result.valid, true);
  assert.equal(result.mode, 'managed');
  assert.equal(result.layout, 'portable-v1');
  assert.equal(result.python, path.win32.join(root, 'python-main', 'python.exe'));
  assert.equal(result.manifest, path.win32.join(root, 'runtime-manifest.json'));
  assert.equal(result.packageLayers, null);
  assert.equal(result.pythonPath, null);
});

test('inspectRuntimeRoot consumes portable-v2 manifest paths and Python layers', () => {
  const root = 'D:\\DigiBox Runtime';
  const entries = [
    'python\\python.exe',
    'runtime-manifest.json',
    'scripts\\run_local_stream.py',
    'src',
    'artifacts\\main',
    'models',
    'packages\\main',
    'packages\\shared',
    'packages\\cosyvoice',
    'packages\\feynobg',
    'third_party\\CosyVoice',
    'third_party\\CosyVoice\\third_party\\Matcha-TTS',
  ].map((entry) => path.win32.join(root, entry));

  const result = inspectRuntimeRoot(root, {
    existsSync: fakeExists(entries),
    readFileSync: () => JSON.stringify(portableV2Manifest()),
    platform: 'win32',
  });

  assert.equal(result.valid, true);
  assert.equal(result.mode, 'managed');
  assert.equal(result.layout, 'portable-v2');
  assert.equal(result.python, path.win32.join(root, 'python', 'python.exe'));
  assert.equal(result.script, path.win32.join(root, 'scripts', 'run_local_stream.py'));
  assert.deepEqual(result.packageLayers.main, [
    path.win32.join(root, 'packages', 'main'),
    path.win32.join(root, 'packages', 'shared'),
    path.win32.join(root, 'src'),
  ]);
  assert.deepEqual(result.packageLayers.cosyvoice, [
    path.win32.join(root, 'packages', 'cosyvoice'),
    path.win32.join(root, 'packages', 'shared'),
    path.win32.join(root, 'third_party', 'CosyVoice'),
    path.win32.join(root, 'third_party', 'CosyVoice', 'third_party', 'Matcha-TTS'),
    path.win32.join(root, 'src'),
  ]);
  assert.equal(result.pythonPath.main, result.packageLayers.main.join(';'));
  assert.equal(result.pythonPath.cosyvoice, result.packageLayers.cosyvoice.join(';'));
  assert.equal(result.pythonPath.feynobg, result.packageLayers.feynobg.join(';'));
});

test('inspectRuntimeRoot rejects portable-v2 when a declared package layer is absent', () => {
  const root = 'D:\\DigiBox Runtime';
  const entries = [
    'python\\python.exe',
    'runtime-manifest.json',
    'scripts\\run_local_stream.py',
    'src',
    'artifacts\\main',
    'models',
    'packages\\main',
    'packages\\shared',
    'packages\\cosyvoice',
    'third_party\\CosyVoice',
    'third_party\\CosyVoice\\third_party\\Matcha-TTS',
  ].map((entry) => path.win32.join(root, entry));

  const result = inspectRuntimeRoot(root, {
    existsSync: fakeExists(entries),
    readFileSync: () => JSON.stringify(portableV2Manifest()),
    platform: 'win32',
  });

  assert.equal(result.valid, false);
  assert.ok(result.missing.some((item) => (
    item.includes('python.packageLayers.feynobg') && item.includes('packages\\feynobg')
  )));
});

test('inspectRuntimeRoot reports exact missing components', () => {
  const root = 'D:\\broken';
  const result = inspectRuntimeRoot(root, {
    existsSync: () => false,
    platform: 'win32',
  });

  assert.equal(result.valid, false);
  assert.ok(result.missing.includes('Python runtime'));
  assert.ok(result.missing.includes('scripts/run_local_stream.py'));
  assert.ok(result.missing.includes('src'));
  assert.ok(result.missing.includes('artifacts/main'));
});

test('resolveRuntimeRoot follows explicit, environment, persisted, packaged, then app priority', () => {
  const valid = new Set(['D:\\explicit', 'D:\\environment', 'D:\\persisted']);
  const inspect = (root) => ({
    root,
    valid: valid.has(root),
    missing: valid.has(root) ? [] : ['invalid'],
  });

  assert.equal(resolveRuntimeRoot({
    explicitRoot: 'D:\\explicit',
    envRoot: 'D:\\environment',
    persistedRoot: 'D:\\persisted',
    resourcesPath: 'C:\\Program Files\\AVTR-1\\resources',
    appPath: 'F:\\AVTR-1',
    inspect,
  }).root, 'D:\\explicit');

  assert.equal(resolveRuntimeRoot({
    envRoot: 'D:\\environment',
    persistedRoot: 'D:\\persisted',
    inspect,
  }).root, 'D:\\environment');
});
