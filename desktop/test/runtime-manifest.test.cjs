'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const {
  classifyTensorRt,
  resolveRuntimeManifest,
  validateRuntimeManifest,
} = require('../lib/runtime-manifest.cjs');

function manifest(overrides = {}) {
  return {
    schemaVersion: 1,
    runtimeId: 'avtr1-win64-test',
    python: { main: '3.12.9', cosyvoice: '3.10.17', feynobg: '3.12.10' },
    tensorrt: {
      version: '10.11.0.33',
      cudaMajor: 12,
      computeCapability: '12.0',
      engines: ['encode', 'decode', 'hubert', 'decoder', 'warp', 'modnet', 'stitch'],
    },
    ...overrides,
  };
}

function portableV2(overrides = {}) {
  return {
    schemaVersion: 2,
    layout: 'portable-v2',
    runtimeId: 'digibox-win64-test',
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
    tensorrt: {
      version: '10.11.0.33',
      cudaMajor: 12,
      computeCapability: null,
      engines: [],
    },
    ...overrides,
  };
}

test('runtime manifest requires versioned standalone Python entries', () => {
  assert.deepEqual(validateRuntimeManifest(manifest()), []);
  const errors = validateRuntimeManifest(manifest({ python: { main: '3.12.9' } }));
  assert.ok(errors.some((error) => error.includes('cosyvoice')));
  assert.ok(errors.some((error) => error.includes('feynobg')));
});

test('portable-v2 resolves one Python and preserves each ordered package layer list', () => {
  const source = portableV2();
  assert.deepEqual(validateRuntimeManifest(source), []);

  const resolved = resolveRuntimeManifest(source, 'D:\\DigiBox\\avtr-runtime', {
    platform: 'win32',
  });

  assert.equal(resolved.layout, 'portable-v2');
  assert.equal(resolved.python.executable, 'D:\\DigiBox\\avtr-runtime\\python\\python.exe');
  assert.deepEqual(resolved.python.packageLayers.main, [
    'D:\\DigiBox\\avtr-runtime\\packages\\main',
    'D:\\DigiBox\\avtr-runtime\\packages\\shared',
    'D:\\DigiBox\\avtr-runtime\\src',
  ]);
  assert.deepEqual(resolved.python.packageLayers.cosyvoice, [
    'D:\\DigiBox\\avtr-runtime\\packages\\cosyvoice',
    'D:\\DigiBox\\avtr-runtime\\packages\\shared',
    'D:\\DigiBox\\avtr-runtime\\third_party\\CosyVoice',
    'D:\\DigiBox\\avtr-runtime\\third_party\\CosyVoice\\third_party\\Matcha-TTS',
    'D:\\DigiBox\\avtr-runtime\\src',
  ]);
  assert.equal(
    resolved.python.pythonPath.cosyvoice,
    resolved.python.packageLayers.cosyvoice.join(';'),
  );
  assert.deepEqual(resolved.python.packageLayers.feynobg, [
    'D:\\DigiBox\\avtr-runtime\\packages\\feynobg',
    'D:\\DigiBox\\avtr-runtime\\packages\\shared',
    'D:\\DigiBox\\avtr-runtime\\src',
  ]);
});

test('portable-v2 example fully declares the ordered CosyVoice import path', () => {
  const example = JSON.parse(fs.readFileSync(
    path.join(__dirname, '..', 'runtime-manifest.example.json'),
    'utf8',
  ));

  assert.deepEqual(example.python.packageLayers.cosyvoice, [
    'packages/cosyvoice',
    'packages/shared',
    'third_party/CosyVoice',
    'third_party/CosyVoice/third_party/Matcha-TTS',
    'src',
  ]);
});

test('portable manifests reject absolute paths and paths that escape the runtime root', () => {
  const invalid = portableV2({
    paths: {
      ...portableV2().paths,
      python: 'C:/host/python.exe',
      artifacts: '../private-artifacts',
    },
    python: {
      ...portableV2().python,
      packageLayers: {
        ...portableV2().python.packageLayers,
        cosyvoice: ['src', '../../host-packages'],
      },
    },
  });

  const errors = validateRuntimeManifest(invalid);
  assert.ok(errors.some((error) => error.includes('paths.python') && error.includes('relative')));
  assert.ok(errors.some((error) => error.includes('paths.artifacts') && error.includes('escape')));
  assert.ok(errors.some((error) => error.includes('python.packageLayers.cosyvoice[1]') && error.includes('escape')));
  assert.throws(
    () => resolveRuntimeManifest(invalid, 'D:\\DigiBox\\avtr-runtime', { platform: 'win32' }),
    /invalid runtime manifest/i,
  );
});

test('portable-v1 stays valid while supplied v1 paths receive the same containment checks', () => {
  assert.deepEqual(validateRuntimeManifest(manifest()), []);
  const errors = validateRuntimeManifest(manifest({
    paths: { mainPython: '../python.exe' },
  }));
  assert.ok(errors.some((error) => error.includes('paths.mainPython') && error.includes('escape')));
});

test('TensorRT is ready only after exact runtime, GPU, files, and probe checks', () => {
  const status = classifyTensorRt({
    manifest: manifest(),
    system: { nvidia: true, tensorrtVersion: '10.11.0.33', cudaMajor: 12, computeCapability: '12.0' },
    existingEngines: new Set(['encode', 'decode', 'hubert', 'decoder', 'warp', 'modnet', 'stitch']),
    deserializeProbe: { ok: true },
  });
  assert.equal(status.state, 'ready');
});

test('TensorRT requests a rebuild when hardware or serialized plans differ', () => {
  const status = classifyTensorRt({
    manifest: manifest(),
    system: { nvidia: true, tensorrtVersion: '10.11.0.33', cudaMajor: 12, computeCapability: '8.9' },
    existingEngines: new Set(['encode', 'decode']),
    deserializeProbe: { ok: false, error: 'incompatible plan' },
  });
  assert.equal(status.state, 'rebuild-required');
  assert.ok(status.reasons.some((reason) => reason.includes('compute capability')));
  assert.ok(status.reasons.some((reason) => reason.includes('missing engines')));
  assert.ok(status.reasons.some((reason) => reason.includes('deserialize')));
});

test('TensorRT is unavailable without an NVIDIA GPU', () => {
  const status = classifyTensorRt({
    manifest: manifest(),
    system: { nvidia: false },
    existingEngines: new Set(),
    deserializeProbe: { ok: false },
  });
  assert.equal(status.state, 'unavailable');
});
