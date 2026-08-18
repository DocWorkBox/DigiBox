'use strict';

const path = require('node:path');

const REQUIRED_PYTHON_RUNTIMES = ['main', 'cosyvoice', 'feynobg'];
const REQUIRED_V2_PATHS = ['python', 'orchestrator', 'source', 'artifacts', 'models'];

function pathApi(platform) {
  return platform === 'win32' ? path.win32 : path.posix;
}

function portablePathError(value) {
  if (typeof value !== 'string' || !value.trim()) return 'must be a non-empty relative path';
  const normalized = value.replaceAll('\\', '/');
  if (normalized.startsWith('/') || normalized.includes(':')) {
    return 'must be a relative path';
  }

  let depth = 0;
  for (const component of normalized.split('/')) {
    if (!component || component === '.') continue;
    if (component === '..') {
      if (depth === 0) return 'must not escape the runtime root';
      depth -= 1;
    } else {
      depth += 1;
    }
  }
  return null;
}

function validatePortablePath(errors, label, value, required = false) {
  if (value === undefined || value === null) {
    if (required) errors.push(`${label} is required`);
    return;
  }
  const error = portablePathError(value);
  if (error) errors.push(`${label} ${error}`);
}

function validateComponents(errors, components, required) {
  if (!required && components === undefined) return;
  if (!components || typeof components !== 'object' || Array.isArray(components)) {
    errors.push('components metadata is required');
    return;
  }
  for (const name of [
    'dependenciesIncluded',
    'modelsIncluded',
    'frontendVendorIncluded',
    'tensorRtBuildInputsIncluded',
  ]) {
    if (components[name] !== true) errors.push(`components.${name} must be true`);
  }
}

function validateRuntimeManifest(manifest) {
  const errors = [];
  if (!manifest || typeof manifest !== 'object') return ['manifest must be an object'];
  if (!manifest.runtimeId || typeof manifest.runtimeId !== 'string') errors.push('runtimeId is required');

  const isV2 = manifest.schemaVersion === 2 || manifest.layout === 'portable-v2';
  if (isV2) {
    if (manifest.schemaVersion !== 2) errors.push('schemaVersion must be 2 for portable-v2');
    if (manifest.layout !== 'portable-v2') errors.push('layout must be portable-v2');
    for (const name of REQUIRED_V2_PATHS) {
      validatePortablePath(errors, `paths.${name}`, manifest.paths?.[name], true);
    }
    if (!manifest.python?.version || typeof manifest.python.version !== 'string') {
      errors.push('python.version is required');
    }
    for (const name of REQUIRED_PYTHON_RUNTIMES) {
      const layers = manifest.python?.packageLayers?.[name];
      if (!Array.isArray(layers) || layers.length === 0) {
        errors.push(`python.packageLayers.${name} must be a non-empty array`);
        continue;
      }
      layers.forEach((layer, index) => {
        validatePortablePath(errors, `python.packageLayers.${name}[${index}]`, layer, true);
      });
    }
    validateComponents(errors, manifest.components, true);
  } else {
    if (manifest.schemaVersion !== 1) errors.push('schemaVersion must be 1');
    if (manifest.layout !== undefined && manifest.layout !== 'portable-v1') {
      errors.push('layout must be portable-v1');
    }
    for (const name of REQUIRED_PYTHON_RUNTIMES) {
      if (!manifest.python?.[name] || typeof manifest.python[name] !== 'string') {
        errors.push(`python.${name} is required`);
      }
    }
    validateComponents(errors, manifest.components, false);
  }

  if (manifest.paths && typeof manifest.paths === 'object' && !Array.isArray(manifest.paths)) {
    for (const [name, value] of Object.entries(manifest.paths)) {
      validatePortablePath(errors, `paths.${name}`, value);
    }
  }
  if (!manifest.tensorrt || typeof manifest.tensorrt !== 'object') {
    errors.push('tensorrt metadata is required');
  } else if (!Array.isArray(manifest.tensorrt.engines)) {
    errors.push('tensorrt.engines must be an array');
  }
  return errors;
}

function resolvePathSet(paths, root, values) {
  return Object.fromEntries(Object.entries(values).map(([name, value]) => [
    name,
    paths.resolve(root, value.replaceAll('\\', '/')),
  ]));
}

function resolveRuntimeManifest(manifest, runtimeRoot, options = {}) {
  const errors = validateRuntimeManifest(manifest);
  if (errors.length) {
    const error = new Error(`Invalid runtime manifest: ${errors.join('; ')}`);
    error.details = errors;
    throw error;
  }
  const platform = options.platform || process.platform;
  const paths = pathApi(platform);
  const root = paths.resolve(runtimeRoot);
  const layout = manifest.schemaVersion === 2 ? 'portable-v2' : 'portable-v1';
  const relativePaths = layout === 'portable-v2'
    ? manifest.paths
    : {
        mainPython: 'python-main/python.exe',
        cosyvoicePython: 'python-cosyvoice/python.exe',
        feynobgPython: 'python-feynobg/python.exe',
        orchestrator: 'scripts/run_local_stream.py',
        source: 'src',
        artifacts: 'artifacts/main',
        models: 'models',
        ...(manifest.paths || {}),
      };
  const resolvedPaths = resolvePathSet(paths, root, relativePaths);

  const relativeLayers = layout === 'portable-v2'
    ? manifest.python.packageLayers
    : {
        main: [relativePaths.source],
        cosyvoice: [
          relativePaths.source,
          'third_party/CosyVoice',
          'third_party/CosyVoice/third_party/Matcha-TTS',
        ],
        feynobg: [relativePaths.source],
      };
  const packageLayers = Object.fromEntries(REQUIRED_PYTHON_RUNTIMES.map((name) => [
    name,
    relativeLayers[name].map((layer) => paths.resolve(root, layer.replaceAll('\\', '/'))),
  ]));
  const delimiter = options.delimiter || (platform === 'win32' ? ';' : ':');
  const pythonPath = Object.fromEntries(REQUIRED_PYTHON_RUNTIMES.map((name) => [
    name,
    packageLayers[name].join(delimiter),
  ]));

  return {
    schemaVersion: manifest.schemaVersion,
    layout,
    runtimeId: manifest.runtimeId,
    root,
    paths: resolvedPaths,
    python: {
      version: layout === 'portable-v2' ? manifest.python.version : manifest.python.main,
      executable: layout === 'portable-v2' ? resolvedPaths.python : resolvedPaths.mainPython,
      packageLayers,
      pythonPath,
    },
  };
}

function classifyTensorRt(options = {}) {
  const manifest = options.manifest || {};
  const system = options.system || {};
  if (!system.nvidia) {
    return { state: 'unavailable', reasons: ['NVIDIA GPU was not detected'] };
  }

  const reasons = [];
  const expected = manifest.tensorrt || {};
  if (!system.tensorrtVersion) {
    reasons.push('TensorRT runtime is not installed');
  } else if (system.tensorrtVersion !== expected.version) {
    reasons.push(`TensorRT version differs (${system.tensorrtVersion} != ${expected.version})`);
  }
  if (Number(system.cudaMajor) !== Number(expected.cudaMajor)) {
    reasons.push(`CUDA major version differs (${system.cudaMajor} != ${expected.cudaMajor})`);
  }
  if (String(system.computeCapability) !== String(expected.computeCapability)) {
    reasons.push(`GPU compute capability differs (${system.computeCapability} != ${expected.computeCapability})`);
  }

  const available = options.existingEngines || new Set();
  const missing = (expected.engines || []).filter((engine) => !available.has(engine));
  if (missing.length) reasons.push(`missing engines: ${missing.join(', ')}`);
  if (!options.deserializeProbe?.ok) {
    const suffix = options.deserializeProbe?.error ? `: ${options.deserializeProbe.error}` : '';
    reasons.push(`engine deserialize probe failed${suffix}`);
  }

  return {
    state: reasons.length ? 'rebuild-required' : 'ready',
    reasons,
  };
}

module.exports = {
  classifyTensorRt,
  resolveRuntimeManifest,
  validateRuntimeManifest,
};
