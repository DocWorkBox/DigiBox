'use strict';

const fs = require('node:fs');
const path = require('node:path');
const { resolveRuntimeManifest } = require('./runtime-manifest.cjs');

function pathApi(platform) {
  return platform === 'win32' ? path.win32 : path.posix;
}

function inspectRuntimeRoot(root, options = {}) {
  const existsSync = options.existsSync || fs.existsSync;
  const readFileSync = options.readFileSync || fs.readFileSync;
  const platform = options.platform || process.platform;
  const paths = pathApi(platform);
  const normalizedRoot = paths.resolve(root);
  const manifest = paths.join(normalizedRoot, 'runtime-manifest.json');
  const hasManifest = existsSync(manifest);
  let resolvedManifest = null;
  let manifestError = null;
  if (hasManifest) {
    try {
      const parsed = JSON.parse(readFileSync(manifest, 'utf8'));
      resolvedManifest = resolveRuntimeManifest(parsed, normalizedRoot, { platform });
    } catch (error) {
      manifestError = error instanceof Error ? error.message : String(error);
    }
  }

  const managedPython = resolvedManifest?.python.executable || paths.join(
    normalizedRoot,
    'python-main',
    platform === 'win32' ? 'python.exe' : 'bin/python3',
  );
  const developmentPython = paths.join(
    normalizedRoot,
    '.venv',
    platform === 'win32' ? 'Scripts/python.exe' : 'bin/python',
  );
  const script = resolvedManifest?.paths.orchestrator
    || paths.join(normalizedRoot, 'scripts', 'run_local_stream.py');
  const source = resolvedManifest?.paths.source || paths.join(normalizedRoot, 'src');
  const artifacts = resolvedManifest?.paths.artifacts
    || paths.join(normalizedRoot, 'artifacts', 'main');
  const models = resolvedManifest?.paths.models || paths.join(normalizedRoot, 'models');

  const hasManagedPython = existsSync(managedPython);
  const hasDevelopmentPython = !resolvedManifest && existsSync(developmentPython);
  const missing = [];
  if (manifestError) missing.push(`runtime-manifest.json (${manifestError})`);
  if (!hasManagedPython && !hasDevelopmentPython) missing.push('Python runtime');
  if (!existsSync(script)) missing.push('scripts/run_local_stream.py');
  if (!existsSync(source)) missing.push('src');
  if (!existsSync(artifacts)) missing.push('artifacts/main');
  if (resolvedManifest?.layout === 'portable-v2') {
    if (!existsSync(models)) missing.push('models');
    for (const [profile, layers] of Object.entries(
      resolvedManifest.python.packageLayers,
    )) {
      for (const layer of layers) {
        if (!existsSync(layer)) {
          missing.push(`python.packageLayers.${profile} (${layer})`);
        }
      }
    }
  }

  const usesPackageLayerRouting = resolvedManifest?.layout === 'portable-v2';

  return {
    root: normalizedRoot,
    valid: missing.length === 0,
    mode: hasManagedPython ? 'managed' : 'development',
    layout: resolvedManifest?.layout || null,
    runtimeId: resolvedManifest?.runtimeId || null,
    python: hasManagedPython ? managedPython : developmentPython,
    script,
    source,
    artifacts,
    models,
    manifest: hasManifest ? manifest : null,
    packageLayers: usesPackageLayerRouting ? resolvedManifest.python.packageLayers : null,
    pythonPath: usesPackageLayerRouting ? resolvedManifest.python.pythonPath : null,
    missing,
  };
}

function uniqueCandidates(values, platform) {
  const paths = pathApi(platform);
  const seen = new Set();
  const output = [];
  for (const value of values) {
    if (!value) continue;
    const normalized = paths.resolve(value);
    const key = platform === 'win32' ? normalized.toLowerCase() : normalized;
    if (seen.has(key)) continue;
    seen.add(key);
    output.push(normalized);
  }
  return output;
}

function ancestorCandidates(start, platform, maxDepth = 6) {
  if (!start) return [];
  const paths = pathApi(platform);
  const output = [];
  let current = paths.resolve(start);
  for (let index = 0; index < maxDepth; index += 1) {
    output.push(current);
    const parent = paths.dirname(current);
    if (parent === current) break;
    current = parent;
  }
  return output;
}

function resolveRuntimeRoot(options = {}) {
  const platform = options.platform || process.platform;
  const paths = pathApi(platform);
  const inspect = options.inspect || ((root) => inspectRuntimeRoot(root, options));
  const packagedRuntime = options.resourcesPath
    ? paths.join(options.resourcesPath, 'avtr-runtime')
    : null;
  const siblingRuntime = options.executablePath
    ? paths.join(paths.dirname(options.executablePath), 'avtr-runtime')
    : null;
  const candidates = uniqueCandidates([
    options.explicitRoot,
    options.envRoot,
    options.persistedRoot,
    packagedRuntime,
    siblingRuntime,
    ...ancestorCandidates(options.appPath, platform),
  ], platform);

  const inspected = candidates.map((candidate) => ({
    candidate,
    result: inspect(candidate),
  }));
  const match = inspected.find(({ result }) => result.valid);
  if (match) {
    return {
      ...match.result,
      candidates: inspected,
    };
  }
  return {
    root: null,
    valid: false,
    mode: null,
    missing: inspected[0]?.result?.missing || ['No runtime candidate was configured'],
    candidates: inspected,
  };
}

module.exports = {
  inspectRuntimeRoot,
  resolveRuntimeRoot,
};
