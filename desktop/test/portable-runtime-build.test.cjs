'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawnSync } = require('node:child_process');

const repoRoot = path.join(__dirname, '..', '..');
const requiredMemoryModules = [
  '__init__.py',
  'admin.py',
  'api.py',
  'extractor.py',
  'models.py',
  'paths.py',
  'schema.py',
  'service.py',
  'sqlite_store.py',
  'transfer.py',
  'worklet.py',
];
const runtimeBuilder = path.join(repoRoot, 'scripts', 'desktop', 'build_portable_runtime.ps1');
const desktopBuilder = path.join(repoRoot, 'scripts', 'build_desktop_windows.ps1');
const fullBuilderConfig = path.join(repoRoot, 'electron-builder-full.yml');
const distributionGuide = path.join(repoRoot, 'docs', 'windows-desktop-distribution.md');

function readRequired(file) {
  assert.ok(fs.existsSync(file), `required distribution file is missing: ${file}`);
  return fs.readFileSync(file, 'utf8');
}

function runPowerShellFile(file, args = []) {
  return spawnSync(
    'powershell.exe',
    ['-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', file, ...args],
    { cwd: repoRoot, encoding: 'utf8' },
  );
}

function parsePowerShell(file) {
  const escaped = file.replaceAll("'", "''");
  return spawnSync(
    'powershell.exe',
    [
      '-NoProfile',
      '-Command',
      `$errors=$null; [System.Management.Automation.Language.Parser]::ParseFile('${escaped}',[ref]$null,[ref]$errors) | Out-Null; if($errors.Count){$errors | ForEach-Object { Write-Error $_ }; exit 1}`,
    ],
    { cwd: repoRoot, encoding: 'utf8' },
  );
}

function resolveManagedPython(managedRoot) {
  const escapedBuilder = runtimeBuilder.replaceAll("'", "''");
  const escapedRoot = managedRoot.replaceAll("'", "''");
  return spawnSync(
    'powershell.exe',
    [
      '-NoProfile',
      '-Command',
      [
        '$tokens=$null; $errors=$null',
        `$ast=[System.Management.Automation.Language.Parser]::ParseFile('${escapedBuilder}',[ref]$tokens,[ref]$errors)`,
        "if($errors.Count){ throw ($errors -join [Environment]::NewLine) }",
        "$function=$ast.Find({ param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Resolve-ManagedPythonExecutable' }, $true)",
        "if($null -eq $function){ throw 'Resolve-ManagedPythonExecutable is missing' }",
        'Invoke-Expression $function.Extent.Text',
        `$resolved=Resolve-ManagedPythonExecutable -ManagedRoot '${escapedRoot}'`,
        '[pscustomobject]@{ path=$resolved } | ConvertTo-Json -Compress',
      ].join('; '),
    ],
    { cwd: repoRoot, encoding: 'utf8' },
  );
}

function inspectPortableDependencyCalls({ wheelhouse = '' } = {}) {
  const escapedBuilder = runtimeBuilder.replaceAll("'", "''");
  const escapedRoot = repoRoot.replaceAll("'", "''");
  const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'avtr-portable-deps-'));
  const escapedTemporary = temporary.replaceAll("'", "''");
  const escapedWheelhouse = wheelhouse.replaceAll("'", "''");
  try {
    const harness = path.join(temporary, 'inspect-dependencies.ps1');
    const source = [
      '$ErrorActionPreference="Stop"',
      '$tokens=$null; $errors=$null',
      `$ast=[System.Management.Automation.Language.Parser]::ParseFile('${escapedBuilder}',[ref]$tokens,[ref]$errors)`,
      "if($errors.Count){ throw ($errors -join [Environment]::NewLine) }",
      "$function=$ast.Find({ param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Install-RuntimeDependencies' }, $true)",
      "if($null -eq $function){ throw 'Install-RuntimeDependencies is missing' }",
      'Invoke-Expression $function.Extent.Text',
      '$global:portableDependencyCalls=@()',
      'function Invoke-UvPipInstall {',
      'param([string]$UvPath,[string]$Python,[string]$Target,[string[]]$Arguments,[string]$Step)',
      '$constraintText=$null',
      '$constraintIndex=[array]::IndexOf($Arguments,"-c")',
      'if($constraintIndex -ge 0){$constraintText=Get-Content -LiteralPath $Arguments[$constraintIndex+1] -Raw -Encoding UTF8}',
      '$global:portableDependencyCalls += [pscustomobject]@{target=[string]$Target;arguments=[string[]]$Arguments;step=[string]$Step;constraintText=[string]$constraintText}',
      '}',
      '$isPortableV2=$true',
      `$resolvedDestination='${escapedTemporary}'`,
      `$resolvedSourceRoot='${escapedRoot}'`,
      '$FeyNoBgDevice="cpu"',
      `$resolvedTorchWheelhouse='${escapedWheelhouse}'`,
      'Install-RuntimeDependencies -UvPath "fixture-uv.exe" -PythonPaths @{python="fixture-python.exe"}',
      'ConvertTo-Json -InputObject @($global:portableDependencyCalls) -Depth 3 -Compress',
    ].join('\r\n');
    fs.writeFileSync(harness, source, 'utf8');
    return spawnSync(
      'powershell.exe',
      ['-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', harness],
      { cwd: repoRoot, encoding: 'utf8' },
    );
  } finally {
    fs.rmSync(temporary, { recursive: true, force: true });
  }
}

function invokePortableRuntimePayloadGate({
  relativePath = null,
  directory = false,
  includeMemoryPackage = true,
  missingMemoryModule = null,
} = {}) {
  const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'avtr-runtime-memory-gate-'));
  const python = path.join(temporary, 'python', 'python.exe');
  fs.mkdirSync(path.dirname(python), { recursive: true });
  fs.writeFileSync(python, 'fixture', 'utf8');
  if (includeMemoryPackage) {
    for (const moduleName of requiredMemoryModules) {
      if (moduleName === missingMemoryModule) continue;
      const memoryModule = path.join(
        temporary,
        'src',
        'avaturn_live_streamer',
        'memory',
        moduleName,
      );
      fs.mkdirSync(path.dirname(memoryModule), { recursive: true });
      fs.writeFileSync(memoryModule, '# fixture', 'utf8');
    }
  }
  if (relativePath) {
    const target = path.join(temporary, ...relativePath.split('/'));
    if (directory) {
      fs.mkdirSync(target, { recursive: true });
    } else {
      fs.mkdirSync(path.dirname(target), { recursive: true });
      fs.writeFileSync(target, 'private', 'utf8');
    }
  }

  const escapedBuilder = runtimeBuilder.replaceAll("'", "''");
  const escapedTemporary = temporary.replaceAll("'", "''");
  const result = spawnSync(
    'powershell.exe',
    [
      '-NoProfile',
      '-Command',
      [
        '$tokens=$null; $errors=$null',
        `$ast=[System.Management.Automation.Language.Parser]::ParseFile('${escapedBuilder}',[ref]$tokens,[ref]$errors)`,
        "if($errors.Count){ throw ($errors -join [Environment]::NewLine) }",
        "$function=$ast.Find({ param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Assert-NoForbiddenPayload' }, $true)",
        "if($null -eq $function){ throw 'Assert-NoForbiddenPayload is missing' }",
        'Invoke-Expression $function.Extent.Text',
        `$resolvedDestination='${escapedTemporary}'`,
        '$isPortableV2=$true',
        '$runtimeNames=@("python")',
        'Assert-NoForbiddenPayload',
      ].join('; '),
    ],
    { cwd: repoRoot, encoding: 'utf8' },
  );
  fs.rmSync(temporary, { recursive: true, force: true });
  return result;
}

test('portable distribution files exist and both PowerShell scripts parse', { skip: process.platform !== 'win32' }, () => {
  for (const file of [runtimeBuilder, desktopBuilder, fullBuilderConfig, distributionGuide]) {
    assert.ok(fs.existsSync(file), `missing ${file}`);
  }
  for (const file of [runtimeBuilder, desktopBuilder]) {
    const result = parsePowerShell(file);
    assert.equal(result.status, 0, result.stderr || result.stdout);
  }
});

test('portable-v2 plan contains one CPython 3.12 runtime and four package layers', { skip: process.platform !== 'win32' }, () => {
  readRequired(runtimeBuilder);
  const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'avtr-runtime-plan-'));
  const destination = path.join(temporary, 'avtr-runtime');
  try {
    const result = runPowerShellFile(runtimeBuilder, [
      '-SourceRoot', repoRoot,
      '-Destination', destination,
      '-PlanOnly',
    ]);
    assert.equal(result.status, 0, result.stderr || result.stdout);
    const plan = JSON.parse(result.stdout.trim());
    assert.equal(plan.layout, 'portable-v2');
    assert.deepEqual(plan.python, {
      version: '3.12.9',
      executable: 'python/python.exe',
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
      layerInventory: 'packages/python-layer-inventory.json',
    });
    assert.equal(plan.consolidator, 'scripts/desktop/consolidate_python_layers.py');
    assert.equal(plan.dependencyLinkMode, 'copy');
    assert.equal(plan.feynobgDevice, 'cpu', 'FeyNoBg still runs on CPU by default');
    assert.equal(
      plan.effectiveFeynobgTorch,
      'cuda',
      'portable-v2 shares one CUDA Torch binary stack even when FeyNoBg runs on CPU',
    );
    assert.deepEqual(plan.torch, {
      source: 'pytorch-cu128',
      wheelhouse: null,
      versions: {
        torch: '2.7.1+cu128',
        torchaudio: '2.7.1+cu128',
        torchvision: '0.22.1+cu128',
      },
    });
    assert.equal(fs.existsSync(destination), false, 'PlanOnly must not create the destination');
  } finally {
    fs.rmSync(temporary, { recursive: true, force: true });
  }
});

test('portable-v2 plan excludes target-machine and memory persistence payloads', { skip: process.platform !== 'win32' }, () => {
  const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'avtr-runtime-engine-manifest-plan-'));
  const destination = path.join(temporary, 'avtr-runtime');
  try {
    const result = runPowerShellFile(runtimeBuilder, [
      '-SourceRoot', repoRoot,
      '-Destination', destination,
      '-PlanOnly',
    ]);
    assert.equal(result.status, 0, result.stderr || result.stdout);
    const plan = JSON.parse(result.stdout.trim());
    assert.ok(
      plan.excludedFiles.includes('engine-manifest.json'),
      'target-machine engine manifest must be excluded from a portable Runtime',
    );
    for (const forbiddenFile of [
      'memory.sqlite3',
      'memory.sqlite3-wal',
      'memory.sqlite3-shm',
      'digibox-memory*.json',
    ]) {
      assert.ok(
        plan.excludedFiles.includes(forbiddenFile),
        `memory persistence payload must be excluded: ${forbiddenFile}`,
      );
    }
    assert.ok(
      plan.excludedDirectories.includes('memory\\backups'),
      'memory backups must be excluded from a portable Runtime',
    );
    assert.ok(
      plan.excludedDirectories.includes('memory\\pending-imports'),
      'staged memory imports must be excluded from a portable Runtime',
    );
    assert.equal(fs.existsSync(destination), false, 'PlanOnly must remain read-only');
  } finally {
    fs.rmSync(temporary, { recursive: true, force: true });
  }
});

test('portable Runtime payload gate requires the complete memory package', { skip: process.platform !== 'win32' }, () => {
  const result = invokePortableRuntimePayloadGate({ includeMemoryPackage: false });
  assert.notEqual(result.status, 0);
  assert.match(`${result.stderr}\n${result.stdout}`, /avaturn_live_streamer[\\/]memory[\\/]__init__\.py/i);
});

for (const moduleName of requiredMemoryModules) {
  test(`portable Runtime payload gate rejects missing memory/${moduleName}`, { skip: process.platform !== 'win32' }, () => {
    const result = invokePortableRuntimePayloadGate({ missingMemoryModule: moduleName });
    assert.notEqual(result.status, 0);
    assert.match(`${result.stderr}\n${result.stdout}`, new RegExp(moduleName.replace('.', '\\.'), 'i'));
  });
}

test('portable Runtime payload gate permits memory source without persistent data', { skip: process.platform !== 'win32' }, () => {
  const result = invokePortableRuntimePayloadGate();
  assert.equal(result.status, 0, result.stderr || result.stdout);
});

test('portable Runtime main Python probe imports the memory application entrypoints', () => {
  const source = readRequired(runtimeBuilder);
  for (const moduleName of [
    'avaturn_live_streamer.local_stream_cli',
    'avaturn_live_streamer.memory.admin',
    'avaturn_live_streamer.memory.api',
  ]) {
    assert.ok(source.includes(moduleName), `missing real Runtime import probe for ${moduleName}`);
  }
});

for (const [relativePath, directory] of [
  ['memory.sqlite3', false],
  ['memory.sqlite3-wal', false],
  ['memory.sqlite3-shm', false],
  ['memory/backups', true],
  ['memory/pending-imports', true],
  ['exports/digibox-memory-20260817.json', false],
]) {
  test(`portable Runtime payload gate rejects ${relativePath}`, { skip: process.platform !== 'win32' }, () => {
    const result = invokePortableRuntimePayloadGate({ relativePath, directory });
    assert.notEqual(result.status, 0);
    assert.match(`${result.stderr}\n${result.stdout}`, /forbidden/i);
  });
}

test('portable-v2 plan records an explicit local Torch wheelhouse without touching it', { skip: process.platform !== 'win32' }, () => {
  const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'avtr-torch-wheelhouse-plan-'));
  const wheelhouse = path.join(temporary, 'wheelhouse');
  const destination = path.join(temporary, 'avtr-runtime');
  fs.mkdirSync(wheelhouse);
  for (const wheel of [
    'torch-2.7.1+cu128-cp312-cp312-win_amd64.whl',
    'torchaudio-2.7.1+cu128-cp312-cp312-win_amd64.whl',
    'torchvision-0.22.1+cu128-cp312-cp312-win_amd64.whl',
  ]) {
    fs.writeFileSync(path.join(wheelhouse, wheel), 'fixture', 'utf8');
  }
  try {
    const result = runPowerShellFile(runtimeBuilder, [
      '-SourceRoot', repoRoot,
      '-Destination', destination,
      '-TorchWheelhouse', wheelhouse,
      '-PlanOnly',
    ]);
    assert.equal(result.status, 0, result.stderr || result.stdout);
    const plan = JSON.parse(result.stdout.trim());
    assert.deepEqual(plan.torch, {
      source: 'wheelhouse',
      wheelhouse: path.resolve(wheelhouse),
      versions: {
        torch: '2.7.1+cu128',
        torchaudio: '2.7.1+cu128',
        torchvision: '0.22.1+cu128',
      },
    });
    assert.equal(fs.existsSync(destination), false);
    assert.equal(fs.readdirSync(wheelhouse).length, 3, 'PlanOnly must not modify the wheelhouse');
  } finally {
    fs.rmSync(temporary, { recursive: true, force: true });
  }
});

test('portable-v2 installs the exact local Torch wheels and constrains every later resolution', { skip: process.platform !== 'win32' }, () => {
  const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'avtr-torch-wheelhouse-calls-'));
  const wheelhouse = path.join(temporary, 'wheelhouse');
  fs.mkdirSync(wheelhouse);
  try {
    const result = inspectPortableDependencyCalls({ wheelhouse });
    assert.equal(result.status, 0, result.stderr || result.stdout);
    const calls = JSON.parse(result.stdout.trim());
    assert.equal(calls.length, 10);
    const torchCalls = calls.filter(({ step }) => /PyTorch/i.test(step));
    assert.equal(torchCalls.length, 3);
    const nvidiaVfxCalls = calls.filter(({ step }) => /NVIDIA VFX Runtime/i.test(step));
    assert.equal(nvidiaVfxCalls.length, 1, 'the main package layer must install NVIDIA VFX once');
    for (const { arguments: args } of torchCalls) {
      assert.equal(args.includes('--no-index'), false);
      assert.equal(args.includes('--no-deps'), false);
      assert.ok(args.includes('--find-links'));
      assert.ok(args.includes(path.resolve(wheelhouse)));
      assert.ok(args.includes('--index-url'));
      assert.ok(args.includes('https://pypi.org/simple'));
      assert.ok(args.includes('-c'));
      assert.ok(args.includes('torch==2.7.1+cu128'));
      assert.ok(args.includes('torchaudio==2.7.1+cu128'));
      assert.ok(args.includes('torchvision==0.22.1+cu128'));
    }
    for (const call of calls) {
      assert.match(call.constraintText, /^torch==2\.7\.1\+cu128$/m);
      assert.match(call.constraintText, /^torchaudio==2\.7\.1\+cu128$/m);
      assert.match(call.constraintText, /^torchvision==0\.22\.1\+cu128$/m);
      assert.ok(call.arguments.includes('--find-links'));
      assert.ok(call.arguments.includes(path.resolve(wheelhouse)));
    }
    for (const call of calls.filter(({ step }) => !/PyTorch/i.test(step))) {
      assert.equal(call.arguments.includes('--no-index'), false, `${call.step} still needs PyPI`);
    }
  } finally {
    fs.rmSync(temporary, { recursive: true, force: true });
  }
});

test('portable-v2 default source pins CUDA Torch while allowing PyPI fallback for every resolution', { skip: process.platform !== 'win32' }, () => {
  const result = inspectPortableDependencyCalls();
  assert.equal(result.status, 0, result.stderr || result.stdout);
  const calls = JSON.parse(result.stdout.trim());
  assert.equal(calls.length, 10);
  for (const call of calls) {
    assert.match(call.constraintText, /^torch==2\.7\.1\+cu128$/m);
    assert.ok(call.arguments.includes('--index-url'));
    assert.ok(call.arguments.includes('https://download.pytorch.org/whl/cu128'));
    assert.ok(call.arguments.includes('--extra-index-url'));
    assert.ok(call.arguments.includes('https://pypi.org/simple'));
  }
});

test('managed Python resolver ignores the uv unversioned alias junction', { skip: process.platform !== 'win32' }, () => {
  const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'avtr-uv-python-alias-'));
  const versioned = path.join(temporary, 'cpython-3.12.9-windows-x86_64-none');
  const alias = path.join(temporary, 'cpython-3.12-windows-x86_64-none');
  const python = path.join(versioned, 'python.exe');
  try {
    fs.mkdirSync(versioned);
    fs.writeFileSync(python, 'fixture', 'utf8');
    fs.symlinkSync(versioned, alias, 'junction');

    const result = resolveManagedPython(temporary);
    assert.equal(result.status, 0, result.stderr || result.stdout);
    assert.equal(path.resolve(JSON.parse(result.stdout.trim()).path), path.resolve(python));
  } finally {
    fs.rmSync(temporary, { recursive: true, force: true });
  }
});

test('managed Python resolver still rejects two physical installations', { skip: process.platform !== 'win32' }, () => {
  const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'avtr-uv-python-duplicate-'));
  try {
    for (const name of [
      'cpython-3.12.9-windows-x86_64-none',
      'cpython-3.11.9-windows-x86_64-none',
    ]) {
      const directory = path.join(temporary, name);
      fs.mkdirSync(directory);
      fs.writeFileSync(path.join(directory, 'python.exe'), 'fixture', 'utf8');
    }

    const result = resolveManagedPython(temporary);
    assert.notEqual(result.status, 0);
    assert.match(`${result.stderr}\n${result.stdout}`, /exactly one[\s\S]*found 2/i);
  } finally {
    fs.rmSync(temporary, { recursive: true, force: true });
  }
});

test('LegacyV1 plan remains explicitly available for rollback builds', { skip: process.platform !== 'win32' }, () => {
  const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'avtr-runtime-v1-plan-'));
  const destination = path.join(temporary, 'avtr-runtime');
  try {
    const result = runPowerShellFile(runtimeBuilder, [
      '-SourceRoot', repoRoot,
      '-Destination', destination,
      '-Layout', 'LegacyV1',
      '-PlanOnly',
    ]);
    assert.equal(result.status, 0, result.stderr || result.stdout);
    const plan = JSON.parse(result.stdout.trim());
    assert.equal(plan.layout, 'portable-v1');
    assert.deepEqual(
      plan.runtimes.map(({ name, pythonVersion }) => [name, pythonVersion]),
      [
        ['python-main', '3.12.9'],
        ['python-cosyvoice', '3.10.17'],
        ['python-feynobg', '3.12.9'],
      ],
    );
    assert.equal(fs.existsSync(destination), false);
  } finally {
    fs.rmSync(temporary, { recursive: true, force: true });
  }
});

test('runtime builder refuses non-empty targets and PlanOnly never performs Clean', { skip: process.platform !== 'win32' }, () => {
  readRequired(runtimeBuilder);
  const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'avtr-runtime-clean-'));
  const destination = path.join(temporary, 'avtr-runtime');
  fs.mkdirSync(destination);
  const sentinel = path.join(destination, 'keep.txt');
  fs.writeFileSync(sentinel, 'keep', 'utf8');
  try {
    const withoutClean = runPowerShellFile(runtimeBuilder, [
      '-SourceRoot', repoRoot,
      '-Destination', destination,
      '-PlanOnly',
    ]);
    assert.notEqual(withoutClean.status, 0);
    assert.ok(fs.existsSync(sentinel));

    const unmarkedClean = runPowerShellFile(runtimeBuilder, [
      '-SourceRoot', repoRoot,
      '-Destination', destination,
      '-Clean',
      '-PlanOnly',
    ]);
    assert.notEqual(unmarkedClean.status, 0);
    assert.ok(fs.existsSync(sentinel));

    fs.writeFileSync(
      path.join(destination, '.avtr-portable-runtime.json'),
      JSON.stringify({
        schemaVersion: 1,
        kind: 'avtr1-portable-runtime',
        destination,
      }),
      'utf8',
    );
    const markedPlan = runPowerShellFile(runtimeBuilder, [
      '-SourceRoot', repoRoot,
      '-Destination', destination,
      '-Clean',
      '-PlanOnly',
    ]);
    assert.equal(markedPlan.status, 0, markedPlan.stderr || markedPlan.stdout);
    assert.ok(fs.existsSync(sentinel), 'PlanOnly -Clean must remain read-only');

    fs.writeFileSync(
      path.join(destination, '.avtr-portable-runtime.json'),
      JSON.stringify({
        schemaVersion: 1,
        kind: 'avtr1-portable-runtime',
        destination: path.join(temporary, 'somewhere-else', 'avtr-runtime'),
      }),
      'utf8',
    );
    const copiedMarker = runPowerShellFile(runtimeBuilder, [
      '-SourceRoot', repoRoot,
      '-Destination', destination,
      '-Clean',
      '-PlanOnly',
    ]);
    assert.notEqual(copiedMarker.status, 0, 'a marker copied from another target must not authorize Clean');
    assert.ok(fs.existsSync(sentinel));
  } finally {
    fs.rmSync(temporary, { recursive: true, force: true });
  }
});

test('runtime builder installs dependencies by copy and excludes local or machine-specific state', () => {
  const source = readRequired(runtimeBuilder);
  assert.match(source, /uv[\s\S]*python[\s\S]*install/i);
  assert.match(source, /UV_LINK_MODE[\s\S]*copy/i);
  assert.doesNotMatch(source, /\buv(?:\.Source)?\s+venv\b/i);
  assert.doesNotMatch(source, /-m\s+venv\b/i);
  assert.doesNotMatch(
    source,
    /&\s*\$UvPath\s+python\s+find/i,
    'uv python find can resolve the repository venv instead of the isolated install root',
  );
  assert.match(source, /basePython[\s\S]*managedRoot[\s\S]*StartsWith/i);
  for (const contract of [
    'user_assets',
    'avatars_artifacts\\backgrounds',
    '*.engine',
    '*.plan',
    'grid_sample_3d_plugin*.dll',
    'spk2info.pt',
    '*.incomplete',
    '__pycache__',
    '.cache',
    '.trash',
  ]) {
    assert.ok(source.includes(contract), `missing exclusion contract: ${contract}`);
  }
  assert.doesNotMatch(
    source,
    /\$entry\.Extension\s+-in\s+@\("\.key",\s*"\.pem"\)/,
    'public CA bundles installed by dependencies must not be rejected as private payload',
  );
  for (const isolationContract of [
    'UV_PYTHON_BIN_DIR',
    'UV_PYTHON_INSTALL_REGISTRY',
    '0.8.0',
  ]) {
    assert.ok(source.includes(isolationContract), `missing uv isolation contract: ${isolationContract}`);
  }
  for (const vendorFile of [
    'preact.module.js',
    'preact-hooks.module.js',
    'htm.module.js',
  ]) {
    assert.ok(source.includes(vendorFile), `portable Runtime does not preflight ${vendorFile}`);
  }
  assert.match(source, /Assert-NoForbiddenPayload/);
  assert.match(source, /ReparsePoint/);
  assert.match(source, /runtime-manifest\.json/);
  assert.match(source, /dependenciesIncluded/);
  assert.match(source, /modelsIncluded/);
  assert.match(source, /packages\\main/);
  assert.match(source, /packages\\cosyvoice/);
  assert.match(source, /packages\\feynobg/);
  assert.match(source, /packages[\\/]shared/);
  assert.match(source, /consolidate_python_layers\.py/);
  assert.match(source, /--target/);
  assert.match(
    source,
    /\$portableTorchArguments[\s\S]*torch==2\.7\.1[\s\S]*torchaudio==2\.7\.1[\s\S]*torchvision==0\.22\.1/,
  );
  for (const profileArguments of [
    '$mainTorchArguments',
    '$cosyvoiceTorchArguments',
    '$feynobgTorchArguments',
  ]) {
    assert.ok(
      source.includes(`${profileArguments} = if ($isPortableV2) { $portableTorchArguments }`),
      `${profileArguments} must use the identical portable-v2 binary stack`,
    );
  }
  assert.match(
    source,
    /\$env:PYTHONPATH\s*=\s*\$cosyvoiceTarget[\s\S]*-Arguments\s+\$whisperArguments/,
    'the target-installed legacy build tools must be importable while Whisper builds',
  );
  assert.match(source, /\$whisperArguments[\s\S]*--no-build-isolation/);
  assert.match(source, /Assert-SharedPythonDistributions/);
  assert.match(source, /Assert-PortableLayerBootstrap/);
  assert.match(source, /'sitecustomize'\s+in\s+sys\.modules/);
  assert.match(source, /@\("torch", "torchaudio", "torchvision"\)/);
  const execution = source.slice(source.indexOf("if (-not $SkipDependencies)"));
  assert.ok(
    execution.indexOf('Remove-GeneratedCaches') < execution.indexOf('$consolidator ='),
    'generated caches must be removed before deterministic package size inventory is written',
  );
});

test('runtime builder requires public model payloads needed by local TTS and both TensorRT modes', () => {
  const source = readRequired(runtimeBuilder);
  assert.match(source, /Length\s+-le\s+0/);
  for (const requiredPayload of [
    'llm.pt',
    'flow.pt',
    'hift.pt',
    'speech_tokenizer_v3.onnx',
    'campplus.onnx',
    'model.safetensors',
    'tokenizer_config.json',
    'avtr1.scripted.pt',
    'hubert-lbs-avtr1.onnx',
    'decoder.onnx',
    'modnet.onnx',
    'stitch_network.onnx',
    'warp_network.onnx',
    'warp_network_ori.onnx',
  ]) {
    assert.ok(source.includes(requiredPayload), `missing payload preflight: ${requiredPayload}`);
  }
});

test('desktop build plans select shell-only and full configurations without executing builds', { skip: process.platform !== 'win32' }, () => {
  readRequired(desktopBuilder);
  const standard = runPowerShellFile(desktopBuilder, ['-Edition', 'Standard', '-PlanOnly']);
  assert.equal(standard.status, 0, standard.stderr || standard.stdout);
  const standardPlan = JSON.parse(standard.stdout.trim());
  assert.equal(standardPlan.config, 'electron-builder.yml');
  assert.equal(standardPlan.includesRuntime, false);
  assert.equal(standardPlan.includesModels, false);

  const full = runPowerShellFile(desktopBuilder, [
    '-Edition', 'Full',
    '-SkipRuntimeBuild',
    '-PlanOnly',
  ]);
  assert.equal(full.status, 0, full.stderr || full.stdout);
  const fullPlan = JSON.parse(full.stdout.trim());
  assert.equal(fullPlan.config, 'electron-builder-full.yml');
  assert.equal(fullPlan.includesRuntime, true);
  assert.equal(fullPlan.target, 'archive');
  assert.match(fullPlan.runtimeRoot, /desktop[\\/]staging[\\/]avtr-runtime$/i);

  const source = readRequired(desktopBuilder);
  assert.match(source, /run["']?,?\s*["']vendor:frontend/i);
  assert.ok(
    source.indexOf('vendor:frontend') < source.indexOf('& $runtimeBuilder'),
    'frontend vendoring must happen before src is copied into the portable Runtime',
  );
  assert.ok(source.includes('AVTR_PORTABLE_RUNTIME_SOURCE'));
  assert.ok(source.includes('node_modules\\.bin\\electron-builder.cmd'));
  assert.doesNotMatch(source, /"exec",\s*"--",\s*"electron-builder"/);
  assert.match(source, /dependenciesIncluded/);
  assert.match(source, /modelsIncluded/);
});

test('full desktop build refuses staged private payloads before invoking npm', { skip: process.platform !== 'win32' }, () => {
  const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'avtr-full-private-'));
  const runtime = path.join(temporary, 'avtr-runtime');
  const requiredFiles = [
    'python-main/python.exe',
    'python-cosyvoice/python.exe',
    'python-feynobg/python.exe',
    'scripts/desktop/build_tensorrt.ps1',
    'scripts/desktop/DigiBox-TensorRT-Setup.cmd',
    'scripts/desktop/inspect_runtime.py',
  ];
  try {
    for (const relative of requiredFiles) {
      const target = path.join(runtime, ...relative.split('/'));
      fs.mkdirSync(path.dirname(target), { recursive: true });
      fs.writeFileSync(target, 'x', 'utf8');
    }
    fs.writeFileSync(
      path.join(runtime, 'runtime-manifest.json'),
      JSON.stringify({
        schemaVersion: 1,
        layout: 'portable-v1',
        components: {
          dependenciesIncluded: true,
          modelsIncluded: true,
          frontendVendorIncluded: true,
          tensorRtBuildInputsIncluded: true,
        },
      }),
      'utf8',
    );
    const privateFile = path.join(runtime, 'local_voices', 'reference.wav');
    fs.mkdirSync(path.dirname(privateFile), { recursive: true });
    fs.writeFileSync(privateFile, 'private', 'utf8');

    const result = runPowerShellFile(desktopBuilder, [
      '-Edition', 'Full',
      '-SkipRuntimeBuild',
      '-RuntimeDestination', runtime,
      '-NpmExecutable', process.execPath,
    ]);
    assert.notEqual(result.status, 0);
    assert.match(`${result.stderr}\n${result.stdout}`, /forbidden[\s\S]*local_voices/i);
  } finally {
    fs.rmSync(temporary, { recursive: true, force: true });
  }
});

test('full desktop build rejects a Runtime whose manifest records skipped dependencies or models', { skip: process.platform !== 'win32' }, () => {
  const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'avtr-full-incomplete-'));
  const runtime = path.join(temporary, 'avtr-runtime');
  const requiredFiles = [
    'python-main/python.exe',
    'python-cosyvoice/python.exe',
    'python-feynobg/python.exe',
    'scripts/desktop/build_tensorrt.ps1',
    'scripts/desktop/DigiBox-TensorRT-Setup.cmd',
    'scripts/desktop/inspect_runtime.py',
  ];
  try {
    for (const relative of requiredFiles) {
      const target = path.join(runtime, ...relative.split('/'));
      fs.mkdirSync(path.dirname(target), { recursive: true });
      fs.writeFileSync(target, 'x', 'utf8');
    }
    fs.writeFileSync(
      path.join(runtime, 'runtime-manifest.json'),
      JSON.stringify({
        schemaVersion: 1,
        layout: 'portable-v1',
        components: { dependenciesIncluded: false, modelsIncluded: false },
      }),
      'utf8',
    );
    const result = runPowerShellFile(desktopBuilder, [
      '-Edition', 'Full',
      '-SkipRuntimeBuild',
      '-RuntimeDestination', runtime,
      '-NpmExecutable', process.execPath,
    ]);
    assert.notEqual(result.status, 0);
    assert.match(`${result.stderr}\n${result.stdout}`, /incomplete[\s\S]*dependenciesIncluded/i);
  } finally {
    fs.rmSync(temporary, { recursive: true, force: true });
  }
});

test('full electron-builder config embeds the runtime and TensorRT assistant defensively', () => {
  const config = readRequired(fullBuilderConfig);
  assert.ok(config.includes('${env.AVTR_PORTABLE_RUNTIME_SOURCE}'));
  const { expandMacro } = require('app-builder-lib/out/util/macroExpander');
  const previousRuntimeSource = process.env.AVTR_PORTABLE_RUNTIME_SOURCE;
  try {
    process.env.AVTR_PORTABLE_RUNTIME_SOURCE = 'C:\\portable-fixture\\avtr-runtime';
    assert.equal(
      expandMacro('${env.AVTR_PORTABLE_RUNTIME_SOURCE}', null, {}),
      process.env.AVTR_PORTABLE_RUNTIME_SOURCE,
    );
  } finally {
    if (previousRuntimeSource === undefined) {
      delete process.env.AVTR_PORTABLE_RUNTIME_SOURCE;
    } else {
      process.env.AVTR_PORTABLE_RUNTIME_SOURCE = previousRuntimeSource;
    }
  }
  assert.match(config, /to:\s*avtr-runtime/);
  assert.match(config, /target:\s*zip/);
  assert.doesNotMatch(config, /target:\s*nsis/);
  assert.doesNotMatch(config, /compression:\s*store/);
  const desktopSource = readRequired(desktopBuilder);
  assert.match(desktopSource, /build_tensorrt\.ps1/);
  assert.match(desktopSource, /DigiBox-TensorRT-Setup\.cmd/);
  assert.match(desktopSource, /inspect_runtime\.py/);
  assert.match(desktopSource, /Get-FileHash/);
  assert.match(desktopSource, /Language\.Parser\]::ParseFile/);
  assert.doesNotMatch(config, /from:\s*scripts\/desktop/);
  for (const forbidden of [
    '*.engine',
    '*.plan',
    'grid_sample_3d_plugin*.dll',
    'user_assets',
    'artifacts/main/avatars_artifacts/backgrounds',
    'local_voices',
    'voice_clones',
    'reference_audio',
    '.trash',
    '.avtr-portable-runtime.json',
    '*.key',
    'src/**/*.pem',
    'spk2info.pt',
  ]) {
    assert.ok(config.includes(forbidden), `full builder lacks defense-in-depth exclusion: ${forbidden}`);
  }
  assert.match(config, /DigiBox-Full-/);
});

test('distribution guide separates desktop editions, TensorRT modes, privacy, and licenses', () => {
  const guide = readRequired(distributionGuide);
  for (const requiredText of [
    '标准桌面版',
    '完整桌面版',
    'TensorRT 标准模式',
    'TensorRT 完整模式',
    'Visual Studio 2022',
    'CUDA Toolkit',
    'TensorRT 10.11',
    'user_assets',
    'spk2info.pt',
    'PolyForm Noncommercial',
    'LICENSE-MODEL.md',
    'LICENSE-RENDERER.md',
    'LICENSE-STREAMER.md',
  ]) {
    assert.ok(guide.includes(requiredText), `distribution guide is missing: ${requiredText}`);
  }
});
