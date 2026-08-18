'use strict';

const {
  app,
  BrowserWindow,
  dialog,
  ipcMain,
  session,
  shell,
} = require('electron');
const fs = require('node:fs');
const path = require('node:path');
const { spawn } = require('node:child_process');
const { pathToFileURL } = require('node:url');

const {
  buildBackendEnvironment,
  childExitMarker,
  ensureBackend,
  isAudioMediaPermissionAllowed,
  isTrustedSplashSender,
  resolveMemoryRoot,
  runTaskkill,
  stopBackend,
} = require('./lib/backend-supervisor.cjs');
const { probeAvtrService, waitForAvtrService } = require('./lib/health.cjs');
const {
  APP_ORIGIN,
  classifyNavigation,
} = require('./lib/navigation.cjs');
const { resolveRuntimeRoot } = require('./lib/runtime-paths.cjs');

const APP_URL = `${APP_ORIGIN}/`;
const HEALTH_URL = `${APP_ORIGIN}/health`;

let mainWindow = null;
let backendSession = null;
let ownedChild = null;
let runtime = null;
let stopping = false;
let relaunchingQuit = false;
let startupPromise = null;
let startupAbortController = null;
let logStream = null;
let activeLogPath = null;
let stopFile = null;

let desktopState = {
  phase: 'idle',
  message: '正在准备 DigiBox…',
  runtime: null,
  runtimeMode: null,
  logPath: null,
  error: null,
};

function emitState(patch = {}) {
  desktopState = { ...desktopState, ...patch };
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send('desktop:state', desktopState);
  }
}

function parseRuntimeArgument(argv) {
  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    if (argument.startsWith('--runtime-root=')) return argument.slice('--runtime-root='.length);
    if (argument === '--runtime-root' && argv[index + 1]) return argv[index + 1];
  }
  return null;
}

function configPath() {
  return path.join(app.getPath('userData'), 'desktop-config.json');
}

function readPersistedRuntime() {
  try {
    const value = JSON.parse(fs.readFileSync(configPath(), 'utf8'));
    return typeof value.runtimeRoot === 'string' ? value.runtimeRoot : null;
  } catch {
    return null;
  }
}

function persistRuntime(root) {
  fs.mkdirSync(path.dirname(configPath()), { recursive: true });
  const temporary = `${configPath()}.${process.pid}.tmp`;
  fs.writeFileSync(temporary, `${JSON.stringify({ runtimeRoot: root }, null, 2)}\n`, 'utf8');
  fs.renameSync(temporary, configPath());
}

function resolveRuntime() {
  return resolveRuntimeRoot({
    explicitRoot: parseRuntimeArgument(process.argv),
    envRoot: process.env.AVTR1_DESKTOP_RUNTIME,
    persistedRoot: readPersistedRuntime(),
    resourcesPath: process.resourcesPath,
    executablePath: process.execPath,
    appPath: app.getAppPath(),
    platform: process.platform,
  });
}

function openLogStream() {
  if (logStream) return logStream;
  const logDirectory = path.join(app.getPath('userData'), 'logs');
  fs.mkdirSync(logDirectory, { recursive: true });
  const stamp = new Date().toISOString().replaceAll(':', '-').replaceAll('.', '-');
  activeLogPath = path.join(logDirectory, `desktop-${stamp}.log`);
  logStream = fs.createWriteStream(activeLogPath, { flags: 'a', encoding: 'utf8' });
  emitState({ logPath: activeLogPath });
  return logStream;
}

function appendLog(streamName, chunk) {
  const line = Buffer.isBuffer(chunk) ? chunk.toString('utf8') : String(chunk);
  openLogStream().write(`[${new Date().toISOString()}] [${streamName}] ${line}`);
  const significant = line.split(/\r?\n/).map((part) => part.trim()).filter(Boolean).at(-1);
  if (significant && desktopState.phase === 'starting') {
    emitState({ message: significant.slice(0, 240) });
  }
}

function managedWorker(root, directory) {
  const candidate = path.join(root, directory, 'python.exe');
  return fs.existsSync(candidate) ? candidate : null;
}

function spawnBackend() {
  stopFile = path.join(
    app.getPath('temp'),
    `avtr1-desktop-${process.pid}-${Date.now()}.stop`,
  );
  try { fs.rmSync(stopFile, { force: true }); } catch { /* best effort */ }

  const environment = buildBackendEnvironment({
    baseEnvironment: process.env,
    runtime,
    stopFile,
    delimiter: path.delimiter,
    userDataRoot: app.getPath('userData'),
    memoryRoot: resolveMemoryRoot(process.env),
  });
  const sharedWorkerPython = runtime.layout === 'portable-v2' ? runtime.python : null;
  const cosyvoice = sharedWorkerPython || managedWorker(runtime.root, 'python-cosyvoice');
  const feynobg = sharedWorkerPython || managedWorker(runtime.root, 'python-feynobg');
  if (cosyvoice) environment.AVTR1_COSYVOICE_PYTHON = cosyvoice;
  if (feynobg) environment.AVTR1_FEYNOBG_PYTHON = feynobg;

  appendLog('desktop', `Starting ${runtime.python} ${runtime.script}\n`);
  const child = spawn(runtime.python, [runtime.script], {
    cwd: runtime.root,
    env: environment,
    windowsHide: true,
    shell: false,
    stdio: ['pipe', 'pipe', 'pipe'],
  });
  if (typeof child.pid === 'number') ownedChild = child;
  child.stdout.on('data', (chunk) => appendLog('backend', chunk));
  child.stderr.on('data', (chunk) => appendLog('backend:error', chunk));
  child.on('error', (error) => {
    if (ownedChild === child && typeof child.pid !== 'number') ownedChild = null;
    appendLog('backend:error', `${error.stack || error}\n`);
  });
  child.on('exit', (code, signalName) => {
    if (ownedChild === child) ownedChild = null;
    appendLog('desktop', `Backend exited code=${code} signal=${signalName || 'none'}\n`);
    if (!stopping && desktopState.phase === 'ready') {
      showSplash();
      emitState({
        phase: 'error',
        message: 'DigiBox 后端意外退出',
        error: `退出码：${code ?? '未知'}。请打开日志查看详细原因。`,
      });
    }
  });
  return child;
}

function waitForChildExit(child, timeoutMs = 20000) {
  if (child.exitCode !== null || child.signalCode !== null) return Promise.resolve(true);
  return new Promise((resolve) => {
    let settled = false;
    let timer = null;
    const finish = (value) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      child.removeListener('exit', onExit);
      resolve(value);
    };
    const onExit = () => finish(true);
    timer = setTimeout(() => finish(false), timeoutMs);
    child.once('exit', onExit);
  });
}

function forceKillTree(child) {
  return runTaskkill(child, { spawnProcess: spawn });
}

async function requestCooperativeStop() {
  if (!stopFile) return;
  try {
    fs.writeFileSync(stopFile, `stop ${new Date().toISOString()}\n`, 'utf8');
  } catch (error) {
    appendLog('desktop:error', `Could not write stop file: ${error}\n`);
  }
}

function stopOwnedSession(sessionToStop) {
  if (!sessionToStop || sessionToStop.ownership !== 'desktop') {
    return Promise.resolve({ stopped: false, forced: false });
  }
  return stopBackend(sessionToStop, {
    requestStop: requestCooperativeStop,
    waitForExit: (child) => waitForChildExit(child, 20000),
    confirmExit: (child) => waitForChildExit(child, 5000),
    killTree: forceKillTree,
  });
}

function terminateOwnedChild(child) {
  return stopOwnedSession({ ownership: 'desktop', child });
}

function currentOwnedSession() {
  if (backendSession?.ownership === 'desktop' && backendSession.child) return backendSession;
  if (ownedChild) return { ownership: 'desktop', child: ownedChild };
  return null;
}

function splashUrl() {
  const splashPath = path.join(__dirname, 'splash.html');
  return pathToFileURL(splashPath).href;
}

function showSplash() {
  if (!mainWindow || mainWindow.isDestroyed()) return;
  const splashPath = path.join(__dirname, 'splash.html');
  const current = mainWindow.webContents.getURL();
  const targetUrl = pathToFileURL(splashPath).href;
  if (current !== targetUrl) {
    void mainWindow.loadFile(splashPath).catch((error) => {
      if (!stopping) appendLog('desktop:error', `Could not load splash: ${error}\n`);
    });
  }
}

async function startDesktopBackend() {
  if (startupPromise) return startupPromise;
  const controller = new AbortController();
  startupAbortController = controller;
  startupPromise = (async () => {
    showSplash();
    emitState({ phase: 'resolving', message: '正在检查 Windows 运行环境…', error: null });
    runtime = resolveRuntime();
    if (!runtime.valid) {
      emitState({
        phase: 'runtime-missing',
        message: '尚未找到可用的 DigiBox Runtime',
        runtime: null,
        runtimeMode: null,
        error: `缺少：${runtime.missing.join('、')}`,
      });
      return;
    }
    emitState({
      phase: 'starting',
      message: '正在启动数字人渲染与语音服务…',
      runtime: runtime.root,
      runtimeMode: runtime.mode,
      error: null,
    });
    try {
      if (backendSession?.ownership === 'desktop') {
        if (childExitMarker(backendSession.child) === null) {
          await stopOwnedSession(backendSession);
        }
        backendSession = null;
      }
      backendSession = await ensureBackend({
        signal: controller.signal,
        probe: () => probeAvtrService({
          url: HEALTH_URL,
          requestTimeoutMs: 1200,
          signal: controller.signal,
        }),
        spawnBackend,
        waitForReady: (child) => waitForAvtrService({
          url: HEALTH_URL,
          timeoutMs: 300000,
          intervalMs: 500,
          signal: controller.signal,
          processPoll: () => childExitMarker(child),
        }),
        onStartupFailure: terminateOwnedChild,
      });
      emitState({
        phase: 'ready',
        message: backendSession.ownership === 'external'
          ? '已连接当前正在运行的 DigiBox 服务'
          : 'DigiBox 已就绪',
        error: null,
      });
      await mainWindow.loadURL(APP_URL);
    } catch (error) {
      let cleanupError = null;
      if (backendSession?.ownership === 'desktop') {
        try {
          await stopOwnedSession(backendSession);
        } catch (failure) {
          cleanupError = failure;
          appendLog('desktop:error', `Could not stop failed backend: ${failure}\n`);
        }
      }
      if (!cleanupError) backendSession = null;
      if (stopping || controller.signal.aborted) return;
      showSplash();
      emitState({
        phase: 'error',
        message: 'DigiBox 启动失败',
        error: [error, cleanupError]
          .filter(Boolean)
          .map((item) => (item instanceof Error ? item.message : String(item)))
          .join('\n'),
      });
    }
  })().finally(() => {
    if (startupAbortController === controller) startupAbortController = null;
    startupPromise = null;
  });
  return startupPromise;
}

function trustedSender(event) {
  return Boolean(mainWindow && !mainWindow.isDestroyed() && event.sender.id === mainWindow.webContents.id);
}

function trustedSplashSender(event) {
  return isTrustedSplashSender(event, mainWindow, splashUrl());
}

function installIpcHandlers() {
  ipcMain.handle('desktop:get-state', (event) => {
    if (!trustedSender(event)) throw new Error('Untrusted IPC sender');
    return desktopState;
  });
  ipcMain.handle('desktop:retry', async (event) => {
    if (!trustedSplashSender(event)) throw new Error('Untrusted IPC sender');
    await startDesktopBackend();
    return desktopState;
  });
  ipcMain.handle('desktop:select-runtime', async (event) => {
    if (!trustedSplashSender(event)) throw new Error('Untrusted IPC sender');
    const selection = await dialog.showOpenDialog(mainWindow, {
      title: '选择 DigiBox Runtime 文件夹',
      properties: ['openDirectory'],
    });
    if (selection.canceled || !selection.filePaths[0]) return desktopState;
    persistRuntime(selection.filePaths[0]);
    await startDesktopBackend();
    return desktopState;
  });
  ipcMain.handle('desktop:open-logs', async (event) => {
    if (!trustedSplashSender(event)) throw new Error('Untrusted IPC sender');
    const target = activeLogPath
      ? path.dirname(activeLogPath)
      : path.join(app.getPath('userData'), 'logs');
    fs.mkdirSync(target, { recursive: true });
    return shell.openPath(target);
  });
}

function installSessionSecurity() {
  session.defaultSession.setPermissionCheckHandler((_webContents, permission, requestingOrigin, details) => (
    isAudioMediaPermissionAllowed(permission, requestingOrigin, details, APP_ORIGIN)
  ));
  session.defaultSession.setPermissionRequestHandler((webContents, permission, callback, details) => {
    const requestingUrl = details?.requestingUrl || webContents.getURL();
    callback(isAudioMediaPermissionAllowed(
      permission,
      requestingUrl,
      { mediaTypes: details?.mediaTypes },
      APP_ORIGIN,
    ));
  });
}

function createWindow() {
  mainWindow = new BrowserWindow({
    title: 'DigiBox',
    width: 1440,
    height: 900,
    minWidth: 900,
    minHeight: 620,
    show: false,
    autoHideMenuBar: true,
    backgroundColor: '#03070d',
    webPreferences: {
      preload: path.join(__dirname, 'preload.cjs'),
      nodeIntegration: false,
      contextIsolation: true,
      sandbox: true,
      webSecurity: true,
      backgroundThrottling: false,
    },
  });
  mainWindow.webContents.on('will-navigate', (event, url) => {
    const classification = classifyNavigation(url);
    if (classification === 'app') return;
    event.preventDefault();
    if (classification === 'external') void shell.openExternal(url);
  });
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    if (classifyNavigation(url) === 'external') void shell.openExternal(url);
    return { action: 'deny' };
  });
  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
    mainWindow.maximize();
  });
  showSplash();
  void startDesktopBackend();
}

const hasSingleInstanceLock = app.requestSingleInstanceLock();
if (!hasSingleInstanceLock) {
  app.quit();
} else {
  app.on('second-instance', () => {
    if (!mainWindow) return;
    if (mainWindow.isMinimized()) mainWindow.restore();
    mainWindow.show();
    mainWindow.focus();
  });

  app.whenReady().then(() => {
    installIpcHandlers();
    installSessionSecurity();
    createWindow();
  });

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });

  app.on('window-all-closed', () => app.quit());
  app.on('before-quit', (event) => {
    if (relaunchingQuit) return;
    const startupToWait = startupPromise;
    const ownedSession = currentOwnedSession();
    if (!startupToWait && !ownedSession) return;
    event.preventDefault();
    if (stopping) return;
    stopping = true;
    startupAbortController?.abort();
    emitState({ phase: 'stopping', message: '正在安全关闭 DigiBox 服务…' });
    void (async () => {
      if (startupToWait) {
        try { await startupToWait; } catch { /* the startup path logs its own failure */ }
      }
      const sessionToStop = currentOwnedSession() || ownedSession;
      if (sessionToStop) await stopOwnedSession(sessionToStop);
    })().catch((error) => {
      appendLog('desktop:error', `Could not stop backend cleanly: ${error}\n`);
    }).finally(() => {
      backendSession = null;
      ownedChild = null;
      relaunchingQuit = true;
      if (logStream) logStream.end();
      app.quit();
    });
  });
}
