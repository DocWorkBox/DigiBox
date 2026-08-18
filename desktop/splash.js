'use strict';

const title = document.querySelector('#title');
const message = document.querySelector('#message');
const details = document.querySelector('#details');
const runtime = document.querySelector('#runtime');
const runtimeMode = document.querySelector('#runtime-mode');
const error = document.querySelector('#error');
const retry = document.querySelector('#retry');
const selectRuntime = document.querySelector('#select-runtime');
const openLogs = document.querySelector('#open-logs');

function render(state) {
  const labels = {
    idle: '正在准备数字人运行环境',
    resolving: '正在检查 Windows 运行环境',
    starting: '正在启动 DigiBox',
    ready: 'DigiBox 已就绪',
    stopping: '正在安全关闭',
    'runtime-missing': '需要选择 DigiBox Runtime',
    error: 'DigiBox 启动失败',
  };
  title.textContent = labels[state.phase] || 'DigiBox';
  message.textContent = state.message || '';
  details.hidden = !state.runtime;
  runtime.textContent = state.runtime || '—';
  runtimeMode.textContent = state.runtimeMode === 'managed' ? '便携 Runtime' : '开发环境';
  error.hidden = !state.error;
  error.textContent = state.error || '';
  retry.hidden = !['error', 'runtime-missing'].includes(state.phase);
  selectRuntime.hidden = !['error', 'runtime-missing'].includes(state.phase);
  retry.disabled = state.phase === 'resolving' || state.phase === 'starting';
  selectRuntime.disabled = retry.disabled;
}

async function run(action, button) {
  button.disabled = true;
  try { render(await action()); }
  finally { button.disabled = false; }
}

retry.addEventListener('click', () => run(() => window.avtrDesktop.retry(), retry));
selectRuntime.addEventListener('click', () => run(() => window.avtrDesktop.selectRuntime(), selectRuntime));
openLogs.addEventListener('click', () => window.avtrDesktop.openLogs());
window.avtrDesktop.onState(render);
window.avtrDesktop.getState().then(render);
