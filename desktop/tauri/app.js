"use strict";

const { invoke } = window.__TAURI__.core;
const { listen } = window.__TAURI__.event;
const title = document.querySelector("#title");
const message = document.querySelector("#message");
const details = document.querySelector("#details");
const runtime = document.querySelector("#runtime");
const runtimeMode = document.querySelector("#runtime-mode");
const error = document.querySelector("#error");
const retry = document.querySelector("#retry");
const selectRuntime = document.querySelector("#select-runtime");
const openLogs = document.querySelector("#open-logs");

function render(state) {
  const labels = {
    idle: "正在准备数字人运行环境",
    resolving: "正在检查 Windows 运行环境",
    starting: "正在启动 DigiBox",
    ready: "DigiBox 已就绪",
    stopping: "正在安全关闭",
    "runtime-missing": "需要选择 DigiBox Runtime",
    error: "DigiBox 启动失败",
  };
  title.textContent = labels[state.phase] || "DigiBox";
  message.textContent = state.message || "";
  details.hidden = !state.runtime;
  runtime.textContent = state.runtime || "—";
  runtimeMode.textContent = state.runtimeMode === "managed" ? "便携 Runtime" : "开发环境";
  error.hidden = !state.error;
  error.textContent = state.error || "";
  retry.hidden = !["error", "runtime-missing"].includes(state.phase);
  selectRuntime.hidden = retry.hidden;
}

async function run(command, button) {
  button.disabled = true;
  try {
    render(await invoke(command));
  } catch (failure) {
    render({
      phase: "error",
      message: "桌面命令执行失败",
      error: failure instanceof Error ? failure.message : String(failure),
    });
  } finally {
    button.disabled = false;
  }
}

retry.addEventListener("click", () => run("retry_startup", retry));
selectRuntime.addEventListener("click", () => run("select_runtime", selectRuntime));
openLogs.addEventListener("click", () => invoke("open_logs"));
listen("desktop-state", ({ payload }) => render(payload));
invoke("get_desktop_state").then(render).catch((failure) => render({
  phase: "error",
  message: "无法读取桌面启动状态",
  error: failure instanceof Error ? failure.message : String(failure),
}));
