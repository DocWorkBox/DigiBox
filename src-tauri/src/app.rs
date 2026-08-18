use std::{
    env, fs,
    path::{Path, PathBuf},
    process::ExitStatus,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Mutex,
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};
use tauri::{
    webview::NewWindowResponse, AppHandle, Emitter, Manager, RunEvent, State, Url, WebviewWindow,
    WebviewWindowBuilder,
};
use tauri_plugin_dialog::DialogExt;
use tauri_plugin_opener::OpenerExt;
use tokio::sync::Mutex as AsyncMutex;

use crate::{
    health::{probe_avtr_service, HealthFailure, HealthProbe},
    navigation::{classify_navigation, NavigationDecision},
    runtime::{
        inspect_runtime_root, resolve_runtime_root, RuntimeInfo, RuntimeMode, RuntimeResolveOptions,
    },
    supervisor::{
        BackendOwnership, BackendSession, RuntimeLaunch, RuntimeLaunchMode, StartupProgress,
    },
};

const SPLASH_LABEL: &str = "splash";
const MAIN_LABEL: &str = "main";
const APP_URL: &str = "http://127.0.0.1:7860/";
const HEALTH_URL: &str = "http://127.0.0.1:7860/health";
const STARTUP_TIMEOUT: Duration = Duration::from_secs(300);
const HEALTH_INTERVAL: Duration = Duration::from_millis(500);
const HEALTH_REQUEST_TIMEOUT: Duration = Duration::from_millis(1_200);
const OWNED_MONITOR_INTERVAL: Duration = Duration::from_millis(500);
const OWNED_MONITOR_FAILURE_LIMIT: u8 = 5;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AppRootFailure {
    RequestFailed(String),
    HttpStatus(u16),
}

pub async fn probe_app_root(
    client: &reqwest::Client,
    url: &str,
    request_timeout: Duration,
) -> Result<(), AppRootFailure> {
    let response = client
        .get(url)
        .timeout(request_timeout)
        .send()
        .await
        .map_err(|error| AppRootFailure::RequestFailed(error.to_string()))?;
    let status = response.status();
    if !status.is_success() {
        return Err(AppRootFailure::HttpStatus(status.as_u16()));
    }
    Ok(())
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DesktopState {
    phase: String,
    message: String,
    runtime: Option<String>,
    runtime_mode: Option<String>,
    log_path: Option<String>,
    error: Option<String>,
}

impl DesktopState {
    fn initial(log_dir: &Path) -> Self {
        Self {
            phase: "idle".to_owned(),
            message: "正在准备 DigiBox 桌面运行环境…".to_owned(),
            runtime: None,
            runtime_mode: None,
            log_path: Some(log_dir.display().to_string()),
            error: None,
        }
    }
}

#[derive(Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct PersistedDesktopConfig {
    runtime_root: Option<PathBuf>,
}

pub struct DesktopApp {
    state: Mutex<DesktopState>,
    session: Mutex<Option<BackendSession>>,
    startup_gate: AsyncMutex<()>,
    selected_runtime: Mutex<Option<PathBuf>>,
    exit_started: AtomicBool,
    final_exit: AtomicBool,
    startup_generation: AtomicU64,
    client: reqwest::Client,
    config_path: PathBuf,
    log_dir: PathBuf,
    stop_dir: PathBuf,
    user_assets_dir: PathBuf,
    cosyvoice_speaker_cache: PathBuf,
    memory_root: PathBuf,
    resources_dir: Option<PathBuf>,
    executable_path: Option<PathBuf>,
    development_roots: Vec<PathBuf>,
    explicit_runtime: Option<PathBuf>,
}

impl DesktopApp {
    fn new(app: &AppHandle) -> Result<Self, String> {
        let config_dir = app.path().app_config_dir().map_err(display_error)?;
        let log_dir = app.path().app_log_dir().map_err(display_error)?;
        let local_data = app.path().app_local_data_dir().map_err(display_error)?;
        let local_app_data = app.path().local_data_dir().map_err(display_error)?;
        fs::create_dir_all(&config_dir).map_err(display_error)?;
        fs::create_dir_all(&log_dir).map_err(display_error)?;
        let stop_dir = local_data.join("stops");
        fs::create_dir_all(&stop_dir).map_err(display_error)?;
        let user_assets_dir = local_data.join("user_assets");
        let cosyvoice_speaker_cache = local_data.join("cosyvoice").join("spk2info.pt");
        let memory_root = memory_root_from_local_app_data(&local_app_data);

        let config_path = config_dir.join("desktop-config.json");
        let persisted = read_persisted_config(&config_path)
            .ok()
            .and_then(|config| config.runtime_root);
        let selected_runtime = persisted.clone();
        let mut development_roots = Vec::new();
        if let Ok(current) = env::current_dir() {
            development_roots.push(current);
        }
        if let Some(source_root) = Path::new(env!("CARGO_MANIFEST_DIR")).parent() {
            development_roots.push(source_root.to_path_buf());
        }

        Ok(Self {
            state: Mutex::new(DesktopState::initial(&log_dir)),
            session: Mutex::new(None),
            startup_gate: AsyncMutex::new(()),
            selected_runtime: Mutex::new(selected_runtime),
            exit_started: AtomicBool::new(false),
            final_exit: AtomicBool::new(false),
            startup_generation: AtomicU64::new(0),
            client: reqwest::Client::new(),
            config_path,
            log_dir,
            stop_dir,
            user_assets_dir,
            cosyvoice_speaker_cache,
            memory_root,
            resources_dir: app.path().resource_dir().ok(),
            executable_path: env::current_exe().ok(),
            development_roots,
            explicit_runtime: parse_runtime_argument(env::args_os()),
        })
    }

    fn snapshot(&self) -> DesktopState {
        lock(&self.state).clone()
    }
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn display_error(error: impl std::fmt::Display) -> String {
    error.to_string()
}

fn parse_runtime_argument(
    arguments: impl IntoIterator<Item = std::ffi::OsString>,
) -> Option<PathBuf> {
    let arguments = arguments.into_iter().collect::<Vec<_>>();
    for (index, argument) in arguments.iter().enumerate() {
        let value = argument.to_string_lossy();
        if let Some(root) = value.strip_prefix("--runtime-root=") {
            return (!root.is_empty()).then(|| PathBuf::from(root));
        }
        if value == "--runtime-root" {
            return arguments.get(index + 1).map(PathBuf::from);
        }
    }
    None
}

fn read_persisted_config(path: &Path) -> Result<PersistedDesktopConfig, String> {
    if !path.is_file() {
        return Ok(PersistedDesktopConfig::default());
    }
    let body = fs::read(path).map_err(display_error)?;
    serde_json::from_slice(&body).map_err(display_error)
}

async fn persist_runtime(path: &Path, runtime_root: &Path) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(display_error)?;
    }
    let config = PersistedDesktopConfig {
        runtime_root: Some(runtime_root.to_path_buf()),
    };
    let body = serde_json::to_vec_pretty(&config).map_err(display_error)?;
    tokio::fs::write(path, body).await.map_err(display_error)
}

fn update_state(app: &AppHandle, change: impl FnOnce(&mut DesktopState)) -> DesktopState {
    let state = app.state::<DesktopApp>();
    let snapshot = {
        let mut current = lock(&state.state);
        change(&mut current);
        current.clone()
    };
    let _ = app.emit_to(SPLASH_LABEL, "desktop-state", &snapshot);
    snapshot
}

fn append_desktop_log(app: &AppHandle, message: &str) {
    let state = app.state::<DesktopApp>();
    if fs::create_dir_all(&state.log_dir).is_err() {
        return;
    }
    let path = state.log_dir.join("avtr-desktop.log");
    if let Ok(mut file) = fs::OpenOptions::new().create(true).append(true).open(path) {
        use std::io::Write;
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |value| value.as_secs());
        let _ = writeln!(file, "[{stamp}] {message}");
    }
}

fn runtime_mode_name(mode: Option<RuntimeMode>) -> Option<String> {
    match mode {
        Some(RuntimeMode::Managed) => Some("managed".to_owned()),
        Some(RuntimeMode::Development) => Some("development".to_owned()),
        None => None,
    }
}

fn resolve_runtime(state: &DesktopApp) -> Result<RuntimeInfo, String> {
    let persisted_root = lock(&state.selected_runtime).clone();
    let resolution = resolve_runtime_root(&RuntimeResolveOptions {
        explicit_root: state.explicit_runtime.clone(),
        environment_root: env::var_os("AVTR1_DESKTOP_RUNTIME").map(PathBuf::from),
        persisted_root,
        resources_dir: state.resources_dir.clone(),
        executable_path: state.executable_path.clone(),
        development_roots: state.development_roots.clone(),
    });
    resolution
        .selected
        .map(|selected| selected.runtime)
        .ok_or_else(|| {
            let details = resolution
                .candidates
                .iter()
                .map(|candidate| {
                    format!(
                        "{}（缺少：{}）",
                        candidate.runtime.root.display(),
                        candidate.runtime.missing.join("、")
                    )
                })
                .collect::<Vec<_>>();
            if details.is_empty() {
                "未找到 DigiBox Runtime；请选择包含 Python、scripts、src 和 artifacts/main 的目录。"
                    .to_owned()
            } else {
                format!("候选 Runtime 均不完整：\n{}", details.join("\n"))
            }
        })
}

pub fn launch_from_runtime(runtime: &RuntimeInfo) -> Result<RuntimeLaunch, String> {
    let mode = match runtime.mode {
        Some(RuntimeMode::Managed) => RuntimeLaunchMode::Managed,
        Some(RuntimeMode::Development) => RuntimeLaunchMode::Development,
        None => return Err("Runtime 没有可用的 Python 解释器。".to_owned()),
    };
    let python = runtime
        .python
        .clone()
        .ok_or_else(|| "Runtime 没有可用的 Python 解释器。".to_owned())?;
    Ok(RuntimeLaunch {
        mode,
        manifest_layout: runtime.manifest_layout,
        root: runtime.root.clone(),
        python,
        orchestrator: runtime.orchestrator.clone(),
        source_dir: runtime.source_dir.clone(),
        package_layers: runtime.package_layers.clone(),
    })
}

fn classify_health_conflict(probe: &HealthProbe) -> Option<String> {
    match &probe.failure {
        Some(HealthFailure::RequestFailed(_)) | None => None,
        Some(HealthFailure::ServiceStatus(_)) => None,
        Some(HealthFailure::IdentityMismatch) => {
            Some("端口 7860 已被非 AVTR-1 服务占用；桌面程序不会接管或关闭该进程。".to_owned())
        }
        Some(HealthFailure::HttpStatus(status)) => Some(format!(
            "端口 7860 上的服务返回 HTTP {status}，无法确认其 AVTR-1 身份。"
        )),
        Some(HealthFailure::InvalidJson(_)) => {
            Some("端口 7860 上的服务不是可识别的 AVTR-1 服务。".to_owned())
        }
    }
}

fn new_owned_instance_id(generation: u64) -> String {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |value| value.as_nanos());
    format!("desktop-{}-{generation}-{timestamp}", std::process::id())
}

pub fn memory_root_from_local_app_data(local_app_data: &Path) -> PathBuf {
    local_app_data.join("DigiBox").join("memory")
}

pub fn backend_environment(
    base_environment: impl IntoIterator<Item = (std::ffi::OsString, std::ffi::OsString)>,
    instance_id: &str,
    user_assets_dir: &Path,
    cosyvoice_speaker_cache: &Path,
    memory_root: &Path,
) -> Vec<(std::ffi::OsString, std::ffi::OsString)> {
    let mut environment = base_environment.into_iter().collect::<Vec<_>>();
    environment.retain(|(key, _)| !key.eq_ignore_ascii_case("AVTR1_MEMORY_ROOT"));
    environment.push(("AVTR1_DESKTOP_INSTANCE_ID".into(), instance_id.into()));
    environment.push((
        "AVTR1_USER_ASSETS_ROOT".into(),
        user_assets_dir.as_os_str().to_owned(),
    ));
    environment.push((
        "AVTR1_COSYVOICE_SPEAKER_CACHE".into(),
        cosyvoice_speaker_cache.as_os_str().to_owned(),
    ));
    environment.push((
        "AVTR1_MEMORY_ROOT".into(),
        memory_root.as_os_str().to_owned(),
    ));
    environment
}

async fn wait_for_existing_service(app: &AppHandle) -> Result<(), String> {
    let state = app.state::<DesktopApp>();
    let deadline = tokio::time::Instant::now() + STARTUP_TIMEOUT;
    loop {
        if state.exit_started.load(Ordering::Acquire) {
            return Err("桌面程序正在退出。".to_owned());
        }
        let probe = probe_avtr_service(&state.client, HEALTH_URL, HEALTH_REQUEST_TIMEOUT).await;
        if probe.healthy {
            return Ok(());
        }
        if let Some(error) = classify_health_conflict(&probe) {
            return Err(error);
        }
        if tokio::time::Instant::now() >= deadline {
            return Err("等待已有 AVTR-1 服务就绪超时。".to_owned());
        }
        tokio::time::sleep(HEALTH_INTERVAL).await;
    }
}

fn ensure_app_root_wait_active(state: &DesktopApp, generation: u64) -> Result<(), String> {
    if state.exit_started.load(Ordering::Acquire) {
        return Err("桌面程序正在退出。".to_owned());
    }
    if state.startup_generation.load(Ordering::Acquire) != generation {
        return Err("DigiBox 启动请求已被新的启动流程取消。".to_owned());
    }
    if let Some(status) = current_owned_exit(state)? {
        return Err(format!("DigiBox 后端在页面就绪前退出：{status}"));
    }
    Ok(())
}

async fn wait_for_app_root(app: &AppHandle, generation: u64) -> Result<(), String> {
    let state = app.state::<DesktopApp>();
    let deadline = tokio::time::Instant::now() + STARTUP_TIMEOUT;
    loop {
        ensure_app_root_wait_active(&state, generation)?;
        let root_probe = probe_app_root(&state.client, APP_URL, HEALTH_REQUEST_TIMEOUT).await;
        ensure_app_root_wait_active(&state, generation)?;
        match root_probe {
            Ok(()) => return Ok(()),
            Err(error) if tokio::time::Instant::now() >= deadline => {
                return Err(format!("DigiBox 页面尚不可用：{error:?}"));
            }
            Err(_) => tokio::time::sleep(HEALTH_INTERVAL).await,
        }
    }
}

fn current_owned_exit(state: &DesktopApp) -> Result<Option<ExitStatus>, String> {
    let mut session = lock(&state.session);
    let Some(session) = session.as_mut() else {
        return Ok(None);
    };
    if session.ownership() != BackendOwnership::Owned {
        return Ok(None);
    }
    session.try_wait().map_err(display_error)
}

async fn stop_current_session(app: &AppHandle) -> Result<(), String> {
    let session = {
        let state = app.state::<DesktopApp>();
        let session = lock(&state.session).take();
        session
    };
    let Some(mut session) = session else {
        return Ok(());
    };
    if session.ownership() == BackendOwnership::External {
        return Ok(());
    }
    let (session, result) = tauri::async_runtime::spawn_blocking(move || {
        let result = session.stop();
        (session, result)
    })
    .await
    .map_err(display_error)?;
    if let Err(error) = result {
        let state = app.state::<DesktopApp>();
        *lock(&state.session) = Some(session);
        return Err(error.to_string());
    }
    Ok(())
}

fn progress_message(progress: StartupProgress) -> &'static str {
    match progress {
        StartupProgress::PreparingLogs => "正在准备后端日志…",
        StartupProgress::SpawningOrchestrator => "正在启动 Python 服务编排器…",
        StartupProgress::AssigningJob => "正在建立 Windows 进程树监管…",
        StartupProgress::Started => "后端已启动，正在等待 GPU 服务就绪…",
    }
}

async fn wait_for_owned_service(app: &AppHandle, expected_instance_id: &str) -> Result<(), String> {
    let state = app.state::<DesktopApp>();
    let deadline = tokio::time::Instant::now() + STARTUP_TIMEOUT;
    loop {
        if state.exit_started.load(Ordering::Acquire) {
            return Err("桌面程序正在退出。".to_owned());
        }
        if let Some(status) = current_owned_exit(&state)? {
            return Err(format!("DigiBox 后端在就绪前退出：{status}"));
        }
        let probe = probe_avtr_service(&state.client, HEALTH_URL, HEALTH_REQUEST_TIMEOUT).await;
        if probe.matches_owned_instance(expected_instance_id) {
            return Ok(());
        }
        if probe.healthy {
            return Err(
                "端口 7860 上出现了另一个 AVTR-1 实例；桌面程序不会把它误认为自己启动的后端。"
                    .to_owned(),
            );
        }
        if let Some(error) = classify_health_conflict(&probe) {
            return Err(error);
        }
        if tokio::time::Instant::now() >= deadline {
            return Err("等待 DigiBox 后端就绪超时，请打开日志查看原因。".to_owned());
        }
        tokio::time::sleep(HEALTH_INTERVAL).await;
    }
}

fn open_external(app: &AppHandle, url: &Url) {
    let _ = app.opener().open_url(url.as_str(), None::<&str>);
}

fn create_main_window(app: &AppHandle) -> Result<WebviewWindow, String> {
    if let Some(window) = app.get_webview_window(MAIN_LABEL) {
        window.show().map_err(display_error)?;
        window.maximize().map_err(display_error)?;
        window.set_focus().map_err(display_error)?;
        return Ok(window);
    }
    let mut config = app
        .config()
        .app
        .windows
        .iter()
        .find(|config| config.label == MAIN_LABEL)
        .cloned()
        .ok_or_else(|| "缺少 main WebView 配置。".to_owned())?;
    config.url = tauri::WebviewUrl::External(APP_URL.parse().map_err(display_error)?);
    let navigation_app = app.clone();
    let new_window_app = app.clone();
    let window = WebviewWindowBuilder::from_config(app, &config)
        .map_err(display_error)?
        .on_navigation(move |url| match classify_navigation(url.as_str(), "") {
            NavigationDecision::Application => true,
            NavigationDecision::External => {
                open_external(&navigation_app, url);
                false
            }
            NavigationDecision::Splash | NavigationDecision::Deny => false,
        })
        .on_new_window(move |url, _features| {
            if classify_navigation(url.as_str(), "") == NavigationDecision::External {
                open_external(&new_window_app, &url);
            }
            NewWindowResponse::Deny
        })
        .build()
        .map_err(display_error)?;
    window.show().map_err(display_error)?;
    window.maximize().map_err(display_error)?;
    window.set_focus().map_err(display_error)?;
    Ok(window)
}

fn create_splash_window(app: &AppHandle) -> Result<WebviewWindow, String> {
    if let Some(window) = app.get_webview_window(SPLASH_LABEL) {
        window.show().map_err(display_error)?;
        window.unminimize().map_err(display_error)?;
        window.set_focus().map_err(display_error)?;
        return Ok(window);
    }
    let config = app
        .config()
        .app
        .windows
        .iter()
        .find(|config| config.label == SPLASH_LABEL)
        .cloned()
        .ok_or_else(|| "缺少 splash WebView 配置。".to_owned())?;
    let window = WebviewWindowBuilder::from_config(app, &config)
        .map_err(display_error)?
        .build()
        .map_err(display_error)?;
    window.show().map_err(display_error)?;
    window.set_focus().map_err(display_error)?;
    Ok(window)
}

async fn show_ready_window(app: &AppHandle, generation: u64) -> Result<(), String> {
    wait_for_app_root(app, generation).await?;
    let _main = create_main_window(app)?;
    if let Some(splash) = app.get_webview_window(SPLASH_LABEL) {
        splash.destroy().map_err(display_error)?;
    }
    Ok(())
}

async fn transition_backend_failure(app: AppHandle, generation: u64, reason: String) {
    let state = app.state::<DesktopApp>();
    if state.exit_started.load(Ordering::Acquire)
        || state
            .startup_generation
            .compare_exchange(
                generation,
                generation + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_err()
    {
        return;
    }

    let _startup_gate = state.startup_gate.lock().await;
    if state.exit_started.load(Ordering::Acquire) {
        return;
    }

    let ownership = lock(&state.session).as_ref().map(BackendSession::ownership);
    let owns_backend = ownership == Some(BackendOwnership::Owned);
    update_state(&app, |current| {
        current.phase = "error".to_owned();
        current.message = if owns_backend {
            "DigiBox 后端连接已中断，正在清理进程…".to_owned()
        } else {
            "DigiBox 外部服务连接已中断。".to_owned()
        };
        current.error = Some(reason.clone());
    });
    let splash_ready = if let Err(error) = create_splash_window(&app) {
        append_desktop_log(
            &app,
            &format!("could not recreate splash after backend failure: {error}"),
        );
        false
    } else {
        true
    };
    if splash_ready {
        if let Some(main) = app.get_webview_window(MAIN_LABEL) {
            let _ = main.destroy();
        }
    }

    let cleanup_error = if ownership == Some(BackendOwnership::Owned) {
        stop_current_session(&app).await.err()
    } else {
        if ownership == Some(BackendOwnership::External) {
            lock(&state.session).take();
        }
        None
    };
    append_desktop_log(&app, &format!("backend supervision failed: {reason}"));
    update_state(&app, |current| {
        current.phase = "error".to_owned();
        current.message = "DigiBox 后端已停止。".to_owned();
        current.error = Some(match cleanup_error {
            Some(cleanup) => format!("{reason}\n清理后端时又发生错误：{cleanup}"),
            None => reason,
        });
    });
}

fn start_external_backend_monitor(app: AppHandle, generation: u64) {
    tauri::async_runtime::spawn(async move {
        let mut consecutive_health_failures = 0_u8;
        loop {
            tokio::time::sleep(OWNED_MONITOR_INTERVAL).await;
            let state = app.state::<DesktopApp>();
            if state.exit_started.load(Ordering::Acquire)
                || state.startup_generation.load(Ordering::Acquire) != generation
            {
                return;
            }
            let still_external = lock(&state.session)
                .as_ref()
                .is_some_and(|session| session.ownership() == BackendOwnership::External);
            if !still_external {
                return;
            }

            let probe = probe_avtr_service(&state.client, HEALTH_URL, HEALTH_REQUEST_TIMEOUT).await;
            if probe.healthy {
                consecutive_health_failures = 0;
                continue;
            }
            consecutive_health_failures = consecutive_health_failures.saturating_add(1);
            if consecutive_health_failures < OWNED_MONITOR_FAILURE_LIMIT {
                continue;
            }

            transition_backend_failure(
                app.clone(),
                generation,
                "DigiBox 外部服务健康检查连续失败，连接已丢失。".to_owned(),
            )
            .await;
            return;
        }
    });
}

fn start_owned_backend_monitor(app: AppHandle, generation: u64, instance_id: String) {
    tauri::async_runtime::spawn(async move {
        let mut consecutive_health_failures = 0_u8;
        loop {
            tokio::time::sleep(OWNED_MONITOR_INTERVAL).await;
            let state = app.state::<DesktopApp>();
            if state.exit_started.load(Ordering::Acquire)
                || state.startup_generation.load(Ordering::Acquire) != generation
            {
                return;
            }

            match current_owned_exit(&state) {
                Ok(Some(status)) => {
                    transition_backend_failure(
                        app.clone(),
                        generation,
                        format!("DigiBox 后端进程已退出：{status}"),
                    )
                    .await;
                    return;
                }
                Err(error) => {
                    transition_backend_failure(
                        app.clone(),
                        generation,
                        format!("无法检查 DigiBox 后端进程：{error}"),
                    )
                    .await;
                    return;
                }
                Ok(None) => {}
            }

            let still_owned = lock(&state.session)
                .as_ref()
                .is_some_and(|session| session.ownership() == BackendOwnership::Owned);
            if !still_owned {
                return;
            }

            let probe = probe_avtr_service(&state.client, HEALTH_URL, HEALTH_REQUEST_TIMEOUT).await;
            if probe.matches_owned_instance(&instance_id) {
                consecutive_health_failures = 0;
                continue;
            }
            consecutive_health_failures = consecutive_health_failures.saturating_add(1);
            if consecutive_health_failures < OWNED_MONITOR_FAILURE_LIMIT {
                continue;
            }

            transition_backend_failure(
                app.clone(),
                generation,
                "DigiBox 后端健康检查连续失败，连接已丢失。".to_owned(),
            )
            .await;
            return;
        }
    });
}

async fn startup_inner(app: &AppHandle, generation: u64) -> Result<(), String> {
    let state = app.state::<DesktopApp>();
    let initial_probe = probe_avtr_service(&state.client, HEALTH_URL, HEALTH_REQUEST_TIMEOUT).await;
    if initial_probe.healthy {
        *lock(&state.session) = Some(BackendSession::external());
        update_state(app, |current| {
            current.phase = "ready".to_owned();
            current.message = "已连接当前正在运行的 DigiBox 服务。".to_owned();
            current.error = None;
        });
        show_ready_window(app, generation).await?;
        start_external_backend_monitor(app.clone(), generation);
        return Ok(());
    }
    if matches!(
        initial_probe.failure,
        Some(HealthFailure::ServiceStatus(ref status)) if status == "starting"
    ) {
        wait_for_existing_service(app).await?;
        *lock(&state.session) = Some(BackendSession::external());
        update_state(app, |current| {
            current.phase = "ready".to_owned();
            current.message = "已有 DigiBox 服务现已就绪。".to_owned();
            current.error = None;
        });
        show_ready_window(app, generation).await?;
        start_external_backend_monitor(app.clone(), generation);
        return Ok(());
    }
    if let Some(error) = classify_health_conflict(&initial_probe) {
        return Err(error);
    }

    let runtime = resolve_runtime(&state)?;
    let launch = launch_from_runtime(&runtime)?;
    update_state(app, |current| {
        current.phase = "starting".to_owned();
        current.message = "正在检查 DigiBox 后端启动条件…".to_owned();
        current.runtime = Some(runtime.root.display().to_string());
        current.runtime_mode = runtime_mode_name(runtime.mode);
        current.error = None;
    });

    if state.exit_started.load(Ordering::Acquire) {
        return Err("桌面程序正在退出。".to_owned());
    }
    let second_probe = probe_avtr_service(&state.client, HEALTH_URL, HEALTH_REQUEST_TIMEOUT).await;
    if second_probe.healthy {
        *lock(&state.session) = Some(BackendSession::external());
        update_state(app, |current| {
            current.phase = "ready".to_owned();
            current.message = "已连接刚刚启动的外部 DigiBox 服务。".to_owned();
        });
        show_ready_window(app, generation).await?;
        start_external_backend_monitor(app.clone(), generation);
        return Ok(());
    }
    if matches!(
        second_probe.failure,
        Some(HealthFailure::ServiceStatus(ref status)) if status == "starting"
    ) {
        wait_for_existing_service(app).await?;
        *lock(&state.session) = Some(BackendSession::external());
        update_state(app, |current| {
            current.phase = "ready".to_owned();
            current.message = "并发启动的外部 DigiBox 服务现已就绪。".to_owned();
        });
        show_ready_window(app, generation).await?;
        start_external_backend_monitor(app.clone(), generation);
        return Ok(());
    }
    if let Some(error) = classify_health_conflict(&second_probe) {
        return Err(error);
    }
    if state.exit_started.load(Ordering::Acquire) {
        return Err("桌面程序正在退出。".to_owned());
    }

    let stop_file = state
        .stop_dir
        .join(format!("backend-{}-{generation}.stop", std::process::id()));
    let instance_id = new_owned_instance_id(generation);
    let environment = backend_environment(
        env::vars_os(),
        &instance_id,
        &state.user_assets_dir,
        &state.cosyvoice_speaker_cache,
        &state.memory_root,
    );
    let log_dir = state.log_dir.clone();
    let progress_app = app.clone();
    let session = tauri::async_runtime::spawn_blocking(move || {
        BackendSession::spawn_owned(
            &launch,
            &log_dir,
            &stop_file,
            environment,
            move |progress| {
                let message = progress_message(progress);
                update_state(&progress_app, |current| {
                    current.phase = "starting".to_owned();
                    current.message = message.to_owned();
                });
            },
        )
    })
    .await
    .map_err(display_error)?
    .map_err(display_error)?;
    *lock(&state.session) = Some(session);

    wait_for_owned_service(app, &instance_id).await?;
    update_state(app, |current| {
        current.phase = "ready".to_owned();
        current.message = "DigiBox 已就绪。".to_owned();
        current.error = None;
    });
    show_ready_window(app, generation).await?;
    start_owned_backend_monitor(app.clone(), generation, instance_id);
    Ok(())
}

async fn run_startup(app: AppHandle) -> DesktopState {
    let state = app.state::<DesktopApp>();
    let _startup_gate = state.startup_gate.lock().await;
    let generation = state.startup_generation.fetch_add(1, Ordering::AcqRel) + 1;
    if state.exit_started.load(Ordering::Acquire) {
        return state.snapshot();
    }

    if let Err(error) = stop_current_session(&app).await {
        append_desktop_log(&app, &format!("could not stop previous backend: {error}"));
        return update_state(&app, |current| {
            current.phase = "error".to_owned();
            current.message = "无法清理上一次后端进程。".to_owned();
            current.error = Some(error);
        });
    }
    update_state(&app, |current| {
        current.phase = "resolving".to_owned();
        current.message = "正在检查 Windows Runtime 和后端服务…".to_owned();
        current.error = None;
    });

    if let Err(error) = startup_inner(&app, generation).await {
        let cleanup_error = stop_current_session(&app).await.err();
        if state.exit_started.load(Ordering::Acquire) {
            return update_state(&app, |current| {
                current.phase = "stopping".to_owned();
                current.message = "正在安全关闭 DigiBox…".to_owned();
                current.error = cleanup_error;
            });
        }
        append_desktop_log(&app, &format!("startup failed: {error}"));
        return update_state(&app, |current| {
            current.phase = if error.contains("Runtime") || error.contains("候选") {
                "runtime-missing".to_owned()
            } else {
                "error".to_owned()
            };
            current.message = "DigiBox 启动失败。".to_owned();
            current.error = Some(match cleanup_error {
                Some(cleanup) => format!("{error}\n清理后端时又发生错误：{cleanup}"),
                None => error,
            });
        });
    }
    state.snapshot()
}

pub fn is_trusted_splash_url(url: &Url) -> bool {
    let trusted_origin = (url.scheme() == "tauri" && url.host_str() == Some("localhost"))
        || (matches!(url.scheme(), "http" | "https") && url.host_str() == Some("tauri.localhost"));
    trusted_origin && matches!(url.path(), "/" | "/index.html")
}

fn ensure_trusted_splash(window: &WebviewWindow) -> Result<(), String> {
    if window.label() != SPLASH_LABEL {
        return Err("此桌面命令仅允许由内置启动页调用。".to_owned());
    }
    let url = window.url().map_err(display_error)?;
    if !is_trusted_splash_url(&url) {
        return Err("当前页面无权调用桌面命令。".to_owned());
    }
    Ok(())
}

#[tauri::command]
async fn get_desktop_state(
    window: WebviewWindow,
    state: State<'_, DesktopApp>,
) -> Result<DesktopState, String> {
    ensure_trusted_splash(&window)?;
    Ok(state.snapshot())
}

#[tauri::command]
async fn retry_startup(window: WebviewWindow, app: AppHandle) -> Result<DesktopState, String> {
    ensure_trusted_splash(&window)?;
    Ok(run_startup(app).await)
}

#[tauri::command]
async fn select_runtime(window: WebviewWindow, app: AppHandle) -> Result<DesktopState, String> {
    ensure_trusted_splash(&window)?;
    let dialog_app = app.clone();
    let selection = tauri::async_runtime::spawn_blocking(move || {
        dialog_app.dialog().file().blocking_pick_folder()
    })
    .await
    .map_err(display_error)?;
    let Some(selection) = selection else {
        return Ok(app.state::<DesktopApp>().snapshot());
    };
    let root = selection.into_path().map_err(display_error)?;
    let runtime = inspect_runtime_root(&root);
    if !runtime.is_valid() {
        return Ok(update_state(&app, |current| {
            current.phase = "runtime-missing".to_owned();
            current.message = "所选目录不是完整的 DigiBox Runtime。".to_owned();
            current.error = Some(format!("缺少：{}", runtime.missing.join("、")));
        }));
    }
    let state = app.state::<DesktopApp>();
    persist_runtime(&state.config_path, &runtime.root).await?;
    *lock(&state.selected_runtime) = Some(runtime.root);
    Ok(run_startup(app).await)
}

#[tauri::command]
async fn open_logs(window: WebviewWindow, app: AppHandle) -> Result<(), String> {
    ensure_trusted_splash(&window)?;
    let log_dir = app.state::<DesktopApp>().log_dir.clone();
    tokio::fs::create_dir_all(&log_dir)
        .await
        .map_err(display_error)?;
    app.opener()
        .open_path(log_dir.to_string_lossy(), None::<&str>)
        .map_err(display_error)
}

fn install_exit_handler(app: &AppHandle, event: &RunEvent) {
    let RunEvent::ExitRequested { api, code, .. } = event else {
        return;
    };
    let state = app.state::<DesktopApp>();
    if state.final_exit.load(Ordering::Acquire) {
        return;
    }
    api.prevent_exit();
    if state.exit_started.swap(true, Ordering::AcqRel) {
        return;
    }
    update_state(app, |current| {
        current.phase = "stopping".to_owned();
        current.message = "正在安全关闭 DigiBox 后端…".to_owned();
        current.error = None;
    });
    let app = app.clone();
    let exit_code = code.unwrap_or(0);
    tauri::async_runtime::spawn(async move {
        let state = app.state::<DesktopApp>();
        let _startup_gate = state.startup_gate.lock().await;
        if let Err(error) = stop_current_session(&app).await {
            append_desktop_log(&app, &format!("backend shutdown failed: {error}"));
        }
        state.final_exit.store(true, Ordering::Release);
        app.exit(exit_code);
    });
}

fn request_exit_from_window(app: &AppHandle, event: &RunEvent) {
    let RunEvent::WindowEvent { label, event, .. } = event else {
        return;
    };
    if !matches!(label.as_str(), SPLASH_LABEL | MAIN_LABEL) {
        return;
    }
    if let tauri::WindowEvent::CloseRequested { api, .. } = event {
        let state = app.state::<DesktopApp>();
        if !state.final_exit.load(Ordering::Acquire) {
            api.prevent_close();
            app.exit(0);
        }
    }
}

fn focus_existing_window(app: &AppHandle) {
    let window = app
        .get_webview_window(MAIN_LABEL)
        .or_else(|| app.get_webview_window(SPLASH_LABEL));
    if let Some(window) = window {
        let _ = window.show();
        let _ = window.unminimize();
        let _ = window.set_focus();
    }
}

pub fn run() {
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_single_instance::init(
            |app, _arguments, _cwd| {
                focus_existing_window(app);
            },
        ))
        .plugin(tauri_plugin_dialog::init())
        .plugin(
            tauri_plugin_opener::Builder::new()
                .open_js_links_on_click(false)
                .build(),
        )
        .invoke_handler(tauri::generate_handler![
            get_desktop_state,
            retry_startup,
            select_runtime,
            open_logs
        ])
        .setup(|app| {
            let state = DesktopApp::new(app.handle())?;
            app.manage(state);
            let handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                let _ = run_startup(handle).await;
            });
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("failed to build DigiBox Tauri desktop application");

    app.run(|app, event| {
        request_exit_from_window(app, &event);
        install_exit_handler(app, &event);
    });
}
