use std::collections::HashMap;
use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};
use std::time::Duration;

use avtr1_desktop::app::{
    backend_environment, launch_from_runtime, memory_root_from_local_app_data,
};
use avtr1_desktop::runtime::{
    PortableRuntimeLayout, RuntimeInfo, RuntimeMode, RuntimePackageLayers,
};
use avtr1_desktop::supervisor::{
    build_backend_environment, build_spawn_spec, stop_with_control, taskkill_arguments,
    BackendOwnership, BackendSession, RuntimeLaunch, RuntimeLaunchMode, ShutdownControl,
    ShutdownResult, StartupProgress, GRACEFUL_SHUTDOWN_TIMEOUT,
};

fn temporary_runtime(name: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!(
        "avtr1-supervisor-{name}-{}-{}",
        std::process::id(),
        std::thread::current().name().unwrap_or("contract")
    ));
    std::fs::create_dir_all(root.join("src")).expect("create source directory");
    root
}

fn managed_runtime(root: &Path) -> RuntimeLaunch {
    RuntimeLaunch {
        mode: RuntimeLaunchMode::Managed,
        manifest_layout: Some(PortableRuntimeLayout::V1),
        root: root.to_path_buf(),
        python: root.join("python-main/python.exe"),
        orchestrator: root.join("scripts/run_local_stream.py"),
        source_dir: root.join("src"),
        package_layers: RuntimePackageLayers {
            main: vec![root.join("src")],
            cosyvoice: vec![
                root.join("src"),
                root.join("third_party/CosyVoice"),
                root.join("third_party/CosyVoice/third_party/Matcha-TTS"),
            ],
            feynobg: vec![root.join("src")],
        },
    }
}

fn managed_v2_runtime(root: &Path) -> RuntimeLaunch {
    RuntimeLaunch {
        mode: RuntimeLaunchMode::Managed,
        manifest_layout: Some(PortableRuntimeLayout::V2),
        root: root.to_path_buf(),
        python: root.join("python/python.exe"),
        orchestrator: root.join("scripts/run_local_stream.py"),
        source_dir: root.join("src"),
        package_layers: RuntimePackageLayers {
            main: vec![
                root.join("packages/main"),
                root.join("packages/shared"),
                root.join("src"),
            ],
            cosyvoice: vec![
                root.join("packages/cosyvoice"),
                root.join("packages/shared"),
                root.join("third_party/CosyVoice"),
                root.join("third_party/CosyVoice/third_party/Matcha-TTS"),
                root.join("src"),
            ],
            feynobg: vec![
                root.join("packages/feynobg"),
                root.join("packages/shared"),
                root.join("src"),
            ],
        },
    }
}

fn development_test_python_from(configured: Option<OsString>) -> PathBuf {
    configured
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("python.exe"))
}

fn development_test_python() -> PathBuf {
    development_test_python_from(std::env::var_os("AVTR1_TEST_PYTHON"))
}

fn development_runtime(root: &Path, orchestrator: PathBuf) -> RuntimeLaunch {
    RuntimeLaunch {
        mode: RuntimeLaunchMode::Development,
        manifest_layout: None,
        root: root.to_path_buf(),
        python: development_test_python(),
        orchestrator,
        source_dir: root.join("src"),
        package_layers: RuntimePackageLayers {
            main: vec![root.join("src")],
            cosyvoice: vec![
                root.join("src"),
                root.join("third_party/CosyVoice"),
                root.join("third_party/CosyVoice/third_party/Matcha-TTS"),
            ],
            feynobg: vec![root.join("src")],
        },
    }
}

#[test]
fn development_runtime_python_can_be_injected_without_a_repository_venv() {
    let full_python = PathBuf::from(r"C:\DigiBox\avtr-runtime\python\python.exe");

    assert_eq!(
        development_test_python_from(Some(full_python.clone().into_os_string())),
        full_python
    );
    assert_eq!(
        development_test_python_from(Some(OsString::new())),
        PathBuf::from("python.exe")
    );
    assert_eq!(
        development_test_python_from(None),
        PathBuf::from("python.exe")
    );
}

#[test]
fn external_service_has_no_process_ownership() {
    let mut control = RecordingControl::default();

    let result = stop_with_control(BackendOwnership::External, None, &mut control)
        .expect("external shutdown must be a no-op");

    assert_eq!(result, ShutdownResult::NotOwned);
    assert!(control.actions.is_empty());
}

#[test]
fn external_session_never_contains_an_owned_pid() {
    let session = BackendSession::external();

    assert_eq!(session.ownership(), BackendOwnership::External);
    assert_eq!(session.pid(), None);
}

#[test]
fn tauri_memory_root_is_derived_from_local_app_data_not_the_runtime() {
    let local_app_data = PathBuf::from(r"C:\Users\Alice\AppData\Local");

    let memory_root = memory_root_from_local_app_data(&local_app_data);

    assert_eq!(
        memory_root,
        PathBuf::from(r"C:\Users\Alice\AppData\Local\DigiBox\memory")
    );
    assert!(!memory_root.starts_with(r"D:\DigiBox-Runtime"));
}

#[test]
fn tauri_backend_environment_replaces_a_stale_memory_root() {
    let memory_root = PathBuf::from(r"C:\Users\Alice\AppData\Local\DigiBox\memory");
    let environment = backend_environment(
        vec![
            (
                OsString::from("PATH"),
                OsString::from(r"C:\Windows\System32"),
            ),
            (
                OsString::from("avtr1_memory_root"),
                OsString::from(r"D:\DigiBox-Runtime\memory"),
            ),
        ],
        "desktop-test",
        Path::new(r"C:\Users\Alice\AppData\Local\DigiBox\user_assets"),
        Path::new(r"C:\Users\Alice\AppData\Local\DigiBox\cosyvoice\spk2info.pt"),
        &memory_root,
    );

    let memory_values = environment
        .iter()
        .filter(|(key, _)| key.eq_ignore_ascii_case("AVTR1_MEMORY_ROOT"))
        .collect::<Vec<_>>();
    assert_eq!(memory_values.len(), 1);
    assert_eq!(memory_values[0].0, "AVTR1_MEMORY_ROOT");
    assert_eq!(memory_values[0].1, memory_root.as_os_str());
}

#[test]
fn managed_environment_isolated_from_host_python_and_configures_workers() {
    let root = temporary_runtime("managed-env");
    std::fs::create_dir_all(root.join("python-cosyvoice")).unwrap();
    std::fs::create_dir_all(root.join("python-feynobg")).unwrap();
    std::fs::write(root.join("python-cosyvoice/python.exe"), []).unwrap();
    std::fs::write(root.join("python-feynobg/python.exe"), []).unwrap();
    let stop_file = root.join("state/desktop.stop");
    let environment = build_backend_environment(
        &managed_runtime(&root),
        &stop_file,
        HashMap::from([
            (
                OsString::from("PATH"),
                OsString::from(r"C:\Windows\System32"),
            ),
            (
                OsString::from("PYTHONHOME"),
                OsString::from(r"C:\HostPython"),
            ),
            (
                OsString::from("PythonPath"),
                OsString::from(r"C:\HostModules"),
            ),
            (
                OsString::from("VIRTUAL_ENV"),
                OsString::from(r"C:\host-venv"),
            ),
            (OsString::from("CONDA_PREFIX"), OsString::from(r"C:\conda")),
            (
                OsString::from("AVTR1_MAIN_PYTHONPATH"),
                OsString::from(r"C:\stale-main-layer"),
            ),
            (
                OsString::from("AVTR1_COSYVOICE_PYTHONPATH"),
                OsString::from(r"C:\stale-cosyvoice-layer"),
            ),
            (
                OsString::from("AVTR1_FEYNOBG_PYTHONPATH"),
                OsString::from(r"C:\stale-feynobg-layer"),
            ),
        ]),
    );

    assert_eq!(
        environment.get(OsStr::new("PATH")).unwrap(),
        r"C:\Windows\System32"
    );
    assert_eq!(
        environment.get(OsStr::new("PYTHONPATH")).unwrap(),
        root.join("src").as_os_str()
    );
    assert!(!environment
        .keys()
        .any(|key| key.eq_ignore_ascii_case("PYTHONHOME")));
    assert!(!environment
        .keys()
        .any(|key| key.eq_ignore_ascii_case("VIRTUAL_ENV")));
    assert!(!environment
        .keys()
        .any(|key| key.eq_ignore_ascii_case("CONDA_PREFIX")));
    assert_eq!(
        environment.get(OsStr::new("AVTR1_RUNTIME_ROOT")).unwrap(),
        root.as_os_str()
    );
    assert_eq!(
        environment.get(OsStr::new("AVTR1_APP_ROOT")).unwrap(),
        root.as_os_str()
    );
    assert_eq!(
        environment
            .get(OsStr::new("AVTR1_DESKTOP_STOP_FILE"))
            .unwrap(),
        stop_file.as_os_str()
    );
    assert_eq!(
        environment.get(OsStr::new("AVTR1_SINGLE_ENV")).unwrap(),
        "1"
    );
    assert_eq!(
        environment
            .get(OsStr::new("AVTR1_COSYVOICE_PYTHON"))
            .unwrap(),
        root.join("python-cosyvoice").join("python.exe").as_os_str()
    );
    assert_eq!(
        environment.get(OsStr::new("AVTR1_FEYNOBG_PYTHON")).unwrap(),
        root.join("python-feynobg").join("python.exe").as_os_str()
    );
    assert!(!environment.contains_key(OsStr::new("AVTR1_MAIN_PYTHONPATH")));
    assert!(!environment.contains_key(OsStr::new("AVTR1_COSYVOICE_PYTHONPATH")));
    assert!(!environment.contains_key(OsStr::new("AVTR1_FEYNOBG_PYTHONPATH")));

    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn managed_portable_v2_routes_ordered_layers_through_one_python() {
    let root = temporary_runtime("managed-v2-env");
    let runtime = managed_v2_runtime(&root);
    let environment = build_backend_environment(
        &runtime,
        &root.join("state/desktop.stop"),
        HashMap::from([
            (
                OsString::from("PYTHONPATH"),
                OsString::from(r"C:\HostModules"),
            ),
            (
                OsString::from("AVTR1_COSYVOICE_PYTHON"),
                OsString::from(r"C:\stale\python.exe"),
            ),
        ]),
    );

    let main = std::env::join_paths(&runtime.package_layers.main).unwrap();
    let cosyvoice = std::env::join_paths(&runtime.package_layers.cosyvoice).unwrap();
    let feynobg = std::env::join_paths(&runtime.package_layers.feynobg).unwrap();
    assert_eq!(environment.get(OsStr::new("PYTHONPATH")), Some(&main));
    assert_eq!(
        environment.get(OsStr::new("AVTR1_MAIN_PYTHONPATH")),
        Some(&main)
    );
    assert_eq!(
        environment.get(OsStr::new("AVTR1_COSYVOICE_PYTHONPATH")),
        Some(&cosyvoice)
    );
    assert_eq!(
        environment.get(OsStr::new("AVTR1_FEYNOBG_PYTHONPATH")),
        Some(&feynobg)
    );
    assert_eq!(
        environment
            .get(OsStr::new("AVTR1_COSYVOICE_PYTHON"))
            .unwrap(),
        runtime.python.as_os_str()
    );
    assert_eq!(
        environment.get(OsStr::new("AVTR1_FEYNOBG_PYTHON")).unwrap(),
        runtime.python.as_os_str()
    );

    let _ = std::fs::remove_dir_all(root);
}

#[cfg(windows)]
#[test]
fn managed_portable_v2_removes_verbatim_prefixes_from_native_python_paths() {
    let root = PathBuf::from(r"\\?\F:\DigiBox\avtr-runtime");
    let runtime = managed_v2_runtime(&root);
    let environment =
        build_backend_environment(&runtime, &root.join("state/desktop.stop"), HashMap::new());

    for key in [
        "PYTHONPATH",
        "AVTR1_MAIN_PYTHONPATH",
        "AVTR1_COSYVOICE_PYTHONPATH",
        "AVTR1_FEYNOBG_PYTHONPATH",
    ] {
        let value = environment
            .get(OsStr::new(key))
            .unwrap_or_else(|| panic!("missing {key}"));
        assert!(
            !value.to_string_lossy().contains(r"\\?\"),
            "{key} leaked a Windows verbatim prefix: {}",
            value.to_string_lossy()
        );
    }

    let main = environment.get(OsStr::new("PYTHONPATH")).unwrap();
    let main_paths = std::env::split_paths(main).collect::<Vec<_>>();
    assert_eq!(
        main_paths,
        vec![
            PathBuf::from(r"F:\DigiBox\avtr-runtime\packages\main"),
            PathBuf::from(r"F:\DigiBox\avtr-runtime\packages\shared"),
            PathBuf::from(r"F:\DigiBox\avtr-runtime\src"),
        ]
    );
}

#[test]
fn app_launch_preserves_manifest_layout_and_every_ordered_package_layer() {
    let root = temporary_runtime("app-runtime-launch-v2");
    let expected = managed_v2_runtime(&root);
    let runtime = RuntimeInfo {
        root: root.clone(),
        mode: Some(RuntimeMode::Managed),
        python: Some(expected.python.clone()),
        orchestrator: expected.orchestrator.clone(),
        source_dir: expected.source_dir.clone(),
        artifacts_dir: root.join("artifacts/main"),
        models_dir: root.join("models"),
        manifest: Some(root.join("runtime-manifest.json")),
        manifest_layout: expected.manifest_layout,
        package_layers: expected.package_layers.clone(),
        missing: Vec::new(),
    };

    let launch = launch_from_runtime(&runtime).expect("convert an inspected v2 runtime");

    assert_eq!(launch, expected);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn development_environment_preserves_explicit_host_pythonpath_after_project_source() {
    let root = temporary_runtime("development-env");
    let mut runtime = managed_runtime(&root);
    runtime.mode = RuntimeLaunchMode::Development;
    let environment = build_backend_environment(
        &runtime,
        &root.join("desktop.stop"),
        HashMap::from([(
            OsString::from("PYTHONPATH"),
            OsString::from(r"C:\DeveloperModules"),
        )]),
    );

    let expected = std::env::join_paths([root.join("src"), PathBuf::from(r"C:\DeveloperModules")])
        .expect("join development PYTHONPATH");
    assert_eq!(
        environment.get(OsStr::new("PYTHONPATH")).unwrap(),
        &expected
    );

    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn spawn_spec_launches_exactly_one_python_orchestrator_and_separates_logs() {
    let root = temporary_runtime("spawn-spec");
    let log_dir = root.join("user-logs");
    let stop_file = root.join("state/desktop.stop");
    let runtime = managed_runtime(&root);

    let spec = build_spawn_spec(&runtime, &log_dir, &stop_file, std::iter::empty())
        .expect("build spawn spec");

    assert_eq!(spec.program, runtime.python);
    assert_eq!(spec.arguments, vec![runtime.orchestrator.into_os_string()]);
    assert_eq!(spec.current_dir, root);
    assert_eq!(spec.stdout_log, log_dir.join("avtr-backend.stdout.log"));
    assert_eq!(spec.stderr_log, log_dir.join("avtr-backend.stderr.log"));
    assert_eq!(
        spec.environment
            .get(OsStr::new("AVTR1_DESKTOP_STOP_FILE"))
            .unwrap(),
        stop_file.as_os_str()
    );
}

#[test]
fn graceful_owned_shutdown_writes_stop_file_and_waits_twenty_seconds_without_forcing() {
    let mut control = RecordingControl::with_waits([true]);

    let result = stop_with_control(BackendOwnership::Owned, Some(4242), &mut control)
        .expect("graceful shutdown");

    assert_eq!(result, ShutdownResult::Graceful);
    assert_eq!(
        control.actions,
        vec![Action::WriteStop, Action::Wait(GRACEFUL_SHUTDOWN_TIMEOUT)]
    );
}

#[test]
fn stubborn_owned_shutdown_terminates_job_before_exact_pid_taskkill_fallback() {
    let mut control = RecordingControl::with_waits([false, false, true]);

    let result = stop_with_control(BackendOwnership::Owned, Some(7391), &mut control)
        .expect("forced shutdown");

    assert_eq!(result, ShutdownResult::Taskkill);
    assert_eq!(
        control.actions,
        vec![
            Action::WriteStop,
            Action::Wait(Duration::from_secs(20)),
            Action::TerminateJob,
            Action::Wait(Duration::from_secs(2)),
            Action::Taskkill(7391),
            Action::Wait(Duration::from_secs(2)),
        ]
    );
    assert_eq!(
        taskkill_arguments(7391),
        ["/PID", "7391", "/T", "/F"].map(OsString::from)
    );
}

#[test]
fn shutdown_cleanup_failures_are_combined_after_exact_pid_fallback_completes() {
    let mut control = RecordingControl::with_waits([false, false, true]);
    control.write_error = Some("stop-file access denied");
    control.terminate_error = Some("job handle invalid");

    let error = stop_with_control(BackendOwnership::Owned, Some(8675), &mut control)
        .expect_err("cleanup failures must be reported after all fallbacks run");

    assert_eq!(
        control.actions,
        vec![
            Action::WriteStop,
            Action::Wait(Duration::from_secs(20)),
            Action::TerminateJob,
            Action::Wait(Duration::from_secs(2)),
            Action::Taskkill(8675),
            Action::Wait(Duration::from_secs(2)),
        ]
    );
    let message = error.to_string();
    assert!(message.contains("write stop file: stop-file access denied"));
    assert!(message.contains("terminate job: job handle invalid"));
}

#[cfg(windows)]
#[test]
fn windows_job_is_created_with_kill_on_close_policy() {
    use avtr1_desktop::windows_job::{
        kill_on_close_limit, suspended_creation_flags, KillOnCloseJob,
    };
    use windows_sys::Win32::System::JobObjects::JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    use windows_sys::Win32::System::Threading::{CREATE_NO_WINDOW, CREATE_SUSPENDED};

    assert_eq!(kill_on_close_limit(), JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE);
    assert_eq!(
        suspended_creation_flags(),
        CREATE_NO_WINDOW | CREATE_SUSPENDED,
        "the orchestrator must be born suspended so it cannot create workers before Job assignment"
    );
    let _job = KillOnCloseJob::new().expect("create configured Windows Job Object");
}

#[cfg(windows)]
#[test]
fn owned_session_spawns_one_orchestrator_logs_and_stops_cooperatively() {
    let root = temporary_runtime("real-owned-session");
    let source_dir = root.join("src");
    let orchestrator = root.join("orchestrator.py");
    std::fs::write(
        &orchestrator,
        concat!(
            "import os, pathlib, time\n",
            "print('orchestrator-started', flush=True)\n",
            "stop = pathlib.Path(os.environ['AVTR1_DESKTOP_STOP_FILE'])\n",
            "while not stop.exists(): time.sleep(0.02)\n",
            "print('orchestrator-stopped', flush=True)\n",
        ),
    )
    .unwrap();
    let runtime = development_runtime(&root, orchestrator);
    assert_eq!(runtime.source_dir, source_dir);
    let log_dir = root.join("logs");
    let stop_file = root.join("state/desktop.stop");
    let mut progress = Vec::new();

    let mut session = BackendSession::spawn_owned(
        &runtime,
        &log_dir,
        &stop_file,
        std::iter::empty(),
        |event| progress.push(event),
    )
    .expect("spawn owned Python orchestrator");

    assert_eq!(session.ownership(), BackendOwnership::Owned);
    assert!(session.pid().is_some());
    assert_eq!(
        progress,
        vec![
            StartupProgress::PreparingLogs,
            StartupProgress::SpawningOrchestrator,
            StartupProgress::AssigningJob,
            StartupProgress::Started,
        ]
    );
    assert_eq!(
        session.stop().expect("cooperative stop"),
        ShutdownResult::Graceful
    );
    assert!(stop_file.is_file());
    let stdout = std::fs::read_to_string(log_dir.join("avtr-backend.stdout.log")).unwrap();
    assert!(stdout.contains("orchestrator-started"));
    assert!(stdout.contains("orchestrator-stopped"));

    let _ = std::fs::remove_dir_all(root);
}

#[cfg(windows)]
#[test]
fn owned_session_reports_early_exit_without_blocking_and_caches_status() {
    let root = temporary_runtime("early-exit-session");
    let orchestrator = root.join("early_exit.py");
    std::fs::write(
        &orchestrator,
        concat!(
            "import sys, time\n",
            "time.sleep(0.1)\n",
            "print('early-exit', flush=True)\n",
            "sys.exit(23)\n",
        ),
    )
    .unwrap();
    let runtime = development_runtime(&root, orchestrator);
    let mut session = BackendSession::spawn_owned(
        &runtime,
        &root.join("logs"),
        &root.join("state/desktop.stop"),
        std::iter::empty(),
        |_| {},
    )
    .expect("spawn early-exit orchestrator");

    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    let status = loop {
        if let Some(status) = session.try_wait().expect("poll child without blocking") {
            break status;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "orchestrator did not report its early exit"
        );
        std::thread::sleep(Duration::from_millis(20));
    };

    assert_eq!(status.code(), Some(23));
    assert!(!session.is_running().expect("query cached child state"));
    assert_eq!(
        session
            .try_wait()
            .expect("read cached exit status")
            .and_then(|cached| cached.code()),
        Some(23)
    );

    drop(session);
    let _ = std::fs::remove_dir_all(root);
}

#[cfg(windows)]
#[test]
fn suspended_orchestrator_cannot_escape_job_with_an_immediate_child() {
    use windows_sys::Win32::Foundation::{CloseHandle, WAIT_OBJECT_0, WAIT_TIMEOUT};
    use windows_sys::Win32::System::Threading::{OpenProcess, WaitForSingleObject};

    const SYNCHRONIZE_ACCESS: u32 = 0x0010_0000;

    fn wait_for_process_exit(pid: u32, timeout: Duration) -> bool {
        let process = unsafe { OpenProcess(SYNCHRONIZE_ACCESS, 0, pid) };
        if process.is_null() {
            return true;
        }
        let timeout_ms = u32::try_from(timeout.as_millis()).unwrap_or(u32::MAX);
        let result = unsafe { WaitForSingleObject(process, timeout_ms) };
        unsafe { CloseHandle(process) };
        match result {
            WAIT_OBJECT_0 => true,
            WAIT_TIMEOUT => false,
            _ => false,
        }
    }

    let root = temporary_runtime("suspended-immediate-child");
    let orchestrator = root.join("spawn_immediately.py");
    let child_pid_file = root.join("child.pid");
    std::fs::write(
        &orchestrator,
        concat!(
            "import os, pathlib, subprocess, sys, time\n",
            "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n",
            "pathlib.Path(sys.argv[1]).write_text(str(child.pid), encoding='ascii')\n",
            "stop = pathlib.Path(os.environ['AVTR1_DESKTOP_STOP_FILE'])\n",
            "while not stop.exists(): time.sleep(0.01)\n",
        ),
    )
    .unwrap();
    let runtime = development_runtime(&root, orchestrator.clone());

    // The production API intentionally starts only the orchestrator argument.
    // Use a tiny wrapper so the immediate worker can publish its PID.
    let wrapper = root.join("wrapper.py");
    std::fs::write(
        &wrapper,
        format!(
            "import runpy, sys\nsys.argv = [r'{}', r'{}']\nrunpy.run_path(r'{}', run_name='__main__')\n",
            orchestrator.display(),
            child_pid_file.display(),
            orchestrator.display(),
        ),
    )
    .unwrap();
    let runtime = RuntimeLaunch {
        orchestrator: wrapper,
        ..runtime
    };
    let mut session = BackendSession::spawn_owned(
        &runtime,
        &root.join("logs"),
        &root.join("state/desktop.stop"),
        std::iter::empty(),
        |_| {},
    )
    .expect("spawn suspended orchestrator inside Job Object");

    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    let child_pid = loop {
        if let Ok(value) = std::fs::read_to_string(&child_pid_file) {
            if let Ok(pid) = value.parse::<u32>() {
                break pid;
            }
        }
        assert!(
            std::time::Instant::now() < deadline,
            "immediate worker did not publish its PID"
        );
        std::thread::sleep(Duration::from_millis(20));
    };
    assert!(!wait_for_process_exit(child_pid, Duration::ZERO));

    assert_eq!(
        session.stop().expect("stop orchestrator cooperatively"),
        ShutdownResult::Graceful
    );
    drop(session); // Closing the Job is the safety net for an orphaned worker.
    assert!(
        wait_for_process_exit(child_pid, Duration::from_secs(5)),
        "worker PID {child_pid} escaped the orchestrator Job Object"
    );

    let _ = std::fs::remove_dir_all(root);
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Action {
    WriteStop,
    Wait(Duration),
    TerminateJob,
    Taskkill(u32),
}

#[derive(Default)]
struct RecordingControl {
    actions: Vec<Action>,
    waits: std::collections::VecDeque<bool>,
    write_error: Option<&'static str>,
    terminate_error: Option<&'static str>,
}

impl RecordingControl {
    fn with_waits(waits: impl IntoIterator<Item = bool>) -> Self {
        Self {
            actions: Vec::new(),
            waits: waits.into_iter().collect(),
            write_error: None,
            terminate_error: None,
        }
    }
}

impl ShutdownControl for RecordingControl {
    fn write_stop_file(&mut self) -> std::io::Result<()> {
        self.actions.push(Action::WriteStop);
        if let Some(message) = self.write_error {
            return Err(std::io::Error::other(message));
        }
        Ok(())
    }

    fn wait_for_exit(&mut self, timeout: Duration) -> std::io::Result<bool> {
        self.actions.push(Action::Wait(timeout));
        Ok(self.waits.pop_front().unwrap_or(false))
    }

    fn terminate_job(&mut self) -> std::io::Result<()> {
        self.actions.push(Action::TerminateJob);
        if let Some(message) = self.terminate_error {
            return Err(std::io::Error::other(message));
        }
        Ok(())
    }

    fn taskkill_exact_pid(&mut self, pid: u32) -> std::io::Result<()> {
        self.actions.push(Action::Taskkill(pid));
        Ok(())
    }
}
