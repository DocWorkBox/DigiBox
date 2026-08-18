use std::collections::HashMap;
use std::ffi::{OsStr, OsString};
use std::fs::{self, File};
use std::io;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::thread;
use std::time::{Duration, Instant, SystemTime};

#[cfg(windows)]
use std::os::windows::ffi::{OsStrExt, OsStringExt};

use crate::runtime::{PortableRuntimeLayout, RuntimePackageLayers};

#[cfg(windows)]
use crate::windows_job::{suspended_creation_flags, KillOnCloseJob};

pub const GRACEFUL_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(20);
const FORCED_SHUTDOWN_CONFIRMATION_TIMEOUT: Duration = Duration::from_secs(2);
const PROCESS_POLL_INTERVAL: Duration = Duration::from_millis(50);

const MANAGED_PYTHON_ENVIRONMENT_KEYS: &[&str] = &[
    "CONDA_DEFAULT_ENV",
    "CONDA_PREFIX",
    "CONDA_PREFIX_1",
    "CONDA_PROMPT_MODIFIER",
    "AVTR1_COSYVOICE_PYTHONPATH",
    "AVTR1_FEYNOBG_PYTHONPATH",
    "AVTR1_MAIN_PYTHONPATH",
    "PYTHONHOME",
    "PYTHONPATH",
    "PYTHONSTARTUP",
    "VIRTUAL_ENV",
    "VIRTUAL_ENV_PROMPT",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendOwnership {
    External,
    Owned,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeLaunchMode {
    Managed,
    Development,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeLaunch {
    pub mode: RuntimeLaunchMode,
    pub manifest_layout: Option<PortableRuntimeLayout>,
    pub root: PathBuf,
    pub python: PathBuf,
    pub orchestrator: PathBuf,
    pub source_dir: PathBuf,
    pub package_layers: RuntimePackageLayers,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpawnSpec {
    pub program: PathBuf,
    pub arguments: Vec<OsString>,
    pub current_dir: PathBuf,
    pub environment: HashMap<OsString, OsString>,
    pub stdout_log: PathBuf,
    pub stderr_log: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StartupProgress {
    PreparingLogs,
    SpawningOrchestrator,
    AssigningJob,
    Started,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShutdownResult {
    NotOwned,
    AlreadyExited,
    Graceful,
    JobTerminated,
    Taskkill,
}

pub trait ShutdownControl {
    fn write_stop_file(&mut self) -> io::Result<()>;
    fn wait_for_exit(&mut self, timeout: Duration) -> io::Result<bool>;
    fn terminate_job(&mut self) -> io::Result<()>;
    fn taskkill_exact_pid(&mut self, pid: u32) -> io::Result<()>;
}

pub fn stop_with_control(
    ownership: BackendOwnership,
    pid: Option<u32>,
    control: &mut impl ShutdownControl,
) -> io::Result<ShutdownResult> {
    if ownership == BackendOwnership::External {
        return Ok(ShutdownResult::NotOwned);
    }
    let Some(pid) = pid else {
        return Ok(ShutdownResult::AlreadyExited);
    };

    let mut errors = Vec::new();
    if let Err(error) = control.write_stop_file() {
        errors.push(format!("write stop file: {error}"));
    }
    match control.wait_for_exit(GRACEFUL_SHUTDOWN_TIMEOUT) {
        Ok(true) => return shutdown_outcome(ShutdownResult::Graceful, errors),
        Ok(false) => {}
        Err(error) => errors.push(format!("wait for graceful exit: {error}")),
    }

    if let Err(error) = control.terminate_job() {
        errors.push(format!("terminate job: {error}"));
    }
    match control.wait_for_exit(FORCED_SHUTDOWN_CONFIRMATION_TIMEOUT) {
        Ok(true) => return shutdown_outcome(ShutdownResult::JobTerminated, errors),
        Ok(false) => {}
        Err(error) => errors.push(format!("wait after job termination: {error}")),
    }

    if let Err(error) = control.taskkill_exact_pid(pid) {
        errors.push(format!("taskkill exact pid {pid}: {error}"));
    }
    match control.wait_for_exit(FORCED_SHUTDOWN_CONFIRMATION_TIMEOUT) {
        Ok(true) => shutdown_outcome(ShutdownResult::Taskkill, errors),
        Ok(false) => {
            errors.push(format!(
                "backend process tree {pid} remained alive after taskkill"
            ));
            Err(combined_shutdown_error(errors))
        }
        Err(error) => {
            errors.push(format!("wait after taskkill: {error}"));
            Err(combined_shutdown_error(errors))
        }
    }
}

fn shutdown_outcome(result: ShutdownResult, errors: Vec<String>) -> io::Result<ShutdownResult> {
    if errors.is_empty() {
        Ok(result)
    } else {
        Err(combined_shutdown_error(errors))
    }
}

fn combined_shutdown_error(errors: Vec<String>) -> io::Error {
    io::Error::other(format!(
        "backend shutdown encountered errors: {}",
        errors.join("; ")
    ))
}

pub fn build_backend_environment(
    runtime: &RuntimeLaunch,
    stop_file: &Path,
    base_environment: impl IntoIterator<Item = (OsString, OsString)>,
) -> HashMap<OsString, OsString> {
    let mut environment: HashMap<OsString, OsString> = base_environment.into_iter().collect();
    let host_python_path = get_case_insensitive(&environment, "PYTHONPATH").cloned();

    if runtime.mode == RuntimeLaunchMode::Managed {
        environment.retain(|key, _| {
            !MANAGED_PYTHON_ENVIRONMENT_KEYS
                .iter()
                .any(|blocked| key.eq_ignore_ascii_case(blocked))
        });
    } else {
        remove_case_insensitive(&mut environment, "PYTHONPATH");
    }

    insert_environment(&mut environment, "AVTR1_DESKTOP_STOP_FILE", stop_file);
    insert_environment(&mut environment, "AVTR1_RUNTIME_ROOT", &runtime.root);
    insert_environment(&mut environment, "AVTR1_APP_ROOT", &runtime.root);
    insert_environment(&mut environment, "AVTR1_SINGLE_ENV", "1");
    insert_environment(&mut environment, "PYTHONNOUSERSITE", "1");
    insert_environment(&mut environment, "PYTHONUNBUFFERED", "1");
    insert_environment(&mut environment, "PYTHONUTF8", "1");

    let managed_v2 = runtime.mode == RuntimeLaunchMode::Managed
        && runtime.manifest_layout == Some(PortableRuntimeLayout::V2);
    let python_path = if managed_v2 {
        join_runtime_paths(&runtime.package_layers.main, &runtime.source_dir)
    } else if runtime.mode == RuntimeLaunchMode::Managed {
        runtime.source_dir.as_os_str().to_os_string()
    } else {
        join_python_paths(&runtime.source_dir, host_python_path.as_deref())
    };
    insert_environment(&mut environment, "PYTHONPATH", &python_path);

    if managed_v2 {
        insert_environment(&mut environment, "AVTR1_MAIN_PYTHONPATH", python_path);
        insert_environment(
            &mut environment,
            "AVTR1_COSYVOICE_PYTHONPATH",
            join_runtime_paths(&runtime.package_layers.cosyvoice, &runtime.source_dir),
        );
        insert_environment(
            &mut environment,
            "AVTR1_FEYNOBG_PYTHONPATH",
            join_runtime_paths(&runtime.package_layers.feynobg, &runtime.source_dir),
        );
        insert_environment(&mut environment, "AVTR1_COSYVOICE_PYTHON", &runtime.python);
        insert_environment(&mut environment, "AVTR1_FEYNOBG_PYTHON", &runtime.python);
    } else {
        for (key, directory) in [
            ("AVTR1_COSYVOICE_PYTHON", "python-cosyvoice"),
            ("AVTR1_FEYNOBG_PYTHON", "python-feynobg"),
        ] {
            let interpreter = runtime.root.join(directory).join("python.exe");
            if interpreter.is_file() {
                insert_environment(&mut environment, key, interpreter);
            }
        }
    }

    environment
}

pub fn build_spawn_spec(
    runtime: &RuntimeLaunch,
    log_dir: &Path,
    stop_file: &Path,
    base_environment: impl IntoIterator<Item = (OsString, OsString)>,
) -> io::Result<SpawnSpec> {
    fs::create_dir_all(log_dir)?;
    if let Some(parent) = stop_file.parent() {
        fs::create_dir_all(parent)?;
    }
    if stop_file.exists() {
        fs::remove_file(stop_file)?;
    }

    Ok(SpawnSpec {
        program: runtime.python.clone(),
        arguments: vec![runtime.orchestrator.clone().into_os_string()],
        current_dir: runtime.root.clone(),
        environment: build_backend_environment(runtime, stop_file, base_environment),
        stdout_log: log_dir.join("avtr-backend.stdout.log"),
        stderr_log: log_dir.join("avtr-backend.stderr.log"),
    })
}

pub fn taskkill_arguments(pid: u32) -> [OsString; 4] {
    ["/PID", &pid.to_string(), "/T", "/F"].map(OsString::from)
}

pub struct BackendSession {
    ownership: BackendOwnership,
    child: Option<Child>,
    exit_status: Option<ExitStatus>,
    stop_file: Option<PathBuf>,
    #[cfg(windows)]
    job: Option<KillOnCloseJob>,
}

impl BackendSession {
    pub fn external() -> Self {
        Self {
            ownership: BackendOwnership::External,
            child: None,
            exit_status: None,
            stop_file: None,
            #[cfg(windows)]
            job: None,
        }
    }

    pub fn ownership(&self) -> BackendOwnership {
        self.ownership
    }

    pub fn pid(&self) -> Option<u32> {
        self.child.as_ref().map(Child::id)
    }

    pub fn try_wait(&mut self) -> io::Result<Option<ExitStatus>> {
        if let Some(status) = self.exit_status {
            return Ok(Some(status));
        }
        let Some(child) = self.child.as_mut() else {
            return Ok(None);
        };
        let Some(status) = child.try_wait()? else {
            return Ok(None);
        };
        self.child = None;
        self.exit_status = Some(status);
        Ok(Some(status))
    }

    pub fn is_running(&mut self) -> io::Result<bool> {
        let _ = self.try_wait()?;
        Ok(self.child.is_some())
    }

    #[cfg(windows)]
    pub fn spawn_owned(
        runtime: &RuntimeLaunch,
        log_dir: &Path,
        stop_file: &Path,
        base_environment: impl IntoIterator<Item = (OsString, OsString)>,
        mut report_progress: impl FnMut(StartupProgress),
    ) -> io::Result<Self> {
        report_progress(StartupProgress::PreparingLogs);
        let spec = build_spawn_spec(runtime, log_dir, stop_file, base_environment)?;
        let stdout = File::options()
            .create(true)
            .append(true)
            .open(&spec.stdout_log)?;
        let stderr = File::options()
            .create(true)
            .append(true)
            .open(&spec.stderr_log)?;

        report_progress(StartupProgress::SpawningOrchestrator);
        let mut command = Command::new(&spec.program);
        command
            .args(&spec.arguments)
            .current_dir(&spec.current_dir)
            .env_clear()
            .envs(&spec.environment)
            .stdin(Stdio::null())
            .stdout(Stdio::from(stdout))
            .stderr(Stdio::from(stderr));
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            command.creation_flags(suspended_creation_flags());
        }
        let mut child = command.spawn()?;

        report_progress(StartupProgress::AssigningJob);
        let job = match KillOnCloseJob::new().and_then(|job| {
            job.assign_child(&child)?;
            job.resume_child(&child)?;
            Ok(job)
        }) {
            Ok(job) => job,
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(error);
            }
        };

        report_progress(StartupProgress::Started);
        Ok(Self {
            ownership: BackendOwnership::Owned,
            child: Some(child),
            exit_status: None,
            stop_file: Some(stop_file.to_path_buf()),
            job: Some(job),
        })
    }

    pub fn stop(&mut self) -> io::Result<ShutdownResult> {
        if self.ownership == BackendOwnership::External {
            return Ok(ShutdownResult::NotOwned);
        }
        let ownership = self.ownership;
        let pid = self.pid();
        let mut control = SessionShutdownControl { session: self };
        stop_with_control(ownership, pid, &mut control)
    }
}

struct SessionShutdownControl<'a> {
    session: &'a mut BackendSession,
}

impl ShutdownControl for SessionShutdownControl<'_> {
    fn write_stop_file(&mut self) -> io::Result<()> {
        let Some(path) = &self.session.stop_file else {
            return Ok(());
        };
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, format!("stop requested at {:?}\n", SystemTime::now()))
    }

    fn wait_for_exit(&mut self, timeout: Duration) -> io::Result<bool> {
        let Some(child) = self.session.child.as_mut() else {
            return Ok(true);
        };
        if let Some(status) = wait_for_child(child, timeout)? {
            self.session.child = None;
            self.session.exit_status = Some(status);
            return Ok(true);
        }
        Ok(false)
    }

    fn terminate_job(&mut self) -> io::Result<()> {
        #[cfg(windows)]
        if let Some(job) = self.session.job.as_ref() {
            return job.terminate(1);
        }
        Ok(())
    }

    fn taskkill_exact_pid(&mut self, pid: u32) -> io::Result<()> {
        run_taskkill(pid)
    }
}

fn wait_for_child(child: &mut Child, timeout: Duration) -> io::Result<Option<ExitStatus>> {
    let deadline = Instant::now() + timeout;
    loop {
        if let Some(status) = child.try_wait()? {
            return Ok(Some(status));
        }
        if Instant::now() >= deadline {
            return Ok(None);
        }
        thread::sleep(
            PROCESS_POLL_INTERVAL.min(deadline.saturating_duration_since(Instant::now())),
        );
    }
}

fn run_taskkill(pid: u32) -> io::Result<()> {
    #[cfg(windows)]
    let executable = Path::new(r"C:\Windows\System32\taskkill.exe");
    #[cfg(not(windows))]
    let executable = Path::new("taskkill");
    let output = Command::new(executable)
        .args(taskkill_arguments(pid))
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .output()?;
    if output.status.success() {
        return Ok(());
    }
    let detail = String::from_utf8_lossy(&output.stderr).trim().to_owned();
    Err(io::Error::other(if detail.is_empty() {
        format!("taskkill exited with {}", output.status)
    } else {
        format!("taskkill exited with {}: {detail}", output.status)
    }))
}

fn get_case_insensitive<'a>(
    environment: &'a HashMap<OsString, OsString>,
    wanted: &str,
) -> Option<&'a OsString> {
    environment
        .iter()
        .find_map(|(key, value)| key.eq_ignore_ascii_case(wanted).then_some(value))
}

fn remove_case_insensitive(environment: &mut HashMap<OsString, OsString>, wanted: &str) {
    environment.retain(|key, _| !key.eq_ignore_ascii_case(wanted));
}

fn insert_environment(
    environment: &mut HashMap<OsString, OsString>,
    key: impl Into<OsString>,
    value: impl AsRef<OsStr>,
) {
    let key = key.into();
    remove_case_insensitive(environment, &key.to_string_lossy());
    environment.insert(key, value.as_ref().to_os_string());
}

fn join_python_paths(source_dir: &Path, host_python_path: Option<&OsStr>) -> OsString {
    let mut paths = vec![source_dir.to_path_buf()];
    if let Some(host) = host_python_path.filter(|value| !value.is_empty()) {
        paths.extend(std::env::split_paths(host));
    }
    std::env::join_paths(paths).unwrap_or_else(|_| source_dir.as_os_str().to_os_string())
}

fn join_runtime_paths(paths: &[PathBuf], fallback: &Path) -> OsString {
    let native_paths = paths
        .iter()
        .map(|path| native_loader_compatible_path(path))
        .collect::<Vec<_>>();
    std::env::join_paths(native_paths)
        .unwrap_or_else(|_| native_loader_compatible_path(fallback).into_os_string())
}

fn native_loader_compatible_path(path: &Path) -> PathBuf {
    #[cfg(windows)]
    {
        let units = path.as_os_str().encode_wide().collect::<Vec<_>>();
        let slash = b'\\' as u16;
        let verbatim = [slash, slash, b'?' as u16, slash];
        if !units.starts_with(&verbatim) {
            return path.to_path_buf();
        }

        let is_verbatim_unc = units.len() >= 8
            && matches!(units[4], value if value == b'U' as u16 || value == b'u' as u16)
            && matches!(units[5], value if value == b'N' as u16 || value == b'n' as u16)
            && matches!(units[6], value if value == b'C' as u16 || value == b'c' as u16)
            && units[7] == slash;
        let normalized = if is_verbatim_unc {
            let mut value = vec![slash, slash];
            value.extend_from_slice(&units[8..]);
            value
        } else {
            units[4..].to_vec()
        };
        PathBuf::from(OsString::from_wide(&normalized))
    }

    #[cfg(not(windows))]
    {
        path.to_path_buf()
    }
}
