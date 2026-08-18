use std::{
    collections::HashSet,
    env, fs,
    path::{Path, PathBuf},
};

use serde::Deserialize;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PortableRuntimeLayout {
    V1,
    V2,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RuntimePackageLayers {
    pub main: Vec<PathBuf>,
    pub cosyvoice: Vec<PathBuf>,
    pub feynobg: Vec<PathBuf>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct PortableRuntimeManifest {
    schema_version: u64,
    layout: String,
    #[serde(default)]
    paths: PortableRuntimePaths,
    python: Option<serde_json::Value>,
    components: PortableRuntimeComponents,
}

#[derive(Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PortableRuntimePaths {
    python: Option<String>,
    main_python: Option<String>,
    cosyvoice_python: Option<String>,
    feynobg_python: Option<String>,
    orchestrator: Option<String>,
    source: Option<String>,
    artifacts: Option<String>,
    models: Option<String>,
    tensor_rt_assistant: Option<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct PortableV2Python {
    version: String,
    package_layers: PortableV2PackageLayers,
}

#[derive(Deserialize)]
struct PortableV2PackageLayers {
    main: Vec<String>,
    cosyvoice: Vec<String>,
    feynobg: Vec<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct PortableRuntimeComponents {
    dependencies_included: bool,
    models_included: bool,
    frontend_vendor_included: bool,
    tensor_rt_build_inputs_included: bool,
}

struct ParsedPortableRuntimeManifest {
    layout: PortableRuntimeLayout,
    python: PathBuf,
    orchestrator: PathBuf,
    source_dir: PathBuf,
    artifacts_dir: PathBuf,
    models_dir: PathBuf,
    package_layers: RuntimePackageLayers,
}

fn normalized_relative_path(value: &str) -> Option<PathBuf> {
    if value.trim().is_empty() || value.contains('\0') {
        return None;
    }
    let portable = value.replace('\\', "/");
    if portable.starts_with('/') || portable.contains(':') {
        return None;
    }

    let mut normalized = PathBuf::new();
    for component in portable.split('/') {
        match component {
            "" | "." => {}
            ".." => {
                if !normalized.pop() {
                    return None;
                }
            }
            value => normalized.push(value),
        }
    }
    Some(normalized)
}

fn resolve_manifest_path(root: &Path, value: &str) -> Option<PathBuf> {
    normalized_relative_path(value).map(|relative| root.join(relative))
}

fn resolve_optional_manifest_path(
    root: &Path,
    value: Option<&str>,
    default: &str,
) -> Option<PathBuf> {
    resolve_manifest_path(root, value.unwrap_or(default))
}

fn components_are_complete(components: &PortableRuntimeComponents) -> bool {
    components.dependencies_included
        && components.models_included
        && components.frontend_vendor_included
        && components.tensor_rt_build_inputs_included
}

fn optional_paths_are_contained(paths: &PortableRuntimePaths) -> bool {
    [
        paths.python.as_deref(),
        paths.main_python.as_deref(),
        paths.cosyvoice_python.as_deref(),
        paths.feynobg_python.as_deref(),
        paths.orchestrator.as_deref(),
        paths.source.as_deref(),
        paths.artifacts.as_deref(),
        paths.models.as_deref(),
        paths.tensor_rt_assistant.as_deref(),
    ]
    .into_iter()
    .flatten()
    .all(|value| normalized_relative_path(value).is_some())
}

fn resolve_package_layers(root: &Path, values: Vec<String>) -> Option<Vec<PathBuf>> {
    if values.is_empty() {
        return None;
    }
    values
        .into_iter()
        .map(|value| resolve_manifest_path(root, &value))
        .collect()
}

fn legacy_package_layers(root: &Path, source_dir: &Path) -> RuntimePackageLayers {
    RuntimePackageLayers {
        main: vec![source_dir.to_path_buf()],
        cosyvoice: vec![
            source_dir.to_path_buf(),
            root.join("third_party").join("CosyVoice"),
            root.join("third_party")
                .join("CosyVoice")
                .join("third_party")
                .join("Matcha-TTS"),
        ],
        feynobg: vec![source_dir.to_path_buf()],
    }
}

fn parse_portable_manifest(path: &Path, root: &Path) -> Option<ParsedPortableRuntimeManifest> {
    let Ok(body) = fs::read(path) else {
        return None;
    };
    let Ok(manifest) = serde_json::from_slice::<PortableRuntimeManifest>(&body) else {
        return None;
    };
    if !components_are_complete(&manifest.components)
        || !optional_paths_are_contained(&manifest.paths)
    {
        return None;
    }

    match (manifest.schema_version, manifest.layout.as_str()) {
        (1, "portable-v1") => {
            let python = resolve_optional_manifest_path(
                root,
                manifest.paths.main_python.as_deref(),
                "python-main/python.exe",
            )?;
            let orchestrator = resolve_optional_manifest_path(
                root,
                manifest.paths.orchestrator.as_deref(),
                "scripts/run_local_stream.py",
            )?;
            let source_dir =
                resolve_optional_manifest_path(root, manifest.paths.source.as_deref(), "src")?;
            let artifacts_dir = resolve_optional_manifest_path(
                root,
                manifest.paths.artifacts.as_deref(),
                "artifacts/main",
            )?;
            let models_dir =
                resolve_optional_manifest_path(root, manifest.paths.models.as_deref(), "models")?;
            Some(ParsedPortableRuntimeManifest {
                layout: PortableRuntimeLayout::V1,
                python,
                orchestrator,
                artifacts_dir,
                models_dir,
                package_layers: legacy_package_layers(root, &source_dir),
                source_dir,
            })
        }
        (2, "portable-v2") => {
            let python_metadata =
                serde_json::from_value::<PortableV2Python>(manifest.python?).ok()?;
            if python_metadata.version.trim().is_empty() {
                return None;
            }
            let python = resolve_manifest_path(root, manifest.paths.python.as_deref()?)?;
            let orchestrator =
                resolve_manifest_path(root, manifest.paths.orchestrator.as_deref()?)?;
            let source_dir = resolve_manifest_path(root, manifest.paths.source.as_deref()?)?;
            let artifacts_dir = resolve_manifest_path(root, manifest.paths.artifacts.as_deref()?)?;
            let models_dir = resolve_manifest_path(root, manifest.paths.models.as_deref()?)?;
            let package_layers = RuntimePackageLayers {
                main: resolve_package_layers(root, python_metadata.package_layers.main)?,
                cosyvoice: resolve_package_layers(root, python_metadata.package_layers.cosyvoice)?,
                feynobg: resolve_package_layers(root, python_metadata.package_layers.feynobg)?,
            };
            Some(ParsedPortableRuntimeManifest {
                layout: PortableRuntimeLayout::V2,
                python,
                orchestrator,
                source_dir,
                artifacts_dir,
                models_dir,
                package_layers,
            })
        }
        _ => None,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RuntimeMode {
    Managed,
    Development,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RuntimeSource {
    Explicit,
    Environment,
    Persisted,
    Packaged,
    Sibling,
    Development,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RuntimeInfo {
    pub root: PathBuf,
    pub mode: Option<RuntimeMode>,
    pub python: Option<PathBuf>,
    pub orchestrator: PathBuf,
    pub source_dir: PathBuf,
    pub artifacts_dir: PathBuf,
    pub models_dir: PathBuf,
    pub manifest: Option<PathBuf>,
    pub manifest_layout: Option<PortableRuntimeLayout>,
    pub package_layers: RuntimePackageLayers,
    pub missing: Vec<String>,
}

impl RuntimeInfo {
    pub fn is_valid(&self) -> bool {
        self.missing.is_empty()
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RuntimeResolveOptions {
    pub explicit_root: Option<PathBuf>,
    pub environment_root: Option<PathBuf>,
    pub persisted_root: Option<PathBuf>,
    pub resources_dir: Option<PathBuf>,
    pub executable_path: Option<PathBuf>,
    pub development_roots: Vec<PathBuf>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InspectedRuntimeCandidate {
    pub source: RuntimeSource,
    pub runtime: RuntimeInfo,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolvedRuntime {
    pub source: RuntimeSource,
    pub runtime: RuntimeInfo,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RuntimeResolution {
    pub selected: Option<ResolvedRuntime>,
    pub candidates: Vec<InspectedRuntimeCandidate>,
}

fn absolute_path(path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(path)
    }
}

fn is_file(path: &Path) -> bool {
    fs::metadata(path).is_ok_and(|metadata| metadata.is_file())
}

fn is_dir(path: &Path) -> bool {
    fs::metadata(path).is_ok_and(|metadata| metadata.is_dir())
}

pub fn inspect_runtime_root(root: impl AsRef<Path>) -> RuntimeInfo {
    let root = absolute_path(root.as_ref());
    let legacy_managed_python = root.join("python-main").join("python.exe");
    let development_python = root.join(".venv").join("Scripts").join("python.exe");
    let manifest_path = root.join("runtime-manifest.json");
    let manifest_exists = is_file(&manifest_path);
    let parsed_manifest = manifest_exists
        .then(|| parse_portable_manifest(&manifest_path, &root))
        .flatten();

    let managed_python = parsed_manifest
        .as_ref()
        .map(|manifest| manifest.python.clone())
        .unwrap_or_else(|| legacy_managed_python.clone());
    let orchestrator = parsed_manifest
        .as_ref()
        .map(|manifest| manifest.orchestrator.clone())
        .unwrap_or_else(|| root.join("scripts").join("run_local_stream.py"));
    let source_dir = parsed_manifest
        .as_ref()
        .map(|manifest| manifest.source_dir.clone())
        .unwrap_or_else(|| root.join("src"));
    let artifacts_dir = parsed_manifest
        .as_ref()
        .map(|manifest| manifest.artifacts_dir.clone())
        .unwrap_or_else(|| root.join("artifacts").join("main"));
    let models_dir = parsed_manifest
        .as_ref()
        .map(|manifest| manifest.models_dir.clone())
        .unwrap_or_else(|| root.join("models"));

    let (mode, python) = if is_file(&managed_python) {
        (Some(RuntimeMode::Managed), Some(managed_python))
    } else if is_file(&development_python) {
        (Some(RuntimeMode::Development), Some(development_python))
    } else {
        (None, None)
    };

    let mut missing = Vec::new();
    if python.is_none() {
        missing.push("Python runtime".to_owned());
    }
    if !is_file(&orchestrator) {
        missing.push("scripts/run_local_stream.py".to_owned());
    }
    if !is_dir(&source_dir) {
        missing.push("src".to_owned());
    }
    if !is_dir(&artifacts_dir) {
        missing.push("artifacts/main".to_owned());
    }
    if parsed_manifest
        .as_ref()
        .is_some_and(|manifest| manifest.layout == PortableRuntimeLayout::V2)
        && !is_dir(&models_dir)
    {
        missing.push("models".to_owned());
    }
    if mode == Some(RuntimeMode::Managed) && manifest_exists && parsed_manifest.is_none() {
        missing.push("compatible runtime-manifest.json".to_owned());
    }

    let package_layers = if mode == Some(RuntimeMode::Managed) {
        parsed_manifest
            .as_ref()
            .map(|manifest| manifest.package_layers.clone())
            .unwrap_or_else(|| legacy_package_layers(&root, &source_dir))
    } else {
        legacy_package_layers(&root, &source_dir)
    };
    if parsed_manifest
        .as_ref()
        .is_some_and(|manifest| manifest.layout == PortableRuntimeLayout::V2)
    {
        for layer in package_layers
            .main
            .iter()
            .chain(&package_layers.cosyvoice)
            .chain(&package_layers.feynobg)
        {
            if !is_dir(layer) {
                missing.push(format!("package layer {}", layer.display()));
            }
        }
    }

    RuntimeInfo {
        root,
        mode,
        python,
        orchestrator,
        source_dir,
        artifacts_dir,
        models_dir,
        manifest: manifest_exists.then_some(manifest_path),
        manifest_layout: parsed_manifest.as_ref().map(|manifest| manifest.layout),
        package_layers,
        missing,
    }
}

fn candidate_key(path: &Path) -> String {
    let value = path.to_string_lossy();
    if cfg!(windows) {
        value.to_lowercase()
    } else {
        value.into_owned()
    }
}

pub fn resolve_runtime_root(options: &RuntimeResolveOptions) -> RuntimeResolution {
    let packaged = options
        .resources_dir
        .as_ref()
        .map(|root| root.join("avtr-runtime"));
    let sibling = options
        .executable_path
        .as_ref()
        .and_then(|executable| executable.parent())
        .map(|root| root.join("avtr-runtime"));

    let mut requested = Vec::new();
    if let Some(root) = &options.explicit_root {
        requested.push((RuntimeSource::Explicit, root.clone()));
    }
    if let Some(root) = &options.environment_root {
        requested.push((RuntimeSource::Environment, root.clone()));
    }
    if let Some(root) = &options.persisted_root {
        requested.push((RuntimeSource::Persisted, root.clone()));
    }
    if let Some(root) = packaged {
        requested.push((RuntimeSource::Packaged, root));
    }
    if let Some(root) = sibling {
        requested.push((RuntimeSource::Sibling, root));
    }
    requested.extend(
        options
            .development_roots
            .iter()
            .cloned()
            .map(|root| (RuntimeSource::Development, root)),
    );

    let mut seen = HashSet::new();
    let mut candidates = Vec::new();
    for (source, root) in requested {
        let root = absolute_path(&root);
        if !seen.insert(candidate_key(&root)) {
            continue;
        }
        candidates.push(InspectedRuntimeCandidate {
            source,
            runtime: inspect_runtime_root(root),
        });
    }

    let selected = candidates
        .iter()
        .find(|candidate| candidate.runtime.is_valid())
        .map(|candidate| ResolvedRuntime {
            source: candidate.source,
            runtime: candidate.runtime.clone(),
        });

    RuntimeResolution {
        selected,
        candidates,
    }
}
