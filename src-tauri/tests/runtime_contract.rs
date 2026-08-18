use std::{
    fs,
    path::{Path, PathBuf},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use avtr1_desktop::{
    health::{probe_avtr_service, HealthFailure},
    runtime::{
        inspect_runtime_root, resolve_runtime_root, PortableRuntimeLayout, RuntimeMode,
        RuntimeResolveOptions, RuntimeSource,
    },
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
};

struct TestTree(PathBuf);

impl TestTree {
    fn new(label: &str) -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock must be after Unix epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "avtr1-tauri-{label}-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(&root).expect("create isolated test tree");
        Self(root)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TestTree {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn create_runtime(root: &Path, managed: bool, development: bool) {
    fs::create_dir_all(root.join("scripts")).unwrap();
    fs::create_dir_all(root.join("src")).unwrap();
    fs::create_dir_all(root.join("artifacts/main")).unwrap();
    fs::write(root.join("scripts/run_local_stream.py"), b"# test\n").unwrap();
    if managed {
        fs::create_dir_all(root.join("python-main")).unwrap();
        fs::write(root.join("python-main/python.exe"), b"test").unwrap();
    }
    if development {
        fs::create_dir_all(root.join(".venv/Scripts")).unwrap();
        fs::write(root.join(".venv/Scripts/python.exe"), b"test").unwrap();
    }
}

#[test]
fn managed_python_has_priority_over_the_development_venv() {
    let tree = TestTree::new("managed-priority");
    create_runtime(tree.path(), true, true);
    fs::write(
        tree.path().join("runtime-manifest.json"),
        br#"{
          "schemaVersion": 1,
          "layout": "portable-v1",
          "components": {
            "dependenciesIncluded": true,
            "modelsIncluded": true,
            "frontendVendorIncluded": true,
            "tensorRtBuildInputsIncluded": true
          }
        }"#,
    )
    .unwrap();

    let runtime = inspect_runtime_root(tree.path());

    assert!(runtime.is_valid());
    assert_eq!(runtime.mode, Some(RuntimeMode::Managed));
    assert_eq!(
        runtime.python,
        Some(tree.path().join("python-main/python.exe"))
    );
    assert_eq!(
        runtime.manifest,
        Some(tree.path().join("runtime-manifest.json"))
    );
}

#[test]
fn managed_runtime_rejects_an_incompatible_portable_manifest() {
    let tree = TestTree::new("managed-manifest");
    create_runtime(tree.path(), true, false);
    fs::write(
        tree.path().join("runtime-manifest.json"),
        br#"{
          "schemaVersion": 2,
          "layout": "future-layout",
          "components": {
            "dependenciesIncluded": false,
            "modelsIncluded": false,
            "frontendVendorIncluded": false,
            "tensorRtBuildInputsIncluded": false
          }
        }"#,
    )
    .unwrap();

    let runtime = inspect_runtime_root(tree.path());

    assert!(!runtime.is_valid());
    assert!(runtime
        .missing
        .iter()
        .any(|item| item == "compatible runtime-manifest.json"));
}

#[test]
fn managed_runtime_accepts_the_complete_portable_v1_manifest() {
    let tree = TestTree::new("managed-manifest-complete");
    create_runtime(tree.path(), true, false);
    fs::write(
        tree.path().join("runtime-manifest.json"),
        br#"{
          "schemaVersion": 1,
          "layout": "portable-v1",
          "components": {
            "dependenciesIncluded": true,
            "modelsIncluded": true,
            "frontendVendorIncluded": true,
            "tensorRtBuildInputsIncluded": true
          }
        }"#,
    )
    .unwrap();

    assert!(inspect_runtime_root(tree.path()).is_valid());
}

#[test]
fn managed_runtime_accepts_portable_v2_with_one_python_and_ordered_package_layers() {
    let tree = TestTree::new("managed-manifest-v2");
    fs::create_dir_all(tree.path().join("python")).unwrap();
    fs::create_dir_all(tree.path().join("scripts")).unwrap();
    fs::create_dir_all(tree.path().join("src")).unwrap();
    fs::create_dir_all(tree.path().join("artifacts/main")).unwrap();
    fs::create_dir_all(tree.path().join("models")).unwrap();
    for layer in [
        "packages/main",
        "packages/cosyvoice",
        "packages/feynobg",
        "packages/shared",
        "third_party/CosyVoice",
        "third_party/CosyVoice/third_party/Matcha-TTS",
    ] {
        fs::create_dir_all(tree.path().join(layer)).unwrap();
    }
    fs::write(tree.path().join("python/python.exe"), b"test").unwrap();
    fs::write(tree.path().join("scripts/run_local_stream.py"), b"# test\n").unwrap();
    fs::write(
        tree.path().join("runtime-manifest.json"),
        br#"{
          "schemaVersion": 2,
          "layout": "portable-v2",
          "paths": {
            "python": "python/python.exe",
            "orchestrator": "scripts/run_local_stream.py",
            "source": "src",
            "artifacts": "artifacts/main",
            "models": "models"
          },
          "python": {
            "version": "3.12.9",
            "packageLayers": {
              "main": ["packages/main", "packages/shared", "src"],
              "cosyvoice": [
                "packages/cosyvoice",
                "packages/shared",
                "third_party/CosyVoice",
                "third_party/CosyVoice/third_party/Matcha-TTS",
                "src"
              ],
              "feynobg": ["packages/feynobg", "packages/shared", "src"]
            }
          },
          "components": {
            "dependenciesIncluded": true,
            "modelsIncluded": true,
            "frontendVendorIncluded": true,
            "tensorRtBuildInputsIncluded": true
          }
        }"#,
    )
    .unwrap();

    let runtime = inspect_runtime_root(tree.path());

    assert!(runtime.is_valid(), "missing: {:?}", runtime.missing);
    assert_eq!(runtime.mode, Some(RuntimeMode::Managed));
    assert_eq!(runtime.manifest_layout, Some(PortableRuntimeLayout::V2));
    assert_eq!(runtime.python, Some(tree.path().join("python/python.exe")));
    assert_eq!(
        runtime.package_layers.main,
        vec![
            tree.path().join("packages/main"),
            tree.path().join("packages/shared"),
            tree.path().join("src"),
        ]
    );
    assert_eq!(
        runtime.package_layers.cosyvoice,
        vec![
            tree.path().join("packages/cosyvoice"),
            tree.path().join("packages/shared"),
            tree.path().join("third_party/CosyVoice"),
            tree.path()
                .join("third_party/CosyVoice/third_party/Matcha-TTS"),
            tree.path().join("src"),
        ]
    );
    assert_eq!(
        runtime.package_layers.feynobg,
        vec![
            tree.path().join("packages/feynobg"),
            tree.path().join("packages/shared"),
            tree.path().join("src"),
        ]
    );
}

#[test]
fn managed_runtime_rejects_portable_v2_when_manifest_models_directory_is_missing() {
    let tree = TestTree::new("managed-manifest-v2-models-missing");
    fs::create_dir_all(tree.path().join("python")).unwrap();
    fs::create_dir_all(tree.path().join("scripts")).unwrap();
    fs::create_dir_all(tree.path().join("src")).unwrap();
    fs::create_dir_all(tree.path().join("artifacts/main")).unwrap();
    for layer in [
        "packages/main",
        "packages/cosyvoice",
        "packages/feynobg",
        "packages/shared",
    ] {
        fs::create_dir_all(tree.path().join(layer)).unwrap();
    }
    fs::write(tree.path().join("python/python.exe"), b"test").unwrap();
    fs::write(tree.path().join("scripts/run_local_stream.py"), b"# test\n").unwrap();
    fs::write(
        tree.path().join("runtime-manifest.json"),
        br#"{
          "schemaVersion": 2,
          "layout": "portable-v2",
          "paths": {
            "python": "python/python.exe",
            "orchestrator": "scripts/run_local_stream.py",
            "source": "src",
            "artifacts": "artifacts/main",
            "models": "missing-models"
          },
          "python": {
            "version": "3.12.9",
            "packageLayers": {
              "main": ["packages/main", "packages/shared", "src"],
              "cosyvoice": ["packages/cosyvoice", "packages/shared", "src"],
              "feynobg": ["packages/feynobg", "packages/shared", "src"]
            }
          },
          "components": {
            "dependenciesIncluded": true,
            "modelsIncluded": true,
            "frontendVendorIncluded": true,
            "tensorRtBuildInputsIncluded": true
          }
        }"#,
    )
    .unwrap();

    let runtime = inspect_runtime_root(tree.path());

    assert!(!runtime.is_valid());
    assert_eq!(runtime.models_dir, tree.path().join("missing-models"));
    assert!(runtime.missing.iter().any(|item| item == "models"));
}

#[test]
fn managed_runtime_rejects_manifest_paths_that_escape_the_runtime_root() {
    let tree = TestTree::new("managed-manifest-escape");
    create_runtime(tree.path(), true, false);
    fs::write(
        tree.path().join("runtime-manifest.json"),
        br#"{
          "schemaVersion": 1,
          "layout": "portable-v1",
          "paths": {
            "mainPython": "python-main/python.exe",
            "orchestrator": "scripts/run_local_stream.py",
            "source": "../host-source",
            "artifacts": "artifacts/main",
            "models": "models"
          },
          "components": {
            "dependenciesIncluded": true,
            "modelsIncluded": true,
            "frontendVendorIncluded": true,
            "tensorRtBuildInputsIncluded": true
          }
        }"#,
    )
    .unwrap();

    let runtime = inspect_runtime_root(tree.path());

    assert!(!runtime.is_valid());
    assert!(runtime
        .missing
        .iter()
        .any(|item| item == "compatible runtime-manifest.json"));
}

#[test]
fn runtime_inspection_reports_each_required_component() {
    let tree = TestTree::new("missing-components");

    let runtime = inspect_runtime_root(tree.path());

    assert!(!runtime.is_valid());
    assert_eq!(
        runtime.missing,
        vec![
            "Python runtime",
            "scripts/run_local_stream.py",
            "src",
            "artifacts/main",
        ]
    );
}

#[test]
fn runtime_resolution_uses_the_documented_candidate_precedence() {
    let tree = TestTree::new("precedence");
    let explicit = tree.path().join("explicit");
    let environment = tree.path().join("environment");
    let persisted = tree.path().join("persisted");
    let resources = tree.path().join("resources");
    let packaged = resources.join("avtr-runtime");
    let executable = tree.path().join("installed/AVTR-1.exe");
    let sibling = executable.parent().unwrap().join("avtr-runtime");
    let development = tree.path().join("development");
    for root in [
        &explicit,
        &environment,
        &persisted,
        &packaged,
        &sibling,
        &development,
    ] {
        create_runtime(root, true, false);
    }

    let resolution = resolve_runtime_root(&RuntimeResolveOptions {
        explicit_root: Some(explicit.clone()),
        environment_root: Some(environment.clone()),
        persisted_root: Some(persisted.clone()),
        resources_dir: Some(resources),
        executable_path: Some(executable),
        development_roots: vec![development.clone()],
    });

    let selected = resolution
        .selected
        .expect("a valid runtime must be selected");
    assert_eq!(selected.source, RuntimeSource::Explicit);
    assert_eq!(selected.runtime.root, explicit);
    assert_eq!(
        resolution
            .candidates
            .iter()
            .map(|candidate| candidate.source)
            .collect::<Vec<_>>(),
        vec![
            RuntimeSource::Explicit,
            RuntimeSource::Environment,
            RuntimeSource::Persisted,
            RuntimeSource::Packaged,
            RuntimeSource::Sibling,
            RuntimeSource::Development,
        ]
    );
}

#[test]
fn runtime_resolution_skips_invalid_higher_priority_candidates() {
    let tree = TestTree::new("invalid-first");
    let explicit = tree.path().join("invalid-explicit");
    let environment = tree.path().join("valid-environment");
    fs::create_dir_all(&explicit).unwrap();
    create_runtime(&environment, false, true);

    let resolution = resolve_runtime_root(&RuntimeResolveOptions {
        explicit_root: Some(explicit),
        environment_root: Some(environment.clone()),
        ..RuntimeResolveOptions::default()
    });

    let selected = resolution
        .selected
        .expect("fallback runtime must be selected");
    assert_eq!(selected.source, RuntimeSource::Environment);
    assert_eq!(selected.runtime.mode, Some(RuntimeMode::Development));
    assert_eq!(selected.runtime.root, environment);
}

async fn serve_json(status: u16, body: &'static str) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let mut request = [0_u8; 2048];
        let _ = stream.read(&mut request).await.unwrap();
        let reason = if status == 200 {
            "OK"
        } else {
            "Service Unavailable"
        };
        let response = format!(
            "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        stream.write_all(response.as_bytes()).await.unwrap();
    });
    format!("http://{address}/health")
}

#[tokio::test]
async fn health_probe_accepts_only_the_exact_avtr_service_identity() {
    let client = reqwest::Client::new();
    let valid_url = serve_json(
        200,
        r#"{"service":"avtr1-streamer","status":"ready","instance_id":"owned-123"}"#,
    )
    .await;
    let wrong_url = serve_json(200, r#"{"service":"unrelated","status":"ready"}"#).await;

    let valid = probe_avtr_service(&client, &valid_url, Duration::from_secs(1)).await;
    let wrong = probe_avtr_service(&client, &wrong_url, Duration::from_secs(1)).await;

    assert!(valid.healthy);
    assert_eq!(valid.failure, None);
    assert_eq!(valid.instance_id.as_deref(), Some("owned-123"));
    assert!(!wrong.healthy);
    assert_eq!(wrong.failure, Some(HealthFailure::IdentityMismatch));
}

#[tokio::test]
async fn owned_health_requires_the_expected_backend_instance() {
    let client = reqwest::Client::new();
    let matching_url = serve_json(
        200,
        r#"{"service":"avtr1-streamer","status":"ready","instance_id":"owned-123"}"#,
    )
    .await;
    let external_url = serve_json(200, r#"{"service":"avtr1-streamer","status":"ready"}"#).await;
    let other_url = serve_json(
        200,
        r#"{"service":"avtr1-streamer","status":"ready","instance_id":"other"}"#,
    )
    .await;

    let matching = probe_avtr_service(&client, &matching_url, Duration::from_secs(1)).await;
    let external = probe_avtr_service(&client, &external_url, Duration::from_secs(1)).await;
    let other = probe_avtr_service(&client, &other_url, Duration::from_secs(1)).await;

    assert!(matching.matches_owned_instance("owned-123"));
    assert!(!external.matches_owned_instance("owned-123"));
    assert!(!other.matches_owned_instance("owned-123"));
}

#[tokio::test]
async fn health_probe_accepts_degraded_but_rejects_other_service_states() {
    let client = reqwest::Client::new();
    let degraded_url = serve_json(200, r#"{"service":"avtr1-streamer","status":"degraded"}"#).await;
    let starting_url = serve_json(200, r#"{"service":"avtr1-streamer","status":"starting"}"#).await;

    let degraded = probe_avtr_service(&client, &degraded_url, Duration::from_secs(1)).await;
    let starting = probe_avtr_service(&client, &starting_url, Duration::from_secs(1)).await;

    assert!(degraded.healthy);
    assert!(!starting.healthy);
    assert_eq!(
        starting.failure,
        Some(HealthFailure::ServiceStatus("starting".to_owned()))
    );
}

#[tokio::test]
async fn health_probe_rejects_non_success_http_responses() {
    let client = reqwest::Client::new();
    let url = serve_json(503, r#"{"service":"avtr1-streamer","status":"ready"}"#).await;

    let result = probe_avtr_service(&client, &url, Duration::from_secs(1)).await;

    assert!(!result.healthy);
    assert_eq!(result.failure, Some(HealthFailure::HttpStatus(503)));
}
