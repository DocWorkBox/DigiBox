from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_tauri_product_is_digibox_without_changing_the_stable_data_identity() -> None:
    config = json.loads(_read("src-tauri/tauri.conf.json"))

    assert config["productName"] == "DigiBox"
    assert config["identifier"] == "live.avaturn.avtr1.desktop"
    assert config["bundle"]["publisher"] == "DigiBox"
    windows = {window["label"]: window for window in config["app"]["windows"]}
    assert windows["splash"]["title"] == "DigiBox 正在启动"
    assert windows["main"]["title"] == "DigiBox"
    assert config["bundle"]["shortDescription"].startswith("DigiBox ")
    assert config["bundle"]["longDescription"].startswith("DigiBox ")
    assert config["bundle"]["windows"]["nsis"]["installerIcon"] == "icons/icon.ico"
    assert config["bundle"]["windows"]["nsis"]["uninstallerIcon"] == "icons/icon.ico"


def test_tauri_icon_uses_the_digibox_monogram() -> None:
    # The source icon filename makes the new brand asset explicit; Tauri's
    # generated Windows PNG/ICO files are verified separately by the build.
    assert (ROOT / "src-tauri/icons/digibox-source.png").is_file()


def test_desktop_package_and_rust_binary_are_named_digibox() -> None:
    package = json.loads(_read("package.json"))
    lock = json.loads(_read("package-lock.json"))
    cargo = _read("src-tauri/Cargo.toml")

    assert package["name"] == "digibox-desktop"
    assert package["description"].startswith("DigiBox ")
    assert lock["name"] == "digibox-desktop"
    assert lock["packages"][""]["name"] == "digibox-desktop"
    assert 'name = "digibox-desktop"' in cargo
    # The Rust library name remains stable so existing contract imports keep working.
    assert 'name = "avtr1_desktop"' in cargo


def test_digibox_packages_and_assets_declare_their_component_licenses() -> None:
    package = json.loads(_read("package.json"))
    lock = json.loads(_read("package-lock.json"))
    cargo = _read("src-tauri/Cargo.toml")
    assets = _read("ASSET-NOTICES.md")

    assert package["license"] == "Apache-2.0"
    assert lock["packages"][""]["license"] == "Apache-2.0"
    assert 'license = "Apache-2.0"' in cargo
    assert "src-tauri/icons/" in assets and "LICENSE-DIGIBOX.md" in assets
    assert "src/avtr1_renderer/assets/" in assets
    assert "LICENSE-RENDERER.md" in assets
    assert "PolyForm Noncommercial" in assets


def test_tauri_and_fallback_splashes_show_digibox_branding() -> None:
    for relative in (
        "desktop/tauri/index.html",
        "desktop/tauri/app.js",
        "desktop/splash.html",
        "desktop/splash.js",
    ):
        source = _read(relative)
        assert "DigiBox" in source, relative
        assert ">AVTR-1<" not in source, relative
        assert "AVTR-1 已就绪" not in source, relative
        assert "AVTR-1 启动失败" not in source, relative


def test_tauri_and_fallback_splashes_do_not_show_backend_ownership_footnote() -> None:
    for relative in ("desktop/tauri/index.html", "desktop/splash.html"):
        source = _read(relative)
        assert "只管理自己启动的后端" not in source, relative
        assert "仅接管自己启动的后端" not in source, relative
        assert "外部 AVTR-1 兼容服务" not in source, relative


def test_actual_digital_human_page_uses_digibox_as_its_product_title() -> None:
    source = _read("src/avaturn_live_streamer/local_stream_ui.html")

    assert "<title>DigiBox 本地互动</title>" in source
    assert "<title>AVTR-1 本地互动</title>" not in source


def test_windows_build_outputs_and_tensorrt_assistant_use_digibox_name() -> None:
    tauri_build = _read("scripts/build_tauri_windows.ps1")
    electron_config = _read("electron-builder.yml")
    electron_full_config = _read("electron-builder-full.yml")

    assert "DigiBox-Full-win64" in tauri_build
    assert "AVTR-1-Full-win64" not in tauri_build
    assert "productName: DigiBox" in electron_config
    assert "artifactName: DigiBox-Setup-" in electron_config
    assert "productName: DigiBox" in electron_full_config
    assert "artifactName: DigiBox-Full-" in electron_full_config
    assert (ROOT / "scripts/desktop/DigiBox-TensorRT-Setup.cmd").is_file()
    assert not (ROOT / "scripts/desktop/AVTR-1-TensorRT-Setup.cmd").exists()


def test_electron_fallback_builds_use_the_digibox_windows_icon() -> None:
    expected_win_header = "\nwin:\n  icon: src-tauri/icons/icon.ico\n"

    for relative in ("electron-builder.yml", "electron-builder-full.yml"):
        source = _read(relative)
        assert expected_win_header in source, relative


def test_tauri_startup_messages_use_the_product_name() -> None:
    source = _read("src-tauri/src/app.rs")

    for message in (
        "正在准备 DigiBox 桌面运行环境",
        "DigiBox 已就绪",
        "DigiBox 启动失败",
        "正在安全关闭 DigiBox",
    ):
        assert message in source


def test_current_desktop_distribution_docs_name_digibox_artifacts() -> None:
    tauri_doc = _read("docs/windows-tauri-desktop-distribution.md")
    fallback_doc = _read("docs/windows-desktop-distribution.md")
    readme = _read("README.md")

    assert tauri_doc.startswith("# DigiBox Windows Tauri v2 桌面版构建与分发")
    assert "digibox-desktop.exe" in tauri_doc
    assert "DigiBox-Full-win64" in tauri_doc
    assert fallback_doc.startswith("# DigiBox Windows 桌面发行说明")
    assert "DigiBox-Setup-<version>-x64.exe" in fallback_doc
    assert "DigiBox-Full-<version>-x64.zip" in fallback_doc
    assert "\n# DigiBox\n" in readme[:500]
    assert "powered by the AVTR-1 realtime rendering engine" in readme


def test_technical_compatibility_identifiers_are_not_rebranded() -> None:
    config = json.loads(_read("src-tauri/tauri.conf.json"))
    app_source = _read("src-tauri/src/app.rs")
    health_source = _read("src-tauri/src/health.rs")

    assert config["identifier"] == "live.avaturn.avtr1.desktop"
    assert "AVTR1_DESKTOP_RUNTIME" in app_source
    assert "AVTR1_DESKTOP_INSTANCE_ID" in app_source
    assert 'Some("avtr1-streamer")' in health_source


def test_english_and_chinese_readmes_link_to_each_other() -> None:
    english = _read("README.md")
    chinese_path = ROOT / "README_zh-CN.md"

    assert chinese_path.is_file()
    chinese = chinese_path.read_text(encoding="utf-8")
    assert "[简体中文](README_zh-CN.md)" in english
    assert "[English](README.md)" in chinese


def test_chinese_readme_preserves_setup_and_license_boundaries() -> None:
    chinese = _read("README_zh-CN.md")

    assert "git clone --recurse-submodules https://github.com/DocWorkBox/DigiBox.git" in chinese
    assert "git submodule update --init --recursive" in chinese
    assert "Windows 10 或 Windows 11" in chinese
    assert "NVIDIA RTX 视频超分辨率" in chinese
    assert "本 Git 仓库不包含" in chinese
    assert "portable-v2" in chinese
    assert r"python\python.exe" in chinese
    assert "packages/shared" in chinese
    assert "LegacyV1" in chinese
    assert "DigiBox-Full-win64.zip" in chinese
    assert "公开源码、多许可证" in chinese
    assert "并非整体符合 OSI 开源定义" in chinese
    assert "PolyForm Noncommercial" in chinese
    assert "LICENSE-DIGIBOX.md" in chinese
    assert "Attachment A" in chinese
    assert "不代表整套 DigiBox 可商业使用" in chinese


def test_readmes_list_all_user_downloaded_models_without_coming_soon_items() -> None:
    english = _read("README.md")
    chinese = _read("README_zh-CN.md")

    assert "Technical report (Coming soon)" not in english
    assert "Production-ready back-end (Coming soon)" not in english
    assert "技术报告\uff08即将发布\uff09" not in chinese
    assert "完整生产级后端\uff08即将发布\uff09" not in chinese

    model_urls = (
        "https://huggingface.co/avaturn-live/avtr-1",
        "https://huggingface.co/digital-avatar/ditto-talkinghead",
        "https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        "https://huggingface.co/feyninc/FeyNobg",
    )
    for readme in (english, chinese):
        for model_url in model_urls:
            assert model_url in readme

    assert "two public HF repos" not in english
    assert "## Model download links" in english
    assert "## 模型下载地址" in chinese


def test_readmes_link_the_windows_one_click_package() -> None:
    package_url = "https://pan.quark.cn/s/887c8b103c18"
    english = _read("README.md")
    chinese = _read("README_zh-CN.md")

    assert package_url in english
    assert package_url in chinese
    english_windows = english.index("## 2. DigiBox native Windows desktop")
    chinese_windows = chinese.index("## 2. Windows 原生桌面版")
    assert (
        english_windows
        < english.index(package_url)
        < english.index("### Prerequisites", english_windows)
    )
    assert (
        chinese_windows
        < chinese.index(package_url)
        < chinese.index("### 前置条件", chinese_windows)
    )
