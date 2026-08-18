from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WINDOWS_WORKFLOW_DOCS = (
    ROOT / "README.md",
    ROOT / "README_zh-CN.md",
    ROOT / "docs" / "windows-tauri-desktop-distribution.md",
)


def test_windows_source_docs_use_the_full_single_python_runtime() -> None:
    for path in WINDOWS_WORKFLOW_DOCS:
        text = path.read_text(encoding="utf-8")
        assert "portable-v2" in text, path
        assert "-RuntimeRoot" in text, path
        assert "AVTR1_DEV_RUNTIME_ROOT" in text, path
        assert "scripts\\test_windows.ps1" in text, path
        assert ".\\.venv\\Scripts" not in text, path


def test_readmes_document_source_first_worker_entrypoints() -> None:
    for filename in ("README.md", "README_zh-CN.md"):
        text = (ROOT / filename).read_text(encoding="utf-8")
        assert "scripts\\run_interactive_windows.ps1" in text, filename
        assert "scripts\\run_cosyvoice_windows.ps1" in text, filename
        assert "scripts\\run_feynobg_windows.ps1" in text, filename
        assert "third_party\\CosyVoice" in text, filename
        assert "Matcha-TTS" in text, filename


def test_windows_readmes_keep_distribution_and_model_links() -> None:
    quark_url = "https://pan.quark.cn/s/887c8b103c18"
    model_urls = (
        "https://huggingface.co/avaturn-live/avtr-1",
        "https://huggingface.co/digital-avatar/ditto-talkinghead",
        "https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        "https://huggingface.co/feyninc/FeyNobg",
    )
    for filename in ("README.md", "README_zh-CN.md"):
        text = (ROOT / filename).read_text(encoding="utf-8")
        assert quark_url in text, filename
        for url in model_urls:
            assert url in text, (filename, url)


def test_source_docs_reuse_full_artifacts_and_full_python_login() -> None:
    english = (ROOT / "README.md").read_text(encoding="utf-8")
    chinese = (ROOT / "README_zh-CN.md").read_text(encoding="utf-8")

    assert "keep checkout artifacts" not in english
    assert "artifacts 使用仓库下" not in chinese
    for text in (english, chinese):
        assert "scripts\\login_huggingface_windows.ps1" in text
        assert "scripts\\build_tensorrt_windows.ps1" in text
