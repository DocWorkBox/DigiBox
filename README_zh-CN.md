<div align="center">

# DigiBox

[English](README.md) | [简体中文](README_zh-CN.md)

[DigiBox 产品架构](https://docworkbox.github.io/DigiBox/)

</div>

DigiBox 是一款 Windows 原生实时数字人应用，将多供应商实时对话、本地长期记忆、
云端与本地混合声音、说话/倾听双路动作生成，以及 LivePortrait 渲染整合为一套桌面体验。

DigiBox 是独立产品，与 Goodsize Inc. 不存在隶属或背书关系。项目仅在部分动作与
渲染链路中使用单独授权的 AVTR-1 上游组件，并保留相应许可证及 Required Notice。
如果旧文件中的 SPDX 标识与 [LICENSE.md](LICENSE.md) 的目录级组件映射发生冲突，
应以 `LICENSE.md` 指定的组件许可证及其控制性许可证文件为准。

---

## 仓库与发行包边界

本 Git 仓库包含项目源码、构建/下载工具，以及固定版本的 CosyVoice 子模块；
本 Git 仓库不包含 AVTR-1 权重、CosyVoice/FeyNoBg 模型、TensorRT engine、
Python Runtime、安装包、API Key、用户上传素材或克隆音色。这些内容需要按下文单独下载、在目标机器构建，或由 Full 发行包提供。

`third_party/CosyVoice` 是固定提交的 Git submodule，不是随仓库提交的模型副本。
克隆时必须保留 `--recurse-submodules`。如果已经使用普通方式克隆，可补充执行：

```bash
git submodule update --init --recursive
```

该命令也会初始化 CosyVoice 内部的 Matcha-TTS 子模块。

---

## 已包含内容

- [x] 模型下载与环境配置工具（权重需单独获取）
- [x] 推理代码
- [x] 交互式实时演示

---

## 模型下载地址

Git 源码仓库不包含模型权重。源码安装时只需下载已启用功能对应的模型：

- **核心必需——AVTR-1 动作、HuBERT、MODNet、人物和背景资源：**
  [avaturn-live/avtr-1](https://huggingface.co/avaturn-live/avtr-1)。
  该仓库受限，下载前必须登录并接受使用条款。
- **核心必需——LivePortrait、InsightFace 和渲染器 ONNX 图：**
  [digital-avatar/ditto-talkinghead](https://huggingface.co/digital-avatar/ditto-talkinghead)。
- **可选——本地流式 TTS 与音色克隆：**
  [FunAudioLLM/Fun-CosyVoice3-0.5B-2512](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512)。
- **可选——本地人像抠图：**
  [feyninc/FeyNobg](https://huggingface.co/feyninc/FeyNobg)。

项目的下载和安装脚本会自动取得这些模型；以上地址用于接受授权、查看文件或手动下载。
TensorRT 引擎与目标机器相关，不能从 Hugging Face 通用下载，必须在本机生成。

---

## 目录

- [模型下载地址](#模型下载地址)
1. [快速开始（Linux）](#1-快速开始linux)
2. [Windows 原生桌面版（无需 WSL）](#2-windows-原生桌面版无需-wsl)
3. [性能](#3-性能)
4. [故障排查](#故障排查)
5. [许可证](#许可证)

---

## 1. 快速开始（Linux）

### 前置条件

- Linux
- NVIDIA GPU（建议 Ampere 或更新架构）
- CUDA 12.x 与 TensorRT 10.x
- [pixi](https://prefix.dev/)：`curl -fsSL https://pixi.sh/install.sh | sh`

### 安装

```bash
git clone --recurse-submodules https://github.com/DocWorkBox/DigiBox.git
cd DigiBox
pixi install
```

### 设置存储位置（可选）

```bash
export AVTR1_LOCAL_STORAGE=/path/to/avtr1_storage
```

下载的 AVTR-1 artifacts 和本机构建的推理引擎会写入该基目录下当前 revision 的
子目录（目前为 `main/`）。未设置时，默认使用仓库根目录下的 `artifacts/main/`，
而不是调用命令时的当前工作目录。

### 下载权重

```bash
pixi run download
```

首次运行会通过 `hf auth login` 要求登录 Hugging Face。AVTR-1 仓库是受限仓库，
请先在 [avaturn-live/avtr-1](https://huggingface.co/avaturn-live/avtr-1) 页面接受使用条款。

### 构建 TensorRT 引擎

上一步会下载 AVTR-1 权重和重新封装为 ONNX 图的 LivePortrait 权重。TensorRT
引擎与 GPU 计算能力相关，因此每台机器都需要本地构建一次；输出位于
`$AVTR1_LOCAL_STORAGE`。

```bash
# 一次构建全部引擎
pixi run build-trt-engines

# 或分别构建
pixi run build-trt-engines-avtr1
pixi run build-trt-engines-renderer
pixi run build-trt-engines-hubert
```

### 启动交互式演示

```bash
pixi run interactive-demo
```

### 离线生成

单人说话：

```bash
pixi run generate_offline --speech example/speaker_1.ogg
pixi run generate_offline --speech example/speaker_1.ogg --avatar maria --bg minimal_office
```

双人对话需要交换 `--speech` 和 `--listen` 分别渲染两侧：

```bash
pixi run generate_offline --speech example/speaker_1.ogg --listen example/speaker_2.ogg --avatar elena --out elena.mp4
pixi run generate_offline --speech example/speaker_2.ogg --listen example/speaker_1.ogg --avatar marcus --out marcus.mp4

ffmpeg -i elena.mp4 -i marcus.mp4 -filter_complex \
  "[0:v][1:v]hstack=inputs=2[v];[0:a][1:a]amix=inputs=2[a]" \
  -map "[v]" -map "[a]" dialogue.mp4
```

无音频待机动作：

```bash
pixi run generate_offline --duration 10
```

下载完成后，可用人物名称来自
`$AVTR1_LOCAL_STORAGE/main/avatars_artifacts/reference_frames/` 中不带 `.png` 后缀的文件名。

---

## 2. Windows 原生桌面版（无需 WSL）

Windows 原生版本直接在 PowerShell 中运行，不依赖 WSL、Docker、pixi 或 Linux
`.so` 插件。它使用 PyTorch CUDA 运行 AVTR-1 TorchScript 模型，并使用 ONNX Runtime CUDA 运行 HuBERT 与渲染组件。

Tauri v2 桌面版的构建和发行细节请参阅
[DigiBox Windows Tauri v2 桌面版构建与分发](docs/windows-tauri-desktop-distribution.md)。

### Windows 本地一键安装包

不想从源码安装时，可以下载
[DigiBox Windows 本地一键安装包](https://pan.quark.cn/s/887c8b103c18)。
下载后请完整解压并保留整个目录，不要只复制可执行文件；具体内容和版本以分享页面为准。

### 前置条件

- Windows 10 或 Windows 11（64 位）
- NVIDIA GPU 和较新的 Windows 显卡驱动
- [Git](https://git-scm.com/download/win)
- 已完整解压的 DigiBox **Full** 包，其中必须包含 schema version 2 的
  `portable-v2` Runtime；请保留整个 `avtr-runtime` 目录

克隆源码后，让开发检查脚本复用这套现成 Runtime：

```powershell
git clone --recurse-submodules https://github.com/DocWorkBox/DigiBox.git
cd DigiBox
$runtimeRoot = "D:\DigiBox-Full-win64\avtr-runtime"
powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1 `
  -RuntimeRoot $runtimeRoot
```

### Full 发行包与 portable-v2 单 Python

正式 Full 发行包默认采用 `portable-v2`：整个 Runtime 只有一个
`python\python.exe`（CPython 3.12），依赖分别放在 `packages/main`、
`packages/cosyvoice`、`packages/feynobg` 和 `packages/shared`，并通过每个进程独立的有序 `PYTHONPATH` 隔离。PyTorch、TorchAudio 和 TorchVision 只在共享层保留一份。

源码启动、测试以及 main、CosyVoice、FeyNoBg 三个 profile 也全部复用这个
`python\python.exe`。开发脚本不会创建 `.venv`、`.venv-cosyvoice` 或
`.venv-feynobg`，也不会重新安装 PyTorch 或其他 Python 包。`LegacyV1` 的三个解释器布局仅用于旧发行包回滚，不适用于新的源码开发入口。

`-RuntimeRoot` 可以指向 `avtr-runtime` 本身，也可以指向它的 Full 包父目录。若不想每次传参，可设置：

```powershell
$env:AVTR1_DEV_RUNTIME_ROOT = "D:\DigiBox-Full-win64"
.\scripts\setup_windows.ps1
```

两者都未设置时，脚本会依次检查 `desktop\dist-tauri`、
`desktop\builds` 中最新的匹配构建以及 staging Runtime；LegacyV1 或 manifest
不完整的 Runtime 会被拒绝。

仓库源码始终优先于发行包内的源码副本：main 与 FeyNoBg profile 先加载仓库
`src`；CosyVoice profile 依次先加载仓库 `src`、`third_party\CosyVoice` 和
`third_party\CosyVoice\third_party\Matcha-TTS`，随后才加载 Full Runtime 对应的依赖层。因此修改仓库源码后，无需重打发行包即可直接启动和测试。

三个 setup 脚本只检查所选依赖 profile，并可按参数下载模型数据，不会创建环境或安装 Python 包。只做离线依赖检查时，可给 CosyVoice/FeyNoBg setup 脚本传入 `-SkipModelDownload`。

构建 Full 发行包前，必须先完整下载模型。源码克隆本身不会取得模型：

```powershell
npm install
npm run vendor:frontend
.\scripts\build_tauri_windows.ps1 -Edition Full
```

默认输出为：

```text
desktop\dist-tauri\DigiBox-Full-win64\
desktop\dist-tauri\DigiBox-Full-win64.zip
```

请完整分发或解压整个目录，不要只复制 `digibox-desktop.exe`。详细构建、WebView2、
TensorRT 助手和发行限制见
[DigiBox Windows Tauri v2 桌面版构建与分发](docs/windows-tauri-desktop-distribution.md)。

### 可选的 TensorRT 与 RTX 超分

TensorRT 10.11 是可选组件：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1 `
  -RuntimeRoot $runtimeRoot -EnableTensorRT
```

该参数检查所选 Full Runtime 中已有的 TensorRT，不会在线安装另一套 Python 环境。

NVIDIA RTX 视频超分辨率也是独立的可选功能。在 Full Runtime 中准备好 NVIDIA
官方 VFX Python Runtime 后，可验证导入：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1 `
  -RuntimeRoot $runtimeRoot -EnableNvidiaVfx
```

启用 RTX 超分后，输出映射为 360p → 720p、540p → 1080p、720p → 1080p。
如果官方 `nvvfx` Runtime 不可用，界面会明确显示不可用，不会使用普通缩放冒充 RTX 超分。

### 模型存储与 Hugging Face 登录

Windows 源码启动器复用所选 Full Runtime 的 `models` 与 `artifacts\main`，包括
已下载权重和本机构建引擎。请把解压后的 Full Runtime 放在用户可写目录。用户上传
素材和克隆音色缓存位于 DigiBox 用户级应用数据目录，不写进仓库或发行包。

接受 AVTR-1 模型条款后，执行：

```powershell
.\scripts\login_huggingface_windows.ps1 -RuntimeRoot $runtimeRoot
.\scripts\setup_windows.ps1 -RuntimeRoot $runtimeRoot `
  -CheckHuggingFaceAccess
```

登录命令会交互式安全读取令牌，不会把令牌写进命令历史或进程参数。

也可以在登录后一次完成检查和下载：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1 `
  -RuntimeRoot $runtimeRoot -CheckHuggingFaceAccess -DownloadModels
```

### Windows 离线生成

```powershell
# 单人说话
powershell -ExecutionPolicy Bypass -File .\scripts\run_offline_windows.ps1 `
  -RuntimeRoot $runtimeRoot --speech example\speaker_1.ogg --avatar maria --bg plain_white `
  --out output\maria.mp4

# 待机动作冒烟测试
powershell -ExecutionPolicy Bypass -File .\scripts\run_offline_windows.ps1 `
  -RuntimeRoot $runtimeRoot --duration 1 --avatar maria --bg plain_white --no-mux `
  --out output\windows-smoke.mp4
```

### Windows 交互式演示

本地抠图和本地 CosyVoice 流式语音/音色克隆使用同一个 Full Python 下的隔离依赖 profile，可按需检查并下载模型数据：

```powershell
# FeyNoBg 人像抠图
powershell -ExecutionPolicy Bypass -File .\scripts\setup_feynobg_windows.ps1 `
  -RuntimeRoot $runtimeRoot

# CosyVoice 本地流式 TTS 与音色克隆
powershell -ExecutionPolicy Bypass -File .\scripts\setup_cosyvoice_windows.ps1 `
  -RuntimeRoot $runtimeRoot
```

启动演示：

```powershell
# Auto：存在本机 TensorRT 引擎时优先使用，否则回退到便携后端
powershell -ExecutionPolicy Bypass -File .\scripts\run_interactive_windows.ps1 `
  -RuntimeRoot $runtimeRoot

# 已构建完整 TensorRT 引擎时强制使用 TensorRT
powershell -ExecutionPolicy Bypass -File .\scripts\run_interactive_windows.ps1 `
  -RuntimeRoot $runtimeRoot -Backend tensorrt
```

交互式启动器通常会统一管理整套服务。只需单独排查 Worker 时，可以直接启动；
两者仍使用同一个 Full Python：

```powershell
.\scripts\run_feynobg_windows.ps1 -RuntimeRoot $runtimeRoot
.\scripts\run_cosyvoice_windows.ps1 -RuntimeRoot $runtimeRoot
```

等待服务就绪后打开 [http://localhost:7860](http://localhost:7860)。完整安装存在时，
启动器还会运行 `127.0.0.1:8767` 上的 FeyNoBg 和 `127.0.0.1:8768` 上的
CosyVoice。按 `Ctrl+C` 可以停止 PowerShell 中的服务。

### Windows 源码测试

测试入口使用相同的 main profile，先检查 main、CosyVoice 与 FeyNoBg 都能从仓库
源码导入，再把其余参数传给 pytest：

```powershell
.\scripts\test_windows.ps1 -RuntimeRoot $runtimeRoot `
  -PytestArgs @("-q", "tests")

# Python Tauri 契约，以及离线 Rust fmt/check/clippy/test
.\scripts\test_tauri_windows.ps1 -RuntimeRoot $runtimeRoot
```

Tauri 测试入口还会把 `AVTR1_TEST_PYTHON` 指向 Full Runtime 的 Python，Rust
supervisor 测试不会再回退到仓库虚拟环境。`npm run test:tauri` 调用的是同一入口。

setup、启动器、Worker 和测试脚本均支持 `-PlanOnly`，可以只查看最终解析到的
Python、Runtime、环境变量和参数，而不真正启动进程。

原生源码客户端可直接运行 `npm run tauri:dev`；Electron 回退壳使用
`npm run desktop:dev`。两者都会通过同一解析器启动仓库源码后端并负责关闭它。
若 7860、8000、8767 或 8768 已被占用，开发桥会拒绝启动，避免新版壳再次静默
接入旧后端。

千问 AI 平台与 MiniMax 是相互独立的供应商，需要分别配置 API Key。一个千问
API Key 可运行完整的 Qwen3 ASR、Qwen LLM 与 CosyVoice 语音链路。API Key 保存在当前站点地址对应的浏览器本地存储中；清空输入框即可移除已保存值。

选择本地 CosyVoice 后，设置面板会从 `/v1/audio/voices` 加载可刷新的音色列表。
可以使用清晰的 3–10 秒 WAV、FLAC、OGG 或 MP3 参考音频和精确文本创建或覆盖音色。系统会在写入说话人缓存前拒绝不支持的容器、静音和截断文件。

如果希望使用本机 Codex 登录而不是 OpenAI API Key，请保持 Codex Desktop 已登录，
在页面中选择 **Codex GPT-Live（实验功能）** 后开始会话。浏览器不会为此模式请求或保存 OpenAI API Key。该集成依赖尚未稳定公开的 Codex realtime app-server 协议，因此仍属于实验功能。

### 首次转换与本机 TensorRT

便携后端首次启动时会准备 warp、HuBERT、MODNet 和 stitch 的动态 ONNX Runtime
图，并将结果缓存到模型旁边。HuBERT 转换体积最大，首次运行请预留数分钟和数 GB 额外空间。

Windows 自动后端会优先使用本机 TensorRT 引擎；如果缺少 AVTR-1 引擎，则回退到便携后端。发布包中的 Linux 引擎和 `.so` 插件不能在 Windows 上使用，因此必须针对目标机器的 GPU 本地构建。

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_tensorrt_windows.ps1 `
  -RuntimeRoot $runtimeRoot
powershell -ExecutionPolicy Bypass -File .\scripts\run_offline_windows.ps1 `
  -RuntimeRoot $runtimeRoot -Backend tensorrt `
  --duration 1 --avatar maria --bg plain_white --no-mux `
  --out output\windows-trt-smoke.mp4
```

默认混合模式将 volumetric warp 保留在 ONNX Runtime CUDA，其余阶段使用 TensorRT。
若要构建包括 warp 插件在内的完整 TensorRT 管线：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_tensorrt_windows.ps1 `
  -RuntimeRoot $runtimeRoot -IncludeWarp
```

Windows 原生版本当前要求 NVIDIA CUDA 和 ONNX Runtime 的
`CUDAExecutionProvider`，不支持纯 CPU 渲染。首次模型下载需要数 GB 空间，交互式使用还需要浏览器麦克风权限和至少一个已配置的对话供应商。

---

## 3. 性能

AVTR-1 每次生成 5 帧。25 fps 下每个分块对应 200 ms 输出，因此单块推理时间低于 200 ms 的 GPU 可以满足实时生成。

| GPU | 每 5 帧延迟 | 实时倍数 |
| --- | ---: | ---: |
| L40 | 84 ms | 2.4× |
| A100 | 91 ms | 2.2× |
| RTX 4060 Ti | 166 ms | 1.2× |
| RTX 3070 | 181 ms | 1.1× |
| L4 | 202 ms | 0.99× |
| RTX 3060 Ti | 206 ms | 0.97× |
| RTX 4060 | 232 ms | 0.86× |

实时倍数 = 200 ms ÷ 延迟；大于或等于 1.0× 表示 GPU 能够跟上 25 fps。

---

## 故障排查

### TURN 中继（可选）

WebRTC 会优先尝试本地 UDP 和 STUN。只有浏览器与流媒体服务之间无法直接穿透
UDP 时才需要 TURN，例如云主机安全组未开放入站 UDP，或任一端位于对称 NAT 后。

项目支持 Cloudflare Realtime TURN，也支持任意标准 TURN 服务。Cloudflare 配置：

```dotenv
CLOUDFLARE_TURN_KEY_ID="<Turn Key ID>"
CLOUDFLARE_TURN_KEY_TOKEN="<API Token>"
```

也可以直接设置 `TURN_URL`，以及可选的 `TURN_USERNAME` 和 `TURN_CREDENTIAL`。
长期 API Token 只保留在服务端；浏览器连接卡片会显示最终使用的是 host、STUN
server-reflexive 还是 TURN relay 路径。

---

## 许可证

本仓库代码属于**公开源码、多许可证**项目，并非整体符合 OSI 开源定义。
特别是上游 Renderer 与 Streamer 仅允许非商业用途。DigiBox 新增的桌面壳、打包代码和测试采用 Apache-2.0，但该许可证不会覆盖或替代任何上游组件许可证。

- **`desktop/`、`src-tauri/`、`tests/`、`docs/`**：DigiBox 新增内容，采用 Apache-2.0，详见 [LICENSE-DIGIBOX.md](LICENSE-DIGIBOX.md)。
- **`scripts/`**：采用 [AVTR-1 Community License](LICENSE-MODEL.md)。年营收低于 1000 万美元的实体仍须遵守该许可证及 Attachment A 的用途限制；达到或超过该阈值需要另行获得商业协议。此范围仅适用于对应组件，不代表整套 DigiBox 可商业使用。AVTR-1 模型权重同样受该许可证约束。
- **`src/avtr1_renderer/`**：采用 [PolyForm Noncommercial License 1.0.0](LICENSE-RENDERER.md)，仅限非商业用途，商业使用需要单独的 Renderer Commercial License。
- **`src/avaturn_live_streamer/`**：采用 [PolyForm Noncommercial License 1.0.0](LICENSE-STREAMER.md)，并受 [PATENTS.md](PATENTS.md) 的专利保留条款约束；仅限非商业用途。

管线使用的 InsightFace 预训练 SCRFD 检测器与 2D106 关键点模型仅授权用于非商业研究。
商业使用必须从 InsightFace 获得商业许可，或替换为许可证兼容的实现。其他第三方依赖、FeyNoBg 模型及 NVIDIA VFX Runtime 的许可说明见
[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md)。

完整组件映射见 [LICENSE.md](LICENSE.md)。如果本摘要与具体许可证文件发生冲突，以具体许可证文件为准。

**商业授权咨询：** hello@avaturn.me
