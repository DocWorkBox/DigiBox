# DigiBox Windows Tauri v2 桌面版构建与分发

本文说明当前 `windows-native` 分支中的 Tauri v2 迁移方案。Tauri 壳负责选择并启动现有 Windows Runtime，后端和数字人页面仍由 Python、CUDA/TensorRT 与 `http://127.0.0.1:7860/` 提供。

> 当前状态：Tauri 是迁移候选，Electron 仍作为临时回退。只有完成本文末尾的真实 Windows 验收后，才能把 Tauri 设为唯一默认桌面端或删除 Electron 文件。本文不会把尚未执行的构建、签名、麦克风/WebRTC 或安装测试描述为已通过。

## 1. 发行版选择

| 发行版 | 包含内容 | 默认交付 | 适用场景 |
| --- | --- | --- | --- |
| Standard | Tauri 壳、离线 WebView2 安装载荷、项目许可证；不含 Python、模型和 Runtime | 当前用户级 NSIS 安装器 | 目标电脑已经有开发目录或单独分发的 `avtr-runtime` |
| Full | Tauri 壳、一套 standalone CPython 3.12、四个隔离依赖层、项目源码、公开模型/构建输入、TensorRT 助手和许可证 | Zip64 归档；也可输出未封装目录 | 解压后直接选择同目录 Runtime，不重新创建 venv |

Full 不生成单文件 NSIS/MSI。完整 Runtime 可能达到数十 GiB，脚本会拒绝 `-Edition Full -Target Installer` 和 `-Target Msi`，只接受 `Archive` 或 `Unpacked`。Full 也不会携带本机生成的 TensorRT engine、API Key、上传人物、参考音频或克隆音色。

## 2. 构建要求

构建机至少需要：

- Windows 10/11 x64；
- Node.js/npm；
- Rust stable MSVC 工具链；
- Visual Studio 2022 Build Tools 的 C++ 桌面工作负载与 Windows SDK；
- 构建 Full 时还需要 `uv`、网络连接、足够的 NTFS 空间，以及已经下载完整的项目模型；
- 构建安装器时需要联网取得 Rust/npm 依赖；WebView2 采用离线安装载荷，但这不代表构建过程本身完全离线。

先安装锁定的前端依赖：

```powershell
Set-Location C:\path\to\DigiBox
npm install
npm run vendor:frontend
```

只查看计划而不构建、不暂存 Runtime：

```powershell
.\scripts\build_tauri_windows.ps1 -Edition Standard -PlanOnly
.\scripts\build_tauri_windows.ps1 -Edition Full -PlanOnly
```

`-PlanOnly` 输出 JSON，可用于核对发行版、目标、Runtime 来源、隐私排除项和许可证清单。

## 3. Standard：壳安装器

构建当前用户级 NSIS 安装器：

```powershell
Set-Location C:\path\to\DigiBox
.\scripts\build_tauri_windows.ps1 -Edition Standard
```

Tauri 的 NSIS 产物位于：

```text
src-tauri\target\release\bundle\nsis\
```

需要先测试未封装 EXE 时：

```powershell
.\scripts\build_tauri_windows.ps1 -Edition Standard -Target Unpacked
& .\src-tauri\target\release\digibox-desktop.exe
```

Standard 不包含 Runtime。启动时应从下列候选中选择第一个结构完整的 Runtime，优先级由高到低为：

1. 命令行 `--runtime-root`；
2. 当前进程的 `AVTR1_DESKTOP_RUNTIME`；
3. 设置窗口中选中过并持久化的目录；
4. Tauri resource 目录内的 `avtr-runtime`；
5. EXE 同级的 `avtr-runtime`；
6. 开发目录。

开发时最简单的启动方式是显式设置环境变量：

```powershell
Set-Location C:\path\to\DigiBox
$env:AVTR1_DEV_RUNTIME_ROOT = "D:\DigiBox-Full-win64\avtr-runtime"
npm run tauri:dev
```

`tauri:dev` 和 `desktop:dev` 通过 `scripts\desktop_dev_windows.ps1` 先启动一个
由当前命令拥有的源码后端，再让桌面壳以 External 模式接入。后端仍使用 Full 的
单 Python 和依赖层，但入口脚本与三个 profile 都优先加载仓库源码。开发桥会先
确认 7860、8000、8767、8768 全部空闲，退出桌面命令时再通过唯一 stop file 停止
自己启动的进程树，避免误接或误停其他版本的服务。

发布 EXE 也可在同一个 PowerShell 会话中使用同一环境变量。设置窗口选择成功后，路径保存为 Tauri 应用配置目录中的 `desktop-config.json`，后续启动不需要重复选择。该文件只保存 Runtime 路径，不应写入供应商 API Key。

有效 Runtime 当前至少需要：

```text
<runtime>\python\python.exe
<runtime>\packages\main\
<runtime>\packages\cosyvoice\
<runtime>\packages\feynobg\
<runtime>\packages\shared\
<runtime>\scripts\run_local_stream.py
<runtime>\src\
<runtime>\artifacts\main\
<runtime>\models\
<runtime>\runtime-manifest.json
```

正式 Full Runtime 默认使用 `portable-v2`：只有 `python\python.exe` 一个 CPython 3.12，主渲染、CosyVoice 和 FeyNoBg 通过 manifest 中的有序 `packageLayers` 选择各自的 profile 层与 `packages\shared`，不依赖构建机上的 venv。新的 Windows 源码启动与测试入口同样要求 schema version 2 的 `portable-v2` Full Runtime；LegacyV1 仅保留给旧发行包回滚。

当前 portable-v2 生成器固定写入 `paths.artifacts = artifacts/main` 和 `paths.models = models`，Full 载荷检查、运行环境与 TensorRT 助手也按这两个标准位置工作。发行包不应手工改写这两个 manifest 路径；要支持自定义位置，需先同步扩展壳、编排器、Full 校验和 TensorRT 助手。

### 源码开发复用 Full Runtime

源码仓库不再维护三套开发 venv。先完整解压 Full 包，然后把包目录或其中的
`avtr-runtime` 传给 `-RuntimeRoot`：

```powershell
Set-Location C:\path\to\DigiBox
$runtimeRoot = "D:\DigiBox-Full-win64\avtr-runtime"

.\scripts\setup_windows.ps1 -RuntimeRoot $runtimeRoot
.\scripts\setup_cosyvoice_windows.ps1 -RuntimeRoot $runtimeRoot -SkipModelDownload
.\scripts\setup_feynobg_windows.ps1 -RuntimeRoot $runtimeRoot -SkipModelDownload
.\scripts\run_interactive_windows.ps1 -RuntimeRoot $runtimeRoot
```

这些 setup 脚本检查 Full Runtime 的 main、CosyVoice、FeyNoBg 依赖 profile；若
CosyVoice 关键文件缺失，其 setup 会修复所选 Full Runtime 的模型目录。它们不会
创建虚拟环境、运行 `uv venv` 或重新安装 Python 包。源码启动会直接复用 Full
Runtime 的 `models` 与 `artifacts\main`，因此解压目录必须可写。需要检查解析结果时可加 `-PlanOnly`。

不想重复传入 `-RuntimeRoot` 时，可设置源码开发专用环境变量：

```powershell
$env:AVTR1_DEV_RUNTIME_ROOT = "D:\DigiBox-Full-win64"
.\scripts\run_interactive_windows.ps1
```

解析优先级为显式 `-RuntimeRoot`、`AVTR1_DEV_RUNTIME_ROOT`、
`desktop\dist-tauri\DigiBox-Full-win64\avtr-runtime`、`desktop\builds` 中最新的
Full dist，最后是对应 staging Runtime。解析器会校验 manifest、路径边界、单一
Python 和完整组件标记，不会把 LegacyV1 当作源码开发 Runtime。

三个 profile 都调用同一个 `python\python.exe`，但使用不同的有序
`PYTHONPATH`。main 与 FeyNoBg 先加载仓库 `src`；CosyVoice 先加载仓库 `src`、
`third_party\CosyVoice` 和 `third_party\CosyVoice\third_party\Matcha-TTS`；随后才加入 Full Runtime 的 profile 与 shared 依赖层。发行包中的源码、CosyVoice 和 Matcha 副本会从源码开发路径中排除，保证仓库改动优先生效。

交互式启动器通常负责统一启动 main、CosyVoice 和 FeyNoBg。需要单独排查可选
Worker 时，也可以直接运行：

```powershell
.\scripts\run_cosyvoice_windows.ps1 -RuntimeRoot $runtimeRoot
.\scripts\run_feynobg_windows.ps1 -RuntimeRoot $runtimeRoot
```

测试必须通过同一入口运行；它会先对三个 profile 做源码优先的 import smoke，再把参数传给 pytest：

```powershell
.\scripts\test_windows.ps1 -RuntimeRoot $runtimeRoot `
  -PytestArgs @("-q", "tests")
```

## 4. Full：可携带 Runtime

默认构建 Runtime、未封装 Tauri EXE，并生成 Zip64：

```powershell
Set-Location C:\path\to\DigiBox
.\scripts\build_tauri_windows.ps1 -Edition Full
```

`portable-v2` 是新构建的默认布局。如果迁移期间需要回滚到旧的三解释器布局，可以单独构建 LegacyV1 Runtime，再让 Full 构建复用它：

```powershell
.\scripts\desktop\build_portable_runtime.ps1 -Layout LegacyV1
.\scripts\build_tauri_windows.ps1 -Edition Full -SkipRuntimeBuild
```

`LegacyV1` 仅用于回滚兼容，不是新发行包的推荐布局。

默认结果：

```text
desktop\dist-tauri\DigiBox-Full-win64\
desktop\dist-tauri\DigiBox-Full-win64.zip
```

只输出未封装目录：

```powershell
.\scripts\build_tauri_windows.ps1 -Edition Full -Target Unpacked
```

复用已经完成并通过脚本检查的暂存 Runtime：

```powershell
.\scripts\build_tauri_windows.ps1 -Edition Full `
  -RuntimeDestination .\desktop\staging\avtr-runtime `
  -SkipRuntimeBuild
```

不要对不完整或来源不明的 Runtime 使用 `-SkipRuntimeBuild`。构建脚本默认核对 `runtime-manifest.json` 的 `portable-v2` 单 Python/四层布局，同时保留对 `LegacyV1` 回滚包的兼容；两种布局都必须通过依赖/模型/前端/TensorRT 构建输入标记、关键文件、Python import、Runtime 检查器以及 TensorRT 助手 SHA-256 校验。

最终目录包含 `digibox-desktop.exe`、同级 `avtr-runtime` 和 `licenses`。分发时必须保留整个目录；只复制 EXE 无法运行。启动方式：

```powershell
& .\desktop\dist-tauri\DigiBox-Full-win64\digibox-desktop.exe
```

## 5. WebView2 离线策略

`src-tauri/tauri.conf.json` 把 Windows WebView2 安装模式设为 `offlineInstaller`。因此 Standard NSIS 会携带 WebView2 离线安装载荷，目标电脑无需在安装时在线下载 WebView2；代价是安装器体积更大。

Full 当前使用 `tauri build --no-bundle`，然后由 PowerShell 组装 Zip64/目录。这个流程不会把 NSIS 中的 WebView2 离线安装步骤一并复制进 Full 目录。因此 Full 目标电脑目前必须已经安装受支持的 WebView2 Runtime；若要做到真正离线的裸机部署，发布者还需在交付物中单独提供并执行微软 WebView2 Evergreen Standalone Installer，并完成相应许可证审查。开发模式同样依赖本机已有 WebView2。

## 6. 启动、健康检查和进程所有权

Tauri 先显示本地 splash，再检查精确地址 `http://127.0.0.1:7860/health`。只有服务身份为 `avtr1-streamer` 且状态为 `ready` 或 `degraded` 时，才加载数字人页面。

- 如果 7860 已有符合身份的服务，Tauri 只连接，不取得所有权，退出桌面端时也不得关闭该外部服务。
- 如果没有服务，Tauri 只启动一个 `python\python.exe scripts\run_local_stream.py` 编排进程；CosyVoice 与 FeyNoBg 使用同一 CPython，由编排器按 manifest 传入各自有序依赖层。LegacyV1 回滚 Runtime 才使用三个独立 Python 路径。
- 后端标准输出和错误输出分别写入 `avtr-backend.stdout.log` 与 `avtr-backend.stderr.log`。
- 退出时先写 `AVTR1_DESKTOP_STOP_FILE`，最多等待 20 秒；随后才使用 Windows Job Object 和精确 PID 的 `taskkill /T /F` 兜底。

缺失 Runtime 或启动失败时，应停留在 splash，可重试、重新选择 Runtime 或打开日志目录。数字人页面位于 loopback origin，不应获得 shell/filesystem Tauri capability；只有内置 splash 可以调用受限的桌面命令。

## 7. 配置、日志和用户数据

三类状态必须分开处理：

1. **桌面壳配置与日志**：Runtime 路径放在 Tauri 的用户级应用配置目录；后端日志放在 Tauri 的用户级应用日志目录。实际 Windows 路径由 Tauri/系统解析，排障时优先点击 splash 的“打开日志”，不要依赖手写固定路径。
2. **网页设置**：主题、性能面板开关、服务商设置等仍由 `http://127.0.0.1:7860` 的 WebView2 profile/localStorage 持久化。Tauri 更新必须保留相同 identifier 与 WebView2 数据目录，否则这些设置会看起来“丢失”。API Key 不进入发行包或 Runtime manifest。
3. **Runtime 可变数据**：由 Tauri 自己启动后端时，上传人物、背景和本地音色缓存分别通过 `AVTR1_USER_ASSETS_ROOT` 与 `AVTR1_COSYVOICE_SPEAKER_CACHE` 写入 Tauri 的用户级应用数据目录。首次启用会从旧 Runtime 无覆盖迁移一次，后续删除不会被旧副本重新导入。TensorRT engines 仍由目标机上的构建助手写入所选 Runtime 的明确 engine 路径；因此 Full 应解压到用户可写位置，更新 Runtime 时必须保留或重新构建 engines。若 Tauri 只接入一个已在外部启动的 AVTR-1 服务，则该外部服务沿用它自己的存储环境，桌面壳不会擅自重定向。

Full 构建会主动排除：

```text
user_assets、local_voices、voice_clones、reference_audio
spk2info.pt、.env*、*.key、*.pem
*.engine、*.plan、grid_sample_3d_plugin*.dll
.cache、__pycache__、.engine-staging、.engine-backups
```

这既避免把构建机的私有数据/API Key 发给别人，也意味着接收方需要导入自己的人物、音色和凭据，并在自己的 GPU 上构建 TensorRT engine。

## 8. TensorRT 构建助手

TensorRT engine 与 GPU 计算能力、TensorRT/CUDA 版本、插件 ABI 和驱动环境绑定，不能把构建机上的 `.engine` 当作通用 Runtime 分发。

运行助手前先退出 DigiBox，确认 7860、8000、8767、8768 没有 AVTR 服务监听。Full 目录中可双击：

```text
avtr-runtime\scripts\desktop\DigiBox-TensorRT-Setup.cmd
```

或显式运行：

```powershell
.\avtr-runtime\scripts\desktop\build_tensorrt.ps1 `
  -RuntimeRoot .\avtr-runtime `
  -Mode Standard
```

`Standard` 是推荐模式：构建 speech-to-motion、HuBERT、decoder、MODNet 和 stitch，不编译自定义 Warp 插件。`Full` 额外构建 Warp，需要 Visual Studio 2022 C++ Build Tools、CMake、Ninja、Git、兼容的 CUDA Toolkit 和 TensorRT Windows SDK；当前插件构建还可能需要访问 GitHub，因此不能宣称为完全离线。

助手先在 `.engine-staging` 生成并反序列化验证，再安装到 `*_runtime_artifacts_cc_win64`。旧文件会备份到 `.engine-backups`，失败时尝试回滚，成功后写入 `artifacts\main\engine-manifest.json`。完成后重新启动 DigiBox。

## 9. 签名、Updater 与许可证

当前 Tauri 配置没有生产代码签名，也没有启用 `tauri-plugin-updater`。因此：

- 本地构建不能视为已签名发布物；正式发布前应使用可信 Windows 代码签名证书对应用和安装器签名并加时间戳，在干净 Windows 账户上验证 SmartScreen/UAC 行为。
- 不要发布不存在的 updater 清单或声称支持自动更新。以后启用 Tauri Updater 时，还需要独立的更新签名密钥、公钥配置、签名产物和回滚测试；Windows Authenticode 与 Tauri 更新签名是两套不同机制。
- 建议把“壳更新”和“大型 Runtime/模型更新”分开版本化。Standard 只更新壳；Full 的更新器不得直接覆盖用户的上传素材、音色、设置、API Key 或本机 TensorRT engines。

Standard bundle 配置和 Full 组装脚本都要求保留：

```text
LICENSE.md
LICENSE-MODEL.md
LICENSE-RENDERER.md
LICENSE-STREAMER.md
PATENTS.md
THIRD-PARTY-NOTICES.md
```

这不替代第三方依赖、模型权重、CPython、CosyVoice/Matcha-TTS 和 NVIDIA/Microsoft 组件的再分发许可审查。Renderer、Streamer 和 InsightFace 相关限制以仓库中的原始许可证及 notice 为准；代码签名也不会授予商业使用权。

## 10. Electron 临时回退

在 Tauri 完成真实验收前，保留现有 Electron 入口：

```powershell
npm run desktop:dev
.\scripts\build_desktop_windows.ps1 -Edition Standard
.\scripts\build_desktop_windows.ps1 -Edition Full
```

Electron 与 Tauri 共享 portable Runtime builder，但配置目录和 WebView profile 不是同一个实现。回退测试时不要假设两个壳的 localStorage、麦克风授权或 Runtime 路径已经自动迁移。

## 11. Windows Job Object 启动顺序

Rust supervisor 现在以 `CREATE_NO_WINDOW | CREATE_SUSPENDED` 创建 Python 编排器，在它运行任何 Python 代码前先绑定 kill-on-close Job Object，再恢复主线程。Windows 集成测试会让编排器在启动后立即派生一个长寿命子进程，并验证关闭 Job 后父子都被回收。因此，早期实现中“Python 可能在 Job 绑定前派生逃逸子进程”的窗口已被消除；退出仍保留 stop-file、Job terminate 与精确 PID `taskkill /T /F` 的分层兜底。

## 12. 发布前必须完成的真实验收

以下是发布门槛，不是本文任务已经完成的结果：

```powershell
$runtimeRoot = "D:\DigiBox-Full-win64\avtr-runtime"
.\scripts\test_tauri_windows.ps1 -RuntimeRoot $runtimeRoot
npm run test:desktop
```

`test_tauri_windows.ps1` 先用 Full Runtime 的 main profile 运行两组 Python
Tauri contract，再依次执行 `cargo fmt`、`cargo check --offline`、
`cargo clippy --offline -D warnings` 和 `cargo test --offline`。Rust supervisor
测试也通过 `AVTR1_TEST_PYTHON` 使用同一个 Full Python。可加 `-PlanOnly` 只核对
Runtime、环境变量和全部测试步骤；`npm run test:tauri` 是同一入口的快捷命令。

还必须在真实安装/未封装 EXE 上人工验证：

- splash 到数字人 UI、缺失 Runtime、错误重试和单实例；
- 麦克风授权、设备枚举、WebRTC、音频播放、主题、人物显示和缩放；
- 主题、性能面板、API Key 等设置重启后仍在；
- 关闭 Tauri 后，其自启的 7860/8000/8767/8768 被释放；预先外部启动的服务保持运行；
- Standard 在无网络目标机上安装 WebView2，Full 在已安装 WebView2 的目标机上启动；
- 生产签名、许可证、更新/回滚和干净 Windows 用户账户。

只有这些检查有新鲜、可复现的结果后，才能宣布迁移完成并考虑移除 Electron。
