# DigiBox Windows 桌面发行说明

本文说明如何把当前原生 Windows 部署制作成桌面发行物，以及接收方第一次运行前还必须完成什么。这里的“便携”表示 Python 和依赖不再绑定构建电脑上的绝对 venv 路径；它不表示 TensorRT 引擎可以跨 GPU 直接复用。

## 两种桌面发行物

### 标准桌面版

标准桌面版只封装 Electron 桌面壳、启动管理和许可证文件，不包含 Python、模型或 AVTR Runtime。它适合已经部署好 Runtime 的开发机，或单独分发桌面壳的场景。

```powershell
npm install
.\scripts\build_desktop_windows.ps1 -Edition Standard
```

默认输出为 `desktop/dist/DigiBox-Setup-<version>-x64.exe`。首次启动时，桌面程序会寻找显式配置的 Runtime、环境变量 `AVTR1_DESKTOP_RUNTIME`，或开发目录中的 Runtime。

### 完整桌面版

完整桌面版把一套 standalone CPython 3.12、`main`/`cosyvoice`/`feynobg`/`shared` 四个依赖层、`src`、运行/构建脚本、必要的 CosyVoice/Matcha-TTS 源码、公开模型文件、AVTR 的非引擎构建输入和全部根许可证放入 Electron 的 `resources/avtr-runtime`。它不会复制 `.venv`、`.venv-cosyvoice` 或 `.venv-feynobg`。完整桌面版默认交付 Zip64 离线归档，而不是 NSIS 单文件安装器。

```powershell
npm install
.\scripts\build_desktop_windows.ps1 -Edition Full
```

默认 Runtime 暂存目录为 `desktop/staging/avtr-runtime`，完整离线归档输出到 `desktop/dist-full/DigiBox-Full-<version>-x64.zip`。Runtime 很大，当前模型和 CUDA Python 依赖可能令归档达到数十 GiB；构建机和接收方应使用 NTFS 并预留足够空间。[NSIS 官方说明](https://nsis.sourceforge.io/I_get_an_error_when_compiling_large_installers)其大型安装器存在约 2 GiB 的理论限制，因此 Full 版不能可靠地生成单一 NSIS EXE；标准桌面版仍使用 NSIS。调试或需要最稳妥的大文件交付时可生成未封装目录：

```powershell
.\scripts\build_desktop_windows.ps1 -Edition Full -Target Unpacked
```

Full Zip 解压后直接运行其中的 `DigiBox.exe`。不要只复制 EXE；`resources/avtr-runtime` 必须和应用目录一起保留。若归档工具或传输平台不支持超大 Zip64，使用 `-Target Unpacked` 后再用支持大文件和分卷的归档工具封装整个输出目录。

若已安全构建过 Runtime，可以避免重复下载和安装依赖：

```powershell
.\scripts\build_desktop_windows.ps1 -Edition Full -SkipRuntimeBuild
```

## 构建便携 Runtime

构建机需要 Windows x64、PowerShell 5.1 或更高版本、uv 0.8.0 或更高版本、Node.js/npm 和网络访问。脚本使用 uv 管理的 `python-build-standalone` CPython；这些发行版是自包含、可迁移的 Windows Python，不是 venv。最低 uv 版本用于把临时 Python 的可执行 shim 和 Windows 注册表写入隔离在构建临时目录中。默认布局如下：

```text
avtr-runtime/
├─ python/            唯一的 CPython 3.12.9
├─ packages/
│  ├─ main/          主渲染与 WebRTC 专用依赖
│  ├─ cosyvoice/     CosyVoice 专用依赖
│  ├─ feynobg/       FeyNoBg 专用依赖
│  └─ shared/        RECORD 证明逐字节相同的共享依赖
├─ src/
├─ scripts/
├─ third_party/CosyVoice/
├─ artifacts/main/
├─ models/
└─ runtime-manifest.json
```

当前 portable-v2 生成器固定写入 `paths.artifacts = artifacts/main` 和 `paths.models = models`。Electron/Tauri 壳、Full 载荷检查和 TensorRT 助手都以这两个标准位置为当前发行契约，不要在交付包中手工改写它们。

只查看计划，不下载、不写入也不删除：

```powershell
.\scripts\desktop\build_portable_runtime.ps1 -PlanOnly
```

实际构建：

```powershell
npm install
npm run vendor:frontend
.\scripts\desktop\build_portable_runtime.ps1
```

通过 `build_desktop_windows.ps1` 构建时会在复制 `src` 前自动执行 `npm run vendor:frontend`；直接调用 Runtime builder 时则必须先生成本地 Preact/HTM 文件，否则脚本会拒绝构建。`-SkipDependencies` 和 `-SkipModels` 仅用于 Runtime 布局诊断；带有这些标记的 Runtime 不会被 Full 桌面构建接受。

依赖通过 uv 的 `copy` link mode 安装到三个 profile 层；运行时不引用构建机的 uv cache。去重器只把 distribution 名称与版本相同、`RECORD` 拥有文件且逐字节一致的副本移到 `packages/shared`；命名空间和路径冲突仍留在 profile 层。编排器为主渲染、CosyVoice 和 FeyNoBg 分别设置有序 `PYTHONPATH`，profile 依赖优先于 shared 依赖。

默认 `portable-v2` 为了让 PyTorch 主体真正只存在一份，三个 profile 使用同一套 CUDA 12.8 `torch`/`torchaudio`/`torchvision` wheel。如果迁移期间需要回滚到旧的三解释器布局，可显式构建：

```powershell
.\scripts\desktop\build_portable_runtime.ps1 -Layout LegacyV1
```

`LegacyV1` 保留 `python-main`/`python-cosyvoice`/`python-feynobg` 三套 Python，仅作回滚兼容；新 Full 发行包应继续使用默认 `portable-v2`。

脚本不会默认清空非空目标。只有目标目录名严格等于 `avtr-runtime`、目录内存在绑定到该规范化绝对路径的有效 `.avtr-portable-runtime.json` 标记并且显式传入 `-Clean` 时，才会递归清理该精确目录：

```powershell
.\scripts\desktop\build_portable_runtime.ps1 -Clean
```

不要手工把标记文件复制到无关目录；`-Clean` 是不可恢复操作。`-PlanOnly -Clean` 只验证清理资格，不执行删除。该构建标记只留在暂存目录，不会进入 Full 发行归档，避免暴露构建机的绝对路径。

## 为什么接收方仍需构建 TensorRT

TensorRT 序列化 plan 会受到 GPU 计算能力、TensorRT/CUDA 版本、插件 ABI 和部分驱动环境影响。把 RTX 5070 Ti 上生成的 `.engine` 发给不同显卡，可能无法反序列化，也可能得到错误性能或结果。因此发行构建明确排除：

- 所有 `*.engine` 和 `*.plan`；
- 所有 `grid_sample_3d_plugin*.dll`（包括旧备份和调试变体）；
- engine staging、备份和构建缓存。

完整桌面版包含 `scripts/desktop/DigiBox-TensorRT-Setup.cmd` 和 `scripts/desktop/build_tensorrt.ps1`。解压后，在目标电脑的 `resources/avtr-runtime/scripts/desktop` 中运行 CMD 助手，或直接执行 PowerShell 脚本。

### TensorRT 标准模式

标准模式构建 speech-to-motion、HuBERT、decoder、MODNet 和 stitch 引擎，不编译自定义 Warp 插件。便携 Runtime 已包含 PyTorch CUDA 12.8 wheel 和 `tensorrt-cu12==10.11.0.33` Python 包；目标电脑还需要：

- 受支持的 NVIDIA GPU；
- 能运行 CUDA 12.x/12.8 Runtime 的 NVIDIA 驱动；
- 足够的显存和磁盘空间；
- 完整离线归档中携带的 AVTR 构建输入。

调用方式：

```powershell
.\scripts\desktop\build_tensorrt.ps1 -RuntimeRoot <avtr-runtime路径> -Mode Standard
```

该模式不要求本地 Visual Studio 或 CUDA Toolkit，但仍必须在目标 GPU 上完成构建和反序列化检查。

### TensorRT 完整模式

完整模式在标准模式基础上编译 GridSample3D 插件并生成 Warp TensorRT 引擎。除标准模式条件外还需要：

- Visual Studio 2022 C++ Build Tools（Desktop development with C++）；
- CMake 和 Ninja；
- Git；
- 与 PyTorch/TensorRT 组合兼容的 CUDA Toolkit；
- NVIDIA TensorRT 10.11 Windows ZIP SDK 的 C++ headers 和 import libraries，并正确设置脚本可发现的 SDK 路径；
- 更多构建时间、磁盘和显存余量。

```powershell
.\scripts\desktop\build_tensorrt.ps1 -RuntimeRoot <avtr-runtime路径> -Mode Full
```

助手先在 staging 目录生成完整集合，通过 smoke/deserialization 检查后才安装；旧引擎会备份到 `artifacts/main/.engine-backups`。硬件不稳定、显存不足或出现 WHEA/PCIe 错误时，应停止完整模式，而不是把半成品引擎写入发行包。

标准模式只使用归档中携带的构建输入，可以离线生成；完整 Warp 模式当前会在新的 BuildRoot 中通过 Git 克隆插件源码，因此目标电脑还需要访问 GitHub。若发布环境必须完全离线，需要另行审核并随发行物提供固定 revision 的插件源码，再调整助手从本地源码构建。

## 隐私与可迁移性边界

Runtime 构建和完整 electron-builder 配置采用两层排除规则，不包含：

- `artifacts/main/user_assets` 中的上传人物、背景和待机素材；
- 已从新主题界面移除的 `avatars_artifacts/backgrounds` 旧默认背景；
- CosyVoice 的 `spk2info.pt`，因为它可能保存本地克隆音色的说话人特征；
- 本地参考音频、`local_voices`、`voice_clones`、`reference_audio` 和任何 `.trash` 回收目录；
- `.env`、私钥、证书、Hugging Face 下载 cache 和 `*.incomplete`；
- 本机 API Key。桌面设置保存在用户自己的 Electron userData 中，不从构建机打包。

因此接收方需要在自己的电脑上重新创建本地音色、上传人物，并填写自己的 API Key。公开 CosyVoice 模型会复制，但修改过的 `spk2info.pt` 不会复制；模型加载时会使用空的说话人表，直到接收方创建获授权的音色。

当前 Full 归档安装 `nobg` 运行代码，但不会把构建机 Hugging Face cache 中的 `feyninc/FeyNobg` 权重重新打包。第一次使用 FeyNoBg 抠图仍需要联网下载该固定 revision，并接受其适用条款。要制作完全离线的 FeyNoBg 发行物，需要先确认权重再分发许可，再让 worker 显式加载发行包内的规范模型目录；不能直接复制构建机 cache。

构建脚本不会代替发布者判断模型文件是否允许再分发，也不会接受 Hugging Face 条款。发布前必须确认公开模型下载完整、来源明确且拥有再分发权。

## 许可证和商业发行边界

发行安装包必须保留 `LICENSE.md`、`LICENSE-MODEL.md`、`LICENSE-RENDERER.md`、`LICENSE-STREAMER.md`、`PATENTS.md`、`THIRD-PARTY-NOTICES.md`，以及 standalone CPython、CosyVoice、Matcha-TTS 和各 Python wheel 自带的许可证/metadata。

- AVTR-1 模型受 `LICENSE-MODEL.md` 约束。年收入达到 USD 10,000,000 的实体进行商业使用时需要另行取得模型商业许可；再分发还必须遵守该协议的通知、完整协议、使用限制和归属要求。
- Renderer 受 `LICENSE-RENDERER.md` 的 PolyForm Noncommercial License 1.0.0 约束。任何商业使用，不论收入规模，都需要 Goodsize Inc. 的独立书面 Renderer 商业许可。
- Streamer 受 `LICENSE-STREAMER.md` 的 PolyForm Noncommercial License 1.0.0 和专利通知约束。任何商业使用都需要独立书面 Streamer 商业许可。
- `THIRD-PARTY-NOTICES.md` 指明 InsightFace 预训练模型仅限非商业研究；商业发行必须获得相应许可或替换这些模型。
- CosyVoice 源码随其 Apache-2.0 `third_party/CosyVoice/LICENSE` 分发，但 CosyVoice/Qwen 模型权重和其他第三方 wheel 可能具有各自条款，发布者仍需逐项审核。

本构建流程只保证文件编排和隐私排除，不构成法律意见，也不会自动授予商标、模型、专利或商业使用权。若要向客户、公司内部商业项目或付费产品分发，应先完成所有适用的商业许可和第三方许可证审查。

## 发布前核对

1. `build_portable_runtime.ps1 -PlanOnly` 的源和目标路径正确。
2. 模型下载已经完成，`models` 中没有只存在于 cache 的半成品。
3. 完整 Runtime 内不存在 `user_assets`、`spk2info.pt`、`.engine` 或 Warp DLL。
4. 在干净 Windows 用户账户中安装并启动桌面程序。
5. 在目标 NVIDIA GPU 上运行 TensorRT 标准模式；需要完整 Warp 加速时再满足完整模式工具链。
6. 检查 `engine-manifest.json` 和反序列化探针结果。
7. 保留全部许可证和 Required Notice，并完成商业/第三方许可审核。
