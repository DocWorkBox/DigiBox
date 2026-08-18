# Third-Party Notices

> Modified by DigiBox contributors to document downstream bundled and optional
> dependencies.

This repository bundles third-party model weights and binaries under their
upstream licenses. The AVTR-1 weights and avatar assets are licensed
separately — see [LICENSE.md](LICENSE.md).

Bundled components (HuBERT, MODNet, grid_sample plugin) are Apache-2.0. The
renderer pipeline depends on LivePortrait (MIT) and InsightFace
(MIT code, **non-commercial research only** for pretrained models). The
AVTR-1 motion-generator architecture derives from MIT-licensed research code.

---

## HuBERT (`build_artifacts/hubert-lbs-avtr1.onnx`)

Fine-tuned derivative of **HuBERT base ls960**.

- **Upstream:** https://huggingface.co/facebook/hubert-base-ls960
- **Original authors:** Wei-Ning Hsu, Benjamin Bolte, Yao-Hung Hubert Tsai,
  Kushal Lakhotia, Ruslan Salakhutdinov, Abdelrahman Mohamed (Meta AI)
- **License:** Apache-2.0
- **Modifications:** Fine-tuned by Goodsize Inc. for lip-sync feature
  extraction; exported to ONNX with dynamic-batch input profile suitable for
  TensorRT.

```
Copyright (c) Meta Platforms, Inc. and affiliates.
Licensed under the Apache License, Version 2.0.
```

---

## MODNet (`build_artifacts/modnet.onnx`)

Portrait matting model from the official MODNet release.

- **Upstream:** https://github.com/ZHKKKe/MODNet
- **Original authors:** Zhanghan Ke, Jiayu Sun, Kaican Li, Qiong Yan,
  Rynson W.H. Lau
- **License:** Apache-2.0
- **Modifications:** Exported to ONNX with batch-aware Reshape surgery on the
  SE-block path so the engine accepts dynamic batch sizes.

```
Copyright (c) 2020 ZHKKKe (Zhanghan Ke et al.)
Licensed under the Apache License, Version 2.0.
```

---

## grid_sample 3D TensorRT Plugin (`renderer_runtime_artifacts/libgrid_sample_3d_plugin.so`)

Compiled TensorRT plugin implementing 3D grid_sample for the warp network.

- **Upstream:** https://github.com/SeanWangJS/grid-sample3d-trt-plugin
- **Original author:** Sean Wang (SeanWangJS)
- **License:** Apache-2.0
- **Modifications:** Compiled binary for x86_64 Linux against CUDA 12 / TRT 10
  ABI. Source code unchanged.

```
Copyright (c) Sean Wang
Licensed under the Apache License, Version 2.0.
```

---

## Apache-2.0 License Text

The Apache-2.0 license text covering the components above is available at:
https://www.apache.org/licenses/LICENSE-2.0

In compliance with Section 4 of the Apache License:

- Each derivative work (where applicable) preserves the upstream copyright and
  license notice.
- Modifications made by Goodsize Inc. are documented in each entry above.
- No trademark or endorsement rights are claimed from any upstream project.

---

## Vendored web UI libraries

- **Preact 10.22.0** — https://github.com/preactjs/preact — MIT License.
  Copyright (c) 2015-present Jason Miller. The complete license is included at
  `src/avaturn_live_streamer/vendor/LICENSE-PREACT.txt`.
- **HTM 3.1.1** — https://github.com/developit/htm — Apache License 2.0.
  The complete license is included at
  `src/avaturn_live_streamer/vendor/LICENSE-HTM.txt`.

## Optional externally obtained components

- **FeyNoBg model weights** (`feyninc/FeyNobg`) are downloaded separately and
  are licensed under CC BY-NC 4.0. They are not stored in this repository.
- **NVIDIA Video Effects / nvidia-vfx** is an optional proprietary NVIDIA
  runtime obtained separately from NVIDIA. It is not licensed as open source
  by this repository.

---

## Pipeline Dependencies

These are not bundled here but are pulled at build time and required for
AVTR-1 to function.

### LivePortrait

The AVTR-1 renderer pipeline is built on LivePortrait. Goodsize does not
redistribute LivePortrait checkpoints; build scripts pull repackaged ONNX
graphs from
[digital-avatar/ditto-talkinghead](https://huggingface.co/digital-avatar/ditto-talkinghead)
(Apache-2.0), since the upstream LivePortrait release does not ship portable
ONNX. The 203-point landmark model (`landmark203.onnx`) is part of
LivePortrait.

- **Upstream:** https://github.com/KwaiVGI/LivePortrait
- **License:** MIT, © 2024 Kuaishou Visual Generation and Interaction Center

### InsightFace

The pipeline uses two InsightFace pretrained models, pulled at build time:

- `insightface_det.onnx` — SCRFD face detector
- `landmark106.onnx` — 2D106 facial landmark detector

InsightFace's **code** is MIT (no commercial restriction). InsightFace's
**pretrained models** are licensed for non-commercial research purposes
only. Commercial users must obtain a commercial license from InsightFace
(deepinsight@gmail.com) or replace these models with permissively-licensed
alternatives (e.g., MediaPipe BlazeFace + Face Landmarker).

- **Upstream:** https://github.com/deepinsight/insightface

---

## Architectural Lineage

The AVTR-1 motion generator (`build_artifacts/avtr1.scripted.pt`) is a
TorchScript-compiled model. The training and inference source code is not
redistributed in this repository. We nevertheless acknowledge the upstream
codebase that served as the initial intellectual starting point for the
project.

### S2G-MDDiffusion

Early prototyping of the AVTR-1 motion generator was scaffolded from
**S2G-MDDiffusion**, which we acknowledge as the project's starting point.

- **Upstream:** https://github.com/thuhcsi/S2G-MDDiffusion
- **Original author:** Xu He
- **License:** MIT
- **Relationship to AVTR-1:** Over the course of development the
  architecture was reimplemented substantially and now diverges from
  S2G-MDDiffusion in generative paradigm, conditioning, inference pattern,
  and target task. Code-level overlap with S2G-MDDiffusion in the released
  model is minimal; we attribute the upstream nonetheless.

```
MIT License

Copyright (c) 2024 Xu He

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Upstream chain

S2G-MDDiffusion itself acknowledges the following codebases, which form the
broader provenance chain for the diffusion-based motion-generation approach:

- **EDGE: Editable Dance Generation From Music** — https://github.com/Stanford-TML/EDGE (MIT)
- **lucidrains' diffusion implementations** (`denoising-diffusion-pytorch`,
  `imagen-pytorch`) — https://github.com/lucidrains (MIT). These provided
  the foundational PyTorch diffusion-model code that EDGE and downstream
  projects adapted.
- **Thin-Plate Spline Motion Model** — https://github.com/yoyo-nb/Thin-Plate-Spline-Motion-Model

The precise per-file lineage through these intermediate projects is partially
obscured by their independent evolution. We attribute S2G-MDDiffusion as the
direct upstream and acknowledge the chain above for completeness.

---

*Last updated: May 20, 2026.*
