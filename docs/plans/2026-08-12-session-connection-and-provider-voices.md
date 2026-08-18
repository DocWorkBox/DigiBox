# 会话连接自动恢复与千问克隆音色列表实现计划

> **For Codex:** REQUIRED SUB-SKILL: Use test-driven-development for every production change and verification-before-completion before declaring this plan complete.

**Goal:** 会话结束后自动重建可用的后端预连接；千问连接成功后自动查询历史克隆音色并以下拉列表呈现；创建或切换音色后自动刷新并重连，无需用户再次进入设置手动操作。

**Architecture:** 保留后端一次性 `engine connection` 的所有权模型（`/offer` 消费连接、会话结束关闭引擎），在前端增加明确的“自动重建意图”状态机，区分正常结束/远端结束、显式断开与配置变更。千问克隆音色清单通过新的仅回环 POST 接口查询，密钥只放请求体且错误脱敏；列表查询失败不影响已建立的核心会话连接。音色按 `target_model` 过滤，因为供应商的克隆音色不能跨模型使用。

**Tech Stack:** FastAPI、Pydantic、httpx、Preact/HTM、pytest。

---

### Task 1: 锁定一次性连接和自动恢复契约

**Files:**
- Modify: `tests/test_local_stream_ui.py`
- Modify: `src/avaturn_live_streamer/local_stream_ui.html`

1. 先添加失败测试：无论“停止后释放模型”开关是否开启，正常/远端结束都登记自动重建；显式“断开”必须取消该意图。
2. 添加失败测试：自动重建必须等待模型释放完成，并避免并发或重复连接。
3. 最小实现统一的重建调度状态，删除 `!autoReleaseModels` 限制。
4. 运行 `tests/test_local_stream_ui.py`，确认 RED 后转 GREEN。

### Task 2: 千问历史克隆音色查询客户端与安全接口

**Files:**
- Modify: `src/avaturn_live_streamer/conversation_engines/aliyun_bailian_client.py`
- Modify: `src/avaturn_live_streamer/local_stream_cli.py`
- Modify: `tests/conversation_engines/test_aliyun_bailian_client.py`
- Modify: `tests/localrtc/test_provider_voice_clones.py`

1. 根据千问官方 Query Voice List 契约添加失败测试，覆盖请求动作、鉴权、workspace、响应解析、模型过滤和错误脱敏。
2. 添加仅允许 loopback 的 `POST /provider-voice-clones/query` 失败测试，确保密钥不出现在响应或日志安全错误中。
3. 最小实现 provider 客户端与 API；列表失败只返回明确安全错误，不影响 `/engine-connections`。
4. 运行上述两份测试，确认 RED 后转 GREEN。

### Task 3: 云端音色下拉列表、创建后入列与选择后自动重连

**Files:**
- Modify: `src/avaturn_live_streamer/local_stream_ui.html`
- Modify: `tests/test_local_stream_ui.py`

1. 先添加失败测试：千问音色 ID 使用 `<select>`，连接成功后查询列表，包含默认/当前音色与匹配当前 TTS 模型的克隆音色。
2. 先添加失败测试：新克隆成功后立即加入列表、自动选中；更换云端或本地音色时，若已有预连接则自动替换连接。
3. 实现独立于核心连接成功与否的音色清单状态与刷新逻辑；保留清单查询错误提示。
4. 实现通用音色变更处理器，通过状态提交后的 effect 自动重连，避免旧配置竞态。
5. 运行 `tests/test_local_stream_ui.py`，确认 RED 后转 GREEN。

### Task 4: 组合回归、运行态更新与验收

**Files:**
- Verify only: all modified production/test files

1. 运行聚焦测试：
   `\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider --basetemp .pytest_session_voice_final tests/conversation_engines/test_aliyun_bailian_client.py tests/localrtc/test_provider_voice_clones.py tests/test_local_stream_ui.py`
2. 运行 Ruff、`compileall` 与 `git diff --check`。
3. 运行全量 pytest，使用新的唯一 `--basetemp`。
4. 安全重启当前 Windows 服务进程树，确认 7860/8000/8767/8768 健康。
5. 用真实页面验证：结束会话后状态自动回到“已连接”；千问连接后历史音色进入下拉框；创建/切换音色后无需手动重连。

不创建提交、不推送；当前工作树包含用户与既有任务的共享改动，所有修改限定在上述文件并保留其他变化。
