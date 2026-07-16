# 阶段 6 · MCP 与插件

> 周期：2-3 天
> 目标：君君作为 MCP 客户端调用外部工具；迁移 vrchat_agent 与 TTS 为 skill 插件。
> 修订（2026-07-16）：原「阶段4」顺延编号；MCP 转换固定用 `langchain-mcp-adapters`（与 LangChain 1.x 配套）；relationship 工具与阶段 4 画像共用同一存储；补 TTS 插件（原计划遗漏）。

---

## 一、技术要点

### 1.1 MCP 客户端 `junjun_mcp_client`
- 用 `langchain-mcp-adapters`（或 `mcp` SDK + 手动转换）连接多 server。
- `config/mcp_servers.toml`：声明各 MCP server 的 `command`/`args`/`cwd`/`env`（对齐原 `mcp_config.json` 结构）。
- 启动时：
  1. 逐个连接 server。
  2. 拉取工具清单（`list_tools`）。
  3. 转换为 LangChain `BaseTool` 注入 `junjun_skills/registry`。
  4. 失败的 server 降级跳过，不阻塞启动。
- 工具命名空间：`mcp_<server>_<tool>`，避免与内置 skill 冲突。

### 1.2 MCP 服务端保留
- 迁移 `src/mcp_server/relationship_mcp_server.py` 到 `junjun_mcp_server/`（新独立包）。
- 工具面（对齐原实现）：`apply_relationship_penalty` / `update_user_impression` / `add_user_tag` / `set_user_name`。
- **存储对齐**：读写阶段 4 的 PersonInfo/memory_points 同一套表，不另建存储——MCP 只是访问通道。
- 作为**服务端**保留，由君君自身作为客户端调用，验证 MCP 闭环。
- `mcp_servers.toml` 中登记该 server，`cwd` 指向新仓库根。

### 1.3 vrchat_agent 插件迁移
- 原 `plugins/vrchat_agent/`：`anya_client.py`/`pose_library.py`/`tools.py`/`plugin.py`。
- 迁移为 `junjun_skills/plugins/vrchat_agent/`：
  - `tools.py` 里的工具函数改造成 `JunJunSkill` 子类（继承 LangChain `BaseTool`）。
  - `anya_client.py`/`pose_library.py` 作为 skill 的依赖模块。
  - `config.toml` 迁移为 skill 局部配置，由 registry 注入。
  - `_manifest.json` 对齐新插件规范（name/version/skills 列表）。
- 可用性按会话过滤：仅特定群/私聊启用 VRChat skill（复用 registry 的 `get_tools(session)` 过滤接口）。

### 1.4 TTS 插件迁移（原计划遗漏）
- 原 `plugins/built_in/tts_plugin`（豆包 TTS，`DOUBAO_TTS_API_KEY` 已在 .env）→ `junjun_skills/plugins/tts/`。
- skill `send_voice`：文本 → TTS API → voice 段发送；失败降级文本回复。

### 1.5 插件规范固化
- `junjun_skills/plugins/` 静态加载：启动扫描目录，`_manifest.json`（name/version/skills/依赖探测）；依赖缺失禁用该插件并 WARN，不崩启动。
- 热加载不做（留 WebUI 阶段按需评估）。

---

## 二、避坑与潜在 Bug

| 风险 | 说明 | 对策 |
|------|------|------|
| MCP server 启动失败阻塞 | server 子进程崩溃导致客户端 await 卡死 | 连接超时 10s + 失败降级跳过；监控子进程存活 |
| 工具 schema 不兼容 | MCP 工具的 JSON schema 与 LangChain 期望不一致 | 用 `langchain-mcp-adapters` 的转换器；手动校验 `args_schema` |
| 工具命名冲突 | MCP 工具名与内置 skill 重名 | 加 `mcp_<server>_` 前缀；registry 启动时检测重名报错 |
| relationship server 路径硬编码 | 原 `mcp_config.json` 写死 `E:\MaiM\MaiM-with-u\MaiBot` | 新配置用相对路径或 `${REPO_ROOT}` 变量 |
| MCP server 端口/stdio 冲突 | 多个 server 同时起冲突 | 用 stdio 传输（默认），避免端口；或分配不同端口 |
| vrchat skill 误触发 | 群聊误调 VRChat 动作 | skill `available_for` 按会话白名单过滤；LLM 不可见的 skill 不进 tools |
| AnyaDance 依赖缺失 | `anya_client` 依赖 VRChat SDK / 网络连接 | skill 启动时探测依赖，缺失则禁用并 WARN |
| 插件热加载冲突 | 运行时重新加载插件导致状态丢失 | 阶段 4 先做静态加载（启动扫描）；热加载留后续 |
| MCP 工具超时 | 远程工具响应慢拖垮 Agent | 单工具超时 30s，超时返回错误信息给 Agent |
| 工具结果过大 | MCP 返回大文本撑爆上下文 | 结果截断（如 >2000 字符摘要）后再喂给 Agent |

---

## 三、验收标准

- [ ] 启动时 MCP client 连接 relationship server 成功，工具出现在 registry。
- [ ] Agent 能调 relationship MCP 工具读取/写入关系数据（与画像同库）。
- [ ] MCP server 断开时，Agent 降级（该工具不可用，其他正常）。
- [ ] vrchat_agent skill 在白名单会话可用，可触发 VRChat 动作。
- [ ] 非 VRChat 会话不出现 VRChat 工具。
- [ ] TTS skill 可发语音，API 失败时降级文本。

---

## 四、交付物

- `junjun_mcp_client/{client.py,config.py}`
- `junjun_mcp_server/relationship_mcp_server.py`
- `junjun_skills/plugins/{vrchat_agent/,tts/}` + 插件规范（`_manifest.json` 加载器）
- `config/mcp_servers.toml`
