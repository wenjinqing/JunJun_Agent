# JunJun_Agent · 君君

君君 QQ 聊天机器人的现代 Agent 架构版。由原 MaiBot（Astron bot）架构升级：LangChain 1.x `create_agent` + Function Calling + MCP + Langfuse 可观测，monorepo 分层。

## 架构

```
NapCat (ws://8095)
   │ OneBot 11
   ▼
junjun_adapter_napcat ──── maim_message WS ────▶ Gateway (ws://8092)
                                                    │ 黑白名单/会话/契约
                                                    ▼
                                          三级决策漏斗（省 token）
                                          L1 规则门: 自消息/@ 旁路/talk_value 概率   (0 token)
                                          L2 语义门: 小模型 reply|no_reply|until_call (小 token)
                                          L3 主 Agent: create_agent + 15+ skill      (主模型)
                                                    │
                                          回复后处理: 分条/错别字/引用/打字延迟
                                                    ▼
                                                 QQ 群/私聊
```

| 包 | 职责 |
|---|---|
| `junjun_core` | 网关、配置、数据契约、DB（peewee+WAL）、可观测 |
| `junjun_adapter_napcat` | OneBot 11 ↔ maim_message 协议转换 |
| `junjun_agent` | 决策漏斗、persona、回复后处理、定时任务（提醒/主动/统计/清理） |
| `junjun_llm` | 任务槽模型（gate/agent/utils，provider fallback）+ Langfuse v3 |
| `junjun_memory` | 三层记忆：窗口 → 话题摘要 → faiss 长期库；用户画像 |
| `junjun_skills` | skill 注册表 + 内置 12 个 + 插件（vrchat/TTS）+ MCP 工具注入 |
| `junjun_express` | 情绪、表情包（偷图/注册/发送）、黑话、表达学习 |
| `junjun_mcp_client` / `junjun_mcp_server` | MCP 双向：调外部 server + relationship 服务端 |
| `junjun_webui` | FastAPI：配置热改/日志实时流/统计/数据管理 |

依赖单向向下，层间只走数据契约（`ReplySet`/`InboundMeta`），core 不 import 上层（processor 注入模式）。

## 快速开始

```bash
# 1. 依赖（Python 3.11+，uv）
uv venv && uv pip install -e .

# 2. 配置
copy .env.example .env
#    必填: MAIBOT_QQ_ACCOUNT（bot 的 QQ 号）、DEEPSEEK_API_KEY
#    建议: SILICONFLOW_API_KEY（embedding 向量记忆，缺省降级关键词检索）
#    可选: LANGFUSE_*（可观测）、DOUBAO_TTS_API_KEY（语音）、WEBUI_TOKEN

# 3. NapCat：确认 onebot11 配置里 websocketClients 指向 ws://127.0.0.1:8095
#    （messagePostFormat=array, reportSelfMessage=false）
#    ⚠️ 确认旧 MaiBot adapter 没在运行（netstat -ano | findstr 8095）

# 4. 启动（两个窗口）
.venv\Scripts\python.exe scripts\run_junjun.py                 # 网关+Agent
.venv\Scripts\python.exe -m junjun_adapter_napcat.main         # Adapter

# 5.（可选）WebUI: .env 设 WEBUI_ENABLED=true → http://127.0.0.1:8002
```

## 配置说明

- `config/bot_config.toml` — 人设/漏斗频控/回复后处理/情绪/表情包/主动/提醒（字段对齐原 MaiBot，注释即文档）
- `config/model_config.toml` — 任务槽模型：每槽 models 按序 fallback
- `config/mcp_servers.toml` — MCP server 声明（stdio，`${REPO_ROOT}` 插值）

## 测试

```bash
.venv\Scripts\python.exe -m pytest tests\ -q          # 167 单测（无网络依赖）
.venv\Scripts\python.exe scripts\test_e2e_fake_napcat.py   # 全链路 E2E（fake NapCat）
.venv\Scripts\python.exe scripts\smoke_agent.py       # 真实 LLM 冒烟（需 API key）
.venv\Scripts\python.exe scripts\smoke_memory.py      # 记忆链路冒烟
```

## 故障排查

| 现象 | 排查 |
|---|---|
| QQ 无响应 | ① netstat 查 8095 是否被旧 adapter 占用 ② adapter 日志是否连上 8092 ③ 名单过滤（启动日志打印生效名单） |
| 双回复 | 旧 MaiBot adapter 还在跑（可能开机自启），杀掉 |
| 全部沉默 | talk_value 配置/时段规则；日志 DEBUG 看每级漏斗拦截原因 |
| MCP 工具缺失 | server 子进程必须 `-u`；日志看连接失败原因（10s 超时降级） |
| Langfuse 无 trace | `LANGFUSE_ENABLED=true` + 自托管服务可达；SDK 降级不影响主流程 |
| Docker/Langfuse 起不来 | 见 `docs/Langfuse排障.md`（Win11 家庭版需启用虚拟机平台组件） |

## 文档

- 升级计划与各阶段验收：`docs/00_总目标.md` ~ `docs/阶段7_WebUI与收尾.md`
- 踩坑清单：`docs/踩坑清单与常见Bug.md`
- 全功能回归记录：`docs/回归记录.md`
