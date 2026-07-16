# 阶段 2 · Agent 最小可用

> 周期：2-3 天
> 目标：LangChain 1.x `create_agent` 替换手写决策，三级决策漏斗成型，接真实 LLM，Langfuse v3 trace 可见，pytest 骨架落地。
> 前置：阶段 1 echo 链路已通（NapCat→Adapter→Gateway→QQ）。
> 修订（2026-07-16）：技术选型由 `AgentExecutor` 改为 LangChain 1.x `create_agent`；Langfuse 明确 v3 SDK；新增决策漏斗与测试骨架（原计划缺失）。

---

## 一、技术要点

### 1.0 依赖落地（当前 venv 还没装 langchain！）
- `uv add`：`langchain>=1.0`、`langchain-openai`、`langgraph`、`langfuse>=3.0`。
- 版本锁进 `uv.lock`；`langchain.agents.create_agent` 可 import 即为正确版本线。
- **禁用清单**（评审否决的旧 API，出现即打回）：`AgentExecutor`、`create_openai_tools_agent`、`initialize_agent`、langfuse v2 的 `langfuse.callback.CallbackHandler` 导入路径。

### 1.1 LLM 层 `junjun_llm`
- `models.py`：`ChatOpenAI` 按任务构造，读 `config/model_config.toml`（新增，结构对齐原 `model_task_config`）：
  - 任务槽（本阶段先落 3 个，后续阶段补齐到原项目 17 个）：`gate`（小模型，如 deepseek-chat/qwen-turbo）、`agent`（主模型，必须支持 function calling）、`utils`。
  - provider：deepseek / siliconflow / bailian / google，baseURL+key 从 env 注入。
  - 每任务槽支持 model_list 顺序 fallback（对齐原 LLMRequest 故障转移语义，简化实现：try 下一个）。
- `tracing.py`：Langfuse **v3** `CallbackHandler`；`get_callbacks() -> list`，未启用/不可达时返回 `[]`。业务层永远只拿 callbacks 列表，不 import langfuse。

### 1.2 决策漏斗 `junjun_agent/funnel`（新增，省 token 的关键）
原项目语义：`v2_native_planner_gate`（reply/no_reply/no_reply_until_call）+ talk_value 概率 + @ 旁路。落成三级：

- **L1 规则门 `rule_gate.py`**（0 token，纯函数）：
  1. 自消息/命令消息 → 丢弃。
  2. `mentioned_bot_reply=true` 且被 @ 或昵称直呼 → 直通 L3（旁路 L2）。
  3. 私聊 → 直通 L2（原 Brain 语义：私聊基本都回，仍过 gate 防刷）。
  4. 群聊非 @ → `random() < talk_value * boost` 未命中 → no_reply 终止。
- **L2 语义门 `llm_gate.py`**（小模型单次调用）：输入近几条上下文，输出三值 `reply | no_reply | no_reply_until_call`；`until_call` 置会话沉默标记，直到被 @/直呼才解除（对齐原 `_SILENCE_ACTIONS`）。JSON 输出 + 解析失败默认 no_reply。
- **L3 主 Agent**：只有穿过漏斗的消息才进，天然控频控本钱。
- 漏斗每级决策打结构化日志（`gate=L1 result=drop reason=talk_value`），后续接 Langfuse span。

### 1.3 Agent 核心 `junjun_agent/agent.py`
- `build_junjun_agent(session) -> agent`：`from langchain.agents import create_agent`（LangGraph runtime）。
  - `model`：任务槽 `agent` 的 ChatOpenAI。
  - `tools`：来自 `junjun_skills.registry`。
  - `system_prompt`：由 `persona/` 组装（本阶段最简版：personality + reply_style + 当前时间）。
- 递归/迭代控制：`recursion_limit` 映射原 `max_agent_iterations=5`（LangGraph 语义为 `2*iterations+1`）。
- **每会话独立 agent 实例 + 独立消息历史**（阶段 4 换 checkpointer，本阶段 session 内存列表即可），防跨会话串味。
- 输出协议：agent 最终文本 → `ReplySet`；空文本或 `do_not_reply` 工具被调 → `should_reply=False`。

### 1.4 最小 Skill `junjun_skills`
- `registry.py`：`@tool` 装饰器 + pydantic args_schema；`register(skill)` 重名报错；`get_tools(session) -> list[BaseTool]` 按会话过滤（本阶段全量返回，接口先定）。
- `builtin/get_time.py`：当前时间。
- `builtin/do_not_reply.py`：显式沉默。返回哨兵对象记入 session 状态，**网关按状态判断，不靠文本匹配**（防哨兵文本泄漏）。

### 1.5 最小记忆 `junjun_memory/short_term.py`
- 滑动窗口（`max_context_size=80`），按 `ChatSession` 组织：`add / render_for_prompt / trim`。
- 群聊消息渲染带昵称前缀（`nickname: text`），Agent 能分清谁在说话。
- 本阶段只做窗口，长期检索留阶段 4。

### 1.6 可观测打通
- Langfuse v3：`observability/langfuse_client.py` 升级为 v3 `get_client()` 封装，保留 Noop 降级。
- trace 结构：`trace(chat_id)` → span(L1/L2 gate) → generation(agent) → tool spans；trace_id 写入日志。
- token 用量从 `response_metadata`/`usage_metadata` 提取，先打日志（落库留阶段 3 的 LLMUsage 表）。

### 1.7 测试骨架 `tests/`（新增，原计划完全没有）
- `uv add --dev pytest pytest-asyncio`；`tests/conftest.py` 提供 fake config fixture。
- 本阶段必备单测（全部纯逻辑，不打真实 LLM）：
  - `test_rule_gate.py`：@ 旁路 / talk_value 概率（注入固定 random）/ 自消息丢弃 / 私聊直通。
  - `test_llm_gate.py`：mock LLM 返回三值解析 + 脏输出 fallback no_reply。
  - `test_contracts.py`：`ReplySet.to_message_base` 段组装（reply 段、seglist）。
  - `test_short_term.py`：窗口裁剪、昵称渲染。
  - `test_registry.py`：重名注册报错、tool schema 生成合法（`tool.args`）。
- 真实 LLM 冒烟脚本放 `scripts/smoke_agent.py`（手动跑，不进 CI）。

---

## 二、避坑与潜在 Bug

| 风险 | 说明 | 对策 |
|------|------|------|
| 装到 LangChain 0.x/旧 API | 教程/补全惯性给 `AgentExecutor` | 锁 `langchain>=1.0`；代码评审对照 1.0 禁用清单 |
| Langfuse SDK/服务器版本错配 | v2 SDK 打 v3 服务器（E:\MaiM\langfuse 为 v3 镜像）| `langfuse>=3.0`，callback 走 v3 导入路径；启动时 `auth_check()` 失败则降级 Noop |
| Agent 无限循环 | 工具反复调用不收敛 | `recursion_limit` 兜底；超限捕获 `GraphRecursionError` 返回 no_reply |
| `do_not_reply` 哨兵泄漏 | 哨兵文本被当回复发出去 | 状态标记而非文本匹配；网关只认 `should_reply` |
| LLM 不返回 tool_call | 模型直接输出文本 | 主模型必须支持 function calling（deepseek-chat 支持）；文本直出也是合法回复路径 |
| gate 小模型 JSON 脏输出 | 前后缀废话/代码块包裹 | 宽松解析（正则抽 JSON）+ 失败默认 no_reply + WARN |
| Langfuse 阻塞主循环 | 服务不可达时卡住 | v3 SDK 异步批量上报；callbacks 注入包 try/except |
| 多会话状态串 | agent/历史跨会话复用 | 每 `ChatSession` 独立实例与历史；单测覆盖 |
| 上下文超长 | 窗口未裁剪 token 溢出 | 严格 `max_context_size`；超长从最旧丢弃 |
| 群聊消息无主 | 上下文不带说话人 | 渲染带昵称前缀，@bot 的消息显式标注 |
| 漏斗全拦 | talk_value 配错导致永久沉默 | 启动打印生效漏斗参数；每级拦截打 DEBUG 带原因 |

---

## 三、验收标准

- [ ] `uv run pytest` 全绿（≥5 个测试文件）。
- [ ] @君君"几点了" → 走 L1 旁路 → agent 调 `get_time` → 回复时间。
- [ ] 非 @ 随机消息 → 多数被 L1/L2 拦截沉默，日志可见拦截原因。
- [ ] 连续对话上下文连贯（窗口生效，昵称可辨）。
- [ ] Langfuse 后台可见完整 trace：gate span + agent generation + tool span + token 用量。
- [ ] 连续 20 条消息不崩、无无限循环、无哨兵泄漏。
- [ ] 切换 deepseek/siliconflow 两个 provider 均能回复；gate 与 agent 用不同模型。

---

## 四、交付物

- `junjun_llm/{models.py,tracing.py}` + `config/model_config.toml`
- `junjun_agent/{agent.py,context.py,funnel/{rule_gate.py,llm_gate.py},persona/}`
- `junjun_skills/{registry.py,builtin/{get_time.py,do_not_reply.py}}`
- `junjun_memory/short_term.py`
- `junjun_core/observability/langfuse_client.py`（v3 化）
- `tests/`（5+ 文件）+ `scripts/smoke_agent.py`
