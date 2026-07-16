# 踩坑清单与常见 Bug

> 跨阶段汇总，按类别组织。执行时对照检查。

---

## 一、环境与依赖

| 陷阱 | 现象 | 对策 |
|------|------|------|
| **LangChain 旧 API**（评审新增，最高优先） | 用了 `AgentExecutor`/`create_openai_tools_agent`（已弃用） | 锁 `langchain>=1.0`，只用 `create_agent`；评审对照禁用清单 |
| **Langfuse v2/v3 错配**（评审新增） | v2 SDK 打自托管 v3 服务器，callback 导入路径报错或数据丢 | `langfuse>=3.0`；启动 `auth_check()` 失败降级 Noop |
| **quick_algo 私有依赖**（评审新增） | LPMM 图算法库新环境装不上 | 阶段 4 先探测；降级 networkx PageRank，接口隔离 |
| Python 版本 | 原项目 `>=3.10`，但部分依赖需 3.11+ | 锁定 3.11，写进 `.python-version` |
| 依赖版本漂移 | LangChain/Langfuse API 变动 | `uv.lock` 锁版本，CI 校验 |
| venv 混用 | 原 MaiBot 有 `.venv` + `venv` 两套 | 新仓库单一 `.venv`，`.gitignore` 忽略 |
| Windows 控制台编码 | 中文日志乱码 | `sys.stdout.reconfigure(encoding='utf-8')` |
| 工作目录错位 | 相对路径读不到配置 | 入口 `chdir` 到仓库根；路径用 `Path(__file__).resolve()` 推导 |
| 包找不到 | monorepo 子包未安装 | workspace editable install，`uv sync` 统一 |

---

## 二、协议与接入

| 陷阱 | 现象 | 对策 |
|------|------|------|
| WS 重连风暴 | NapCat 重启后疯狂重连 | 指数退避，最大间隔 30s |
| 自消息回环 | Agent 把自己发的消息当输入 | `message_id_echo` + `reportSelfMessage=false` |
| 群消息段解析错 | @ 和图片混在一起解析失败 | 按 `messagePostFormat=array` 逐段解析 |
| 黑白名单误杀 | whitelist 漏配导致全静默 | 启动打印生效名单，被拦截打 DEBUG |
| 端口冲突 | 8091/8095 被原项目占 | 新端口 8092（网关），NapCat ws client 同步改 |
| 心跳超时误判 | Adapter 判 NapCat 离线 | 心跳对齐 30s，超时阈值 90s |

---

## 三、Agent 与 LLM

| 陷阱 | 现象 | 对策 |
|------|------|------|
| Agent 无限循环 | 工具反复调用 | `max_iterations=5` + `generate` 兜底 |
| tool schema 不合规 | LLM 拒绝调用 | pydantic args_schema，`tool.args` 验证 |
| 模型不支持 function calling | 直接输出文本 | 主模型必须支持（deepseek-chat 等）；文本兜底解析 |
| Langfuse 阻塞 | 服务不可达卡住 | SDK 默认异步批量 + try/except |
| token 用量缺失 | 统计不准 | 手动从 `response_metadata` 提取上报 |
| 多会话状态串 | Agent 跨会话复用混淆 | 每会话独立 `AgentExecutor` |
| 上下文超长 | token 溢出报错 | 严格裁剪 `max_context_size` |
| `do_not_reply` 哨兵泄漏 | 哨兵文本被发出去 | 网关过滤，skill 标记 `should_reply=False` |
| persona emoji 污染 | system prompt emoji 干扰 schema | `_strip_emoji_for_system_prompt` |
| 消息堆积 | LLM 慢导致排队 | 每会话队列 + 丢弃 >60s 过期消息 |

---

## 四、记忆与数据

| 陷阱 | 现象 | 对策 |
|------|------|------|
| faiss 索引未落盘 | 重启记忆丢失 | `data/` 持久化，启动 load |
| 用户画像并发写 | 覆盖丢失 | peewee 事务 + 字段级 merge |
| 黑话误判 | 正常词被永久标记 | 阈值宽松 + 可手动删除 |
| 提醒重启丢失 | 进程重启未恢复 | 启动 `load_pending_reminders()` |
| DB 清理误删 | messages 被删 | 只清 `llm_usage` + 超长 jargon |
| 大表查询慢 | 统计卡顿 | 按 `created_at` 索引 + 分页 |

---

## 五、MCP 与插件

| 陷阱 | 现象 | 对策 |
|------|------|------|
| MCP server 启动阻塞 | 子进程崩导致卡死 | 10s 超时 + 降级跳过 |
| 工具命名冲突 | 与内置 skill 重名 | `mcp_<server>_` 前缀 + 启动检测 |
| relationship 路径硬编码 | 旧绝对路径失效 | 相对路径 / `${REPO_ROOT}` |
| vrchat 误触发 | 群聊误调 VRChat | skill 按会话白名单过滤 |
| 工具结果过大 | 撑爆上下文 | >2000 字符摘要 |
| 工具超时 | 远程慢拖垮 Agent | 30s 超时返回错误 |

---

## 六、主动系统与频率

| 陷阱 | 现象 | 对策 |
|------|------|------|
| 主动发言打扰 | 夜间被骂 | `silent_hours` + 日限额 |
| timing gate 死锁 | wait 态循环 | 超时强制 continue，窗口内只评一次 |
| 重复检测误杀 | 正常复读被拦 | 阈值 0.8 + 连续 3 条才触发 |
| @ 与 talk_value 冲突 | 低 talk_value 时 @ 不回 | @ 走 `mentioned_bot_reply` 旁路 |
| 表情刷屏 | 表情轰炸 | 独立冷却 60s/会话 |
| 频率加成漏接 | 昵称直呼未加成 | `frequency_boost_when_addressed` 生效校验 |

---

## 七、架构与工程

| 陷阱 | 现象 | 对策 |
|------|------|------|
| **repeat 语义搞反**（评审新增） | 把「主动参与复读」做成「防复读拦截」 | 原 repeat_plugin 是参与复读的动作；防刷由冷却承担 |
| **SQLite 并发写锁**（评审新增） | 多协程写库 `database is locked` | WAL + 单写队列 |
| **expression 只学不用**（评审新增） | 只迁 learner 漏 selector | 三件套齐迁，selector 接入回复链 |
| 双轨残留 | HeartF/Brain 逻辑混入 | 统一 `ChatSession`，差异靠 persona + skill 集 |
| 跨层 import | Adapter 反向依赖 Agent | 层间只走数据契约，lint 禁止逆向 |
| 配置插值失败 | `${VAR}` 未替换 | 加载器实现替换，缺失报错 |
| 循环 import | observability 反向依赖 | observability 零业务依赖 |
| 热改不生效 | WebUI 改配置无重载 | 事件总线 + 监听重载 |
| 静态文件 404 | dist 路径错 | StaticFiles mount `/`，确认 dist 存在 |
