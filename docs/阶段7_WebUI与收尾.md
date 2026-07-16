# 阶段 7 · WebUI 与收尾

> 周期：2 天
> 目标：迁移 WebUI、DB 自动清理、完善文档，全功能回归对照，整体可交付。
> 修订（2026-07-16）：原「阶段5」顺延编号；新增全功能回归清单（对照总目标第二节 13 项逐项打勾）。

---

## 一、技术要点

### 1.1 WebUI `junjun_webui`
- 迁移 `src/webui` 的必要路由（按价值裁剪，不全量搬）：
  - `config_routes.py`：在线改配置（人设/频率/ talk_value）。
  - `logs_routes.py` + `logs_ws.py`：日志实时流（WebSocket）。
  - `statistics_routes.py`：token 用量 / 发言统计。
  - `plugin_routes.py`：skill/插件管理（启用/禁用）。
  - `person_routes.py`：用户画像查看。
  - `chat_routes.py`：会话调试（查看上下文/记忆）。
- 前端 `webui/dist` 复用原静态构建（不重写前端），后端 API 路径对齐。
- 鉴权 `auth.py` 迁移（token 管理）。
- 独立 FastAPI app，与网关同进程或独立进程（默认同进程，`WEBUI_ENABLED=true`）。

### 1.2 数据库自动清理
- 迁移 `bot_config.toml [database]`：
  - `enable_auto_cleanup` / `cleanup_retention_days` / `cleanup_interval_hours`。
- 后台定时任务：清理 `llm_usage`（按保留天数）+ 超长 jargon 数据；功能数据（messages/已确认黑话）不动。
- 失败不影响主进程。

### 1.3 可观测收尾
- Langfuse trace 与日志关联：trace_id 写入结构化日志，WebUI 日志页可点击跳转 Langfuse。
- token 用量统计落库 + WebUI 展示。

### 1.4 文档
- `README.md`：快速开始（依赖/配置/启动/停止）。
- 架构图（Mermaid，对齐总目标文档）。
- 配置说明（`bot_config.toml` / `mcp_servers.toml` / `.env`）。
- 故障排查指南。

---

## 二、避坑与潜在 Bug

| 风险 | 说明 | 对策 |
|------|------|------|
| WebUI 前端 API 不匹配 | 原前端调旧 API 路径，新后端路径变了 | 后端路由路径与原 `src/webui` 对齐，不擅自改 URL |
| WebUI 端口冲突 | 原 8001 被占 | 可配置，默认 8002 |
| 鉴权缺失被刷 | 无 token 直接访问 | 迁移 `auth.py` token 校验；生产环境强制 token |
| DB 清理误删功能数据 | 清理逻辑误删 messages | 严格只清 `llm_usage` + 超长 jargon；清理前打 WARN 日志 |
| 日志 WebSocket 内存泄漏 | 长连接不释放 | 心跳 + 超时断开；广播器限制最大连接数 |
| 前端静态文件 404 | dist 路径不对 | StaticFiles mount 到 `/`，确认 `webui/dist` 存在 |
| 配置热改不生效 | WebUI 改了 config 但 Agent 用旧缓存 | 配置变更发事件，Agent/persona 监听重载 |
| 统计数据查询慢 | 大表无索引 | `llm_usage` 按 `created_at` 建索引；分页查询 |

---

## 三、验收标准

- [ ] WebUI 可访问，登录后可见配置/日志/统计页。
- [ ] WebUI 改 talk_value 后 Agent 行为立即变化。
- [ ] 日志页实时滚动，trace_id 可见。
- [ ] DB 自动清理任务按周期执行，只清 `llm_usage`。
- [ ] `README.md` 按步骤可从零启动君君。
- [ ] 全流程冒烟：QQ 收发 → 漏斗决策 → 工具调用 → 后处理分条 → 主动发言 → 提醒触发，Langfuse trace 完整。
- [ ] **全功能回归对照**：总目标「第二节 必须复刻功能清单」13 项逐项验证打勾，留档 `docs/回归记录.md`。
- [ ] `uv run pytest` 全绿；`ruff check` 零告警。

---

## 四、交付物

- `junjun_webui/`（路由 + 静态资源）
- `junjun_core/database/cleanup.py`
- `README.md` / 架构图 / 配置说明
