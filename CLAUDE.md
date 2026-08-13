# JunJun_Agent 仓库守则

QQ 机器人「君君」单仓 monorepo。北极星：做一个优秀的通用 agent
（2026-08-13 用户拍板换轨——原「像真人」降为前端体验指标之一；
QQ 是前端之一，不是全部）。

## 硬安全约束（任何任务都必须遵守）

- `data/junjun.db` 是**生产库，只读**：`sqlite3.connect("file:data/junjun.db?mode=ro", uri=True)`。
- **测试绝不写生产库**：触达 `junjun_core.database` 模型的测试必须用
  `peewee.SqliteDatabase(tmp_path/"t.db")` + `db.bind_ctx([...])` + `db.create_tables`。
  （2026-08-04 两次事故：selfmood、Images 表被测试污染。）
- `.env`、`data/sf_keys.txt` 含活密钥：展示时只给前缀（sk-xxx...），PIXIV_COOKIE、
  Langfuse key 永不打印。
- gitignored 不入库：`config/bot_config.toml`（只提交 `.example`）、`data/`、
  各插件 `config.toml`、`config/bot_config.toml.bak_*`。
- `ADMIN_QQ` 是管理员信任根；R18 订阅标题群推送必须打码（URL 保留）。
- **模型加字段必须带默认值或 `null=True`**：`init_database` 会自动
  `ALTER TABLE ADD COLUMN` 对齐旧库（`_ensure_columns`），SQLite 加列只支持
  常量默认——无默认且非空的列会被跳过并告警（得人工迁移），改/删列不自动做。

## 工作方式

- **边做边 commit**，方便出错回滚；提交信息用中文、说明为什么。
- 全量回归：`uv run python -m pytest tests/ -q`（规模以实际输出为准，
  2026-08-13 约 1749 条——数字会漂移，别拿旧基线当判据）。
- 决策评测（真实 LLM，花 API 额度）：`uv run python scripts/eval_golden.py`
  ——golden case 在 `tests/eval/golden_cases.jsonl`，改 prompt/工具掩码/模型前后各跑一次对比。
- 功能体检：`uv run python scripts/functional_check.py --pytest`。
- 控制台 GBK：打印中文用 `unicode_escape` 或写文件，避免乱码报错。
- Agent 技能手册（md skills）：`junjun_skills/agent_skills/<name>.md`，frontmatter
  必须带 `name` + `when`；新增后跑 `tests/test_agent_skills.py`。
- **加宽命中面必须配误判回归测试**：router 词表、意图组关键词、守卫 pattern、
  工具掩码阈值——凡是「让更多输入命中」的改动，必须同 commit 带「不该命中的
  日常句子」断言。项目事故全在误伤方向（2026-08-06 审查：eval 驱动补缺口时
  连续四处宽化回归）。
- 估算/判定宁可保守方向的反面教训记牢：token 估算宁高估勿低估（低估=预算
  形同虚设）；号池文件故障宁用旧 key 勿清空（错杀=全线硬失败）。

## 关键结构

| 目录 | 职责 |
|---|---|
| junjun_core | 网关 / 配置 / DB 模型 / 安全 |
| junjun_adapter_napcat | NapCat(OneBot) 适配器 |
| junjun_agent | 决策漏斗 / 人设 / 各 loop |
| junjun_llm | LLM 槽位（agent / utils / vlm…） |
| junjun_memory | 三层记忆 + echo 防复读 |
| junjun_skills | 工具注册表 + plugins/ 27 个插件 |
| junjun_express | 表达层（情绪/口吻） |
| docs/ | 路线图、自设、踩坑清单、验证清单 |
