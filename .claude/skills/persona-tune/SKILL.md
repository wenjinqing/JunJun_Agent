---
name: persona-tune
description: 调整君君人设/口吻/情绪表达（personality、behavior_examples、persona_brief、reply_style、emotion_style）。用户说"人设不对/说话不像人/改性格/语气问题"时使用。
---

# 人设调优

## 方法论（人格调色盘，docs/人格调色盘 (1).txt）

- 主色调/底色/性格点缀 + **性格衍生**（具体行为机制，矛盾衍生制造深度）+
  **作者二次解释**（防止 AI 自行脑补成套路）。手写 > AI 生成。
- 行为层 > 心理描写；不写绝对化台词（"姐姐疼你"这类句子是一次性调味料，
  写进 prompt 就会被复读）。
- 当前生效自设存档：`docs/君君自设_2026-08-04.md`（温柔学姐 v3）。

## 配置落点（config/bot_config.toml，gitignored；改前留 .bak）

| 键 | 作用 |
|---|---|
| `[personality] personality` | 设定卡正文（~500 字内有效） |
| `[personality] behavior_examples` | 示例集，拼接在设定卡后，标题「感受分布，不要照抄原句」 |
| `[personality] persona_brief` | 一句话口吻摘要，注入全部 utils 单发 prompt（intention/reminder/perception_followup/proactive/async_jobs）——**一个声音**的关键 |
| `reply_style` / `emotion_style` | 回复风格 / 情绪表达（emotion_style 不得点名具体心情） |

## 铁律（踩坑沉淀）

- **prompt 里出现的每一句具体台词都会被复读**——示例必须多而杂（≥10 条），
  靠 echo guard + 口头禅自动检测兜底（junjun_memory/echo.py），不要试图维护语录库。
- 固定短语列表（如 repeat.py 打断语、tasks.py 模板池）会变人设词汇表，改动要当人设改。
- 情绪系统：平静是默认态不进 SelfMood；负面情绪只影响语气不压人格。
- 改完跑 `tests/test_persona_role.py` + `tests/test_echo_guard.py` +
  `tests/test_w2_human_like.py`，再全量回归。
- 生效需重启 bot（restart-deploy skill），生产观察对照 prod-verify 的口吻指标。
