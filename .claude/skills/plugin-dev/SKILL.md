---
name: plugin-dev
description: 新建或修改 junjun_skills 插件（LLM 工具/命令/拦截器/定时任务）。用户要求"加个插件/新工具/新功能给 Agent 调用"时使用。
---

# 插件开发

## 解剖（以 plugins/pixiv、plugins/fun_texts 为范本）

```
junjun_skills/plugins/<name>/
├── _manifest.json   # 必需：{"name","version","module","tools_attr","available_for","admin_only"}
├── __init__.py
├── tools.py         # @tool 装饰的函数 + TOOLS = [...] 列表
└── config.toml      # 可选；含密钥/白名单时必须加进 .gitignore，另存 config.toml.example
```

- `_manifest.json`：`module` 是导入路径（如 `junjun_skills.plugins.pixiv.tools`），
  `tools_attr` 默认 `"TOOLS"`；`available_for` 会话白名单（空=全会话）；
  `admin_only` 走管理员门。
- 可选钩子：模块级 `probe_available() -> bool`，依赖缺失时返回 False，
  加载器 WARN 禁用而不崩启动。
- 拦截器 / 定时任务 / job handler 靠 import 副作用注册（见 pixiv、async_task）。
- 持久禁用：`bot_config.toml` 的 `[plugins].disabled` 列表。

## 工具 docstring 铁律（模型选工具的唯一依据）

- ≥15 字，**必须含「何时使用」触发场景**（"用户说xxx时使用"）。
- 与相邻工具职责重叠时，**互相标注边界**（范本：search_knowledge 注明
  "区别于 recall_memory 查聊天记忆"）。
- 只读工具可加入 `scripts/functional_check.py` 的 LIVE_PROBES；有副作用的
  加 SKIP_REASONS 并写明原因。

## 收尾

1. `uv run python -c "from junjun_skills.registry import load_builtin,get_tools; from junjun_skills.plugin_loader import load_plugins; load_builtin(); load_plugins(); print(len(get_tools()))"` 确认装载数与日志行。
2. 测试用内存库 bind_ctx（见 CLAUDE.md 硬约束），跑全套件回归。
3. 边做边 commit。
