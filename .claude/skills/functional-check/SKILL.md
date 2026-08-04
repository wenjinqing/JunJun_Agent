---
name: functional-check
description: 全功能体检——环境矩阵/插件审计/工具静态检查/只读实调/LLM 活性/pytest，输出测试报告并按需生成修复计划。用户要求"全面测试/体检/检查所有功能是否正常"时使用。
---

# 全功能体检

## 执行

```bash
uv run python scripts/functional_check.py            # 快速体检（约 40s）
uv run python scripts/functional_check.py --pytest   # 全量（含套件，约 2min）
uv run python scripts/functional_check.py --json     # 机器可读
```

输出为 markdown 表格 + 状态汇总（PASS/FAIL/DEGRADED/SKIP）。

## 解读纪律

- **SKIP 不是失败**：发消息/写库/花钱的工具按安全约束不实调，核对 SKIP 原因列是否合理即可。
- **DEGRADED 多为配置缺口**（缺 key），是配置决策不是代码 bug——列进报告"建议"节，不改代码。
- **FAIL 先怀疑探测脚本**：参数名写错会导致误报（2026-08-04 有 4 例：
  query_jargon/abbreviation_translate 用 `term`，search_knowledge 用 `question`，
  bilibili_summary 用 `url`）。修正脚本后复跑再下结论。
- 静态检查硬性标准：每个工具必须有 docstring 且 ≥15 字、含「何时使用」触发场景——
  docstring 是模型选工具的唯一依据。

## 产出

1. 测试报告写 `docs/功能测试报告_<日期>.md`：总览表 / 实调明细 / 环境缺口 /
   插件审计 / 结论 / 复跑方法。参考 `docs/功能测试报告_2026-08-04.md`。
2. 有真问题时写 `docs/修复计划_<日期>.md`：P0（就地已修）/P1（代码修复，
   按 LLM 决策影响排序）/P2（配置建议），每条带验收标准，修完全套件回归后 commit。
