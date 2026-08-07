"""ContextBudget 预算管理器单元测试（Phase 2）。"""

import pytest

from junjun_agent.context_budget import BudgetBlock, ContextBudget


def test_estimate_tokens_basic():
    cb = ContextBudget(1000)
    # 纯中文：cl100k 约 1.5 字/token
    assert cb.estimate_tokens("你好世界") == 2  # 4/1.5 = 2.67 -> int = 2
    # 纯英文：约 4 字符/token
    assert cb.estimate_tokens("hello world") == 2  # 11/4 = 2.75 -> 2


def test_estimate_tokens_chinese_not_underestimated():
    """回归：中文块曾被低估 10 倍（word 估算把无空格全文算 1 个词取 min）。
    驱逐逻辑因此永不触发、Langfuse 指标失真（2026-08-06 审查实锤）。"""
    cb = ContextBudget(1000)
    # 真实形态的中文记忆块（多行、无空格）
    block = "\n".join(f"群友{i}：今天聊的话题还挺有意思的嘛" for i in range(30))
    est = cb.estimate_tokens(block)
    cjk = sum(1 for ch in block if "一" <= ch <= "鿿")
    assert est >= cjk / 2, f"中文估算失真：{cjk} 字估出 {est} token"
    # 18 字无空格中文句不许估成 1
    assert cb.estimate_tokens("今天天气不错我们一起去公园散步吧") >= 10


def test_fit_keeps_all_when_under_budget():
    cb = ContextBudget(1000)
    blocks = [
        BudgetBlock(name="system", content="x" * 100, priority=1, required=True),
        BudgetBlock(name="memory", content="m" * 100, priority=2),
    ]
    fit = cb.fit(blocks)
    assert [b.name for b in fit.blocks] == ["system", "memory"]
    assert not fit.evicted
    assert fit.total_tokens <= 1000


def test_fit_evicts_low_priority_first():
    cb = ContextBudget(50)
    blocks = [
        BudgetBlock(name="system", content="x" * 30, priority=1, required=True),
        BudgetBlock(name="relation", content="r" * 200, priority=3),
        BudgetBlock(name="memory", content="m" * 200, priority=2),
    ]
    fit = cb.fit(blocks)
    assert "system" in [b.name for b in fit.blocks]
    # 优先级低的 relation 应被驱逐
    assert "relation" in [b.name for b in fit.evicted]


def test_required_block_survives():
    cb = ContextBudget(10)
    blocks = [
        BudgetBlock(name="latest", content="x" * 500, priority=1, required=True),
    ]
    fit = cb.fit(blocks)
    assert [b.name for b in fit.blocks] == ["latest"]
    assert not fit.evicted


def test_fit_preserves_original_order():
    cb = ContextBudget(50)
    blocks = [
        BudgetBlock(name="c", content="c", priority=2),
        BudgetBlock(name="a", content="a", priority=1),
        BudgetBlock(name="b", content="b", priority=2),
    ]
    fit = cb.fit(blocks)
    # 全部保留，顺序与传入一致
    assert [b.name for b in fit.blocks] == ["c", "a", "b"]


def test_reserve_tokens_reduces_effective_budget():
    cb = ContextBudget(100, reserve_tokens=80)
    blocks = [
        BudgetBlock(name="system", content="x" * 100, priority=1, required=True),
        BudgetBlock(name="memory", content="m" * 100, priority=2),
    ]
    fit = cb.fit(blocks)
    # 有效预算只有 20，非必需 memory 装不下
    assert "memory" in [b.name for b in fit.evicted]
    assert fit.metrics["effective_budget"] == 20


def test_metrics_contains_block_sizes():
    cb = ContextBudget(100)
    blocks = [
        BudgetBlock(name="system", content="你好", priority=1),
    ]
    fit = cb.fit(blocks)
    assert "block_sizes" in fit.metrics
    assert "system" in fit.metrics["block_sizes"]
    assert "evicted_names" in fit.metrics
