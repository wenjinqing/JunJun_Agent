"""ContextBudget 预算管理器单元测试（Phase 2）。"""

import pytest

from junjun_agent.context_budget import BudgetBlock, ContextBudget


def test_estimate_tokens_basic():
    cb = ContextBudget(1000)
    # 纯中文：每个字约 1/3 token
    assert cb.estimate_tokens("你好世界") == 1  # 4/3 = 1.33 -> int = 1
    # 纯英文单词
    assert cb.estimate_tokens("hello world") == 2  # 2 words / 0.75 = 2.67 -> 2


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
