"""工具健康度测试（P5-4）：降级阈值/TTL 恢复/成功清零/持久化/上下文注入/registry 挂钩。"""

import json
import time

import pytest
from langchain_core.tools import tool

import junjun_skills.health as health


@pytest.fixture(autouse=True)
def clean_state(tmp_path, monkeypatch):
    monkeypatch.setattr(health, "_STATE_PATH", tmp_path / "tool_health.json")
    # registry 错误包装还会写 patches 失败日志——一并隔离
    # （2026-08-06 实锤：exploding_tool/flaky_tool 污染了生产 data/tool_failures.jsonl）
    from junjun_skills import patches
    monkeypatch.setattr(patches, "_LOG_PATH", tmp_path / "tool_failures.jsonl")
    health._reset_for_test()
    yield
    health._reset_for_test()


class TestDegradation:
    def test_below_threshold_not_degraded(self):
        health.record_fail("ai_draw", "网络", "timeout")
        health.record_fail("ai_draw", "网络", "timeout")
        assert health.degraded_tools() == []
        assert health.health_block() == ""

    def test_threshold_degrades(self):
        for _ in range(3):
            health.record_fail("ai_draw", "限流", "429")
        d = health.degraded_tools()
        assert len(d) == 1 and d[0]["tool"] == "ai_draw"
        assert d[0]["kind"] == "限流" and d[0]["fails"] == 3

    def test_success_recovers(self):
        for _ in range(3):
            health.record_fail("ai_draw", "限流")
        assert health.degraded_tools()
        health.record_ok("ai_draw")
        assert health.degraded_tools() == []

    def test_success_resets_consecutive_count(self):
        health.record_fail("ai_draw", "网络")
        health.record_fail("ai_draw", "网络")
        health.record_ok("ai_draw")          # 打断连续计数
        health.record_fail("ai_draw", "网络")
        health.record_fail("ai_draw", "网络")
        assert health.degraded_tools() == []  # 只有 2 次连续，不降级

    def test_ttl_expires(self, monkeypatch):
        for _ in range(3):
            health.record_fail("ai_draw", "限流")
        # 把最近失败时间拨到 25h 前 -> 不再宣称故障
        health._STATE["ai_draw"]["last_fail_at"] = time.time() - 25 * 3600
        assert health.degraded_tools() == []
        assert "ai_draw" not in health._STATE  # 顺手清理

    def test_persistence(self):
        for _ in range(3):
            health.record_fail("ai_draw", "限流")
        data = json.loads(health._STATE_PATH.read_text(encoding="utf-8"))
        assert data["ai_draw"]["consec_fails"] == 3
        # 模拟重启：清空内存重新加载
        health._STATE.clear()
        health._loaded = False
        assert len(health.degraded_tools()) == 1


class TestHealthBlock:
    def test_block_content(self):
        for _ in range(3):
            health.record_fail("ai_draw", "限流")
        block = health.health_block()
        assert "系统状态" in block and "ai_draw" in block
        assert "在修" in block and "不要主动提议" in block
        assert "限流" in block

    def test_block_empty_when_healthy(self):
        assert health.health_block() == ""


class TestRegistryHook:
    @pytest.mark.asyncio
    async def test_fail_then_recover_via_tool_calls(self, monkeypatch):
        """通过 registry 包装的工具调用驱动健康状态。"""
        from junjun_skills import breaker, registry
        # 健康度记账需要 3 次连续失败——P2 熔断（阈值 2）会短路第 3 次，
        # 这里测的是健康度不是熔断，关掉熔断干扰
        monkeypatch.setattr(breaker, "is_open", lambda *a, **kw: False)
        registry.clear()

        @tool("exploding_tool")
        async def _t(x: str = "") -> str:
            """测试工具。"""
            raise ConnectionError("boom")

        registry.register(_t)
        for _ in range(2):
            await registry._registry["exploding_tool"].ainvoke({"x": "1"})
        assert health.degraded_tools() == []     # 2 次未降级
        out = await registry._registry["exploding_tool"].ainvoke({"x": "1"})
        assert "[TOOL_ERROR kind=网络" in out     # P0-13 行为不变（P1 起带 suggestion）
        assert len(health.degraded_tools()) == 1  # 第 3 次降级
        registry.clear()

    @pytest.mark.asyncio
    async def test_success_clears(self, monkeypatch):
        from junjun_skills import breaker, registry
        monkeypatch.setattr(breaker, "is_open", lambda *a, **kw: False)  # 同上：隔熔断
        registry.clear()
        calls = {"n": 0}

        @tool("flaky_tool")
        async def _t(x: str = "") -> str:
            """测试工具。"""
            calls["n"] += 1
            if calls["n"] <= 3:
                raise ConnectionError("boom")
            return "ok"

        registry.register(_t)
        for _ in range(3):
            await registry._registry["flaky_tool"].ainvoke({"x": "1"})
        assert len(health.degraded_tools()) == 1
        out = await registry._registry["flaky_tool"].ainvoke({"x": "1"})
        assert out == "ok"
        assert health.degraded_tools() == []
        registry.clear()


class TestPersonaInjection:
    def test_prompt_contains_health_state(self):
        from junjun_agent.persona import build_system_prompt
        for _ in range(3):
            health.record_fail("ai_draw", "限流")
        prompt = build_system_prompt(is_group=True)
        assert "系统状态" in prompt and "ai_draw" in prompt

    def test_prompt_clean_when_healthy(self):
        from junjun_agent.persona import build_system_prompt
        prompt = build_system_prompt(is_group=True)
        assert "系统状态" not in prompt


class TestCapabilitiesReport:
    def test_get_capabilities_shows_degraded(self):
        from junjun_skills.builtin.capability_skills import get_capabilities
        for _ in range(3):
            health.record_fail("ai_draw", "限流")
        text = get_capabilities.invoke({"query_type": "all"})
        assert "故障中" in text and "ai_draw" in text
