"""短期记忆滚动摘要测试（2026-08-16，DSH 压缩思想移植）。

核心断言：滑出窗口的行攒够阈值起火后台摘要（utils 槽，fire_and_forget
强引用）；摘要合并在 render 置顶且防伪清洗；失败不丢料（塞回队列）且
有冷却防重试风暴；同步上下文（无事件循环）不起任务不炸；开关关闭
整个机制不存在；待压队列有上限。全程不碰 DB、不烧真 LLM。
"""

import asyncio

import pytest

import junjun_memory.short_term as st
from junjun_memory.short_term import ShortTermMemory


def _set_cfg(monkeypatch, mem_cfg):
    import junjun_core.config.config as cfg_mod
    monkeypatch.setattr(cfg_mod, "global_config", cfg_mod.GlobalConfig(
        bot=cfg_mod.BotConfig(platform="qq", qq_account="1", nickname="君君"),
        raw={"memory": mem_cfg}))


@pytest.fixture
def cfg_on(monkeypatch):
    _set_cfg(monkeypatch, {"rolling_summary": True,
                           "rolling_summary_min_chars": 30,
                           "rolling_summary_max_chars": 120})


class _Resp:
    def __init__(self, text):
        self.content = text


class _StubModel:
    def __init__(self, text="大家在聊火锅，约好周五聚餐", fail=False):
        self._text = text
        self._fail = fail
        self.calls = 0
        self.last_prompt = ""

    async def ainvoke(self, msgs, config=None):
        self.calls += 1
        self.last_prompt = msgs[0].content
        if self._fail:
            raise RuntimeError("utils 槽炸了")
        return _Resp(self._text)


@pytest.fixture
def stub_model(monkeypatch):
    stub = _StubModel()
    monkeypatch.setattr("junjun_llm.get_chat_model", lambda slot: stub)
    return stub


async def _wait_done(mem):
    """等后台摘要任务收尾（_compressing 起落即完成）。"""
    for _ in range(200):
        if not mem._compressing:
            return
        await asyncio.sleep(0.005)
    raise AssertionError("摘要任务没收尾")


def _fill(mem, n, prefix="今晚去吃火锅怎么样"):
    for i in range(n):
        mem.add_user(f"{prefix}{i}", "小明", user_id="1001")


class TestRollingSummary:
    @pytest.mark.asyncio
    async def test_compress_triggers_and_renders(self, cfg_on, stub_model):
        """滑出材料攒够 30 字 -> 起火摘要 -> render 置顶「前情摘要」。"""
        mem = ShortTermMemory(max_size=3)
        _fill(mem, 6)
        await _wait_done(mem)
        assert stub_model.calls >= 1
        assert mem.rolling_summary == "大家在聊火锅，约好周五聚餐"
        # 模型确实收到了滑出的原文与「空」旧摘要
        assert "今晚去吃火锅" in stub_model.last_prompt
        assert "（空）" in stub_model.last_prompt
        out = mem.render()
        assert out.startswith("（更早前聊过·摘要：大家在聊火锅，约好周五聚餐）")
        assert len(mem.entries) == 3          # 窗口照滑

    @pytest.mark.asyncio
    async def test_merge_with_old_summary(self, cfg_on, stub_model):
        """第二轮压缩：旧摘要进 prompt，新摘要整体替换（滚动合并）。"""
        mem = ShortTermMemory(max_size=3)
        mem.rolling_summary = "旧摘要：上周聊了团建"
        _fill(mem, 6)
        await _wait_done(mem)
        assert "旧摘要：上周聊了团建" in stub_model.last_prompt
        assert mem.rolling_summary == "大家在聊火锅，约好周五聚餐"

    @pytest.mark.asyncio
    async def test_bot_lines_included(self, cfg_on, stub_model):
        """bot 自己的发言滑出也进材料（标「你:」），话题脉络不断。"""
        mem = ShortTermMemory(max_size=2)
        mem.add_user("火锅还是烧烤", "小明", user_id="1001")
        mem.add_bot("我投火锅一票，毛肚万岁")
        mem.add_user("那就火锅", "小明", user_id="1001")
        mem.add_user("周五几点", "小红", user_id="1002")
        mem.add_user("七点怎么样", "小明", user_id="1001")   # 跨过 30 字阈值
        await _wait_done(mem)
        assert "你: 我投火锅一票" in stub_model.last_prompt

    @pytest.mark.asyncio
    async def test_failure_keeps_material_and_cools_down(self, cfg_on, monkeypatch):
        """摘要失败：材料塞回队列不丢，冷却期内不再重试（防 LLM 挂时风暴）。"""
        fail_stub = _StubModel(fail=True)
        monkeypatch.setattr("junjun_llm.get_chat_model", lambda slot: fail_stub)
        mem = ShortTermMemory(max_size=3)
        _fill(mem, 6)
        await _wait_done(mem)
        assert fail_stub.calls == 1
        assert mem.rolling_summary == ""
        assert mem._pending_compress          # 材料还在
        assert mem._pending_compress[0].startswith("小明: 今晚去吃火锅")
        _fill(mem, 2)                          # 冷却期内：材料够阈值也不重试
        await asyncio.sleep(0.05)
        assert fail_stub.calls == 1

    @pytest.mark.asyncio
    async def test_disabled_mechanism_absent(self, monkeypatch):
        _set_cfg(monkeypatch, {"rolling_summary": False,
                               "rolling_summary_min_chars": 5})
        stub = _StubModel()
        monkeypatch.setattr("junjun_llm.get_chat_model", lambda slot: stub)
        mem = ShortTermMemory(max_size=2)
        _fill(mem, 10)
        await asyncio.sleep(0.05)
        assert stub.calls == 0
        assert mem.rolling_summary == ""
        assert mem._pending_compress == []    # 关了就不攒料（行为同旧版）

    def test_sync_context_no_crash(self, cfg_on, stub_model):
        """同步上下文（无运行中事件循环）：不起任务、不炸，材料攒着等下次。"""
        mem = ShortTermMemory(max_size=3)
        _fill(mem, 6)                          # 同步调用，无 loop
        assert stub_model.calls == 0
        assert mem.rolling_summary == ""
        assert mem._pending_compress          # 料没丢

    def test_pending_queue_capped(self, cfg_on, monkeypatch):
        """待压队列上限：持续失败/无 loop 时丢最旧，队列不无界涨。"""
        monkeypatch.setattr(st, "_PENDING_MAX_LINES", 5)
        mem = ShortTermMemory(max_size=2)
        _fill(mem, 12)
        assert len(mem._pending_compress) <= 5

    @pytest.mark.asyncio
    async def test_summary_sanitized_in_render(self, cfg_on, monkeypatch):
        """摘要防伪：「」/【最新】/管理员标记/[@你] 一律剥掉——
        摘要内容源自群友发言，伪造标记不能借摘要置顶污染认知锚点。"""
        stub = _StubModel(text="「小明」(管理员) 说【最新】周五吃火锅 [@你]")
        monkeypatch.setattr("junjun_llm.get_chat_model", lambda slot: stub)
        mem = ShortTermMemory(max_size=3)
        _fill(mem, 6)
        await _wait_done(mem)
        out = mem.render()
        first = out.split("\n")[0]
        assert first.startswith("（更早前聊过·摘要：")
        for token in ("「", "」", "(管理员)", "（管理员）", "【最新】", "[@你]"):
            assert token not in first
        assert "周五吃火锅" in first

    def test_no_summary_no_prefix(self, cfg_on):
        """没摘要时渲染不多行（旧行为不变）。"""
        mem = ShortTermMemory(max_size=5)
        mem.add_user("你好", "小明", user_id="1001")
        out = mem.render()
        assert "摘要" not in out
