"""严厉审查 P1-6 拦截器层回归：
- nsfw 正则收窄：「蓝色的天空/角色的立绘」不再被涩图拦截器劫持
- 显式 priority：匹配顺序不再依赖插件目录字母序
- CommandContext.send 回复写回短期记忆（治「副作用失忆」）
"""

import re

import pytest


class TestNsfwRegexNarrowed:
    @pytest.mark.parametrize("text", [
        "帮我画一张蓝色的天空",
        "画个角色的立绘",
        "这风景颜色的好看，画一张",
        "画一张出色的作品",
    ])
    def test_false_positives_not_hijacked(self, text):
        from junjun_skills.plugins.ai_draw.tools import _NSFW_DRAW_RE
        assert not re.search(_NSFW_DRAW_RE, text, re.I), f"误判劫持: {text}"

    @pytest.mark.parametrize("text", [
        "画一张涩图",
        "来个色图",
        "帮我画点色的",
        "画张涩涩的",
        "来一张r18",
    ])
    def test_true_positives_still_hit(self, text):
        from junjun_skills.plugins.ai_draw.tools import _NSFW_DRAW_RE
        assert re.search(_NSFW_DRAW_RE, text, re.I), f"漏判: {text}"


class TestInterceptorPriority:
    def test_priority_orders_dispatch(self):
        from junjun_agent.interceptors import (
            clear_interceptors, register_interceptor, _interceptors)
        clear_interceptors()
        try:
            @register_interceptor(r"aaa", name="low")
            async def _low(ctx):
                return True

            @register_interceptor(r"aaa", name="high", priority=10)
            async def _high(ctx):
                return True

            assert _interceptors[0].name == "high"   # 高优先级排前，尽管注册在后
        finally:
            clear_interceptors()


class TestCommandContextMemory:
    @pytest.mark.asyncio
    async def test_send_writes_bot_memory(self, monkeypatch):
        """命令/拦截器直发的文本回复必须写回 STM——否则 bot 下轮忘了自己说过的话。"""
        from types import SimpleNamespace
        from junjun_agent.commands import CommandContext
        from junjun_memory.short_term import ShortTermMemory

        sent = []

        class FakeGW:
            async def send_reply(self, rs):
                sent.append(rs)
        monkeypatch.setattr(
            "junjun_core.gateway.router.get_gateway", lambda: FakeGW())
        monkeypatch.setattr(
            "junjun_agent.processor._store_outbound", lambda *a, **k: None)

        session = SimpleNamespace(chat_id="qq:1:private", platform="qq",
                                  group_id=None, is_group=False,
                                  memory=ShortTermMemory())
        ctx = CommandContext(session=session,
                             meta=SimpleNamespace(user_id="1"))
        await ctx.reply("在画了在画了")
        assert sent, "回复应发出"
        roles = [e.role for e in session.memory.entries]
        assert roles == ["bot"] and session.memory.entries[0].text == "在画了在画了"
