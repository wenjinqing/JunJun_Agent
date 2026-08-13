"""重复调用熔断测试（2026-08-14 trace bc95cd3b 实锤：use_skill×3/manage_mood×2
交替空转撞穿迭代上限，run_code 一次没真调）。

RepeatCallGuardMiddleware：同（工具,参数）每轮限次，第三次起短路成
「禁止复读、立刻推进」的结构化文本并拍回上次结果摘要；不同参数不受影响；
按尝试计数；0=关闭。
"""

from types import SimpleNamespace

import pytest
from langchain_core.messages import ToolMessage

from junjun_agent.loop.repeat_guard import RepeatCallGuardMiddleware, _limit_cfg
from junjun_core.config import get_global_config


def _req(name, args=None, call_id="c1"):
    return SimpleNamespace(tool_call={"name": name, "args": args or {}, "id": call_id})


class _Handler:
    """记录型假执行体：被调即记一次（工具名+参数），返回正常 ToolMessage。"""

    def __init__(self):
        self.calls = []

    async def __call__(self, request):
        tc = request.tool_call
        self.calls.append((tc["name"], dict(tc.get("args") or {})))
        return ToolMessage(content=f"{tc['name']} 的执行结果正文",
                           tool_call_id=tc["id"], name=tc["name"])


@pytest.fixture()
def limit3(monkeypatch):
    monkeypatch.setitem(get_global_config().raw, "agent", {"repeat_call_limit": 3})


class TestRepeatGuard:
    @pytest.mark.asyncio
    async def test_first_two_pass_third_blocked(self, limit3):
        mw, h = RepeatCallGuardMiddleware(), _Handler()
        await mw.awrap_tool_call(_req("use_skill", {"name": "code-lab"}), h)
        await mw.awrap_tool_call(_req("use_skill", {"name": "code-lab"}, "c2"), h)
        r3 = await mw.awrap_tool_call(_req("use_skill", {"name": "code-lab"}, "c3"), h)
        assert len(h.calls) == 2, "第三次同参调用不许真执行"
        assert "复读熔断" in r3.content and "推进" in r3.content
        assert "执行结果正文" in r3.content          # 上次结果摘要拍回
        assert r3.tool_call_id == "c3" and r3.name == "use_skill"  # 图状态一致性

    @pytest.mark.asyncio
    async def test_alternating_loop_caught(self, limit3):
        """还原病灶 trace：use_skill/manage_mood 交替——不是连续重复，
        按（工具,参数）总量计，第三次 use_skill(code-lab) 照样熔断。"""
        mw, h = RepeatCallGuardMiddleware(), _Handler()
        await mw.awrap_tool_call(_req("use_skill", {"name": "code-lab"}), h)
        await mw.awrap_tool_call(_req("manage_mood", {"action": "set"}, "c2"), h)
        await mw.awrap_tool_call(_req("use_skill", {"name": "code-lab"}, "c3"), h)
        await mw.awrap_tool_call(_req("manage_mood", {"action": "set"}, "c4"), h)
        r = await mw.awrap_tool_call(_req("use_skill", {"name": "code-lab"}, "c5"), h)
        assert "复读熔断" in r.content
        assert len(h.calls) == 4   # 前两次 use_skill + 前两次 manage_mood 真执行

    @pytest.mark.asyncio
    async def test_different_args_not_blocked(self, limit3):
        """误判回归：同工具不同参数是合法探索（如换关键词搜索），不许熔断。"""
        mw, h = RepeatCallGuardMiddleware(), _Handler()
        for i, q in enumerate(["关键词甲", "关键词乙", "关键词丙", "关键词丁"]):
            r = await mw.awrap_tool_call(
                _req("web_search", {"query": q}, f"c{i}"), h)
            assert "复读熔断" not in r.content
        assert len(h.calls) == 4

    @pytest.mark.asyncio
    async def test_blocked_repeats_keep_blocking(self, limit3):
        """第四、五次同参调用继续熔断（不给「再试一次就放行」的口子）。"""
        mw, h = RepeatCallGuardMiddleware(), _Handler()
        for i in range(2):
            await mw.awrap_tool_call(_req("use_skill", {"name": "x"}, f"c{i}"), h)
        for i in range(2, 5):
            r = await mw.awrap_tool_call(_req("use_skill", {"name": "x"}, f"c{i}"), h)
            assert "复读熔断" in r.content
        assert len(h.calls) == 2

    @pytest.mark.asyncio
    async def test_new_instance_resets(self, limit3):
        """实例随 _build_agent 每轮新建：新一轮会话计数清零。"""
        h = _Handler()
        mw1 = RepeatCallGuardMiddleware()
        for i in range(2):
            await mw1.awrap_tool_call(_req("use_skill", {"name": "x"}, f"a{i}"), h)
        mw2 = RepeatCallGuardMiddleware()
        r = await mw2.awrap_tool_call(_req("use_skill", {"name": "x"}, "b0"), h)
        assert "复读熔断" not in r.content

    @pytest.mark.asyncio
    async def test_zero_disables(self, monkeypatch):
        monkeypatch.setitem(get_global_config().raw, "agent", {"repeat_call_limit": 0})
        mw, h = RepeatCallGuardMiddleware(), _Handler()
        for i in range(5):
            r = await mw.awrap_tool_call(_req("use_skill", {"name": "x"}, f"c{i}"), h)
            assert "复读熔断" not in r.content
        assert len(h.calls) == 5

    def test_bad_config_falls_back_to_default(self, monkeypatch):
        monkeypatch.setitem(get_global_config().raw, "agent",
                            {"repeat_call_limit": "不是数字"})
        assert _limit_cfg() == 3

    def test_default_three_when_missing(self, monkeypatch):
        monkeypatch.delitem(get_global_config().raw, "agent", raising=False)
        assert _limit_cfg() == 3
