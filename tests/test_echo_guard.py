"""复读自检（echo guard）测试（2026-08-04 用户反馈：Agent 重复话术污染上下文）。

自我污染循环：bot 复读 -> 话术落进短期记忆 -> 下一轮 context 堆 N 次
-> 模型当成自己的说话习惯继续复读。代码层两处确定性打断：
1) 输入端：ShortTermMemory.render bot 历史行去重（同一句只留最近一次）
2) 出口端：agent.process 撞车追问重说，仍撞车则沉默（被@发重试稿）
"""

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from junjun_memory.echo import extract_catchphrases, is_echo, normalize_echo
from junjun_memory.short_term import ShortTermMemory


class TestExtractCatchphrases:
    def test_phrase_in_three_messages_flagged(self):
        texts = ["来，姐姐疼你", "今天姐姐疼你一次", "姐姐疼你还不乐意？"]
        cps = extract_catchphrases(texts, min_count=3)
        assert any("姐姐疼你" in cp for cp in cps)

    def test_pure_laughter_exempt(self):
        texts = ["哈哈哈哈笑死", "哈哈哈哈哈哈", "笑死哈哈哈哈"]
        assert extract_catchphrases(texts, min_count=3) == []

    def test_varied_speech_no_catchphrase(self):
        texts = ["今晚吃什么呢", "昨天睡得好吗", "新番更新了", "这把排位稳了"]
        assert extract_catchphrases(texts, min_count=3) == []

    def test_below_threshold_not_flagged(self):
        texts = ["姐姐疼你", "姐姐疼你哦"]
        assert extract_catchphrases(texts, min_count=3) == []

    def test_keeps_longest_representative(self):
        texts = ["姐姐疼你一次", "姐姐疼你两次", "姐姐疼你三次"]
        cps = extract_catchphrases(texts, min_count=3)
        # 「姐姐疼你」被更长的「姐姐疼你一/两/三」包含时只留代表，不重复上报
        assert len([cp for cp in cps if "姐姐疼你" in cp]) >= 1
        assert all(not (a != b and a in b) for a in cps for b in cps)


class TestNormalize:
    def test_punctuation_and_emoji_ignored(self):
        assert normalize_echo("杂鱼就是杂鱼！") == normalize_echo("杂鱼就是杂鱼~")
        assert normalize_echo("晚安咯😴") == normalize_echo("晚安咯")

    def test_case_and_whitespace(self):
        assert normalize_echo("Hello World") == normalize_echo("helloworld")


class TestIsEcho:
    def test_exact_repeat_after_normalize(self):
        hit = is_echo("杂鱼就是杂鱼！", ["杂鱼就是杂鱼"], similarity=0.85)
        assert hit == "杂鱼就是杂鱼"

    def test_near_duplicate_hits(self):
        hit = is_echo("今天天气真好呀大家", ["今天天气真好啊大家"], similarity=0.85)
        assert hit is not None

    def test_containment_hits(self):
        hit = is_echo("杂鱼就是杂鱼，一群杂鱼", ["杂鱼就是杂鱼"], similarity=0.85)
        assert hit is not None

    def test_different_text_passes(self):
        assert is_echo("今晚吃什么呢", ["昨天睡得好吗"], similarity=0.85) is None

    def test_short_text_never_echoes(self):
        """「嗯嗯」「好耶」这类短句天然会重复，不触发拦截。"""
        assert is_echo("好耶", ["好耶"], similarity=0.85) is None
        assert is_echo("嗯嗯嗯", ["嗯嗯嗯"], similarity=0.85) is None

    def test_empty_history_passes(self):
        assert is_echo("任何一句话啦", [], similarity=0.85) is None


class TestRenderDedupe:
    def test_repeated_bot_line_kept_once(self):
        """bot 同一句话说了 3 次：context 里只出现最近一次。"""
        mem = ShortTermMemory()
        mem.add_user("在吗", "甲")
        mem.add_bot("杂鱼就是杂鱼")
        mem.add_user("说句话", "乙")
        mem.add_bot("杂鱼就是杂鱼！")   # 归一化后与上一条相同
        mem.add_user("再来", "甲")
        mem.add_bot("杂鱼就是杂鱼~")   # 最近一次：保留
        mem.add_user("好吧", "乙")
        out = mem.render()
        assert out.count("杂鱼就是杂鱼") == 1
        assert "你(历史): 杂鱼就是杂鱼~" in out  # 留的是最近一次

    def test_distinct_bot_lines_all_kept(self):
        mem = ShortTermMemory()
        mem.add_user("在吗", "甲")
        mem.add_bot("在呢")
        mem.add_user("说句话", "乙")
        mem.add_bot("说啥好呢")
        out = mem.render()
        assert "你(历史): 在呢" in out
        assert "你(历史): 说啥好呢" in out

    def test_user_duplicates_untouched(self):
        """去重只针对 bot 自己的行——用户复读是群聊常态，不能动。"""
        mem = ShortTermMemory()
        mem.add_user("复读这句话好吧", "甲")
        mem.add_user("复读这句话好吧", "乙")
        out = mem.render()
        assert out.count("复读这句话好吧") == 2


def _session_with_memory():
    from junjun_core.gateway.session_manager import ChatSession
    session = ChatSession("qq:1:private", "qq", user_id="1")
    session.memory = ShortTermMemory()
    return session


class _ScriptedAgent:
    """按脚本逐次返回文本的假 agent 图。"""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = 0

    async def ainvoke(self, params, config=None):
        out = self.outputs[min(self.calls, len(self.outputs) - 1)]
        self.calls += 1
        return {"messages": [AIMessage(content=out)]}


class TestAgentEchoGuard:
    @pytest.mark.asyncio
    async def test_echo_triggers_retry_with_fresh_text(self, monkeypatch):
        """首轮复读 -> 追问 -> 重说出新话 -> 发新话。"""
        import junjun_agent.agent as agent_mod

        scripted = _ScriptedAgent(["杂鱼就是杂鱼", "好吧好吧，换个话题呗"])
        monkeypatch.setattr(agent_mod.JunJunAgent, "_build_agent",
                            lambda self, full=False: scripted)
        session = _session_with_memory()
        session.memory.add_bot("杂鱼就是杂鱼")
        agent = agent_mod.JunJunAgent(session, model=object())
        out = await agent.process("甲: 说点什么")
        assert out == "好吧好吧，换个话题呗"
        assert scripted.calls == 2  # 确实追问了一轮

    @pytest.mark.asyncio
    async def test_still_echo_silences_when_unaddressed(self, monkeypatch):
        """重说仍复读 + 非必回 -> 沉默，绝不把复读发出去。"""
        import junjun_agent.agent as agent_mod

        scripted = _ScriptedAgent(["杂鱼就是杂鱼", "杂鱼就是杂鱼！"])
        monkeypatch.setattr(agent_mod.JunJunAgent, "_build_agent",
                            lambda self, full=False: scripted)
        session = _session_with_memory()
        session.memory.add_bot("杂鱼就是杂鱼")
        agent = agent_mod.JunJunAgent(session, model=object())
        assert await agent.process("甲: 说点什么", addressed=False) is None

    @pytest.mark.asyncio
    async def test_still_echo_addressed_sends_retry_draft(self, monkeypatch):
        """被@必回：重说仍复读 -> 发重试稿（至少挣扎过一次），不装死。"""
        import junjun_agent.agent as agent_mod

        scripted = _ScriptedAgent(["杂鱼就是杂鱼", "杂鱼就是杂鱼！"])
        monkeypatch.setattr(agent_mod.JunJunAgent, "_build_agent",
                            lambda self, full=False: scripted)
        session = _session_with_memory()
        session.memory.add_bot("杂鱼就是杂鱼")
        agent = agent_mod.JunJunAgent(session, model=object())
        out = await agent.process("甲: @君君 说话", addressed=True)
        assert out == "杂鱼就是杂鱼！"

    @pytest.mark.asyncio
    async def test_catchphrase_triggers_retry(self, monkeypatch):
        """口头禅命中：整句不与任何历史相似，但嵌着近期用滥的词组 -> 追问。"""
        import junjun_agent.agent as agent_mod

        # 新稿和任何一条历史整句都不像，但都嵌着「姐姐疼你」
        scripted = _ScriptedAgent(["姐姐疼你别哭", "我在呢，慢慢说"])
        monkeypatch.setattr(agent_mod.JunJunAgent, "_build_agent",
                            lambda self, full=False: scripted)
        session = _session_with_memory()
        session.memory.add_bot("来，姐姐疼你")
        session.memory.add_bot("今天姐姐疼你一次")
        session.memory.add_bot("姐姐疼你还不乐意？")
        agent = agent_mod.JunJunAgent(session, model=object())
        out = await agent.process("甲: 我心情不好")
        assert out == "我在呢，慢慢说"
        assert scripted.calls == 2

    @pytest.mark.asyncio
    async def test_fresh_text_no_retry(self, monkeypatch):
        """不复读：不触发追问，一轮结束。"""
        import junjun_agent.agent as agent_mod

        scripted = _ScriptedAgent(["今晚月亮真圆啊"])
        monkeypatch.setattr(agent_mod.JunJunAgent, "_build_agent",
                            lambda self, full=False: scripted)
        session = _session_with_memory()
        session.memory.add_bot("杂鱼就是杂鱼")
        agent = agent_mod.JunJunAgent(session, model=object())
        out = await agent.process("甲: 看月亮")
        assert out == "今晚月亮真圆啊"
        assert scripted.calls == 1

    @pytest.mark.asyncio
    async def test_guard_disabled_by_config(self, monkeypatch):
        """[agent] echo_guard=false 时复读也照发（逃生开关）。"""
        import junjun_agent.agent as agent_mod
        import junjun_core.config.config as cfg_mod

        scripted = _ScriptedAgent(["杂鱼就是杂鱼"])
        monkeypatch.setattr(agent_mod.JunJunAgent, "_build_agent",
                            lambda self, full=False: scripted)
        cfg = cfg_mod.get_global_config()
        monkeypatch.setitem(cfg.raw, "agent", {"echo_guard": False})
        session = _session_with_memory()
        session.memory.add_bot("杂鱼就是杂鱼")
        agent = agent_mod.JunJunAgent(session, model=object())
        out = await agent.process("甲: 说点什么")
        assert out == "杂鱼就是杂鱼"
        assert scripted.calls == 1
