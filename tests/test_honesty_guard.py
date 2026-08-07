"""HonestyGuard 单元测试（Phase 3）。"""

from types import SimpleNamespace

import pytest

from junjun_agent.honesty_guard import (
    record_tool_call, start_decision, verify, _CLAIM_RULES,
)


@pytest.fixture
def session():
    return SimpleNamespace(chat_id="qq:1:group")


def test_no_claim_no_issue(session):
    start_decision(session)
    ok, text, issues = verify(session, "今天天气不错")
    assert ok
    assert text == "今天天气不错"
    assert issues == []


def test_block_draw_claim_without_tool(session):
    start_decision(session)
    ok, text, issues = verify(session, "画好了，等下发给你")
    assert not ok
    assert "ai_draw" in " ".join(issues)
    # 修正稿引用模型实际说出口的短语，不是 regex 源码（2026-08-06 实锤：
    # 用户收到『画[好了|完了|出来]』一脸懵）
    assert "画好" in text
    assert "[" not in text and "系统拦住我了" not in text
    assert "不能骗你" in text
    # 不许承诺「我重新来」——守卫自己不重试，空头承诺恰是它在防的不诚实
    assert "我重新来" not in text


def test_allow_draw_claim_with_tool(session):
    start_decision(session)
    record_tool_call(session, "ai_draw", result="图片任务已接受")
    ok, text, issues = verify(session, "画好了，等下发给你")
    assert ok
    assert text == "画好了，等下发给你"


def test_block_feed_claim(session):
    start_decision(session)
    ok, text, issues = verify(session, "说说已经发好了")
    assert not ok
    assert any("send_feed" in i for i in issues)


def test_allow_feed_claim_with_tool(session):
    start_decision(session)
    record_tool_call(session, "send_feed", result="说说已发布")
    ok, text, issues = verify(session, "说说已经发好了")
    assert ok


def test_only_recent_decision_tools_count(session):
    start_decision(session)
    record_tool_call(session, "ai_draw", result="old")
    # 模拟新一轮决策
    start_decision(session)
    ok, text, issues = verify(session, "画好了")
    assert not ok


def test_old_tool_calls_pruned_but_not_current(session):
    start_decision(session)
    record_tool_call(session, "ai_draw", result="new")
    ok, _, _ = verify(session, "画好了")
    assert ok


class TestCharClassLandmines:
    """字符组误写回归（2026-08-06 实锤）：[a|b] 是字符组不是交替，
    单字命中导致日常表达被当成行为声称误拦。"""

    @pytest.mark.parametrize("innocent", [
        "跟我说说发生了什么",          # 「说说发」≠ 发说说声称
        "帮我画出这道数学题的思路",      # 「画出」≠ 画好了
        "这种事已经发生过了",           # 「已经发」≠ 已经发了
        "语音发不出去啊怎么办",         # 「语音发」≠ 语音已发
        "消息发不出去，急",            # 「消息发」≠ 消息已发
        "你画画了吗",                  # 疑问不是声称
        "他已经取消了行程",            # 第三方取消 ≠ 我取消订阅
    ])
    def test_innocent_phrases_not_intercepted(self, session, innocent):
        start_decision(session)
        ok, text, issues = verify(session, innocent)
        assert ok, f"日常表达被误拦: {innocent} -> {issues}"

    @pytest.mark.parametrize("claim,tool", [
        ("画好了，等下给你", "ai_draw"),
        ("在画了在画了", "ai_draw"),       # 进行中声称也要证据
        ("等下就发给你", "ai_draw"),
        ("语音发好了", "unified_tts"),
        ("说说发了，去看", "send_feed"),
        ("提醒设好了", "set_reminder"),
        ("订阅好了", "subscribe_updates"),
    ])
    def test_real_claims_still_caught(self, session, claim, tool):
        start_decision(session)
        ok, _, issues = verify(session, claim)
        assert not ok, f"真声称漏拦: {claim}"
        assert tool in " ".join(issues)


class _ScriptedGraph:
    """按脚本逐次返回消息列表的假 agent 图。"""

    def __init__(self, runs):
        self.runs = list(runs)
        self.calls = 0

    async def ainvoke(self, params, config=None):
        run = self.runs[min(self.calls, len(self.runs) - 1)]
        self.calls += 1
        return {"messages": run}


class TestRetryPathToolLedger:
    """2026-08-06 生产误拦回归：意图补救轮真调了 ai_draw，台账却只记首轮
    （空的）——诚实的「在画了」被 HonestyGuard 换成系统腔替换稿。"""

    @pytest.mark.asyncio
    async def test_intent_retry_tool_calls_enter_ledger(self, monkeypatch):
        import junjun_agent.agent as agent_mod
        from langchain_core.messages import AIMessage, ToolMessage
        from junjun_core.gateway.session_manager import ChatSession
        from junjun_memory.short_term import ShortTermMemory

        first = [AIMessage(content="好，画好了给你")]  # 首轮嘴炮没调工具
        second = [
            AIMessage(content="", tool_calls=[
                {"name": "ai_draw", "args": {"prompt": "猫"}, "id": "t1"}]),
            ToolMessage(content="图片任务已接受，正在生成",
                        tool_call_id="t1", name="ai_draw"),
            AIMessage(content="在画了，等下发出来"),
        ]
        scripted = _ScriptedGraph([first, second])
        monkeypatch.setattr(agent_mod.JunJunAgent, "_build_agent",
                            lambda self, full=False, **_kw: scripted)
        session = ChatSession("qq:1:private", "qq", user_id="1")
        session.memory = ShortTermMemory()
        agent = agent_mod.JunJunAgent(session, model=object())
        out = await agent.process("甲: 帮我画只猫", latest_text="帮我画只猫",
                                  addressed=True)
        assert out == "在画了，等下发出来"
        assert scripted.calls == 2  # 意图追问了一轮
        ok, _, issues = verify(session, out)
        assert ok, f"补救轮真调了 ai_draw 却被误拦: {issues}"


class TestNovelSentClaims:
    """「发你了《X》/抓取完成/txt 已发」是 pixiv 小说命令回执的专有措辞
    （2026-08-07 实锤：私聊裸链接根本没抓，LLM 模仿上文命令回执
    5 秒报出上一篇标题「发你了，《奴隶神明的最后一天》」）。"""

    @pytest.mark.parametrize("claim", [
        "发你了，《奴隶神明的最后一天》",
        "发你了,《某本小说》",
        "《某某》抓取完成，txt 已发你～",
        "《某某》已发你，查收",
        "txt 已发你邮箱……啊不，私聊",
    ])
    def test_novel_claims_blocked(self, session, claim):
        start_decision(session)
        ok, text, issues = verify(session, claim)
        assert not ok, f"空口小说发送声称漏拦: {claim}"
        assert any("pixiv_download_novel" in i for i in issues)
        assert "不能骗你" in text

    def test_real_download_passes(self, session):
        """真调了 pixiv_download_novel（工具回执本身就是这句模板）不拦。"""
        start_decision(session)
        record_tool_call(session, "pixiv_download_novel",
                         result="《某某》抓取完成，txt 已发你～")
        ok, text, _ = verify(session, "《某某》抓取完成，txt 已发你～")
        assert ok

    @pytest.mark.parametrize("innocent", [
        "我记得《三体》你说过好看",
        "这本《活着》推荐给你，链接在这",
        "《三体》吗？我查查再跟你说",
        "上次那本你看完了吗",
    ])
    def test_innocent_book_mentions_pass(self, session, innocent):
        start_decision(session)
        ok, _, issues = verify(session, innocent)
        assert ok, f"书名讨论被误拦: {innocent} -> {issues}"


class TestFailedToolResultsNotEvidence:
    """失败/拒绝的工具调用不能当「已完成」的证据（2026-08-06 审查实锤：
    台账认名不认结果，群聊拒画也放行「在画了」——防线在最需要它的场景失效）。"""

    @pytest.mark.parametrize("refusal", [
        "群里画不了这种（公共场合 + 账号风控）。笑着让对方私聊你——照这个意思回他，别派单。",
        "[TOOL_ERROR] ai_draw: ModelScopeError 余额不足",
        "拒绝：描述涉及未成年人性内容，不会生成。",
        "上一个还在弄呢，等下吧。",
        "这次画失败了，再试一次？",
        "我这会儿手头的活排满了，等一个弄完了再帮你弄。",
        "画图功能未配置 MODELSCOPE_API_KEY，暂时画不了。",
    ])
    def test_failed_or_refused_call_does_not_unlock_claim(self, session, refusal):
        start_decision(session)
        record_tool_call(session, "ai_draw", result=refusal)
        ok, _, issues = verify(session, "在画了，等下发给你")
        assert not ok, f"拒画/失败结果被当成证据放行: {refusal[:20]}"
        assert any("ai_draw" in i for i in issues)

    def test_success_result_still_counts(self, session):
        start_decision(session)
        record_tool_call(session, "ai_draw", result="在弄了，好了直接发出来。")
        ok, _, _ = verify(session, "在画了，等下发给你")
        assert ok
