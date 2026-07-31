"""意图自检：强意图词命中但对应工具没调 -> 生成系统追问。

背景：弱模型把「帮我盯着xxx」当记忆任务只调 save_memory 或纯口头答应，
动作没生效用户却以为办好了。补救轮比换贵模型便宜一个数量级。
"""

from langchain_core.messages import AIMessage

from junjun_agent.agent import _called_tool_names, _intent_nudge


def _ai_with_tools(*names):
    return AIMessage(content="", tool_calls=[{"name": n, "args": {}, "id": f"t{n}"} for n in names])


ALL_TOOLS = {"subscribe_updates", "unsubscribe", "set_reminder", "save_memory", "do_not_reply"}


class TestIntentNudge:
    def test_subscribe_intent_without_tool_call(self):
        """「帮我盯着p站16689973」只调了 save_memory -> 追问 subscribe_updates。"""
        msgs = [_ai_with_tools("save_memory")]
        nudge = _intent_nudge("@君君 帮我盯着p站作者16689973", msgs, ALL_TOOLS)
        assert nudge and "subscribe_updates" in nudge

    def test_no_nudge_when_tool_called(self):
        """已正确调用 subscribe_updates -> 不追问。"""
        msgs = [_ai_with_tools("subscribe_updates")]
        assert _intent_nudge("帮我盯着p站作者16689973", msgs, ALL_TOOLS) is None

    def test_no_nudge_when_tool_unavailable(self):
        """工具被掩码裁掉 -> 不追问（问了也调不了）。"""
        msgs = [_ai_with_tools("save_memory")]
        assert _intent_nudge("帮我盯着p站16689973", msgs, {"save_memory"}) is None

    def test_unsubscribe_takes_priority(self):
        """「取消订阅」含「订阅」——必须先命中 unsubscribe 规则。"""
        msgs = [_ai_with_tools()]
        nudge = _intent_nudge("取消订阅 3", msgs, ALL_TOOLS)
        assert nudge and "unsubscribe" in nudge

    def test_reminder_intent(self):
        msgs = [_ai_with_tools()]
        nudge = _intent_nudge("明天早上八点提醒我开会", msgs, ALL_TOOLS)
        assert nudge and "set_reminder" in nudge

    def test_no_intent_no_nudge(self):
        msgs = [_ai_with_tools()]
        assert _intent_nudge("今天天气真好", msgs, ALL_TOOLS) is None
        assert _intent_nudge("", msgs, ALL_TOOLS) is None

    def test_called_tool_names(self):
        msgs = [_ai_with_tools("a", "b"), AIMessage(content="你好")]
        assert _called_tool_names(msgs) == {"a", "b"}
