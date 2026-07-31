"""工具掩码的关键词钉住：强触发词命中的工具必须保留在可用集里。

背景：embedding 相关性排序在 LangGraph 工作线程里走不通（get_event_loop
防护），实际生效的是关键词兜底；没有关键词条目的工具（如订阅三件套）
会被掩掉，LLM 拿不到工具只能空口答应——「口头说订好了但库里没有」。
"""

from types import SimpleNamespace

from langchain_core.tools import tool

from junjun_memory.short_term import ShortTermMemory
from junjun_skills import registry


@tool
def subscribe_updates(source: str, target: str) -> str:
    """订阅创作者更新，对方有新作品时你会主动发消息通知。

    Args:
        source: pixiv（P 站小说作者）或 bilibili（B 站 UP 主）
        target: 作者 UID 或 UP 主 mid/昵称
    """
    return "ok"


def _session_with(text: str):
    mem = ShortTermMemory()
    mem.add_user(text, nickname="某人", user_id="1")
    return SimpleNamespace(memory=mem, chat_id="qq:1:group")


def _make_fillers(n: int):
    out = []
    for i in range(n):
        @tool
        def filler(x: str) -> str:
            """凑数工具，和任何话题都无关。

            Args:
                x: 任意输入
            """
            return "ok"
        out.append(filler)
    return out


@tool
def unsubscribe(sub_id: str) -> str:
    """取消订阅。用户说「取消订阅/别盯了」时使用。

    Args:
        sub_id: 订阅编号
    """
    return "ok"


class TestKeywordPinning:
    def test_subscription_pinned_on_trigger_words(self):
        """「帮我盯着p站16689973」→ 订阅工具被钉住，不会被掩掉。"""
        tools = [subscribe_updates] + _make_fillers(20)
        session = _session_with("帮我盯着p站16689973，更新了告诉我")
        kept = registry._mask_by_relevance(tools, session)
        names = {t.name for t in kept}
        assert "subscribe_updates" in names
        # 钉住数量有上限，无关工具被裁掉大部分
        assert len(kept) <= 14

    def test_no_trigger_no_pin(self):
        """无关话题不钉订阅工具。"""
        session = _session_with("今天天气怎么样")
        pinned = registry._pinned_by_keywords([subscribe_updates], "今天天气怎么样")
        assert pinned == []

    def test_unsubscribe_keywords(self):
        """「取消订阅 3」→ unsubscribe 命中（订阅工具顺带命中无妨，取消流程同样需要）。"""
        pinned = registry._pinned_by_keywords([subscribe_updates, unsubscribe], "取消订阅 3")
        assert unsubscribe in pinned
