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


def _named_tool(name):
    @tool(name)
    def _t(x: str) -> str:
        """凑数工具。

        Args:
            x: 任意输入
        """
        return "ok"
    return _t


class TestThreeLayerSubset:
    """P5-2 三层工具子集：CORE 瘦身 / INTENT 整组挂载 / 稳定序。"""

    def test_core_slimmed_to_eight(self):
        """CORE ≤9：原 CORE 的 set_reminder/ai_draw/send_feed 等无话题时不常驻。
        （2026-08-04 第 9 席给 use_skill：prompt 技能包索引引用了它，
        必须常驻防「索引指向被掩码工具」陷阱。）"""
        ex_core = [_named_tool(n) for n in
                   ("set_reminder", "list_reminders", "ai_draw", "unified_tts",
                    "ja_tts", "send_feed", "read_feed", "find_user_id")]
        core_now = [_named_tool(n) for n in registry._CORE_TOOLS]
        # 凑数工具在前：同分（0）时稳定序让它们占满补位名额，隔离 CORE 常驻性验证
        tools = _make_fillers(20) + core_now + ex_core
        session = _session_with("今天天气真好啊")
        kept = registry._mask_by_relevance(tools, session)
        names = {t.name for t in kept}
        assert len(registry._CORE_TOOLS) <= 9
        for n in registry._CORE_TOOLS:
            assert n in names  # CORE 永不掩码
        for n in ("set_reminder", "ai_draw", "send_feed"):
            assert n not in names  # 旧 CORE 无话题时被裁

    def test_intent_mounts_whole_group(self):
        """INTENT 层：「明天提醒我开会」-> 提醒三件套整组挂载（cancel 无关键词也带上）。"""
        trio = [_named_tool(n) for n in
                ("set_reminder", "list_reminders", "cancel_reminder_task")]
        tools = trio + _make_fillers(20)
        session = _session_with("明天早上八点提醒我开会")
        kept = registry._mask_by_relevance(tools, session)
        names = {t.name for t in kept}
        assert {"set_reminder", "list_reminders", "cancel_reminder_task"} <= names

    def test_intent_no_hit_no_mount(self):
        """闲聊不挂意图组。"""
        assert registry._intent_mounted("今天天气真好") == []
        assert registry._intent_mounted("") == []

    def test_canonical_order_stable(self):
        """稳定序：同一子集任意输入顺序 -> 相同输出（CORE 固定序 + 其余字典序）。"""
        a, b, c = _named_tool("ai_draw"), _named_tool("get_time"), _named_tool("web_search")
        o1 = [t.name for t in registry._canonical_order([a, b, c])]
        o2 = [t.name for t in registry._canonical_order([c, a, b])]
        assert o1 == o2 == ["get_time", "web_search", "ai_draw"]  # CORE 在前


class TestPhase2ToolPinning:
    """Phase 2 工具域的关键词钉住 + 误判回归（铁律：加宽命中面同 commit 配误判断言）。"""

    def test_run_code_pinned_on_strong_words(self):
        t = _named_tool("run_code")
        for text in ("把今天聊天记录做成词云", "帮我算一下这个月的开销",
                     "用工作区数据画个趋势图", "这个 csv 帮我统计一下"):
            assert t in registry._pinned_by_keywords([t], text), text

    def test_fetch_page_pinned_on_article_words(self):
        t = _named_tool("fetch_page")
        for text in ("看看这篇文章讲了啥", "这个网页里有写吗", "这个链接的内容帮我读一下"):
            assert t in registry._pinned_by_keywords([t], text), text

    def test_daily_sentences_no_misfire(self):
        """误判回归：日常句子不许钉住 run_code/fetch_page——
        「数据」「统计」「链接」单拎出来都太宽泛，刻意不收。"""
        rc, fp = _named_tool("run_code"), _named_tool("fetch_page")
        for text in ("今天群里讨论数据安全呢", "据统计局发布的消息",
                     "这个数据说房价又涨了", "这个表情包链接发我看看",
                     "我刚看了篇新闻", "今晚吃啥"):
            pinned = registry._pinned_by_keywords([rc, fp], text)
            assert pinned == [], text
