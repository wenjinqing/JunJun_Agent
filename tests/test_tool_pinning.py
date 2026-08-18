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

    def test_save_file_pinned_on_received_file_words(self):
        t = _named_tool("workspace_save_file")
        for text in ("刚发的文件帮我看看", "这个文件能处理吗", "这个表格统计一下",
                     "这个pdf讲了啥"):
            assert t in registry._pinned_by_keywords([t], text), text

    def test_save_file_no_misfire(self):
        """误判回归：「文件/表格/pdf」不带指向词不钉——「发个文件给他」
        「做个表格」「pdf 是什么」都不是收文件场景。"""
        t = _named_tool("workspace_save_file")
        for text in ("把这个结果做成表格发群里", "帮我整理一份文档",
                     "pdf 和 word 有啥区别", "文件夹怎么整理比较好",
                     "这是我今天的作业"):
            assert registry._pinned_by_keywords([t], text) == [], text


class TestCodeLabPinning20260814:
    """2026-08-14（trace ede13923）：任务全文零命中导致 run_code 被裁——
    「沙箱/图表」入钉词，意图组整组挂载 workspace 六件。"""

    def test_run_code_pinned_on_sandbox_chart_words(self):
        t = _named_tool("run_code")
        for text in ("在沙箱里帮我跑一下", "画个图表看看分布"):
            assert t in registry._pinned_by_keywords([t], text), text

    def test_run_code_new_words_no_misfire(self):
        """误判回归：「沙盒游戏」与日常句不钉。"""
        t = _named_tool("run_code")
        for text in ("这个沙盒游戏真好玩", "今晚吃啥", "我刚看了篇新闻"):
            assert t not in registry._pinned_by_keywords([t], text), text

    def test_intent_mounts_workspace_group(self):
        """意图层：沙箱/数据分析诉求整组挂载（只给 run_code 会「算出来发不出」）。"""
        names = registry._intent_mounted("帮我在沙箱里跑代码，画个趋势图")
        for n in ("run_code", "workspace_write", "workspace_read",
                  "workspace_list", "workspace_save_file", "workspace_send"):
            assert n in names, n

    def test_intent_no_misfire_daily(self):
        """误判回归：日常句/插画诉求不挂 code-lab 组。"""
        for text in ("今天好累", "给我画一张猫娘", "提醒我开会",
                     "这个沙盒游戏不错", "查一下天气"):
            names = registry._intent_mounted(text)
            assert "run_code" not in names, text


class TestCatcafeMounting20260818:
    """2026-08-18：小涩猫咖啡厅站点管理接入——站点词整组挂载 catcafe 五件；
    primary=None 只挂载不追问（「咖啡厅/官网」日常高频，追问会抢管闲事）。"""

    def test_intent_mounts_catcafe_group(self):
        for text in ("帮我在咖啡厅发个公告", "小涩猫官网最近咋样",
                     "站点 slogan 换一下", "更新网站上说一声"):
            names = registry._intent_mounted(text)
            for n in ("catcafe_get_content", "catcafe_get_stats",
                      "catcafe_post_notice", "catcafe_set_slogan",
                      "catcafe_set_status"):
                assert n in names, (text, n)

    def test_intent_no_nudge_primary(self):
        """catcafe 组 primary 必须是 None：日常词命中也不产生意图自检追问。"""
        for _keywords, group, primary in registry.intent_groups():
            if any(n.startswith("catcafe_") for n in group):
                assert primary is None

    def test_intent_no_misfire_daily(self):
        """误判回归：裸「网站」「群公告」「咖啡店」类日常句不挂 catcafe 组。"""
        for text in ("这个网站打不开了", "群公告在哪看", "今晚吃啥",
                     "帮我搜一下附近的咖啡店", "提醒我开会"):
            names = registry._intent_mounted(text)
            assert not any(n.startswith("catcafe_") for n in names), text


class TestPeriodicPushIntent20260815:
    """2026-08-15（eval daily-tech-news 实锤）：「以后每天早上给我推科技新闻」
    式周期推送零命中——不含「提醒」二字，set_reminder 被掩码裁掉，模型空口
    「没这个功能」。词形带「给我」锚点防日常句误伤。"""

    def test_periodic_push_mounts_reminder_group(self):
        for text in ("以后每天早上给我推一下当天的科技新闻",
                     "每天晚上给我讲个睡前故事",
                     "每天给我推送群活跃总结"):
            names = registry._intent_mounted(text)
            assert "set_reminder" in names, text

    def test_daily_sentences_no_misfire(self):
        """误判回归：含「每天早/晚」但不是请求的日常句不挂提醒组。"""
        for text in ("他每天早上都来得很早", "我每天晚上都熬夜",
                     "今天好累", "明天早上吃什么好呢", "早上好"):
            names = registry._intent_mounted(text)
            assert "set_reminder" not in names, text


class TestIdentityIntent20260815:
    """2026-08-15（生产实锤）：「你是谁/你基于什么模型」零意图命中——
    introduce_self 不在 CORE、无组无钉词，被掩码裁掉后模型凭印象编出
    「混元大模型」+「框架细节不清楚」。身份组整组挂载给事实锚；
    primary=None 只挂不追问（群里「你是谁」可能是群友问新人，追问=抢答）。"""

    def test_identity_questions_mount_intro(self):
        for text in ("你是谁啊", "介绍一下你自己", "你是什么模型",
                     "谁开发的你", "你的技术栈是啥", "你会什么", "有什么功能",
                     "君君你都会什么呀", "你都会些啥", "你能干啥"):
            names = registry._intent_mounted(text)
            assert "introduce_self" in names, text
            assert "get_capabilities" in names, text

    def test_daily_sentences_no_misfire(self):
        """误判回归：日常句/技术讨论句不挂身份组（挂载虽无追问，也别白绑）。"""
        for text in ("你是做什么工作的", "我是谁我在哪", "你有什么事吗",
                     "这个模型效果不错", "今晚吃啥", "他每天都在群里吹牛"):
            names = registry._intent_mounted(text)
            assert "introduce_self" not in names, text

    def test_no_primary_no_forced_nudge(self):
        """身份组 primary 必须为 None：「你是谁」可能是群友互问，
        追问会逼着 bot 抢答自我介绍。"""
        for kws, _group, primary in registry.intent_groups():
            if "你是谁" in kws:
                assert primary is None
                return
        raise AssertionError("身份意图组不存在")

    def test_topic_keywords_pin_intro(self):
        """embedding 降级路径：钉词兜底也能钉住 introduce_self。"""
        t = _named_tool("introduce_self")
        assert t in registry._pinned_by_keywords([t], "你是什么模型")
