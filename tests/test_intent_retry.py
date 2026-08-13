"""意图自检：强意图词命中但对应工具没调 -> 生成系统追问。

背景：弱模型把「帮我盯着xxx」当记忆任务只调 save_memory 或纯口头答应，
动作没生效用户却以为办好了。补救轮比换贵模型便宜一个数量级。
"""

from langchain_core.messages import AIMessage

import pytest

from junjun_agent.agent import _called_tool_names, _intent_nudge


def _ai_with_tools(*names):
    return AIMessage(content="", tool_calls=[{"name": n, "args": {}, "id": f"t{n}"} for n in names])


ALL_TOOLS = {"subscribe_updates", "unsubscribe", "set_reminder", "save_memory", "do_not_reply"}


class TestIntentNudge:
    def test_subscribe_intent_without_tool_call(self):
        """「帮我盯着p站16689973」只调了 save_memory -> 追问 subscribe_updates。"""
        msgs = [_ai_with_tools("save_memory")]
        nudge = _intent_nudge("@君君 帮我盯着p站作者16689973", msgs, ALL_TOOLS)
        assert nudge and "subscribe_updates" in nudge[0]
        assert nudge[1] is False  # 工具可用，普通补救轮

    def test_no_nudge_when_tool_called(self):
        """已正确调用 subscribe_updates -> 不追问。"""
        msgs = [_ai_with_tools("subscribe_updates")]
        assert _intent_nudge("帮我盯着p站作者16689973", msgs, ALL_TOOLS) is None

    def test_full_bind_when_tool_masked_out(self):
        """工具被掩码裁掉（漏绑）-> 追问 + 全绑补救轮（P5-2 兜底）。

        2026-08-01 实战：模型被追问一个没绑定的工具，如实答「没有这个工具」。
        """
        msgs = [_ai_with_tools("save_memory")]
        nudge = _intent_nudge("帮我盯着p站16689973", msgs, {"save_memory"})
        assert nudge and "subscribe_updates" in nudge[0]
        assert nudge[1] is True  # full_bind：补救轮用全量工具重建 agent

    def test_unsubscribe_takes_priority(self):
        """「取消订阅」含「订阅」——必须先命中 unsubscribe 规则。"""
        msgs = [_ai_with_tools()]
        nudge = _intent_nudge("取消订阅 3", msgs, ALL_TOOLS)
        assert nudge and "unsubscribe" in nudge[0]

    def test_reminder_intent(self):
        msgs = [_ai_with_tools()]
        nudge = _intent_nudge("明天早上八点提醒我开会", msgs, ALL_TOOLS)
        assert nudge and "set_reminder" in nudge[0]

    def test_research_intent_goes_background(self):
        """「调研」「深研」命中深度研究规则：只做了内联搜索 -> 追问 deep_research。"""
        tools = ALL_TOOLS | {"deep_research"}
        msgs = [_ai_with_tools("web_search")]
        nudge = _intent_nudge("帮我调研一下绝区零丹的攻略", msgs, tools)
        assert nudge and "deep_research" in nudge[0]
        # 2026-08-01 trace：用户口语缩写「深研」未命中关键词，当场内联查完
        nudge = _intent_nudge("帮我深研一下绝区零丹的配队", msgs, tools)
        assert nudge and "deep_research" in nudge[0]

    def test_research_no_nudge_when_submitted(self):
        tools = ALL_TOOLS | {"deep_research"}
        msgs = [_ai_with_tools("deep_research")]
        assert _intent_nudge("帮我调研一下绝区零丹的攻略", msgs, tools) is None

    def test_quick_lookup_no_nudge(self):
        """快查（查天气/搜快讯）不触发深度研究规则。"""
        tools = ALL_TOOLS | {"deep_research"}
        msgs = [_ai_with_tools("web_search")]
        assert _intent_nudge("帮我查下明天天气", msgs, tools) is None
        assert _intent_nudge("绝区零丹怎么配队", msgs, tools) is None

    def test_no_intent_no_nudge(self):
        msgs = [_ai_with_tools()]
        assert _intent_nudge("今天天气真好", msgs, ALL_TOOLS) is None
        assert _intent_nudge("", msgs, ALL_TOOLS) is None

    def test_group_nsfw_draw_intent_not_nudged(self):
        """群聊涩图意图不追问（2026-08-06 实锤）：模型不调 ai_draw 是政策正确，
        追问等于把它往违规上推。私聊照常追问。"""
        tools = ALL_TOOLS | {"ai_draw"}
        msgs = [_ai_with_tools()]  # 模型没调任何工具
        assert _intent_nudge("君君来张涩图", msgs, tools, is_group=True) is None
        assert _intent_nudge("给我画个色图", msgs, tools, is_group=True) is None
        # 私聊：意图机制照常工作（NSFW 直通道漏网的措辞靠它兜底）
        nudge = _intent_nudge("帮我画张涩图", msgs, tools, is_group=False)
        assert nudge and "ai_draw" in nudge[0]
        # 群里普通画图意图不受影响
        nudge = _intent_nudge("帮我画一只猫", msgs, tools, is_group=True)
        assert nudge and "ai_draw" in nudge[0]

    def test_called_tool_names(self):
        msgs = [_ai_with_tools("a", "b"), AIMessage(content="你好")]
        assert _called_tool_names(msgs) == {"a", "b"}


class TestSearchIntent:
    """搜索意图自检（2026-08-07 实锤：「上网查一下绝区零配队排名」一轮没调
    工具就空口答 + recursion 后「网有点抽」搪塞——搜索意图此前不在自检清单，
    persona 劝告压不过弱模型的懒惰）。"""

    def test_search_intent_nudged(self):
        tools = ALL_TOOLS | {"web_search"}
        msgs = [_ai_with_tools()]
        nudge = _intent_nudge("上网查一下绝区零当前配队强度排名", msgs, tools)
        assert nudge and "web_search" in nudge[0]

    def test_search_no_nudge_when_called(self):
        tools = ALL_TOOLS | {"web_search"}
        msgs = [_ai_with_tools("web_search")]
        assert _intent_nudge("帮我查一下明天天气", msgs, tools) is None
        assert _intent_nudge("搜一下今天有什么科技新闻", msgs, tools) is None

    def test_no_search_intent_no_nudge(self):
        tools = ALL_TOOLS | {"web_search"}
        msgs = [_ai_with_tools()]
        assert _intent_nudge("今天天气真好", msgs, tools) is None


class TestGroupEvidence:
    """证据语义：组内行动工具任一被调即算办过；list_* 只读不算。"""

    def test_background_task_counts_for_research(self):
        """「调研」派了 run_background_task 就是办成了，不再死磕 deep_research。"""
        tools = ALL_TOOLS | {"deep_research", "run_background_task"}
        msgs = [_ai_with_tools("run_background_task")]
        assert _intent_nudge("帮我调研一下绝区零丹的攻略", msgs, tools) is None

    def test_list_tool_not_evidence(self):
        """只调 list_reminders 查看不算办了「提醒我」这件事。"""
        msgs = [_ai_with_tools("list_reminders")]
        nudge = _intent_nudge("明天早上八点提醒我开会", msgs, ALL_TOOLS)
        assert nudge and "set_reminder" in nudge[0]


class TestAgentRebuild:
    @pytest.mark.asyncio
    async def test_agent_rebuilds_tools_each_round(self, monkeypatch):
        """回归（2026-08-01 trace）：agent 图必须每轮重建。

        曾经构造时绑死工具集——此时 memory 为空，关键词钉不住，
        run_background_task 等非 CORE 工具被裁后整个会话不可用，
        意图自检却按实时掩码追问 -> 模型被追问一个没绑定的工具。
        """
        import junjun_agent.agent as agent_mod
        from langchain_core.language_models.fake_chat_models import (
            FakeMessagesListChatModel)
        from junjun_core.gateway.session_manager import ChatSession

        class _BindableFake(FakeMessagesListChatModel):
            def bind_tools(self, tools, **kwargs):
                return self

        calls = []
        real_get_tools = agent_mod.get_tools

        def counting(session=None):
            calls.append(1)
            return real_get_tools(session)
        monkeypatch.setattr(agent_mod, "get_tools", counting)

        session = ChatSession("qq:1:private", "qq", user_id="1")
        agent = agent_mod.JunJunAgent(
            session, model=_BindableFake(responses=[AIMessage(content="好")]))
        assert calls == []  # 构造时不再绑工具
        await agent.process("甲: 你好")
        assert calls  # process 时按当前会话状态实时构建

    def test_full_bind_rebuild_uses_all_tools(self, monkeypatch):
        """漏绑补救轮：_build_agent(full=True) 用全量工具（get_tools 不带 session）。"""
        import junjun_agent.agent as agent_mod
        from junjun_core.gateway.session_manager import ChatSession

        sessions = []
        real_get_tools = agent_mod.get_tools

        def counting(session=None):
            sessions.append(session)
            return real_get_tools(session)
        monkeypatch.setattr(agent_mod, "get_tools", counting)
        monkeypatch.setattr(agent_mod, "create_agent",
                            lambda model, tools, middleware=None: object())

        session = ChatSession("qq:1:private", "qq", user_id="1")
        agent = agent_mod.JunJunAgent(session, model=object())
        agent._build_agent(full=True)
        assert sessions and sessions[-1] is None  # 全量
        agent._build_agent()
        assert sessions[-1] is session            # 按会话掩码


class TestAddressedFallback:
    @pytest.mark.asyncio
    async def test_addressed_exception_gets_fallback(self, monkeypatch):
        """被 @ 时 agent 炸了（含 recursion limit）回实话，不装死。"""
        import junjun_agent.agent as agent_mod
        from junjun_core.gateway.session_manager import ChatSession

        class _BoomAgent:
            async def ainvoke(self, *a, **kw):
                raise RuntimeError("Recursion limit reached")

        monkeypatch.setattr(agent_mod.JunJunAgent, "_build_agent",
                            lambda self, **_kw: _BoomAgent())
        session = ChatSession("qq:1:private", "qq", user_id="1")
        agent = agent_mod.JunJunAgent(session, model=object())
        out = await agent.process("甲: 帮我盯着p站16689973", addressed=True)
        assert out and "没办成" in out

    @pytest.mark.asyncio
    async def test_unaddressed_exception_stays_silent(self, monkeypatch):
        """未被 @ 时保持沉默（不炸会话）。"""
        import junjun_agent.agent as agent_mod
        from junjun_core.gateway.session_manager import ChatSession

        class _BoomAgent:
            async def ainvoke(self, *a, **kw):
                raise RuntimeError("Recursion limit reached")

        monkeypatch.setattr(agent_mod.JunJunAgent, "_build_agent",
                            lambda self, **_kw: _BoomAgent())
        session = ChatSession("qq:1:group", "qq", group_id="1")
        agent = agent_mod.JunJunAgent(session, model=object())
        assert await agent.process("甲: 随便聊聊", addressed=False) is None


class TestAddressedNoSilence:
    """必回场景结构性摘除 do_not_reply（2026-08-04 trace：管理员私聊 + addressed=true，
    prompt 明写「禁止调用 do_not_reply」，模型照调、代码照吞，output=null）。
    prompt 劝告管不住的，工具集里直接没有这个工具才管得住。"""

    def test_silence_tool_stripped_when_addressed(self, monkeypatch):
        import junjun_agent.agent as agent_mod
        from junjun_core.gateway.session_manager import ChatSession

        captured = {}

        def _fake_create_agent(model, tools, middleware):
            captured["tools"] = tools
            return object()
        monkeypatch.setattr(agent_mod, "create_agent", _fake_create_agent)

        session = ChatSession("qq:1:group", "qq", group_id="1")
        agent = agent_mod.JunJunAgent(session, model=object())

        agent._build_agent(full=True, allow_silence=False)
        names = {t.name for t in captured["tools"]}
        assert "do_not_reply" not in names
        assert "send_message" in names          # 其余工具不受影响

        agent._build_agent(full=True, allow_silence=True)
        assert "do_not_reply" in {t.name for t in captured["tools"]}


class TestBackgroundRoleStructure:
    """严厉审查 P0-3：bot 历史发言必须以 AIMessage 进 context（role 边界是
    防自我模仿的第一道结构防线，混在 user 文本流里只剩「语气模仿」一条路）。"""

    def test_bot_lines_become_ai_messages(self):
        from langchain_core.messages import AIMessage, HumanMessage
        from junjun_agent.agent import _background_to_messages
        bg = "甲: 今天吃啥\n你(历史): 火锅啊火锅\n乙: 又火锅\n你(历史): 怎么了嘛\n多行续行"
        msgs = _background_to_messages(bg)
        roles = [type(m).__name__ for m in msgs]
        assert roles == ["HumanMessage", "AIMessage", "HumanMessage", "AIMessage"]
        assert msgs[1].content == "火锅啊火锅"
        assert "你(历史)" not in msgs[1].content     # 前缀剥离
        assert "多行续行" in msgs[3].content          # 续行并入上一条

    def test_user_only_background(self):
        from langchain_core.messages import HumanMessage
        from junjun_agent.agent import _background_to_messages
        msgs = _background_to_messages("甲: 在吗\n乙: 他不在")
        assert len(msgs) == 1 and isinstance(msgs[0], HumanMessage)


class TestCodeLabIntent:
    """2026-08-14 trace ede13923 实锤：数据分析/沙箱诉求此前无意图组，
    run_code 被掩码裁掉后模型抓 tavily_extract 冒充「跑代码」——
    意图组整组挂载 + 自检追问 + 漏绑全绑补救三件套兜底。"""

    def test_sandbox_intent_nudged(self):
        msgs = [_ai_with_tools("web_search")]
        nudge = _intent_nudge("帮我在沙箱里跑代码画个趋势图", msgs, ALL_TOOLS)
        assert nudge and "run_code" in nudge[0]

    def test_full_bind_when_run_code_masked(self):
        """run_code 被裁掉 -> 追问 + 全绑补救（不再让模型对着空工具带找工具）。"""
        msgs = [_ai_with_tools("web_search")]
        nudge = _intent_nudge("这份数据帮我做个数据分析，画个占比图", msgs,
                              {"web_search"})
        assert nudge and "run_code" in nudge[0]
        assert nudge[1] is True

    def test_chart_words_beat_ai_draw(self):
        """「画个趋势图/图表」是数据图表不是插画——code-lab 组必须先命中。"""
        msgs = [_ai_with_tools()]
        nudge = _intent_nudge("画个趋势图给我看看这个月的数据", msgs, ALL_TOOLS)
        assert nudge and "run_code" in nudge[0]

    def test_no_nudge_when_workspace_tool_called(self):
        """组内任一行动工具调用即算证据（先收文件再跑码是正常链路）。"""
        msgs = [_ai_with_tools("workspace_save_file")]
        assert _intent_nudge("在沙箱里跑代码处理这个表格", msgs, ALL_TOOLS) is None

    def test_no_nudge_on_unrelated(self):
        """误判回归：日常句不触发 code-lab 意图。"""
        msgs = [_ai_with_tools()]
        for text in ("今天好累", "给我画一张猫娘", "这个沙盒游戏不错",
                     "提醒我下午开会", "这个视频讲了啥"):
            nudge = _intent_nudge(text, msgs, ALL_TOOLS)
            assert not (nudge and "run_code" in nudge[0]), text
