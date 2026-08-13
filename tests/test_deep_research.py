"""深度研究流水线测试：规划解析/检索去重/读全文降级/综述/工具入口。

搜索/读全文/LLM 全部打桩（handler 依赖注入），DB 用内存库隔离。
"""

import pytest
from peewee import SqliteDatabase

import junjun_core.config.config as cfg_mod
from junjun_core.database import models as m
from junjun_skills.plugins.async_task import research

test_db = SqliteDatabase(":memory:")

CHAT = "qq:12345:group"


def _set_config(raw: dict):
    cfg_mod.global_config = cfg_mod.GlobalConfig(
        bot=cfg_mod.BotConfig(platform="qq", qq_account="12345", nickname="君君"),
        raw=raw)


@pytest.fixture
def env():
    old = cfg_mod.global_config
    _set_config({"async_task": {"enable": True, "max_concurrent": 2,
                                "job_timeout_seconds": 600, "max_pending_per_chat": 2,
                                "report_max_chars": 1500},
                 "deep_research": {"queries": 3, "pages_per_query": 2,
                                   "fetch_max_chars": 100, "report_max_chars": 500}})
    with test_db.bind_ctx([m.AsyncJob]):
        test_db.create_tables([m.AsyncJob])
        m.AsyncJob.delete().execute()
        yield
    cfg_mod.global_config = old


class _FakeModel:
    """ainvoke 直出固定 content 的假 LLM。"""

    def __init__(self, content):
        self._content = content
        self.prompts = []

    async def ainvoke(self, messages, config=None):
        self.prompts.append(str(messages[-1].content))
        return type("R", (), {"content": self._content})()


class TestParseQueries:
    def test_valid_json(self, env):
        assert research._parse_queries('["q1", "q2"]', "t", 5) == ["q1", "q2"]

    def test_prose_with_json(self, env):
        """模型前后啰嗦时抠出第一个 JSON 数组。"""
        raw = '好的，查询如下：\n["q1", "q2"]\n以上。'
        assert research._parse_queries(raw, "t", 5) == ["q1", "q2"]

    def test_garbage_falls_back_to_topic(self, env):
        assert research._parse_queries("我不知道", "主题本身", 5) == ["主题本身"]
        assert research._parse_queries("", "主题本身", 5) == ["主题本身"]
        assert research._parse_queries("[1, null]", "主题本身", 5) == ["1"]

    def test_truncates_to_n(self, env):
        raw = '["q1","q2","q3","q4","q5"]'
        assert research._parse_queries(raw, "t", 2) == ["q1", "q2"]


class TestPlan:
    @pytest.mark.asyncio
    async def test_plan_uses_model(self, env):
        model = _FakeModel('["A 现状", "A 对比", "A 进展"]')
        queries = await research._plan("研究A", model)
        assert queries == ["A 现状", "A 对比", "A 进展"]
        assert "研究A" in model.prompts[0]

    @pytest.mark.asyncio
    async def test_plan_model_crash_falls_back(self, env):
        class _Boom:
            async def ainvoke(self, *a, **kw):
                raise RuntimeError("api down")
        assert await research._plan("研究A", _Boom()) == ["研究A"]


class TestFetchToolSelection:
    """2026-08-14：fetch MCP（mcp-server-fetch 无 SSRF 防护，模型曾拿它探
    hallucinated localhost 端点）默认禁用后，全文读取必须优先走自带
    SSRF 防护的 fetch_page，全无时降级空串不炸流水线。"""

    @staticmethod
    def _register(name):
        from langchain_core.tools import tool
        from junjun_skills import registry

        @tool(name)
        def _t(x: str = "") -> str:
            """测试工具。

            Args:
                x: 输入
            """
            return "ok"

        registry.register(_t)

    def test_prefers_fetch_page(self):
        self._register("mcp_fetch")
        self._register("fetch_page")
        assert research._fetch_tool().name == "fetch_page"

    def test_mcp_fetch_as_fallback(self):
        self._register("mcp_fetch")
        assert research._fetch_tool().name == "mcp_fetch"

    @pytest.mark.asyncio
    async def test_none_degrades_empty(self, monkeypatch):
        import junjun_skills.registry as reg
        monkeypatch.setattr(reg, "get_tools", lambda *a, **kw: [])
        assert research._fetch_tool() is None
        assert await research._default_fetch("http://example.com", 100) == ""


class TestCollect:
    @pytest.mark.asyncio
    async def test_dedupe_by_url_and_fetch(self, env):
        pages = {
            "q1": [{"title": "甲", "url": "http://a", "snippet": "sa"},
                   {"title": "乙", "url": "http://b", "snippet": "sb"},
                   {"title": "丙", "url": "http://c", "snippet": "sc"}],  # 超出 pages_per_query=2 被丢
            "q2": [{"title": "乙2", "url": "http://b", "snippet": "sb2"},  # 与 q1 重复
                   {"title": "丁", "url": "http://d", "snippet": "sd"}],
        }

        async def fake_search(q, num):
            return pages[q]

        async def fake_fetch(url, max_chars):
            return f"全文:{url}"

        items = await research._collect(["q1", "q2"], search=fake_search, fetch=fake_fetch)
        assert [i["url"] for i in items] == ["http://a", "http://b", "http://d"]
        assert all(i["content"].startswith("全文:") for i in items)

    @pytest.mark.asyncio
    async def test_search_exception_skipped(self, env):
        async def fake_search(q, num):
            if q == "boom":
                raise RuntimeError("engine down")
            return [{"title": "t", "url": "http://ok", "snippet": "s"}]

        async def fake_fetch(url, max_chars):
            return ""

        items = await research._collect(["boom", "fine"], search=fake_search, fetch=fake_fetch)
        assert [i["url"] for i in items] == ["http://ok"]
        assert items[0]["content"] == ""  # fetch 失败降级摘要，不炸流水线

    @pytest.mark.asyncio
    async def test_all_empty_returns_empty(self, env):
        async def fake_search(q, num):
            return []
        assert await research._collect(["q1"], search=fake_search, fetch=None) == []


class TestSynthesize:
    @pytest.mark.asyncio
    async def test_returns_report(self, env):
        model = _FakeModel("报告正文……来源：xxx")
        items = [{"title": "甲", "url": "http://a", "snippet": "sa", "content": "全文a"}]
        out = await research._synthesize("主题", items, model)
        assert "报告正文" in out
        assert "全文a" in model.prompts[0]  # 材料进了综述 prompt

    @pytest.mark.asyncio
    async def test_empty_response_raises(self, env):
        with pytest.raises(RuntimeError):
            await research._synthesize("主题", [{"title": "t", "url": "u",
                                                 "snippet": "s", "content": ""}],
                                       _FakeModel("  "))


class TestHandler:
    def _job(self, title="研究主题"):
        return type("J", (), {"job_id": "abc123", "chat_id": CHAT,
                              "title": title, "kind": "deep_research"})()

    @pytest.mark.asyncio
    async def test_full_chain(self, env):
        async def fake_search(q, num):
            return [{"title": f"t-{q}", "url": f"http://{q}", "snippet": f"s-{q}"}]

        async def fake_fetch(url, max_chars):
            return f"全文:{url}"

        out = await research.deep_research_handler(
            self._job(), {"topic": "绝区零丹怎么配队"},
            plan_model=_FakeModel('["配队", "驱动盘"]'),
            synth_model=_FakeModel("最终报告"),
            search=fake_search, fetch=fake_fetch)
        assert out == "最终报告"

    @pytest.mark.asyncio
    async def test_no_materials_raises(self, env):
        async def fake_search(q, num):
            return []
        with pytest.raises(RuntimeError, match="没查到"):
            await research.deep_research_handler(
                self._job(), {"topic": "查不到的东西"},
                plan_model=_FakeModel('["q"]'), synth_model=_FakeModel("x"),
                search=fake_search, fetch=None)

    @pytest.mark.asyncio
    async def test_empty_topic_raises(self, env):
        job = self._job(title="")
        with pytest.raises(RuntimeError, match="主题为空"):
            await research.deep_research_handler(
                job, {"topic": ""}, plan_model=_FakeModel(""),
                synth_model=_FakeModel(""), search=None, fetch=None)


class TestReflectRound:
    """反思轮（2026-08-09 宁德事故）：材料薄 -> 改写查询再搜，不再一轮定终身。"""

    def _job(self, title="研究主题"):
        return type("J", (), {"job_id": "abc123", "chat_id": CHAT,
                              "title": title, "kind": "deep_research"})()

    class _TwoPhaseModel:
        """第一次调用返回旧查询，第二次（反思）返回新查询。"""

        def __init__(self):
            self.calls = 0

        async def ainvoke(self, messages, config=None):
            self.calls += 1
            content = '["旧查询1", "旧查询2"]' if self.calls == 1 else '["新查询A", "新查询B"]'
            return type("R", (), {"content": content})()

    @pytest.mark.asyncio
    async def test_thin_materials_trigger_second_round(self, env):
        """首轮材料薄 -> 反思改写 -> 第二轮结果合并进综述。"""
        async def fake_search(q, num):
            if q.startswith("旧"):
                return [{"title": "t", "url": "http://old", "snippet": "s"}]
            return [{"title": f"t-{q}", "url": f"http://{q}", "snippet": f"s-{q}"}]

        async def fake_fetch(url, max_chars):
            return f"全文:{url}"

        model = self._TwoPhaseModel()
        out = await research.deep_research_handler(
            self._job(), {"topic": "宁德到深圳怎么去"},
            plan_model=model, synth_model=_FakeModel("反思后报告"),
            search=fake_search, fetch=fake_fetch)
        assert out == "反思后报告"
        assert model.calls == 2  # 规划 + 反思各一次

    @pytest.mark.asyncio
    async def test_duplicate_replan_breaks(self, env):
        """反思给出的查询全是重复的 -> 不再烧一轮检索。"""
        searches = []

        async def fake_search(q, num):
            searches.append(q)
            return [{"title": "t", "url": "http://only", "snippet": "s"}]

        async def fake_fetch(url, max_chars):
            return ""

        out = await research.deep_research_handler(
            self._job(), {"topic": "某主题"},
            plan_model=_FakeModel('["一样的查询"]'),  # 规划和反思返回同一查询
            synth_model=_FakeModel("报告"),
            search=fake_search, fetch=fake_fetch)
        assert out == "报告"
        assert searches == ["一样的查询"]  # 只搜了一轮

    @pytest.mark.asyncio
    async def test_empty_first_round_reflects_before_giving_up(self, env):
        """首轮全空不再直接认输：反思改写后第二轮有货 -> 正常出报告。"""
        async def fake_search(q, num):
            if q == "死路查询":
                return []
            return [{"title": "t", "url": "http://new", "snippet": "s"}]

        async def fake_fetch(url, max_chars):
            return "全文"

        class _Model:
            def __init__(self):
                self.calls = 0

            async def ainvoke(self, messages, config=None):
                self.calls += 1
                return type("R", (), {"content": '["死路查询"]' if self.calls == 1
                                      else '["活路查询"]'})()

        out = await research.deep_research_handler(
            self._job(), {"topic": "冷门主题"},
            plan_model=_Model(), synth_model=_FakeModel("绝处逢生报告"),
            search=fake_search, fetch=fake_fetch)
        assert out == "绝处逢生报告"

    @pytest.mark.asyncio
    async def test_rich_materials_no_waste(self, env):
        """误判回归：首轮材料充足 -> 不触发反思（不白烧规划+检索）。"""
        searches = []

        async def fake_search(q, num):
            searches.append(q)
            return [{"title": f"t-{q}-{i}", "url": f"http://{q}{i}", "snippet": "s"}
                    for i in range(2)]

        async def fake_fetch(url, max_chars):
            return "全文"

        model = _FakeModel('["q1", "q2", "q3"]')
        out = await research.deep_research_handler(
            self._job(), {"topic": "材料充足的主题"},
            plan_model=model, synth_model=_FakeModel("报告"),
            search=fake_search, fetch=fake_fetch)
        assert out == "报告"
        assert len(searches) == 3  # 只有首轮的 3 个查询
        assert len(model.prompts) == 1  # 规划器只被调一次


class TestTool:
    def test_deep_research_tool_submits(self, env):
        """LLM 工具入口：contextvar 路由 -> 落表 pending。"""
        from junjun_skills.plugins.async_task import tools as plugin
        from junjun_skills.builtin.memory_skills import current_chat_id
        from junjun_core.security import current_user_id, current_nickname
        t1 = current_chat_id.set(CHAT)
        t2 = current_user_id.set("111")
        t3 = current_nickname.set("甲")
        try:
            out = plugin.deep_research.invoke({"topic": "调研一下绝区零丹的配队"})
            assert "接单成功" in out and "深度研究" in out
            row = m.AsyncJob.get()
            assert row.kind == "deep_research" and row.status == "pending"
            assert row.chat_id == CHAT and "配队" in row.payload
        finally:
            current_chat_id.reset(t1)
            current_user_id.reset(t2)
            current_nickname.reset(t3)

    def test_short_topic_rejected(self, env):
        from junjun_skills.plugins.async_task import tools as plugin
        out = plugin.deep_research.invoke({"topic": "xx"})
        assert "太短" in out
