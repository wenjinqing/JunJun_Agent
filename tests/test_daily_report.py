"""热点日报管线测试：选题/深研/写稿/人审/发布 + 崩溃续跑 + 启动恢复 + 审批钩子。

素材/选题/写稿/发布全部打桩（build_graph deps 注入），research 纯函数打桩，
sqlite 用 tmp_path，不触生产库、不触真实 QZone。
"""

import asyncio
import json

import pytest

import junjun_core.config.config as cfg_mod
from junjun_skills.plugins.async_task import research
from junjun_skills.plugins.daily_report import graph as dr_graph
from junjun_skills.plugins.daily_report import tools as dr_tools

RID = "dr-2099-01-01"


def _set_config(raw: dict):
    cfg_mod.global_config = cfg_mod.GlobalConfig(
        bot=cfg_mod.BotConfig(platform="qq", qq_account="12345", nickname="君君"),
        raw=raw)


class _Deps:
    """节点依赖桩：计数 + 可切行为。"""

    def __init__(self):
        self.gather_calls = 0
        self.pick_calls = 0
        self.research_collect_calls = 0
        self.write_calls = 0
        self.published: list = []
        self.titles = ["标题A", "标题B"]
        self.topic = "话题X"
        self.collect_items = [{"title": "t", "url": "http://a",
                               "snippet": "s", "content": "全文"}]

    def as_dict(self):
        async def gather():
            self.gather_calls += 1
            return list(self.titles)

        async def pick(titles, recent):
            self.pick_calls += 1
            return self.topic

        async def write(topic, report):
            self.write_calls += 1
            return "说说草稿"

        async def publish(draft):
            self.published.append(draft)
            return "tid-1"

        return {"gather": gather, "pick": pick, "recent_topics": lambda: [],
                "write": write, "publish": publish,
                "plan_model": None, "synth_model": None,
                "search": None, "fetch": None}


@pytest.fixture
def env(tmp_path, monkeypatch):
    old = cfg_mod.global_config
    _set_config({"daily_report": {"enable": True, "time": "21:30",
                                  "approval_timeout_seconds": 600,
                                  "max_titles": 5, "history_days": 7},
                 "deep_research": {"max_rounds": 2, "min_items": 2,
                                   "min_fulltext": 1, "queries": 2,
                                   "pages_per_query": 1, "fetch_max_chars": 100,
                                   "report_max_chars": 300}})
    deps = _Deps()
    d = deps.as_dict()
    # 默认依赖（default_deps 从 tools 取）全部换桩——sqlite 建的图也吃到同一套桩
    monkeypatch.setattr(dr_tools, "gather_materials", d["gather"])
    monkeypatch.setattr(dr_tools, "pick_topic", d["pick"])
    monkeypatch.setattr(dr_tools, "recent_topics", d["recent_topics"])
    monkeypatch.setattr(dr_tools, "write_draft", d["write"])
    monkeypatch.setattr(dr_tools, "publish_draft", d["publish"])
    # research 纯函数打桩（research_node 直接调它们）
    async def fake_plan(topic, model=None):
        return ["q1"]
    async def fake_collect(queries, *, search=None, fetch=None):
        deps.research_collect_calls += 1
        return [dict(it) for it in deps.collect_items]
    async def fake_replan(topic, old, got, model=None):
        return []
    async def fake_synth(topic, items, model=None):
        return "深研综述"
    monkeypatch.setattr(research, "_plan", fake_plan)
    monkeypatch.setattr(research, "_collect", fake_collect)
    monkeypatch.setattr(research, "_replan", fake_replan)
    monkeypatch.setattr(research, "_synthesize", fake_synth)
    # 审批通知打桩
    notices = []
    import junjun_core.security as sec
    async def fake_notify(text):
        notices.append(text)
        return True
    monkeypatch.setattr(sec, "notify_admin", fake_notify)
    # 模块状态隔离
    dr_graph._persist_dir = tmp_path
    dr_graph._graph = None
    dr_graph._recovered = False
    dr_graph._pending.clear()
    monkeypatch.setattr(dr_tools, "DATA_DIR", tmp_path / "dr")
    deps.notices = notices
    yield deps
    for info in dr_graph._pending.values():
        t = info.get("timeout_task")
        if t:
            t.cancel()
    dr_graph._pending.clear()
    dr_graph._graph = None
    dr_graph._persist_dir = None
    dr_graph._recovered = False
    cfg_mod.global_config = old


def _memory_graph(deps):
    from langgraph.checkpoint.memory import MemorySaver
    return dr_graph.build_graph(MemorySaver())  # 默认依赖已在 fixture 换桩


async def _close_graph():
    g = dr_graph._graph
    cp = getattr(g, "checkpointer", None)
    conn = getattr(cp, "conn", None)
    if conn is not None:
        await conn.close()


class TestFlow:
    @pytest.mark.asyncio
    async def test_run_parks_at_approval(self, env):
        dr_graph._graph = _memory_graph(env)
        state = await dr_graph.run(RID, "2099-01-01")
        assert "__interrupt__" in state
        assert RID in dr_graph._pending
        assert env.notices and "话题X" in env.notices[0]
        assert not env.published, "审批前不许发布"

    @pytest.mark.asyncio
    async def test_approve_publishes_and_records(self, env, tmp_path):
        dr_graph._graph = _memory_graph(env)
        await dr_graph.run(RID, "2099-01-01")
        await dr_graph.resume(RID, True)
        assert env.published == ["说说草稿"]
        assert RID not in dr_graph._pending
        assert dr_graph._registry_load() == {}
        hist = json.loads((tmp_path / "dr" / "history.json")
                          .read_text(encoding="utf-8"))
        assert hist[0]["topic"] == "话题X" and hist[0]["tid"] == "tid-1"

    @pytest.mark.asyncio
    async def test_reject_skips_without_publish(self, env, tmp_path):
        dr_graph._graph = _memory_graph(env)
        await dr_graph.run(RID, "2099-01-01")
        await dr_graph.resume(RID, False)
        assert not env.published
        assert dr_graph._registry_load() == {}
        assert not (tmp_path / "dr" / "history.json").exists(), \
            "被驳回的日报不进历史（明天还能再选这个题）"

    @pytest.mark.asyncio
    async def test_timeout_defaults_no_publish(self, env, monkeypatch):
        monkeypatch.setattr(dr_graph, "_cfg",
                            lambda: {"approval_timeout_seconds": 0.05})
        dr_graph._graph = _memory_graph(env)
        await dr_graph.run(RID, "2099-01-01")
        assert RID in dr_graph._pending
        await asyncio.sleep(0.15)
        assert RID not in dr_graph._pending, "超时必须自动结案"
        assert not env.published, "对外发布，超时默认不发（保守方向）"

    @pytest.mark.asyncio
    async def test_no_materials_skips(self, env):
        dr_graph._graph = _memory_graph(env)
        env.titles = []
        state = await dr_graph.run(RID, "2099-01-01")
        assert "__interrupt__" not in state
        assert not env.published and not env.notices
        assert env.pick_calls == 0, "没素材不该烧选题模型"

    @pytest.mark.asyncio
    async def test_no_topic_skips(self, env):
        dr_graph._graph = _memory_graph(env)
        env.topic = ""
        state = await dr_graph.run(RID, "2099-01-01")
        assert state.get("skip_reason")
        assert env.research_collect_calls == 0, "没选题不该烧深研"

    @pytest.mark.asyncio
    async def test_empty_research_skips(self, env):
        dr_graph._graph = _memory_graph(env)
        env.collect_items = []
        state = await dr_graph.run(RID, "2099-01-01")
        assert state.get("skip_reason") == "深研没查到材料"
        assert not env.published and env.write_calls == 0


class TestCrashResume:
    @pytest.mark.asyncio
    async def test_resume_skips_finished_nodes(self, env):
        """图在深研后崩了：续跑不重抓素材、不重选题、不重深研。"""
        try:
            graph = await dr_graph._ensure_graph()
            from langgraph.errors import GraphRecursionError
            with pytest.raises(GraphRecursionError):
                await graph.ainvoke({"report_id": RID, "date": "2099-01-01"},
                                    {"configurable": {"thread_id": RID},
                                     "recursion_limit": 3})
            assert env.gather_calls == 1
            state = await graph.ainvoke(None, dr_graph._thread_cfg(RID))
            assert "__interrupt__" in state, "续跑应一路跑到人审挂起"
            assert env.gather_calls == 1 and env.pick_calls == 1
            assert env.research_collect_calls == 1
            assert env.write_calls == 1
        finally:
            await _close_graph()

    @pytest.mark.asyncio
    async def test_recover_rebuilds_pending_approval(self, env):
        """崩溃时正停在人审：重启 recover 重建待审批并重新通知管理员。"""
        try:
            await dr_graph.run(RID, "2099-01-01")
            assert RID in dr_graph._pending
            assert len(env.notices) == 1
            # 模拟重启：图/内存态全丢，sqlite checkpoint + 注册表还在
            await _close_graph()
            dr_graph._graph = None
            dr_graph._pending.clear()
            dr_graph._recovered = False

            await dr_graph.recover()
            assert RID in dr_graph._pending, "恢复后待审批必须重建"
            assert len(env.notices) == 2, "恢复后必须重新通知管理员"
            # 审批通道照常工作
            await dr_graph.resume(RID, True)
            assert env.published == ["说说草稿"]
            assert dr_graph._registry_load() == {}
        finally:
            await _close_graph()


class TestApprovalHook:
    def _meta(self, user_id, text):
        return type("M", (), {"user_id": user_id, "text": text})()

    def _session(self):
        return type("S", (), {"chat_id": "qq:12345:group"})()

    @pytest.mark.asyncio
    async def test_non_admin_not_consumed(self, env, monkeypatch):
        import junjun_core.security as sec
        monkeypatch.setattr(sec, "is_admin", lambda uid: False)
        dr_graph._pending[RID] = {"topic": "t", "draft": "d"}
        assert await dr_graph.approval_hook(self._session(),
                                            self._meta("999", "发")) is False

    @pytest.mark.asyncio
    async def test_exact_word_only(self, env, monkeypatch):
        """误判回归：「发一下」「发啊」不许触发审批。"""
        import junjun_core.security as sec
        monkeypatch.setattr(sec, "is_admin", lambda uid: True)
        dr_graph._pending[RID] = {"topic": "t", "draft": "d"}
        for text in ("发一下", "发啊", "发表", "算了算了"):
            assert await dr_graph.approval_hook(
                self._session(), self._meta("1", text)) is False
        assert RID in dr_graph._pending

    @pytest.mark.asyncio
    async def test_no_pending_not_consumed(self, env, monkeypatch):
        import junjun_core.security as sec
        monkeypatch.setattr(sec, "is_admin", lambda uid: True)
        assert await dr_graph.approval_hook(self._session(),
                                            self._meta("1", "发")) is False

    @pytest.mark.asyncio
    async def test_approve_consumed_with_ack(self, env, monkeypatch):
        import junjun_core.security as sec
        monkeypatch.setattr(sec, "is_admin", lambda uid: True)
        sent = []
        import junjun_agent.outbound as outbound
        async def fake_send(chat_id, segments, **kw):
            sent.append((chat_id, segments[0].data))
        monkeypatch.setattr(outbound, "send_proactive", fake_send)
        resumed = []
        async def fake_resume(rid, approved):
            resumed.append((rid, approved))
            dr_graph._pending.pop(rid, None)
        monkeypatch.setattr(dr_graph, "resume", fake_resume)
        dr_graph._pending[RID] = {"topic": "t", "draft": "d"}
        assert await dr_graph.approval_hook(self._session(),
                                            self._meta("1", "发")) is True
        await asyncio.sleep(0)  # 让 create_task 的 resume 跑掉
        assert resumed == [(RID, True)]
        assert sent and "发到空间" in sent[0][1]


class TestTick:
    @pytest.mark.asyncio
    async def test_fires_once_per_day(self, env, monkeypatch, tmp_path):
        import time as _time
        monkeypatch.setattr(dr_tools, "_cfg",
                            lambda: {"enable": True,
                                     "time": _time.strftime("%H:%M")})
        runs = []
        async def fake_run(rid, date):
            runs.append(rid)
            return {"skip_reason": "测试"}
        monkeypatch.setattr(dr_graph, "run", fake_run)
        await dr_tools.daily_report_tick()
        await dr_tools.daily_report_tick()
        assert len(runs) == 1, "同一天只触发一次"

    @pytest.mark.asyncio
    async def test_disabled_no_fire(self, env, monkeypatch):
        monkeypatch.setattr(dr_tools, "_cfg", lambda: {"enable": False})
        runs = []
        monkeypatch.setattr(dr_graph, "run",
                            lambda *a: runs.append(a))
        await dr_tools.daily_report_tick()
        assert not runs

    @pytest.mark.asyncio
    async def test_already_published_today_skips(self, env, monkeypatch, tmp_path):
        import time as _time
        from datetime import datetime
        monkeypatch.setattr(dr_tools, "_cfg",
                            lambda: {"enable": True,
                                     "time": _time.strftime("%H:%M")})
        today = datetime.now().strftime("%Y-%m-%d")
        dr_tools._write_json("history.json",
                             [{"date": today, "topic": "旧题", "tid": "x",
                               "ts": _time.time()}])
        runs = []
        async def fake_run(rid, date):
            runs.append(rid)
        monkeypatch.setattr(dr_graph, "run", fake_run)
        await dr_tools.daily_report_tick()
        assert not runs, "今天已发过就不再跑（含重启后手动补触发场景）"
