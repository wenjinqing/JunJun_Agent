"""任务有下文（2026-08-04「图呢」事件）：失败自动重试 / 结局登记 /
决策注入 / 短期记忆回写 / list_background_tasks 合并成品任务。"""

import asyncio

import pytest

from junjun_agent.tasks import TaskManager, task_manager
from junjun_core.contracts import ReplySegment


@pytest.fixture
def tm():
    m = TaskManager()
    sent = []

    async def _send(chat_id, segments):
        sent.append(segments)
        return True
    m._send = _send
    yield m, sent


class TestAutoRetry:
    @pytest.mark.asyncio
    async def test_retry_once_on_failure(self, tm, monkeypatch):
        """首次失败自动重试一次；第二次成功则直发成品。"""
        m, sent = tm
        monkeypatch.setattr(TaskManager, "_auto_retry", staticmethod(lambda: True))
        calls = []

        async def work():
            calls.append(1)
            if len(calls) == 1:
                return None                      # 第一次无产出
            return [ReplySegment(type="image", data="http://x/a.png")]

        ack = await m.submit(kind="ai_draw", work=work, done_text="画好了",
                             fail_text="画砸了", timeout=5, chat_id="qq:1:private")
        assert ack
        await asyncio.sleep(3.5)                 # 重试间隔 3s
        assert len(calls) == 2
        assert sent and sent[0][0].data == "画好了"
        out = m._outcomes["qq:1:private"][-1]
        assert out["status"] == "done"

    @pytest.mark.asyncio
    async def test_retry_disabled_config(self, tm, monkeypatch):
        m, sent = tm
        monkeypatch.setattr(TaskManager, "_auto_retry", staticmethod(lambda: False))
        calls = []

        async def work():
            calls.append(1)
            return None

        await m.submit(kind="ai_draw", work=work, fail_text="画砸了",
                       timeout=5, chat_id="qq:1:private")
        await asyncio.sleep(0.2)
        assert len(calls) == 1                   # 不重试
        assert sent[0][0].data == "画砸了"
        assert m._outcomes["qq:1:private"][-1]["status"] == "failed"

    @pytest.mark.asyncio
    async def test_both_attempts_fail_records_failure(self, tm, monkeypatch):
        m, sent = tm
        monkeypatch.setattr(TaskManager, "_auto_retry", staticmethod(lambda: True))

        async def work():
            raise RuntimeError("api down")

        await m.submit(kind="ai_draw", work=work, fail_text="画砸了",
                       timeout=5, chat_id="qq:1:private")
        await asyncio.sleep(3.5)
        out = m._outcomes["qq:1:private"][-1]
        assert out["status"] == "failed" and "RuntimeError" in out["detail"]
        assert sent[0][0].data == "画砸了"


class TestOutcomeVisibility:
    @pytest.mark.asyncio
    async def test_status_block_running_and_done(self, tm, monkeypatch):
        """在途与结局都进决策注入块。"""
        m, _ = tm
        monkeypatch.setattr(TaskManager, "_auto_retry", staticmethod(lambda: False))
        ev = asyncio.Event()

        async def work_slow():
            await ev.wait()
            return [ReplySegment(type="text", data="x")]

        await m.submit(kind="ai_draw", work=work_slow, timeout=30,
                       chat_id="qq:1:private")
        block = m.task_status_block("qq:1:private")
        assert "画图" in block and "进行中" in block
        ev.set()
        await asyncio.sleep(0.2)
        block = m.task_status_block("qq:1:private")
        assert "完成" in block
        # 别的会话不受影响
        assert m.task_status_block("qq:2:private") == ""

    def test_list_for_chat_empty(self, tm):
        m, _ = tm
        assert m.list_for_chat("qq:9:private") == ""

    @pytest.mark.asyncio
    async def test_list_background_tasks_merges_task_manager(self, tm, monkeypatch):
        """工具合并：成品任务（task_manager）+ 派活任务（async_jobs）都可见。"""
        m, _ = tm
        import junjun_skills.plugins.async_task.tools as att
        monkeypatch.setattr(att, "async_jobs",
                            type("AJ", (), {"list_for_chat": staticmethod(
                                lambda cid: "这个会话还没有后台任务。")}))
        from junjun_agent import tasks as tasks_mod
        monkeypatch.setattr(tasks_mod, "task_manager", m)
        monkeypatch.setattr(TaskManager, "_auto_retry", staticmethod(lambda: False))
        m._outcomes.setdefault("qq:1:private", __import__("collections").deque(maxlen=10)).append(
            {"ts": __import__("time").time(), "kind": "ai_draw",
             "status": "failed", "detail": "超时"})
        from junjun_skills.builtin import memory_skills
        token = memory_skills.current_chat_id.set("qq:1:private")
        try:
            out = att.list_background_tasks.invoke({})
        finally:
            memory_skills.current_chat_id.reset(token)
        assert "成品任务" in out and "画图" in out and "失败" in out
        assert "还没有后台任务" in out           # async_jobs 段也在


class TestNegativeEvidence:
    @pytest.mark.asyncio
    async def test_query_intent_injects_negative_block(self, monkeypatch, tmp_path):
        """问进度但无在途/无记录 -> 注入否定证据（治「顺着旧话编还在画」）。"""
        import junjun_core.config.config as cfg_mod
        monkeypatch.setattr(cfg_mod, "global_config", cfg_mod.GlobalConfig(
            bot=cfg_mod.BotConfig(platform="qq", qq_account="1", nickname="君君"),
            raw={}))
        from types import SimpleNamespace
        from junjun_agent.processor import _build_memory_block
        session = SimpleNamespace(chat_id="qq:777:group", memory=None)
        meta = SimpleNamespace(image_urls=None, sticker_urls=None, voice_records=None,
                               video_urls=None, text="图呢", user_id="1", nickname="甲")
        block, _ = await _build_memory_block(session, meta)
        assert "没有在途任务" in block and "别顺着旧话编" in block

    @pytest.mark.asyncio
    async def test_non_query_gets_no_negative_block(self, monkeypatch):
        import junjun_core.config.config as cfg_mod
        monkeypatch.setattr(cfg_mod, "global_config", cfg_mod.GlobalConfig(
            bot=cfg_mod.BotConfig(platform="qq", qq_account="1", nickname="君君"),
            raw={}))
        from types import SimpleNamespace
        from junjun_agent.processor import _build_memory_block
        session = SimpleNamespace(chat_id="qq:778:group", memory=None)
        meta = SimpleNamespace(image_urls=None, sticker_urls=None, voice_records=None,
                               video_urls=None, text="今天天气真好", user_id="1", nickname="甲")
        block, _ = await _build_memory_block(session, meta)
        assert "没有在途任务" not in block


class TestOutcomePersistence:
    def test_restore_marks_orphan_start_as_interrupted(self, tmp_path):
        """有 start 无 end = 重启中断，恢复为失败结局。"""
        import json as _json
        from junjun_agent import tasks as tasks_mod
        recs = [
            {"op": "start", "chat_id": "qq:1:private", "kind": "ai_draw", "ts": 100.0},
            {"op": "end", "chat_id": "qq:1:private", "kind": "ai_draw",
             "status": "done", "detail": "耗时60s", "ts": 160.0},
            {"op": "start", "chat_id": "qq:1:private", "kind": "tts", "ts": 200.0},
        ]
        f = tmp_path / "out.jsonl"
        f.write_text("\n".join(_json.dumps(r) for r in recs), encoding="utf-8")
        mgr = TaskManager()
        tasks_mod._restore_from_records(mgr, tasks_mod._load_records(f))
        outs = list(mgr._outcomes["qq:1:private"])
        assert len(outs) == 2
        assert outs[0]["status"] == "done" and outs[0]["kind"] == "ai_draw"
        assert outs[1]["status"] == "failed" and "重启" in outs[1]["detail"]
        assert outs[1]["kind"] == "tts"

    def test_no_persist_file_no_writes(self, tmp_path, monkeypatch):
        """测试纪律：未挂接落盘文件时不写任何文件。"""
        from junjun_agent import tasks as tasks_mod
        assert tasks_mod._PERSIST_FILE is None
        tasks_mod._append_rec({"op": "end", "chat_id": "x", "kind": "k"})
        assert list(tmp_path.iterdir()) == []


class TestConcurrencyAndCancel:
    """严厉审查 S3：全局并发上限 + 成品任务可取消。"""

    @pytest.mark.asyncio
    async def test_global_concurrency_cap(self, tm, monkeypatch):
        m, sent = tm
        monkeypatch.setattr(TaskManager, "_max_concurrent", staticmethod(lambda: 1))
        monkeypatch.setattr(TaskManager, "_auto_retry", staticmethod(lambda: False))
        ev = asyncio.Event()

        async def work_slow():
            await ev.wait()
            return None

        ack1 = await m.submit(kind="ai_draw", work=work_slow, timeout=30,
                              chat_id="qq:1:private")
        ack2 = await m.submit(kind="tts", work=work_slow, timeout=30,
                              chat_id="qq:2:private")
        assert "在弄了" in ack1 or ack1          # 第一个接单
        assert "排满" in ack2                    # 第二个被全局上限拒绝
        ev.set()
        await asyncio.sleep(0.2)

    @pytest.mark.asyncio
    async def test_cancel_by_kind_cn_name(self, tm):
        """「别画了」-> cancel_for_chat(chat_id, "画图") 能取消在途任务。"""
        m, _ = tm
        ev = asyncio.Event()

        async def work_slow():
            await ev.wait()
            return None

        await m.submit(kind="ai_draw", work=work_slow, timeout=30,
                       chat_id="qq:1:private")
        n = m.cancel_for_chat("qq:1:private", "画图")   # 中文名也可
        assert n == 1
        await asyncio.sleep(0)                          # 让取消的 done 回调落地
        assert m.running_count() == 0
        out = m._outcomes["qq:1:private"][-1]
        assert out["status"] == "cancelled"
        ev.set()
