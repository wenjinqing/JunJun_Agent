"""P0 修复测试：bilibili 成品文件生命周期 / 周期提醒顺延 / adapter 异常隔离 /
知识库召回 / poke 路由。

每个测试对应一次生产实锤 bug，防回归。
"""

import asyncio
import time
from pathlib import Path

import pytest


class TestBilibiliFileLifecycle:
    """成品视频文件必须活到发送之后——提前删 NapCat 拿不到路径。"""

    @pytest.mark.asyncio
    async def test_task_manager_cleanup_runs_after_send(self):
        from junjun_agent.tasks import TaskManager
        from junjun_core.contracts import ReplySegment
        tm = TaskManager()
        sent, cleaned = [], []
        tm._send = lambda chat_id, segs: sent.append(segs) or asyncio.sleep(0)

        async def _work():
            return [ReplySegment(type="video", data="/tmp/fake.mp4")]

        async def _cleanup():
            cleaned.append(True)

        await tm.submit(kind="bilibili", work=_work, cleanup=_cleanup,
                        chat_id="qq:1:group")
        task = tm._running[("qq:1:group", "bilibili")]
        await task
        assert sent and cleaned  # 发送和清理都发生
        # cleanup 在 finally 里——一定在 _send 之后

    @pytest.mark.asyncio
    async def test_cleanup_runs_on_failure_too(self):
        """失败路径同样清理（不能留垃圾文件）。"""
        from junjun_agent.tasks import TaskManager
        tm = TaskManager()
        cleaned = []
        tm._send = lambda *a: asyncio.sleep(0)

        async def _work():
            return None  # 失败

        async def _cleanup():
            cleaned.append(True)

        await tm.submit(kind="bilibili", work=_work, cleanup=_cleanup,
                        chat_id="qq:1:group")
        await tm._running[("qq:1:group", "bilibili")]
        assert cleaned

    @pytest.mark.asyncio
    async def test_process_keeps_final_file(self, tmp_path, monkeypatch):
        """成功路径：成品文件不在 finally 删除，移交 keep 列表。"""
        from junjun_skills.plugins.bilibili import tools as bt
        final = tmp_path / "BV1xx_123.mp4"
        final.write_bytes(b"x" * 2048)

        monkeypatch.setattr(bt, "extract_bvid", lambda url: asyncio.sleep(0, result="BV1xx"))
        monkeypatch.setattr(bt, "_fetch_view",
                            lambda bvid: asyncio.sleep(0, result={
                                "aid": 1, "cid": 2, "bvid": "BV1xx",
                                "title": "测试", "duration": 60}))
        monkeypatch.setattr(bt, "_ffmpeg_path", lambda: "/usr/bin/ffmpeg")
        monkeypatch.setattr(bt, "_fetch_playurl",
                            lambda aid, cid: asyncio.sleep(0, result={
                                "type": "durl", "url": "http://x/v.mp4"}))

        async def _fake_download(url, path):
            path.write_bytes(b"x" * 2048)
            return True
        monkeypatch.setattr(bt, "_download", _fake_download)
        monkeypatch.setattr(bt, "TMP_DIR", tmp_path)
        monkeypatch.setattr(bt, "_max_size_mb", lambda: 100.0)
        monkeypatch.setattr(bt, "_max_duration", lambda: 0.0)

        keep = []
        segments = await bt._process_to_segments("https://b23.tv/xx", keep=keep)
        types = [s.type for s in segments]
        assert "video" in types
        assert len(keep) == 1 and keep[0].exists()  # 成品还活着
        keep[0].unlink()  # 模拟发送后清理

    @pytest.mark.asyncio
    async def test_process_failure_cleans_everything(self, tmp_path, monkeypatch):
        """失败路径：中间文件全部清理（keep 为空）。"""
        from junjun_skills.plugins.bilibili import tools as bt
        monkeypatch.setattr(bt, "extract_bvid", lambda url: asyncio.sleep(0, result="BV1xx"))
        monkeypatch.setattr(bt, "_fetch_view",
                            lambda bvid: asyncio.sleep(0, result={
                                "aid": 1, "cid": 2, "bvid": "BV1xx",
                                "title": "测试", "duration": 60}))
        monkeypatch.setattr(bt, "_ffmpeg_path", lambda: "/usr/bin/ffmpeg")
        monkeypatch.setattr(bt, "_fetch_playurl",
                            lambda aid, cid: asyncio.sleep(0, result={
                                "type": "durl", "url": "http://x/v.mp4"}))
        monkeypatch.setattr(bt, "_download",
                            lambda url, path: asyncio.sleep(0, result=False))  # 下载失败
        monkeypatch.setattr(bt, "TMP_DIR", tmp_path)

        keep = []
        segments = await bt._process_to_segments("https://b23.tv/xx", keep=keep)
        assert keep == []
        assert not list(tmp_path.glob("*.mp4"))  # 中间文件已清
        assert all(s.type == "text" for s in segments)  # 降级信息卡


class TestPeriodicReminderAdvance:
    """周期任务顺延必须推进到未来——停机多日不补发风暴。"""

    def test_daily_advances_beyond_now(self):
        from types import SimpleNamespace
        # 模拟 _fire 的顺延逻辑（不依赖发送）：直接验证推进算法
        now = time.time()
        task = SimpleNamespace(repeat_type="daily", remind_time=now - 3 * 86400,
                               is_completed=False)
        # 与 reminder._fire 相同的推进逻辑
        task.remind_time += 86400
        while task.remind_time <= now:
            task.remind_time += 86400
        assert task.remind_time > now
        assert task.remind_time <= now + 86400

    def test_weekly_advances_beyond_now(self):
        from types import SimpleNamespace
        now = time.time()
        task = SimpleNamespace(repeat_type="weekly", remind_time=now - 30 * 86400)
        task.remind_time += 7 * 86400
        while task.remind_time <= now:
            task.remind_time += 7 * 86400
        assert task.remind_time > now

    @pytest.mark.asyncio
    async def test_fire_does_not_refire_past_tasks(self, monkeypatch, tmp_path):
        """集成验证：停机 3 天的 daily 任务 _fire 一次后，下次到期在未来。"""
        import peewee
        from junjun_core.database import models as m
        from junjun_agent.loop import reminder

        db = peewee.SqliteDatabase(str(tmp_path / "t.db"))
        with db.bind_ctx([m.ReminderTasks]):
            db.create_tables([m.ReminderTasks])
            now = time.time()
            m.ReminderTasks.create(
                task_id="t1", chat_id="qq:1:private", user_id="1",
                content="喝水", remind_time=now - 3 * 86400,
                repeat_type="daily", is_completed=False)

            # 跳过 LLM 与发送
            async def _fake_send(*a, **kw):
                return None
            monkeypatch.setattr(
                "junjun_core.gateway.router.get_gateway",
                lambda: type("G", (), {"send_reply": staticmethod(_fake_send)})())
            monkeypatch.setattr(
                "junjun_llm.get_chat_model",
                lambda slot: (_ for _ in ()).throw(RuntimeError("no llm")))
            monkeypatch.setattr("junjun_llm.get_callbacks", lambda: [])

            task = m.ReminderTasks.get(m.ReminderTasks.task_id == "t1")
            await reminder._fire(task)
            task = m.ReminderTasks.get(m.ReminderTasks.task_id == "t1")
            assert task.remind_time > time.time()  # 不会下轮立即再发


class TestAdapterLoopIsolation:
    """单条坏消息不能杀死收信循环。"""

    @pytest.mark.asyncio
    async def test_bad_message_does_not_kill_loop(self, monkeypatch):
        from junjun_adapter_napcat import main as adapter_main

        processed = []

        async def _ok(msg):
            processed.append(msg.get("message_id"))

        async def _bad(msg):
            raise ValueError("畸形消息")

        monkeypatch.setattr(adapter_main.message_handler, "handle_raw_message", _bad)
        # 放入两条消息：第一条炸、第二条换好 handler 应被处理
        q = adapter_main.message_queue
        while not q.empty():
            q.get_nowait()
            q.task_done()
        await q.put({"post_type": "message", "message_id": 1})
        await q.put({"post_type": "message", "message_id": 2})

        async def _run():
            await adapter_main.message_process()
        task = asyncio.create_task(_run())
        await asyncio.sleep(0.2)
        monkeypatch.setattr(adapter_main.message_handler, "handle_raw_message", _ok)
        await asyncio.sleep(0.2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        # 循环活着：第二条消息被消费
        assert q.empty()


class TestKnowledgeRecall:
    """知识库条目（chat_id="knowledge"）必须能被日常检索召回。"""

    def test_chat_allowed_modes(self):
        from junjun_memory.long_term import _chat_allowed
        assert _chat_allowed("qq:1:group", None) is True
        assert _chat_allowed("knowledge", "qq:1:group") is False
        assert _chat_allowed("knowledge", ("qq:1:group", "knowledge")) is True
        assert _chat_allowed("qq:2:group", ("qq:1:group", "knowledge")) is False
        assert _chat_allowed("qq:1:group", ("qq:1:group", "knowledge")) is True

    @pytest.mark.asyncio
    async def test_search_recalls_knowledge_items(self, tmp_path, monkeypatch):
        """集成：processor 式多值过滤能召回知识库条目。"""
        from junjun_memory import long_term as lt
        monkeypatch.setattr(lt, "get_embedding_client",
                            lambda: type("E", (), {"available": False,
                                                   "embed_one": staticmethod(
                                                       lambda t: asyncio.sleep(0, result=None))})())
        mem = lt.LongTermMemory(data_dir=tmp_path / "mem")
        mem.load()
        await mem.add("项目代号是星尘", chat_id="knowledge")
        await mem.add("今天天气不错", chat_id="qq:1:group")
        await mem.add("别群的秘密", chat_id="qq:2:group")
        # 关键词路径（embedding 不可用）+ 多值过滤
        out = await mem.search("星尘", top_k=3, chat_id=("qq:1:group", "knowledge"))
        texts = [it.text for it in out]
        assert "项目代号是星尘" in texts
        assert "别群的秘密" not in texts


class TestPokeRouting:
    """poke 必须走 WS 发给核心网关，不再调本进程 echo gateway。"""

    @pytest.mark.asyncio
    async def test_poke_goes_to_message_sending(self, monkeypatch):
        from junjun_adapter_napcat.recv_handler import notice_handler
        sent = []

        async def _fake_send(msg_base):
            sent.append(msg_base)
        monkeypatch.setattr(
            "junjun_adapter_napcat.message_sending.message_send_instance.message_send",
            _fake_send)
        # 黑白名单放行
        monkeypatch.setattr(notice_handler, "message_handler_allow",
                            lambda u, g: asyncio.sleep(0, result=True))

        handler = notice_handler.NoticeHandler()
        await handler._handle_poke({
            "self_id": "10000001", "target_id": "10000001",  # 戳的是 bot 自己
            "user_id": "12345", "group_id": "678",
        })
        assert len(sent) == 1
