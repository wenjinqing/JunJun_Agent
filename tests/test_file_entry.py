"""文件入口 0-token 测试：群里发文件 -> 登记 -> workspace_save_file 存进工作区。

闭环（2026-08-13）：适配器解出 file_ref（name/size/url）-> 网关 InboundMeta.file_refs
-> processor 登记 recent_files -> workspace_save_file 下载进工作区 -> run_code 处理
-> workspace_send 发回。全部不触网、不写生产库：下载/OneBot 调用 monkeypatch，
工作区根指 tmp。
"""

import json

import pytest

from junjun_core.gateway.router import InboundMeta, _extract_files
from junjun_core.contracts import Seg


# ---------------------------------------------------------------- 适配器：文件段解析

class TestAdapterFileSegment:
    def _handler(self):
        from junjun_adapter_napcat.recv_handler.message_handler import MessageHandler
        return MessageHandler()

    @pytest.mark.asyncio
    async def test_file_with_url_emits_ref(self):
        """带 url 的文件段：可读占位文本 + file_ref 结构化引用，不再只剩 [文件]。"""
        h = self._handler()
        segs, at = await h._parse_message_segments(
            [{"type": "file", "data": {"name": "销售表.xlsx", "size": 2048,
                                       "url": "https://cdn.qq.com/f/abc"}}],
            self_id="10000", group_id="999")
        texts = [s.data for s in segs if s.type == "text"]
        refs = [s for s in segs if s.type == "file_ref"]
        assert any("销售表.xlsx" in t and "2.0KB" in t for t in texts)
        assert len(refs) == 1
        ref = json.loads(refs[0].data)
        assert ref == {"name": "销售表.xlsx", "size": 2048,
                       "url": "https://cdn.qq.com/f/abc"}

    @pytest.mark.asyncio
    async def test_group_file_without_url_resolves(self, monkeypatch):
        """群文件段没 url：调 get_group_file_url 补（go-cqhttp 系只有 id/busid）。"""
        calls = []

        class _FakeSender:
            async def send_message_to_napcat(self, action, params):
                calls.append((action, params))
                return {"data": {"url": "https://cdn.qq.com/f/resolved"}}

        from junjun_adapter_napcat.send_handler import nc_sending
        monkeypatch.setattr(nc_sending, "nc_message_sender", _FakeSender())
        h = self._handler()
        segs, _ = await h._parse_message_segments(
            [{"type": "file", "data": {"id": "/file-id-1", "name": "a.pdf",
                                       "size": 100, "busid": 102}}],
            self_id="10000", group_id="999")
        assert calls and calls[0][0] == "get_group_file_url"
        assert calls[0][1]["file_id"] == "/file-id-1"
        refs = [s for s in segs if s.type == "file_ref"]
        assert len(refs) == 1
        assert json.loads(refs[0].data)["url"] == "https://cdn.qq.com/f/resolved"

    @pytest.mark.asyncio
    async def test_resolve_failure_keeps_placeholder(self, monkeypatch):
        """补链失败：占位文本仍在，只是没有 file_ref（存不了但看得见）。"""
        class _BoomSender:
            async def send_message_to_napcat(self, action, params):
                raise RuntimeError("ws down")

        from junjun_adapter_napcat.send_handler import nc_sending
        monkeypatch.setattr(nc_sending, "nc_message_sender", _BoomSender())
        h = self._handler()
        segs, _ = await h._parse_message_segments(
            [{"type": "file", "data": {"id": "/x", "name": "b.txt", "size": 10}}],
            self_id="10000", group_id="999")
        assert any("b.txt" in s.data for s in segs if s.type == "text")
        assert not [s for s in segs if s.type == "file_ref"]

    @pytest.mark.asyncio
    async def test_private_file_no_resolve_call(self, monkeypatch):
        """私聊文件（无 group_id）不调 get_group_file_url——私聊段一般自带 url。"""
        calls = []

        class _FakeSender:
            async def send_message_to_napcat(self, action, params):
                calls.append(action)
                return {"data": {"url": "https://x"}}

        from junjun_adapter_napcat.send_handler import nc_sending
        monkeypatch.setattr(nc_sending, "nc_message_sender", _FakeSender())
        h = self._handler()
        segs, _ = await h._parse_message_segments(
            [{"type": "file", "data": {"id": "/y", "name": "c.zip", "size": 10}}],
            self_id="10000", group_id="")
        assert calls == []
        assert not [s for s in segs if s.type == "file_ref"]

    def test_fmt_size(self):
        from junjun_adapter_napcat.recv_handler.message_handler import _fmt_size
        assert _fmt_size(0) == "大小未知"
        assert _fmt_size(512) == "512B"
        assert _fmt_size(2048) == "2.0KB"
        assert _fmt_size(1_500_000) == "1.4MB"


# ---------------------------------------------------------------- 网关：file_refs 抽取

class TestGatewayExtract:
    def test_extract_files_happy(self):
        seg = Seg(type="seglist", data=[
            Seg(type="text", data="[文件：a.xlsx（1.0KB）]"),
            Seg(type="file_ref", data=json.dumps(
                {"name": "a.xlsx", "size": 1024, "url": "https://u"}, ensure_ascii=False)),
        ])
        refs = _extract_files(seg)
        assert refs == [{"name": "a.xlsx", "size": 1024, "url": "https://u"}]

    def test_extract_files_bad_json_skipped(self):
        """坏 JSON / 缺 url 的段跳过不炸主链路。"""
        seg = Seg(type="seglist", data=[
            Seg(type="file_ref", data="{oops"),
            Seg(type="file_ref", data=json.dumps({"name": "x"})),   # 无 url
            Seg(type="file_ref", data=json.dumps(
                {"name": "ok", "size": 1, "url": "https://u"})),
        ])
        refs = _extract_files(seg)
        assert len(refs) == 1 and refs[0]["name"] == "ok"

    def test_file_only_message_not_dropped(self):
        """误判回归：纯文件消息（占位文本没了也不算空）不得被网关无内容闸丢掉。"""
        refs = _extract_files(Seg(type="file_ref", data=json.dumps(
            {"name": "a", "size": 1, "url": "https://u"})))
        text = ""
        # 与 router.handle_inbound 同一判定式
        dropped = not text and not [] and not [] and not [] and not [] and not refs
        assert not dropped


# ---------------------------------------------------------------- recent_files 登记处

class TestRecentFiles:
    def setup_method(self):
        from junjun_memory import recent_files as rf
        rf._reset_for_test()
        self.rf = rf

    def test_note_and_recent_newest_first(self):
        self.rf.note_recent_file("qq:999:group", {"name": "一.xlsx", "size": 1, "url": "u1"})
        self.rf.note_recent_file("qq:999:group", {"name": "二.pdf", "size": 2, "url": "u2"})
        files = self.rf.recent_files("qq:999:group")
        assert [f["name"] for f in files] == ["二.pdf", "一.xlsx"]
        assert self.rf.recent_file("qq:999:group")["name"] == "二.pdf"

    def test_chat_isolated(self):
        self.rf.note_recent_file("qq:999:group", {"name": "a", "size": 1, "url": "u"})
        assert self.rf.recent_files("qq:888:group") == []
        assert self.rf.recent_file("qq:888:group") is None

    def test_bad_data_ignored(self):
        self.rf.note_recent_file("qq:999:group", {"name": "无链接"})
        self.rf.note_recent_file("", {"name": "a", "url": "u"})
        self.rf.note_recent_file("qq:999:group", None)
        assert self.rf.recent_files("qq:999:group") == []

    def test_ttl_expired(self, monkeypatch):
        self.rf.note_recent_file("qq:999:group", {"name": "a", "size": 1, "url": "u"})
        import time
        real = time.time
        monkeypatch.setattr(self.rf.time, "time", lambda: real() + 601)
        assert self.rf.recent_file("qq:999:group") is None

    def test_max_cap(self):
        for i in range(20):
            self.rf.note_recent_file("c", {"name": f"f{i}", "size": 1, "url": "u"})
        assert len(self.rf.recent_files("c")) <= self.rf._RECENT_MAX


# ---------------------------------------------------------------- processor 投喂

class TestProcessorFeeding:
    @pytest.mark.asyncio
    async def test_file_refs_registered(self):
        """带文件的消息过 _pre_decision 后，登记处能查到（发文件->再@君君 场景）。"""
        from junjun_agent import processor as proc_mod
        from junjun_core.gateway.session_manager import ChatSession
        from junjun_memory import recent_files as rf
        from junjun_memory.short_term import ShortTermMemory
        rf._reset_for_test()
        session = ChatSession("qq:111:private", "qq")
        session.memory = ShortTermMemory()
        meta = InboundMeta(text="[文件：报表.xlsx（1.0KB）]", user_id="111", nickname="甲",
                           group_id=None, message_id="m1", at_bot=False, is_self=False,
                           file_refs=[{"name": "报表.xlsx", "size": 1024,
                                       "url": "https://cdn.qq.com/f/x"}])
        await proc_mod._pre_decision(session, meta)
        ref = rf.recent_file("qq:111:private")
        assert ref and ref["name"] == "报表.xlsx"
        rf._reset_for_test()

    @pytest.mark.asyncio
    async def test_self_message_not_registered(self):
        """bot 自己发的文件不登记（防自我归因）。"""
        from junjun_agent import processor as proc_mod
        from junjun_core.gateway.session_manager import ChatSession
        from junjun_memory import recent_files as rf
        from junjun_memory.short_term import ShortTermMemory
        rf._reset_for_test()
        session = ChatSession("qq:111:private", "qq")
        session.memory = ShortTermMemory()
        meta = InboundMeta(text="[文件]", user_id="111", nickname="君君",
                           group_id=None, message_id="m2", at_bot=False, is_self=True,
                           file_refs=[{"name": "x", "size": 1, "url": "u"}])
        await proc_mod._pre_decision(session, meta)
        assert rf.recent_file("qq:111:private") is None


# ---------------------------------------------------------------- workspace_save_file

class TestWorkspaceSaveFile:
    @pytest.fixture()
    def ws(self, tmp_path, monkeypatch):
        """工作区根指 tmp；会话钉在 qq:111:private；登记处清空。"""
        from junjun_skills.plugins.workspace import tools as wt
        monkeypatch.setattr(wt, "_ROOT", tmp_path / "ws")
        from junjun_skills.builtin.memory_skills import current_chat_id
        token = current_chat_id.set("qq:111:private")
        from junjun_memory import recent_files as rf
        rf._reset_for_test()
        yield tmp_path / "ws"
        current_chat_id.reset(token)
        rf._reset_for_test()

    def _seed(self, name="报表.xlsx", size=5, url="https://cdn.qq.com/f/x"):
        from junjun_memory import recent_files as rf
        rf.note_recent_file("qq:111:private", {"name": name, "size": size, "url": url})

    async def _call(self, args):
        """兼容 registry 包装（异常折 [TOOL_ERROR] 文本）与裸调两种上下文。"""
        from junjun_skills.plugins.workspace.tools import workspace_save_file
        try:
            return await workspace_save_file.ainvoke(args)
        except Exception as e:
            return f"[raised] {type(e).__name__}: {e}"

    @pytest.mark.asyncio
    async def test_save_happy(self, ws, monkeypatch):
        """闭环主路径：登记过文件 -> 下载（假）-> 二进制落盘工作区。"""
        from junjun_skills.plugins.workspace import tools as wt
        self._seed()
        monkeypatch.setattr(wt, "_download_capped", lambda url, cap: self._fake_dl(url, cap, b"PK\x03\x04xlsx"))
        out = await self._call({})
        assert "已把「报表.xlsx」存到工作区" in out
        assert (ws / "qq_111_private" / "报表.xlsx").read_bytes() == b"PK\x03\x04xlsx"

    async def _fake_dl(self, url, cap, data):
        assert url.startswith("https://")
        return data

    @pytest.mark.asyncio
    async def test_save_as_rename(self, ws, monkeypatch):
        from junjun_skills.plugins.workspace import tools as wt
        self._seed()
        monkeypatch.setattr(wt, "_download_capped", lambda url, cap: self._fake_dl(url, cap, b"abc"))
        out = await self._call({"save_as": "data/改名.csv"})
        assert "data/改名.csv" in out
        assert (ws / "qq_111_private" / "data" / "改名.csv").read_bytes() == b"abc"

    @pytest.mark.asyncio
    async def test_no_recent_file_friendly(self, ws):
        """没收到文件时给可转述的提示，不是报错——模型能直接答「先把文件发出来」。"""
        out = await self._call({})
        assert "没收到过文件" in out

    @pytest.mark.asyncio
    async def test_oversize_rejected(self, ws, monkeypatch):
        from junjun_skills.plugins.workspace import tools as wt
        self._seed()

        async def _boom(url, cap):
            raise OverflowError(url)
        monkeypatch.setattr(wt, "_download_capped", _boom)
        out = await self._call({})
        assert "50MB" in out
        assert not (ws / "qq_111_private" / "报表.xlsx").exists()

    @pytest.mark.asyncio
    async def test_traversal_blocked(self, ws, monkeypatch):
        """save_as 穿越必须被 _resolve 挡死（下载不该发生）。"""
        from junjun_skills.plugins.workspace import tools as wt
        self._seed()
        called = []
        monkeypatch.setattr(wt, "_download_capped",
                            lambda url, cap: called.append(url) or b"")
        out = await self._call({"save_as": "../escape.txt"})
        assert called == []
        assert "相对路径" in out or "越出工作区" in out

    @pytest.mark.asyncio
    async def test_bad_url_scheme(self, ws):
        self._seed(url="file:///etc/passwd")
        out = await self._call({})
        assert "http" in out

    @pytest.mark.asyncio
    async def test_download_capped_real_logic(self, monkeypatch):
        """_download_capped 本体：content-length 超帽直接拒；流式累计超帽也拒。"""
        import httpx
        from junjun_skills.plugins.workspace import tools as wt

        class _Resp:
            headers = {"content-length": str(60 * 1024 * 1024)}
            def raise_for_status(self):
                pass
            async def aiter_bytes(self, n):
                yield b"x"

        class _StreamCtx:
            async def __aenter__(self):
                return _Resp()
            async def __aexit__(self, *a):
                return False

        class _Client:
            def __init__(self, **kw):
                pass
            async def __aenter__(self):
                return self
            async def __aexit__(self, *a):
                return False
            def stream(self, method, url):
                return _StreamCtx()

        monkeypatch.setattr(httpx, "AsyncClient", _Client)
        with pytest.raises(OverflowError):
            await wt._download_capped("https://cdn.qq.com/big", 50 * 1024 * 1024)
