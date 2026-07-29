"""Adapter 断线韧性测试：NapCat 掉线等待重连 + 连接状态清理。"""

import asyncio

import pytest
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK

import junjun_adapter_napcat.send_handler.nc_sending as nc
from junjun_adapter_napcat.send_handler.nc_sending import NCMessageSender


class _Conn:
    """假 WS 连接：send 记录 payload，可按需抛断连。"""

    def __init__(self):
        self.sent = []

    async def send(self, payload):
        self.sent.append(payload)


class TestSendResilience:
    @pytest.mark.asyncio
    async def test_send_waits_for_reconnect(self, monkeypatch):
        """无连接时发送挂起，NapCat 重连后消息仍送达（不丢消息）。"""
        sender = NCMessageSender()
        conn = _Conn()

        async def _fake_get_response(request_uuid, timeout=15):
            return {"status": "ok", "echo": request_uuid}

        monkeypatch.setattr(nc, "get_response", _fake_get_response)

        task = asyncio.create_task(sender.send_message_to_napcat("send_msg", {}))
        await asyncio.sleep(0.05)
        assert not task.done()  # 在等待重连，未直接报错
        await sender.set_server_connection(conn)
        result = await task
        assert result["status"] == "ok"
        assert len(conn.sent) == 1  # 消息发出去了

    @pytest.mark.asyncio
    async def test_send_fails_after_wait_timeout(self, monkeypatch):
        """重连等待超时仍未连上：返回 error 而不是无限挂起。"""
        monkeypatch.setattr(nc, "_RECONNECT_WAIT", 0.05)
        sender = NCMessageSender()
        result = await sender.send_message_to_napcat("send_msg", {})
        assert result["status"] == "error"
        assert result["message"] == "no connection"

    @pytest.mark.asyncio
    async def test_send_on_closed_conn_returns_error(self, monkeypatch):
        """连接对象存在但已死：send 抛异常 -> error dict，不向外冒泡。"""
        class _DeadConn:
            async def send(self, payload):
                raise ConnectionClosedError(None, None)

        sender = NCMessageSender()
        await sender.set_server_connection(_DeadConn())
        result = await sender.send_message_to_napcat("send_msg", {})
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_wait_connected_immediate_when_connected(self):
        sender = NCMessageSender()
        await sender.set_server_connection(_Conn())
        assert await sender.wait_connected(timeout=0.01) is True


class _ClosedIter:
    """async for 第一轮即抛断连的假连接。"""

    def __init__(self, exc):
        self._exc = exc

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise self._exc


class TestMessageRecvDisconnect:
    @pytest.mark.asyncio
    async def test_abnormal_close_swallowed_and_cleared(self):
        """ConnectionClosedError 不再炸 traceback；发送器连接被清空。"""
        from junjun_adapter_napcat.main import message_recv
        from junjun_adapter_napcat.send_handler.nc_sending import nc_message_sender

        conn = _ClosedIter(ConnectionClosedError(None, None))
        await message_recv(conn)
        assert nc_message_sender.server_connection is None

    @pytest.mark.asyncio
    async def test_normal_close_swallowed(self):
        from junjun_adapter_napcat.main import message_recv

        conn = _ClosedIter(ConnectionClosedOK(None, None))
        await message_recv(conn)  # 不抛异常
