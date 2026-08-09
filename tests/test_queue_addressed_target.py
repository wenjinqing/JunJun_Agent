"""Q1 回归（2026-08-09 温衿青事故）：
会话队列合并时，决策目标必须回拨到「最新 addressed 消息」——
@bot 的提问不能被后进闲聊顶替（顶替后决策门判沉默，@必回 契约破坏）。

误判回归同步覆盖：全非 addressed 维持取最新（决策门判沉默的闲聊合批路径
不得被弄坏）；无 addressed_fn 保持旧行为；私聊语义不变。
"""

import asyncio
import time
from types import SimpleNamespace

import pytest

from junjun_agent.funnel.session_queue import SessionQueue


def _meta(text, at_bot=False):
    return SimpleNamespace(text=text, at_bot=at_bot)


def _mk_queue(chat_id, handled, pre_seen=None, addressed=True):
    gate = asyncio.Event()

    async def handler(session, meta):
        handled.append(meta.text)
        gate.set()

    async def pre_handler(session, meta):
        pre_seen.append(meta.text)

    addressed_fn = (lambda s, m: m.at_bot) if addressed else None
    q = SessionQueue(chat_id, handler,
                     pre_handler=pre_handler if pre_seen is not None else None,
                     addressed_fn=addressed_fn)
    return q, gate


def _fill(q, session, metas):
    now = time.time()
    for m in metas:
        q._queue.put_nowait((session, m, now))


class TestAddressedTargetSelection:
    @pytest.mark.asyncio
    async def test_at_message_beats_later_chatter(self):
        """事故复现：[闲聊, @bot提问, 闲聊] -> 决策目标必须是 @bot 那条。"""
        handled = []
        session = SimpleNamespace(chat_id="qq:t1:group")
        q, gate = _mk_queue("qq:t1:group", handled)
        _fill(q, session, [_meta("哈哈哈"), _meta("@你 在吗", at_bot=True),
                           _meta("老婆老婆")])
        q.start()
        await asyncio.wait_for(gate.wait(), timeout=5)
        await q.stop()
        assert handled == ["@你 在吗"]

    @pytest.mark.asyncio
    async def test_two_at_messages_pick_latest_at(self):
        """两人先后 @bot -> 回最新 @bot 那条（早的那条已在 STM 上下文里）。"""
        handled = []
        session = SimpleNamespace(chat_id="qq:t2:group")
        q, gate = _mk_queue("qq:t2:group", handled)
        _fill(q, session, [_meta("@你 第一个问题", at_bot=True),
                           _meta("插话"),
                           _meta("@你 第二个问题", at_bot=True)])
        q.start()
        await asyncio.wait_for(gate.wait(), timeout=5)
        await q.stop()
        assert handled == ["@你 第二个问题"]

    @pytest.mark.asyncio
    async def test_all_chatter_keeps_latest(self):
        """误判回归：全非 addressed -> 维持取最新（闲聊合批路径不变）。"""
        handled = []
        session = SimpleNamespace(chat_id="qq:t3:group")
        q, gate = _mk_queue("qq:t3:group", handled)
        _fill(q, session, [_meta("第一条"), _meta("第二条"), _meta("第三条")])
        q.start()
        await asyncio.wait_for(gate.wait(), timeout=5)
        await q.stop()
        assert handled == ["第三条"]

    @pytest.mark.asyncio
    async def test_no_addressed_fn_keeps_old_behavior(self):
        """误判回归：未注入 addressed_fn -> 严格旧行为（取最新）。"""
        handled = []
        session = SimpleNamespace(chat_id="qq:t4:group")
        q, gate = _mk_queue("qq:t4:group", handled, addressed=False)
        _fill(q, session, [_meta("@你 在吗", at_bot=True), _meta("灌水")])
        q.start()
        await asyncio.wait_for(gate.wait(), timeout=5)
        await q.stop()
        assert handled == ["灌水"]

    @pytest.mark.asyncio
    async def test_non_target_messages_all_run_pre(self):
        """被合并的消息（含目标之后的闲聊）都必须过决策前段——命令不丢。"""
        handled, pre_seen = [], []
        session = SimpleNamespace(chat_id="qq:t5:group")
        q, gate = _mk_queue("qq:t5:group", handled, pre_seen=pre_seen)
        _fill(q, session, [_meta("@你 在吗", at_bot=True),
                           _meta("/sub add 123"),   # 目标之后的命令也不能丢
                           _meta("灌水")])
        q.start()
        await asyncio.wait_for(gate.wait(), timeout=5)
        await q.stop()
        assert handled == ["@你 在吗"]
        assert "/sub add 123" in pre_seen
        assert "灌水" in pre_seen

    @pytest.mark.asyncio
    async def test_addressed_fn_exception_falls_back_to_latest(self):
        """addressed 判定异常 -> 保守回退取最新（退化成现状，不丢决策）。"""
        handled = []
        gate = asyncio.Event()

        async def handler(session, meta):
            handled.append(meta.text)
            gate.set()

        def bad_fn(session, meta):
            raise RuntimeError("boom")

        session = SimpleNamespace(chat_id="qq:t6:group")
        q = SessionQueue("qq:t6:group", handler, addressed_fn=bad_fn)
        _fill(q, session, [_meta("@你 在吗", at_bot=True), _meta("灌水")])
        q.start()
        await asyncio.wait_for(gate.wait(), timeout=5)
        await q.stop()
        assert handled == ["灌水"]
