"""短期记忆持久化测试（Phase 2）：重启后上下文可恢复。

DB 用内存库 bind_ctx 隔离，绝不写 data/junjun.db。
"""

import pytest
from peewee import SqliteDatabase

import junjun_core.config.config as cfg_mod
from junjun_core.database import models as m
from junjun_memory.short_term import ShortTermMemory

_mem_db = SqliteDatabase(":memory:")


@pytest.fixture
def db_env():
    old = cfg_mod.global_config
    cfg_mod.global_config = cfg_mod.GlobalConfig(
        bot=cfg_mod.BotConfig(platform="qq", qq_account="12345", nickname="君君"),
        raw={},
    )
    with _mem_db.bind_ctx([m.ShortTermMemory]):
        _mem_db.create_tables([m.ShortTermMemory])
        m.ShortTermMemory.delete().execute()
        yield
    cfg_mod.global_config = old


def test_persist_load_empty(db_env):
    mem = ShortTermMemory(chat_id="qq:1:group", persist=True)
    assert mem.entries == []


def test_persist_save_and_reload(db_env):
    mem1 = ShortTermMemory(chat_id="qq:1:group", persist=True)
    mem1.add_user("你好", "小明", user_id="1001", message_id="m1")
    mem1.add_bot("嗨")
    mem1._flush()  # 3s 节流：测试立即断言需手动补落（生产由 Timer/关停兜底）

    # 模拟重启：新实例读同一 chat_id
    mem2 = ShortTermMemory(chat_id="qq:1:group", persist=True)
    assert len(mem2.entries) == 2
    assert mem2.entries[0].text == "你好"
    assert mem2.entries[0].nickname == "小明"
    assert mem2.entries[1].role == "bot"
    assert mem2.entries[1].text == "嗨"


def test_no_persist_without_chat_id(db_env):
    mem = ShortTermMemory(persist=True)  # chat_id 空
    mem.add_user("你好", "小明")
    assert mem.entries
    # 不报错、不写库
    assert m.ShortTermMemory.select().count() == 0


def test_no_persist_flag_no_db_write(db_env):
    mem = ShortTermMemory(chat_id="qq:2:group", persist=False)
    mem.add_user("你好", "小明")
    assert m.ShortTermMemory.select().count() == 0


def test_persist_respects_max_size(db_env):
    mem1 = ShortTermMemory(chat_id="qq:1:group", persist=True, max_size=2)
    mem1.add_user("a", "u")
    mem1.add_user("b", "u")
    mem1.add_user("c", "u")
    mem1._flush()  # 同上：节流窗口内手动补落
    assert len(mem1.entries) == 2

    mem2 = ShortTermMemory(chat_id="qq:1:group", persist=True, max_size=2)
    assert len(mem2.entries) == 2
    assert mem2.entries[0].text == "b"
    assert mem2.entries[1].text == "c"


def test_save_throttled_and_trailing_flush(db_env):
    """3s 节流：窗口内多次变更只落一次，尾部 Timer 补落（2026-08-09 写放大修复）。"""
    import time as _t
    mem = ShortTermMemory(chat_id="qq:9:group", persist=True)
    mem.add_user("第一条", "u")   # 立即落
    mem.add_user("第二条", "u")   # 节流窗口内 -> 延迟
    # 此刻库里可能还是一条（取决于节流），补落后必须两条
    mem._flush()
    mem2 = ShortTermMemory(chat_id="qq:9:group", persist=True)
    assert [e.text for e in mem2.entries] == ["第一条", "第二条"]


def test_timer_flush_actually_fires(db_env, monkeypatch):
    """尾部 Timer 真会补落（不等手动 _flush）。

    注：:memory: SQLite 连接按线程隔离，真 Timer 线程写的库主线程看不到——
    把 Timer 换成立即执行的假实现，测「调度逻辑」而非线程本身。"""
    import junjun_memory.short_term as st

    class _ImmediateTimer:
        def __init__(self, _interval, fn):
            self._fn = fn
        daemon = True
        def start(self):
            self._fn()

    monkeypatch.setattr(st.threading, "Timer", _ImmediateTimer)
    import time as _t
    mem = ShortTermMemory(chat_id="qq:8:group", persist=True)
    mem._last_save = _t.time()       # 假装刚落过，下一条必走节流
    mem.add_user("延迟保存", "u")     # -> 调度尾部 flush（假 Timer 立即执行）
    mem2 = ShortTermMemory(chat_id="qq:8:group", persist=True)
    assert any(e.text == "延迟保存" for e in mem2.entries), "Timer 尾部补落未生效"
