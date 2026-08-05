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
    assert len(mem1.entries) == 2

    mem2 = ShortTermMemory(chat_id="qq:1:group", persist=True, max_size=2)
    assert len(mem2.entries) == 2
    assert mem2.entries[0].text == "b"
    assert mem2.entries[1].text == "c"
