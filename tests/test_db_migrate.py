"""加列迁移测试（2026-08-13 审查 P1）：_ensure_columns 把模型新字段 ALTER 进旧库。

背景：create_tables(safe=True) 只补缺表不补列——给模型加字段后生产库不跟着长，
peewee 显式列查询直接 OperationalError「no such column」。
绝不碰生产库：全程 tmp 库 + tdb.bind_ctx（CLAUDE.md 硬约束）。
"""

import peewee
import pytest

import junjun_core.database.models as m


@pytest.fixture()
def old_db(tmp_path):
    """手工建一张「旧版」messages 表（缺后加的列）+ 一行老数据，模拟生产老库。"""
    tdb = peewee.SqliteDatabase(str(tmp_path / "t.db"))
    tdb.connect()
    t = m.Messages._meta.table_name
    tdb.execute_sql(
        f"CREATE TABLE {t} (id INTEGER PRIMARY KEY, bot_id VARCHAR(255), "
        "message_id VARCHAR(255), chat_id VARCHAR(255), time REAL, "
        "user_id VARCHAR(255), user_nickname VARCHAR(255), group_id VARCHAR(255), "
        "processed_plain_text TEXT, is_bot INTEGER, is_mentioned INTEGER)")
    # 缺：is_at / reply_to / is_emoji / is_picid（假设是后加的列）
    tdb.execute_sql(
        f"INSERT INTO {t} (bot_id, message_id, chat_id, time) "
        "VALUES ('b', 'm1', 'c1', 1.0)")
    yield tdb
    tdb.close()


class TestEnsureColumns:
    def test_missing_columns_added(self, old_db):
        with old_db.bind_ctx([m.Messages]):
            m._ensure_columns(old_db, [m.Messages])
        cols = {r[1] for r in old_db.execute_sql(
            f"PRAGMA table_info({m.Messages._meta.table_name})")}
        for c in ("is_at", "reply_to", "is_emoji", "is_picid"):
            assert c in cols

    def test_old_rows_defaults_and_new_insert(self, old_db):
        """老行按默认值读出；新行带新列插得进（迁移不是摆设，业务真能跑）。"""
        with old_db.bind_ctx([m.Messages]):
            m._ensure_columns(old_db, [m.Messages])
            row = m.Messages.get(m.Messages.message_id == "m1")
            assert row.reply_to == "" and row.is_at is False
            m.Messages.create(message_id="m2", chat_id="c1", time=2.0,
                              reply_to="m1", is_emoji=True)
            assert m.Messages.get(m.Messages.message_id == "m2").is_emoji is True

    def test_idempotent(self, old_db):
        with old_db.bind_ctx([m.Messages]):
            m._ensure_columns(old_db, [m.Messages])
            m._ensure_columns(old_db, [m.Messages])  # 二刷无操作不炸

    def test_callable_default_column(self, tmp_path):
        """callable 默认（default=time.time）的列也能加——DDL 不带 DEFAULT 子句，
        老行由 peewee 层兜默认，迁移期不许炸。"""
        tdb = peewee.SqliteDatabase(str(tmp_path / "t3.db"))
        tdb.connect()
        t = m.DiaryEntry._meta.table_name
        tdb.execute_sql(
            f"CREATE TABLE {t} (id INTEGER PRIMARY KEY, bot_id VARCHAR(255), "
            "date VARCHAR(255), content TEXT, mood VARCHAR(255))")
        # 缺 created_at（FloatField(default=time.time)）
        with tdb.bind_ctx([m.DiaryEntry]):
            m._ensure_columns(tdb, [m.DiaryEntry])
            new = m.DiaryEntry.create(date="2026-08-13", content="x")
            assert new.created_at > 0
        cols = {r[1] for r in tdb.execute_sql(f"PRAGMA table_info({t})")}
        assert "created_at" in cols
        tdb.close()

    def test_unsafe_column_skipped_not_crash(self, tmp_path, capsys):
        """无默认值且非空的新列：跳过+告警，不炸启动（SQLite 加列只支持常量默认）。"""
        tdb = peewee.SqliteDatabase(str(tmp_path / "t2.db"))
        tdb.connect()

        class Unsafe(peewee.Model):
            id = peewee.AutoField()
            name = peewee.CharField()  # 无默认、非空——自动加列必须拒

            class Meta:
                database = tdb
                table_name = "unsafe_t"

        tdb.execute_sql("CREATE TABLE unsafe_t (id INTEGER PRIMARY KEY)")
        m._ensure_columns(tdb, [Unsafe])
        cols = {r[1] for r in tdb.execute_sql("PRAGMA table_info(unsafe_t)")}
        assert "name" not in cols
        assert "跳过" in capsys.readouterr().out
        tdb.close()

class TestEnsureIndexes:
    def test_missing_index_created(self, old_db):
        """老表缺 user_id 索引（2026-08-13 补的）——启动时自动补建。"""
        with old_db.bind_ctx([m.Messages]):
            m._ensure_indexes(old_db, [m.Messages])
        t = m.Messages._meta.table_name
        idx = {r[0] for r in old_db.execute_sql(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name=?", (t,))}
        assert f"{t}_user_id" in idx
        # 既有数据不受补索引影响
        assert old_db.execute_sql(f"SELECT COUNT(*) FROM {t}").fetchone()[0] == 1

    def test_idempotent(self, old_db):
        with old_db.bind_ctx([m.Messages]):
            m._ensure_indexes(old_db, [m.Messages])
            m._ensure_indexes(old_db, [m.Messages])  # 二刷无操作不炸


class TestMessagesRetention:
    def test_old_messages_purged_new_kept(self, tmp_path):
        """messages_retention_days 窗外的老消息清理，窗内保留。"""
        import time as _time
        from junjun_core.database.cleanup import _do_cleanup
        tdb = peewee.SqliteDatabase(str(tmp_path / "t4.db"))
        with tdb.bind_ctx(m.ALL_TABLES):
            tdb.create_tables(m.ALL_TABLES)
            now = _time.time()
            m.Messages.create(message_id="old", chat_id="c", time=now - 400 * 86400)
            m.Messages.create(message_id="new", chat_id="c", time=now - 10 * 86400)
            _do_cleanup(cutoff=now - 60 * 86400, msg_cutoff=now - 365 * 86400)
            left = {r.message_id for r in m.Messages.select()}
            assert left == {"new"}
        tdb.close()

    def test_zero_cutoff_keeps_everything(self, tmp_path):
        """messages_retention_days=0 -> msg_cutoff=0 -> 一条不动（保守方向）。"""
        import time as _time
        from junjun_core.database.cleanup import _do_cleanup
        tdb = peewee.SqliteDatabase(str(tmp_path / "t5.db"))
        with tdb.bind_ctx(m.ALL_TABLES):
            tdb.create_tables(m.ALL_TABLES)
            now = _time.time()
            m.Messages.create(message_id="ancient", chat_id="c", time=now - 3000 * 86400)
            _do_cleanup(cutoff=now - 60 * 86400, msg_cutoff=0.0)
            assert m.Messages.select().count() == 1
        tdb.close()
