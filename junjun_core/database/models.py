"""peewee 数据库 models（阶段 3 实装建表）。

表结构对齐原 MaiBot database_model.py，全部带 bot_id 字段（单 bot 架构预留，
默认当前 QQ 号）。SQLite WAL 模式；写操作统一走 writer 队列防并发锁。
"""

import os
from pathlib import Path

from peewee import (
    SqliteDatabase, Model, AutoField, CharField, TextField,
    FloatField, BooleanField, IntegerField,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

db = SqliteDatabase(
    str(DATA_DIR / "junjun.db"),
    pragmas={
        "journal_mode": "wal",
        "cache_size": -1024 * 32,
        "foreign_keys": 1,
        "synchronous": 1,
    },
)


def _bot_id() -> str:
    return os.environ.get("MAIBOT_QQ_ACCOUNT", "")


class BaseModel(Model):
    class Meta:
        database = db


class Messages(BaseModel):
    """消息记录（入站 + bot 回复都落）。"""
    id = AutoField()
    bot_id = CharField(default=_bot_id, index=True)
    message_id = CharField(index=True)
    chat_id = CharField(index=True)          # 会话键 platform:id:type
    time = FloatField(index=True)
    user_id = CharField(default="")
    user_nickname = CharField(default="")
    group_id = CharField(default="", index=True)
    processed_plain_text = TextField(default="")
    is_bot = BooleanField(default=False)      # bot 自己的回复
    is_mentioned = BooleanField(default=False)
    is_at = BooleanField(default=False)
    reply_to = CharField(default="")          # 引用的 message_id
    is_emoji = BooleanField(default=False)
    is_picid = BooleanField(default=False)


class Images(BaseModel):
    """图片识别缓存（hash 去重）。"""
    id = AutoField()
    bot_id = CharField(default=_bot_id, index=True)
    image_hash = CharField(unique=True)
    description = TextField(default="")
    timestamp = FloatField()


class LLMUsage(BaseModel):
    """token 用量统计。"""
    id = AutoField()
    bot_id = CharField(default=_bot_id, index=True)
    time = FloatField(index=True)
    model_name = CharField(default="")
    request_type = CharField(default="")      # gate / agent / utils / vlm...
    prompt_tokens = IntegerField(default=0)
    completion_tokens = IntegerField(default=0)
    chat_id = CharField(default="")


class PersonInfo(BaseModel):
    """用户画像（阶段 4 实装逻辑，表先建）。"""
    id = AutoField()
    bot_id = CharField(default=_bot_id, index=True)
    person_id = CharField(unique=True)        # MD5(platform+user_id)
    platform = CharField(default="qq")
    user_id = CharField(index=True)
    person_name = CharField(default="")
    memory_points = TextField(default="[]")   # JSON: ["分类:内容:权重", ...]


class Jargon(BaseModel):
    """黑话（阶段 4）。"""
    id = AutoField()
    bot_id = CharField(default=_bot_id, index=True)
    term = CharField(index=True)
    explanation = TextField(default="")
    chat_id = CharField(default="")           # all_global=true 时为空
    count = IntegerField(default=1)


class Expression(BaseModel):
    """表达学习（阶段 5）。"""
    id = AutoField()
    bot_id = CharField(default=_bot_id, index=True)
    chat_id = CharField(index=True)
    situation = TextField(default="")
    style = TextField(default="")
    count = IntegerField(default=1)
    last_active_time = FloatField(default=0.0)


class Emoji(BaseModel):
    """表情包库（阶段 5）。"""
    id = AutoField()
    bot_id = CharField(default=_bot_id, index=True)
    full_path = CharField(unique=True)
    emoji_hash = CharField(index=True)
    description = TextField(default="")
    emotion = TextField(default="[]")         # JSON list
    usage_count = IntegerField(default=0)


class ReminderTasks(BaseModel):
    """提醒任务（阶段 5，重启恢复依赖此表）。"""
    id = AutoField()
    bot_id = CharField(default=_bot_id, index=True)
    task_id = CharField(unique=True)
    chat_id = CharField(index=True)
    user_id = CharField(default="")
    content = TextField(default="")
    remind_time = FloatField(index=True)
    repeat_type = CharField(default="")       # "" / daily / weekly
    is_completed = BooleanField(default=False)
    is_cancelled = BooleanField(default=False)


class OnlineTime(BaseModel):
    """在线时长记录（对齐原 OnlineTimeRecordTask：每分钟续 end_timestamp）。"""
    id = AutoField()
    bot_id = CharField(default=_bot_id, index=True)
    start_timestamp = FloatField()
    end_timestamp = FloatField(index=True)


ALL_TABLES = [Messages, Images, LLMUsage, PersonInfo, Jargon, Expression, Emoji, ReminderTasks, OnlineTime]


def init_database() -> None:
    """建表（幂等）。"""
    db.connect(reuse_if_open=True)
    db.create_tables(ALL_TABLES, safe=True)
