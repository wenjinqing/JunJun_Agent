"""peewee 数据库 models（阶段 3 实装建表）。

全部带 bot_id 字段（单 bot 架构预留，默认当前 QQ 号）。
SQLite WAL 模式；写操作统一走 writer 队列防并发锁。
"""

import os
import time
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
    # JUNJUN_QQ_ACCOUNT 为准，MAIBOT_QQ_ACCOUNT 为旧名兼容兜底
    return os.environ.get("JUNJUN_QQ_ACCOUNT") or os.environ.get("MAIBOT_QQ_ACCOUNT", "")


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
    user_id = CharField(default="", index=True)   # 2026-08-13 补索引（按人查记录
                                                  # 全表扫）；老库由 _ensure_indexes 补建
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


class Intimacy(BaseModel):
    """好感度（插件迁移：intimacy_query）。按用户累计互动分，0~100。"""
    id = AutoField()
    bot_id = CharField(default=_bot_id, index=True)
    user_id = CharField(index=True)
    platform = CharField(default="qq")
    score = FloatField(default=0.0)           # 好感度 0~100
    interaction_count = IntegerField(default=0)
    last_interaction = FloatField(default=0.0)
    daily_gain = FloatField(default=0.0)      # 当日已涨（防刷，每天重置）
    daily_date = CharField(default="")        # daily_gain 对应日期 YYYY-MM-DD


class SelfMood(BaseModel):
    """全局自我心境：跨场景持续、DB 持久化，由聊天情绪评估与日记共同塑造。"""
    id = AutoField()
    bot_id = CharField(default=_bot_id, unique=True)
    state = CharField(default="平静")
    reason = CharField(default="")        # 心境来源（哪个场景/日记）
    updated_at = FloatField(default=0.0)


class DiaryEntry(BaseModel):
    """日记：每天一篇第一人称自我叙事，同时进长期记忆（self:diary 域）。"""
    id = AutoField()
    bot_id = CharField(default=_bot_id, index=True)
    date = CharField(index=True)          # YYYY-MM-DD
    content = TextField()
    mood = CharField(default="")
    created_at = FloatField(default=time.time)


class SelfIdentity(BaseModel):
    """Identity Core（P6-3）：从日记蒸馏出的自我认知条目（我喜欢/我看不惯/
    我们的梗/最近在乎），人设漂移对冲的第二锚点。旧条目折叠归档不删。"""
    id = AutoField()
    bot_id = CharField(default=_bot_id, index=True)
    category = CharField(default="")      # 我喜欢 / 我看不惯 / 我们的梗 / 最近在乎
    content = CharField(default="")
    weight = FloatField(default=1.0)      # 未再确认则逐轮衰减
    seen_count = IntegerField(default=1)  # 反复出现次数（防一次 emo 固化成人设）
    archived = BooleanField(default=False)
    created_at = FloatField(default=time.time)
    updated_at = FloatField(default=time.time)


class UserSceneProfile(BaseModel):
    """跨场景用户档案（P6-4）：按 user_id 聚合的蒸馏事实（不落原文），
    每条带来源场景标签——隐私生命线：群聊注入前必须过滤私聊来源的条目，
    多群之间默认隔离（A 群的事不在 B 群说）。"""
    id = AutoField()
    bot_id = CharField(default=_bot_id, index=True)
    person_id = CharField(index=True)     # platform:user_id
    platform = CharField(default="qq")
    user_id = CharField(index=True)
    content = CharField(default="")       # 蒸馏事实（一句话，非原文）
    source_scene = CharField(default="group")   # group / private
    source_chat_id = CharField(default="", index=True)  # 来源会话（多群隔离）
    weight = FloatField(default=1.0)
    created_at = FloatField(default=time.time)
    updated_at = FloatField(default=time.time)


class Intention(BaseModel):
    """意向（P7 意向系统）：她想做的事——不是因为被 @。持久化队列，重启不丢。
    生成源三类：事件（有人 emo/订阅更新）、定时（晨起巡检）、反思（日记复盘）。
    评估门（quiet hours/日限额/去重/亲密度）过了才生成消息发出。"""
    id = AutoField()
    bot_id = CharField(default=_bot_id, index=True)
    kind = CharField(index=True)       # care_followup / diary_plan / morning_greet
    chat_id = CharField(index=True)    # 目标会话
    user_id = CharField(default="")    # 关心对象（群聊场景）
    user_nickname = CharField(default="")
    motive = CharField(default="")     # 动机摘要（生成消息的种子）
    priority = IntegerField(default=5) # 1 最高 9 最低
    status = CharField(index=True, default="pending")  # pending/fired/expired/dropped
    created_at = FloatField(default=time.time)
    expires_at = FloatField(default=0.0)   # 过期即焚
    fired_at = FloatField(default=0.0)


class SkillPatch(BaseModel):
    """技能补丁（P8-2 Memento 思路，基座冻结）：从工具失败复盘出的 prompt
    补丁，注入对应工具 description 后缀。版本化 + 可回滚 + 定期合并防膨胀。
    门控：candidate 必须经管理员启用（人工即单测门控的审查环），
    pytest 回放保证注入链路与 source_case 完备。"""
    id = AutoField()
    bot_id = CharField(default=_bot_id, index=True)
    tool = CharField(index=True)       # 目标工具名
    patch = TextField()                # 注入 description 后缀的补丁文本
    source_case = TextField(default="")  # 依据的失败模式摘要（无此不许激活）
    version = IntegerField(default=1)
    status = CharField(index=True, default="candidate")  # candidate/active/rolled_back/merged
    created_at = FloatField(default=time.time)
    updated_at = FloatField(default=time.time)


class ShortTermMemory(BaseModel):
    """短期记忆持久化（Phase 2）：进程重启后可恢复最近对话上下文。"""
    id = AutoField()
    bot_id = CharField(default=_bot_id, index=True)
    chat_id = CharField(unique=True)   # 会话键 platform:id:type
    entries_json = TextField(default="[]")  # JSON: [{role,text,nickname,user_id,message_id,at_bot}]
    updated_at = FloatField(default=time.time)


class AsyncJob(BaseModel):
    """异步任务：接单->后台跑->主动汇报的持久化队列（重启可恢复，区别见 tasks.py 注释）。"""
    id = AutoField()
    bot_id = CharField(default=_bot_id, index=True)
    job_id = CharField(unique=True)       # 短 uuid，对用户展示用
    kind = CharField(index=True)          # agent_task / 未来：video_transcode...
    title = CharField(default="")         # 一句话任务说明（列表/汇报用）
    payload = TextField(default="{}")     # JSON，handler 的输入
    status = CharField(index=True, default="pending")  # pending/running/done/failed/cancelled
    result = TextField(default="")
    error = CharField(default="")
    chat_id = CharField(index=True)       # 汇报到哪个会话
    user_id = CharField(default="")       # 委托人（取消权限判定用）
    user_nickname = CharField(default="")
    attempts = IntegerField(default=0)    # 执行次数（崩溃残留重试计数）
    created_at = FloatField(default=time.time)
    started_at = FloatField(default=0.0)
    finished_at = FloatField(default=0.0)


class Subscription(BaseModel):
    """订阅：Agent 自然语言创建的常驻监视任务（P站作者/B站UP主更新等）。"""
    id = AutoField()
    bot_id = CharField(default=_bot_id, index=True)
    kind = CharField(index=True)          # pixiv_author / bili_up
    target_id = CharField()               # 作者UID / UP主mid
    target_name = CharField(default="")   # 显示名（首次检查时回填）
    chat_id = CharField(index=True)       # 通知到哪个会话
    user_id = CharField(default="")       # 创建者（删除权限判定用）
    user_nickname = CharField(default="")
    last_seen = CharField(default="")     # 已见最新（pixiv 小说id / b站 pubdate 时间戳）
    interval_minutes = IntegerField(default=30)
    enabled = BooleanField(default=True)
    created_at = FloatField(default=time.time)
    last_checked = FloatField(default=0.0)


class OutboxMessage(BaseModel):
    """出站暂存（WS outbox）：adapter 断连期的回复不落空，重连后回放。
    2026-08 观察：gateway→adapter 断连时回复仅记日志丢弃——用户侧表现为
    「君君突然不理人」。TTL + 次数上限防陈旧消息轰炸。"""
    id = AutoField()
    bot_id = CharField(default=_bot_id, index=True)
    platform = CharField(index=True)
    target_group_id = CharField(default="")
    target_user_id = CharField(default="")
    payload_json = TextField()            # MessageBase dict
    created_ts = FloatField(index=True)
    attempts = IntegerField(default=0)    # 回放尝试次数（上限后丢弃）


class BlockedUser(BaseModel):
    """屏蔽名单：管理员按会话拉黑某人（多为其他 bot）。其消息记完记忆即丢弃，
    不进决策/命令/拦截器（0 token）——防 bot 互回循环的硬闸（2026-08-13 用户裁决）。"""
    id = AutoField()
    bot_id = CharField(default=_bot_id, index=True)
    chat_id = CharField(index=True)       # 会话键 platform:id:type
    user_id = CharField()
    created_by = CharField(default="")    # 操作的管理员 QQ
    created_ts = FloatField(default=time.time)

    class Meta:
        indexes = ((("chat_id", "user_id"), True),)


ALL_TABLES = [Messages, Images, LLMUsage, PersonInfo, Jargon, Expression, Emoji, ReminderTasks,
              OnlineTime, Intimacy, SelfMood, DiaryEntry, Subscription, AsyncJob, SelfIdentity,
              UserSceneProfile, Intention, SkillPatch, ShortTermMemory, OutboxMessage,
              BlockedUser]


def _ensure_columns(database, models=None) -> None:
    """轻量加列迁移（2026-08-13 审查 P1）：模型新增字段 -> ALTER TABLE ADD COLUMN。

    create_tables(safe=True) 只补缺表不补列——此前给模型加字段，生产库不会
    跟着长，peewee 显式列查询直接 OperationalError「no such column」。
    只加不改不删；无默认值且非空的列跳过并告警（SQLite 加列只支持常量默认，
    这类列得人工迁移）——宁可跳过一列，也不能让启动炸死。
    database 显式传入：测试用 tmp 库调本函数，绝不许碰全局生产句柄。
    """
    from playhouse.migrate import SqliteMigrator, migrate

    from junjun_core.observability import get_logger
    logger = get_logger("db.migrate")
    migrator = SqliteMigrator(database)
    for model in (models or ALL_TABLES):
        table = model._meta.table_name
        existing = {row[1] for row in
                    database.execute_sql(f"PRAGMA table_info({table})")}
        if not existing:
            continue  # 表还没建（create_tables 会先跑）
        for f in model._meta.sorted_fields:
            if f.column_name in existing:
                continue
            if not f.null and f.default is None:
                logger.warning(f"{table}.{f.column_name} 无默认值且非空，"
                               "自动加列跳过（需人工迁移）")
                continue
            try:
                migrate(migrator.add_column(table, f.column_name, f))
                logger.info(f"表 {table} 自动加列: {f.column_name}")
            except Exception as e:
                logger.warning(f"表 {table} 加列 {f.column_name} 失败（跳过）: "
                               f"{type(e).__name__}: {e}")


def _ensure_indexes(database, models=None) -> None:
    """既有表补建缺失的单列索引（create_tables 不管老表的索引）。

    只补 index=True 的单列索引；复合索引（Meta.indexes）与 unique 约束
    不在自动范围（unique 对存量重复数据会炸，得人工来）。
    """
    from junjun_core.observability import get_logger
    logger = get_logger("db.migrate")
    for model in (models or ALL_TABLES):
        table = model._meta.table_name
        existing = {r[0] for r in database.execute_sql(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name=?",
            (table,))}
        for f in model._meta.sorted_fields:
            if not f.index:
                continue
            name = f"{table}_{f.column_name}"
            if name in existing:
                continue
            try:
                database.execute_sql(
                    f'CREATE INDEX IF NOT EXISTS "{name}" '
                    f'ON "{table}" ("{f.column_name}")')
                logger.info(f"表 {table} 补建索引: {name}")
            except Exception as e:
                logger.warning(f"表 {table} 补建索引 {name} 失败（跳过）: "
                               f"{type(e).__name__}: {e}")


def init_database() -> None:
    """建表（幂等）+ 加列对齐（模型新字段自动 ALTER TABLE）+ 补缺失索引。"""
    db.connect(reuse_if_open=True)
    db.create_tables(ALL_TABLES, safe=True)
    _ensure_columns(db)
    _ensure_indexes(db)
