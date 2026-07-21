"""好感度核心（插件迁移：intimacy_query 的存储/累计层，core 侧）。

按互动累计 0~100：普通发言 +0.1，@bot/直呼 +0.3（与普通发言不叠加），
每日最多 +3.0 防刷。processor 每条用户消息调 note_interaction（core 功能，
不依赖插件启用）；查询面在 plugins/intimacy（命令 + tool）。
"""

import datetime
import time

from junjun_core.observability import get_logger

logger = get_logger("express.intimacy")

MAX_SCORE = 100.0
DAILY_CAP = 3.0
GAIN_NORMAL = 0.1
GAIN_ADDRESSED = 0.3

_LEVELS = [
    (90.0, "挚友"), (70.0, "好朋友"), (50.0, "朋友"),
    (30.0, "熟人"), (10.0, "认识"), (0.0, "陌生"),
]


def level_name(score: float) -> str:
    """好感度等级称号。"""
    for threshold, name in _LEVELS:
        if score >= threshold:
            return name
    return "陌生"


def note_interaction(user_id: str, *, addressed: bool = False) -> None:
    """累计一次互动（db_writer 异步入库，失败静默）。"""
    if not user_id:
        return
    try:
        from junjun_core.database import db_writer
        today = datetime.date.today().isoformat()
        gain = GAIN_ADDRESSED if addressed else GAIN_NORMAL
        db_writer.submit(_accumulate, user_id, gain, today)
    except Exception as e:
        logger.debug(f"好感度累计失败（忽略）: {e}")


def _accumulate(user_id: str, gain: float, today: str) -> None:
    from junjun_core.database import Intimacy
    row = Intimacy.get_or_none(Intimacy.user_id == str(user_id))
    if row is None:
        Intimacy.create(user_id=str(user_id), score=min(gain, DAILY_CAP),
                        interaction_count=1, last_interaction=time.time(),
                        daily_gain=min(gain, DAILY_CAP), daily_date=today)
        return
    daily = 0.0 if row.daily_date != today else row.daily_gain
    actual = max(0.0, min(gain, DAILY_CAP - daily))
    if actual <= 0:
        return
    row.score = min(MAX_SCORE, row.score + actual)
    row.interaction_count += 1
    row.last_interaction = time.time()
    row.daily_gain = daily + actual
    row.daily_date = today
    row.save()


def get_intimacy(user_id: str):
    """查询 (score, interaction_count, level)。无记录返回 (0, 0, 等级)。"""
    from junjun_core.database import Intimacy
    row = Intimacy.get_or_none(Intimacy.user_id == str(user_id))
    if row is None:
        return 0.0, 0, level_name(0.0)
    return row.score, row.interaction_count, level_name(row.score)
