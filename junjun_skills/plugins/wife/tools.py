"""wife 插件：每日「抽老婆」群娱乐（迁移自 wife_plugin，新架构重写）。

命令（raw 关键词）：抽老婆 / 今日老婆
- 每群每天一次：data/wife/{group_id}/{YYYY-MM-DD}.json 记录，已抽则回同一人
- 群成员列表走 junjun_core.napcat_client（NAPCAT_HTTP_BASE 未配置则降级）
- 回复：@本人 + QQ 头像图 + 结果文本
"""

import asyncio
import json
import random
import time
from pathlib import Path

from junjun_agent.commands import register_command
from junjun_core.contracts import ReplySegment
from junjun_core.observability import get_logger

logger = get_logger("plugin.wife")

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "wife"


def _today_record(group_id: str) -> Path:
    import datetime
    return DATA_DIR / str(group_id) / f"{datetime.date.today().isoformat()}.json"


def _load_today(group_id: str):
    p = _today_record(group_id)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _save_today(group_id: str, record: dict) -> None:
    p = _today_record(group_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")


async def _draw_wife(group_id: str, self_qq: str):
    """从群成员里随机抽一个（排除 bot 自己）。失败返回 None。"""
    from junjun_core import napcat_client
    members = await napcat_client.get_group_members(group_id)
    if not members:
        return None
    candidates = [m for m in members if str(m.get("user_id")) != str(self_qq)]
    if not candidates:
        return None
    m = random.choice(candidates)
    return {"user_id": str(m.get("user_id")),
            "nickname": m.get("card") or m.get("nickname") or str(m.get("user_id")),
            "ts": time.time()}


@register_command("抽老婆", aliases=["今日老婆"], raw=True, plugin="wife",
                  description="抽今日群老婆（每群每天一次）")
async def wife_cmd(ctx):
    if not ctx.session.is_group:
        return "抽老婆是群聊玩法，私聊没有群成员哦。"
    group_id = ctx.session.group_id
    record = _load_today(group_id)
    if not record:
        from junjun_core.config import get_global_config
        record = await _draw_wife(group_id, get_global_config().bot.qq_account)
        if not record:
            return "今天抽不了——群成员列表拿不到（NapCat HTTP 未配置或调用失败）。"
        try:
            await asyncio.to_thread(_save_today, group_id, record)
        except Exception as e:
            logger.warning(f"老婆记录写入失败（不影响本次结果）: {e}")
    from junjun_core.napcat_client import qq_avatar_url
    # @ 发命令的人（不是抽中的老婆），文本里写老婆名字
    await ctx.send([
        ReplySegment(type="at", data=ctx.meta.user_id),
        ReplySegment(type="image", data=qq_avatar_url(record["user_id"])),
        ReplySegment(type="text",
                     data=f" 你今天的群老婆是「{record['nickname']}」，好好珍惜～"),
    ])
    return None


TOOLS = []
