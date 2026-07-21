"""表情包管理插件：/emoji add|delete|list + /random_emojis。

对齐旧 MaiBot emoji_manage_plugin 的命令面，但不复制其架构——
存储与注册管线复用新架构的 junjun_express.emoji.emoji_manager
（下载 / hash 去重 / VLM 描述）与 junjun_core.database.Emoji 表，
不另建存储。表情库管理是管理行为，/emoji 系列均为 admin_only。
"""

import hashlib
import json
import random
from pathlib import Path
from typing import Optional, Tuple

from junjun_agent.commands import register_command
from junjun_core.contracts import ReplySegment
from junjun_core.observability import get_logger

logger = get_logger("plugin.emoji_manage")

# 与 steal 路径一致：>2MB 的不当表情包处理
_MAX_BYTES = 2 * 1024 * 1024


def _parse_count(text: str, *, default: int, max_n: int) -> int:
    """解析数量参数，非法值回落默认，钳制到 [1, max_n]。"""
    try:
        n = int((text or "").strip())
    except ValueError:
        n = default
    return max(1, min(n, max_n))


def _load_emotions(row) -> list:
    """解析 Emoji.emotion（JSON list），脏数据降级为空。"""
    try:
        return json.loads(row.emotion or "[]")
    except json.JSONDecodeError:
        return []


async def _register_one(url: str) -> Tuple[bool, str]:
    """注册一张图：下载 -> hash 去重 -> VLM 描述 -> 落盘入库。

    复用 emoji_manager 的内部步骤（_download/_describe/_evict_one），
    上限换血策略与 register_pending 一致。VLM 不可用时 _describe 自带
    降级描述，仍允许入库（与 steal 路径一致）。
    """
    from junjun_core.database import Emoji
    from junjun_express import emoji as emoji_mod

    mgr = emoji_mod.emoji_manager
    data = await mgr._download(url)
    if data is None:
        return False, "下载失败"
    if len(data) > _MAX_BYTES:
        return False, "图片超过 2MB，不像表情包"

    h = hashlib.md5(data).hexdigest()
    if Emoji.get_or_none(Emoji.emoji_hash == h):
        return False, "库里已经有这张了（hash 重复）"

    # 上限换血：与 register_pending 同款策略
    cfg = emoji_mod._cfg()
    if Emoji.select().count() >= int(cfg.get("max_reg_num", 2000)):
        if not cfg.get("do_replace", True):
            return False, "表情库已满且未开启换血（do_replace=false）"
        mgr._evict_one()

    desc, emotions = await mgr._describe(data)
    if desc is None:
        return False, "VLM 无法描述这张图（可能不是有效图片）"

    target = emoji_mod.EMOJI_REG_DIR / f"{h}.img"
    target.write_bytes(data)
    Emoji.create(full_path=str(target), emoji_hash=h,
                 description=desc, emotion=json.dumps(emotions, ensure_ascii=False))
    emo_text = "/".join(emotions) if emotions else "无"
    return True, f"描述「{desc or '(空)'}」，情感: {emo_text}"


async def _delete_by_hash(url: str) -> Tuple[bool, str]:
    """按图片内容 hash 找库中表情并删除。"""
    from junjun_core.database import Emoji
    from junjun_express import emoji as emoji_mod

    data = await emoji_mod.emoji_manager._download(url)
    if data is None:
        return False, "下载失败"
    h = hashlib.md5(data).hexdigest()
    row = Emoji.get_or_none(Emoji.emoji_hash == h)
    if row is None:
        return False, "库里没找到这张图"
    _delete_row(row)
    return True, f"已删除 #{row.id}「{(row.description or '')[:20]}」"


def _delete_row(row) -> None:
    """删文件 + 删表记录。"""
    Path(row.full_path).unlink(missing_ok=True)
    row.delete_instance()


@register_command("random_emojis", plugin="emoji_manage",
                  description="随机来几张表情包（合并转发）：/random_emojis [N]，默认3上限5")
async def random_emojis_cmd(ctx) -> Optional[str]:
    from junjun_core.database import Emoji
    n = _parse_count(ctx.args, default=3, max_n=5)
    rows = list(Emoji.select())
    if not rows:
        return "表情库还是空的，先让管理员用 /emoji add 存几张吧。"

    picked = random.sample(rows, min(n, len(rows)))
    from junjun_core.config import get_global_config
    bot = get_global_config().bot
    nodes = [
        {"type": "node", "data": {
            "user_id": bot.qq_account,
            "nickname": bot.nickname,
            "content": [{"type": "image",
                         "data": {"file": Path(r.full_path).resolve().as_uri()}}],
        }}
        for r in picked
    ]
    for r in picked:  # 与 pick() 一致：发出去就计一次使用
        r.usage_count += 1
        r.save()
    await ctx.send([ReplySegment(type="forward", data=json.dumps(nodes, ensure_ascii=False))])
    return None


# ---------------------------------------------------------------------------
# 命令
# ---------------------------------------------------------------------------
@register_command("emoji", plugin="emoji_manage", admin_only=True,
                  description="表情库管理：/emoji add（带图）| /emoji delete [id]（或带图）| /emoji list [N]")
async def emoji_cmd(ctx) -> Optional[str]:
    parts = ctx.args.split(None, 1)
    action = parts[0] if parts else ""
    rest = parts[1].strip() if len(parts) > 1 else ""

    if action == "add":
        return await _add(ctx)
    if action == "delete":
        return await _delete(ctx, rest)
    if action == "list":
        return _list(rest)
    return "用法：/emoji add（发图时用）| /emoji delete [id]（或发图时用）| /emoji list [N]"


async def _add(ctx) -> str:
    image_urls = getattr(ctx.meta, "image_urls", None) or []
    if not image_urls:
        return "没看到图哦——发图的同时用 /emoji add 才会收录。"
    ok, fail, details = 0, 0, []
    for i, url in enumerate(image_urls, 1):
        try:
            success, msg = await _register_one(url)
        except Exception as e:  # 单张失败不拖垮整批
            logger.warning(f"表情包注册异常: {e}")
            success, msg = False, f"出错了（{type(e).__name__}）"
        ok += success
        fail += not success
        details.append(f"第{i}张 {'成功' if success else '失败'}：{msg}")
    return f"收录完成：成功 {ok} 张，失败 {fail} 张\n" + "\n".join(details)


async def _delete(ctx, rest: str) -> str:
    # 优先按 id 删除
    if rest:
        from junjun_core.database import Emoji
        try:
            emoji_id = int(rest)
        except ValueError:
            return "id 要是数字哦，用法：/emoji delete <id> 或发图时用 /emoji delete"
        row = Emoji.get_or_none(Emoji.id == emoji_id)
        if row is None:
            return f"没找到 #{emoji_id} 这张表情。"
        desc = (row.description or "")[:20]
        _delete_row(row)
        return f"已删除 #{emoji_id}「{desc}」。"

    # 否则按触发消息里的图片 hash 删除
    image_urls = getattr(ctx.meta, "image_urls", None) or []
    if not image_urls:
        return "发图的同时用 /emoji delete，或者 /emoji delete <id> 按编号删（编号见 /emoji list）。"
    ok, fail, details = 0, 0, []
    for i, url in enumerate(image_urls, 1):
        success, msg = await _delete_by_hash(url)
        ok += success
        fail += not success
        details.append(f"第{i}张：{msg}")
    return f"删除完成：成功 {ok} 张，失败 {fail} 张\n" + "\n".join(details)


def _list(rest: str) -> str:
    from junjun_core.database import Emoji
    n = _parse_count(rest, default=10, max_n=30)
    total = Emoji.select().count()
    if total == 0:
        return "表情库还是空的——发图时用 /emoji add 存几张吧。"
    rows = list(Emoji.select().order_by(Emoji.id.desc()).limit(n))
    lines = [f"表情库共 {total} 张，最近 {len(rows)} 张："]
    for r in rows:
        desc = (r.description or "(无描述)")[:30]
        emo = "/".join(_load_emotions(r)) or "无"
        lines.append(f"#{r.id} {desc} [{emo}] 用过{r.usage_count}次")
    if total > len(rows):
        lines.append(f"……还有 {total - len(rows)} 张未列出")
    return "\n".join(lines)


# 本插件只提供命令，不提供 LLM 工具
TOOLS = []
