"""lolicon_setu 插件：二次元图片（迁移自 lolicon_setu_plugin，新架构重写）。

命令：/setu [数量] [#标签]... [横图|竖图|方图] [noai]
  例：/setu 3 #萝莉 #白丝 横图 noai
API：POST https://api.lolicon.app/setu/v2
限制：每会话冷却 30s；R18 关闭；AI 图默认不排除（noai 可排除）。
"""

import time

from junjun_agent.commands import register_command
from junjun_core.contracts import ReplySegment
from junjun_core.observability import get_logger

logger = get_logger("plugin.setu")

_API = "https://api.lolicon.app/setu/v2"
_COOLDOWN = 30.0
_MAX_NUM = 5
_last_use: dict = {}  # chat_id -> ts

_ASPECT = {"横图": "pc", "竖图": "mobile", "方图": "square"}


def _parse_args(args: str) -> dict:
    """解析 /setu 参数。返回 {num, tags, aspect, exclude_ai}。"""
    num, tags, aspect, exclude_ai = 1, [], "", False
    for tok in args.split():
        if tok.isdigit():
            num = max(1, min(_MAX_NUM, int(tok)))
        elif tok.startswith("#") and len(tok) > 1:
            tags.append(tok[1:])
        elif tok in _ASPECT:
            aspect = _ASPECT[tok]
        elif tok.lower() == "noai":
            exclude_ai = True
    return {"num": num, "tags": tags, "aspect": aspect, "exclude_ai": exclude_ai}


async def _fetch_setu(num: int, tags: list, aspect: str, exclude_ai: bool) -> list | None:
    """调 Lolicon API，返回图片 URL 列表；失败 None。"""
    payload = {
        "r18": 0, "num": num, "size": ["regular"],
        "tag": tags, "excludeAI": exclude_ai, "proxy": "i.pixiv.re",
    }
    if aspect:
        payload["aspectRatio"] = aspect
    try:
        import httpx
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(_API, json=payload)
            data = resp.json()
        if data.get("error"):
            logger.warning(f"lolicon API 错误: {data['error']}")
            return None
        urls = []
        for item in data.get("data") or []:
            url = (item.get("urls") or {}).get("regular")
            if url:
                urls.append(url)
        return urls
    except Exception as e:
        logger.warning(f"lolicon 请求失败: {type(e).__name__}: {e}")
        return None


@register_command("setu", aliases=["涩图", "色图"], plugin="lolicon_setu",
                  description="来张二次元图：/setu [数量] [#标签] [横图|竖图|方图] [noai]")
async def setu_cmd(ctx):
    chat_id = ctx.session.chat_id
    now = time.time()
    left = _COOLDOWN - (now - _last_use.get(chat_id, 0))
    if left > 0:
        return f"歇会儿嘛，{int(left)} 秒后再来。"

    req = _parse_args(ctx.args)
    urls = await _fetch_setu(**req)
    if urls is None:
        return "图库请求失败了，稍后再试试吧。"
    if not urls:
        return "没有找到符合要求的图，换个标签试试？"

    _last_use[chat_id] = now
    segs = [ReplySegment(type="text", data="看吧！涩批！")] if ctx.session.is_group else []
    segs += [ReplySegment(type="image", data=u) for u in urls]
    await ctx.send(segs)
    return None


TOOLS = []
