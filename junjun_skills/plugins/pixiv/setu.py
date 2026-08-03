"""setu：P 站官方 API 二次元图（2026-08-03 从 Lolicon 聚合 API 迁移）。

命令：/setu [数量] [标签...] [#标签]... [横图|竖图|方图] [noai]
  例：/setu 3 白丝 横图 noai
迁移要点：
- 统一走 Pixiv 官方 AJAX（client.py，cookie + Cloudflare 绕过），不再依赖
  Lolicon 第三方聚合——搜索/排行榜都是官方数据，图片 URL 走 i.pixiv.re 代理
  （i.pximg.net 需要 Referer，NapCat 拉图会 403）
- bug 修复：裸词（不带 #）以前被 _parse_args 静默丢弃 -> 永远随机图，
  现在裸词和 #标签 一视同仁都当标签
- 带标签：popular_d 搜索（随机页 1-5）随机挑；不带标签：每日插画榜随机挑
- R18 写死过滤（mode=safe + xRestrict 双保险）；noai 过滤 aiType==2
"""

import random
import time
import urllib.parse

from junjun_agent.commands import register_command
from junjun_core.contracts import ReplySegment
from junjun_core.observability import get_logger

from .client import (_cookie, _fetch_json, _fetch_raw, BASE_URL,
                     images_to_b64, pximg_proxy)

logger = get_logger("plugin.pixiv.setu")

_COOLDOWN = 30.0
_MAX_NUM = 5
_last_use: dict = {}  # chat_id -> ts

_RATIO = {"横图": "-0.5", "竖图": "0.5"}   # 方图：客户端按宽高近似过滤
_NO_COOKIE = "涩图功能还没配置 Pixiv Cookie，暂时不可用（让主人在 .env 里设置 PIXIV_COOKIE 吧）。"


def _parse_args(args: str) -> dict:
    """解析 /setu 参数。返回 {num, tags, ratio, square, exclude_ai}。

    裸词和 #标签 都收为标签（2026-08-03 bug：以前裸词被静默丢弃，
    用户输啥都是随机图）。数字=数量；横图/竖图/方图=比例；noai=排除 AI 图。
    """
    num, tags, ratio, square, exclude_ai = 1, [], "", False, False
    for tok in args.split():
        if tok.isdigit():
            num = max(1, min(_MAX_NUM, int(tok)))
        elif tok in ("横图", "竖图"):
            ratio = _RATIO[tok]
        elif tok == "方图":
            square = True
        elif tok.lower() == "noai":
            exclude_ai = True
        else:
            tags.append(tok.lstrip("#"))  # 裸词与 #标签 一视同仁
    return {"num": num, "tags": tags, "ratio": ratio,
            "square": square, "exclude_ai": exclude_ai}


def _ok_item(item: dict, exclude_ai: bool, square: bool) -> bool:
    """过滤：R18 / AI（可选）/ 方图（可选）。"""
    try:
        if int(item.get("xRestrict") or 0) >= 1:
            return False
    except (TypeError, ValueError):
        pass
    if exclude_ai:
        try:
            if int(item.get("aiType") or 1) == 2:
                return False
        except (TypeError, ValueError):
            pass
    if square:
        w, h = item.get("width") or 0, item.get("height") or 0
        if not w or not h or not (0.8 <= w / h <= 1.25):
            return False
    return True


async def _pick_from_tags(tags: list, ratio: str, square: bool,
                          exclude_ai: bool, num: int) -> list:
    """带标签：popular_d 搜索随机页，随机挑 num 个作品 id。"""
    kw = " ".join(tags)
    enc = urllib.parse.quote(kw)
    page = random.randint(1, 5)
    url = (BASE_URL + f"/ajax/search/artworks/{enc}?word={enc}"
           f"&order=popular_d&mode=safe&p={page}&s_mode=s_tag"
           + (f"&ratio={ratio}" if ratio else ""))
    body = await _fetch_json(url, BASE_URL + "/tags/")
    if body.get("error"):
        return []
    data = (body.get("illustManga") or {}).get("data") or []
    picks = [d for d in data if _ok_item(d, exclude_ai, square)]
    random.shuffle(picks)
    return [str(d.get("id")) for d in picks[:num] if d.get("id")]


async def _pick_from_ranking(square: bool, exclude_ai: bool, num: int) -> list:
    """不带标签：每日插画榜随机挑 num 个作品 id。"""
    body = await _fetch_raw(BASE_URL + "/ranking.php?mode=daily&content=illust&p=1&format=json",
                            BASE_URL + "/ranking.php")
    if body.get("error"):
        return []
    contents = body.get("contents") or []
    items = [{"xRestrict": c.get("illust_content_type", 0),
              "width": c.get("width"), "height": c.get("height"),
              "id": c.get("illust_id"), "aiType": 1} for c in contents]
    # ranking 的 illust_content_type: 0=全年龄 1=R18 2=??——只收 0
    picks = [d for d in items if (d.get("xRestrict") or 0) == 0
             and (not square or (d["width"] and d["height"]
                                 and 0.8 <= d["width"] / d["height"] <= 1.25))]
    random.shuffle(picks)
    return [str(d["id"]) for d in picks[:num] if d.get("id")]


async def _illust_image_urls(illust_id: str) -> tuple:
    """作品 -> (标题作者说明, [图片 URL 列表])（regular 尺寸，代理改写）。"""
    body = await _fetch_json(BASE_URL + f"/ajax/illust/{illust_id}",
                             BASE_URL + f"/artworks/{illust_id}")
    if body.get("error"):
        return "", []
    title = body.get("title") or ""
    author = body.get("userName") or ""
    urls = (body.get("urls") or {}).get("regular")
    return f"「{title}」by {author}" if title else "", ([pximg_proxy(urls)] if urls else [])


@register_command("setu", aliases=["涩图", "色图"], plugin="pixiv",
                  description="来张 P 站图：/setu [数量] [标签...] [横图|竖图|方图] [noai]")
async def setu_cmd(ctx):
    chat_id = ctx.session.chat_id
    now = time.time()
    left = _COOLDOWN - (now - _last_use.get(chat_id, 0))
    if left > 0:
        return f"歇会儿嘛，{int(left)} 秒后再来。"
    if not _cookie():
        return _NO_COOKIE

    req = _parse_args(ctx.args)
    if req["tags"]:
        ids = await _pick_from_tags(req["tags"], req["ratio"], req["square"],
                                    req["exclude_ai"], req["num"])
    else:
        ids = await _pick_from_ranking(req["square"], req["exclude_ai"], req["num"])
    if not ids:
        kw = " ".join(req["tags"])
        return (f"没找到符合要求的图，换个标签试试？" if kw
                else "图库请求失败了，稍后再试试吧。")

    _last_use[chat_id] = now
    segs = [ReplySegment(type="text", data="看吧！涩批！")] if ctx.session.is_group else []
    sent_info = []
    for iid in ids:
        info, urls = await _illust_image_urls(iid)
        if info:
            sent_info.append(info)
        # NapCat 拉不到图床（被墙无代理），本侧代下转 base64
        segs += [ReplySegment(type="image", data=b) for b in await images_to_b64(urls)]
    if len(segs) <= (1 if ctx.session.is_group else 0):
        return "图拿到了但下载失败了（图床得走代理），稍后再试试吧。"
    if sent_info:
        segs.append(ReplySegment(type="text", data="\n".join(sent_info)))
    await ctx.send(segs)
    return None
