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
- R18 三层元数据过滤（2026-08-03 用户实锤 mode=safe 会漏）：
  xRestrict==0 + R18 tag 黑名单 + 群聊 sl<=2/排行榜 sexual==0
- 质量：随机图过收藏数门槛（config [features] min_bookmarks，默认 300）——
  详情请求本来就要发，零额外开销；候选过采样 3 倍，被门槛刷掉自动换下一张
"""

import random
import time

from junjun_agent.commands import register_command
from junjun_core.contracts import ReplySegment
from junjun_core.observability import get_logger

from .client import (_cookie, _fetch_json, _fetch_raw, _min_bookmarks,
                     BASE_URL, has_r18_tag, has_ugly_tag, images_to_b64,
                     passes_policy, pximg_proxy, quality_tiers, search_artworks)

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


def _ok_item(item: dict, exclude_ai: bool, square: bool, group: bool) -> bool:
    """过滤：内容政策（群全年龄/私 R18 可 G 不可）+ 低质 tag + AI（可选）+ 方图。"""
    if not passes_policy(item, group) or has_ugly_tag(item):
        return False
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
                          exclude_ai: bool, num: int, group: bool) -> list:
    """带标签：收藏分层池随机页挑候选 id（3 倍过采样，给收藏门槛留余量）。

    免会员质量方案（popular_d 是 Premium 限定，非会员静默降级最新优先）：
    关键词 + 「1000users入り」收藏分层 tag 搜索，按层级逐级放宽兜底。
    """
    kw = " ".join(tags)
    for t in quality_tiers():
        data = await search_artworks(f"{kw} {t}".strip(),
                                     random.randint(1, 5), ratio,
                                     r18_ok=not group)
        picks = [d for d in data if _ok_item(d, exclude_ai, square, group)]
        if picks:
            random.shuffle(picks)
            return [str(d.get("id")) for d in picks[:num * 3] if d.get("id")]
    return []


def _ranking_sexual(content: dict) -> int:
    """ranking 条目的露骨等级：illust_content_type 是 dict（{"sexual": 0|1|2}），
    兼容旧整型形态（0=全年龄 1=R18）。"""
    ict = content.get("illust_content_type") or 0
    try:
        return int(ict.get("sexual") or 0) if isinstance(ict, dict) else int(ict)
    except (TypeError, ValueError):
        return 0


async def _pick_from_ranking(square: bool, exclude_ai: bool, num: int,
                             group: bool) -> list:
    """不带标签：每日插画榜随机挑候选 id（3 倍过采样）。

    群聊 sexual==0 才收（sexual=1 的轻度擦边在群里也是雷区，2026-08-03 用户定）。
    """
    body = await _fetch_raw(BASE_URL + "/ranking.php?mode=daily&content=illust&p=1&format=json",
                            BASE_URL + "/ranking.php")
    if body.get("error"):
        return []
    contents = body.get("contents") or []
    ceiling = 1 if group else 2  # sexual < ceiling
    picks = []
    for c in contents:
        if _ranking_sexual(c) >= ceiling or has_r18_tag(c):
            continue
        w, h = c.get("width") or 0, c.get("height") or 0
        if square and (not w or not h or not (0.8 <= w / h <= 1.25)):
            continue
        if exclude_ai:
            try:
                if int(c.get("ai_illust") or 0) == 1:
                    continue
            except (TypeError, ValueError):
                pass
        if c.get("illust_id"):
            picks.append(str(c["illust_id"]))
    random.shuffle(picks)
    return picks[:num * 3]


async def _illust_image_urls(illust_id: str, group: bool) -> tuple:
    """作品 -> (标题作者说明, [图片 URL 列表])（regular 尺寸，代理改写）。

    详情层终检：R18（xRestrict+tag）+ 收藏数门槛（质量兜底，搭既有请求的便车）。
    """
    body = await _fetch_json(BASE_URL + f"/ajax/illust/{illust_id}",
                             BASE_URL + f"/artworks/{illust_id}")
    if body.get("error"):
        return "", []
    if not passes_policy(body, group) or has_ugly_tag(body):
        return "", []
    try:
        if int(body.get("bookmarkCount") or 0) < _min_bookmarks():
            return "", []
    except (TypeError, ValueError):
        pass
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
    group = bool(ctx.session.is_group)

    _last_use[chat_id] = now
    segs = [ReplySegment(type="text", data="看吧！涩批！")] if group else []
    sent_info, sent, saw_candidate = [], 0, False
    # popular_d 是「近期人气」不是全时期——某些标签整页都是低收藏新作，
    # 门槛会全刷掉，最多换 3 个随机页重试（排行榜库存足，单发即可）
    for _attempt in range(3 if req["tags"] else 1):
        if req["tags"]:
            ids = await _pick_from_tags(req["tags"], req["ratio"], req["square"],
                                        req["exclude_ai"], req["num"], group)
        else:
            ids = await _pick_from_ranking(req["square"], req["exclude_ai"],
                                           req["num"], group)
        for iid in ids:
            saw_candidate = True
            if sent >= req["num"]:
                break
            info, urls = await _illust_image_urls(iid, group)
            if not urls:
                continue  # 详情层被刷（安全/收藏门槛），换下一个候选
            # NapCat 拉不到图床（被墙无代理），本侧代下转 base64
            b64s = await images_to_b64(urls)
            if not b64s:
                continue
            if info:
                sent_info.append(info)
            segs += [ReplySegment(type="image", data=b) for b in b64s]
            sent += 1
        if sent >= req["num"]:
            break
    if sent == 0:
        kw = " ".join(req["tags"])
        if not saw_candidate:
            return (f"没找到符合要求的图，换个标签试试？" if kw
                    else "图库请求失败了，稍后再试试吧。")
        return (f"「{kw}」挑了几页都没过收藏门槛的图，换个热门点的标签？"
                if kw else "挑到的图都没过收藏门槛（丑图过滤），再试一次？")
    if sent_info:
        segs.append(ReplySegment(type="text", data="\n".join(sent_info)))
    await ctx.send(segs)
    return None
