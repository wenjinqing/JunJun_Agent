"""pixiv 命令族：P 站内容统一入口（插画搜索/作品发图/排行榜/作者/关注新图）。

命令：
  /pixiv search <关键词>        插画搜索（编号列表，/pixiv dl <编号> 发图）
  /pixiv illust <作品ID或URL>   直接发这个作品的图（多页最多 3 张）
  /pixiv rank [daily|weekly|monthly|rookie] [illust|manga]  排行榜（默认 daily illust）
  /pixiv author <作者URL或UID>  作者作品（插画 + 小说合并列表）
  /pixiv new                    关注画师的新图（需要 Cookie 登录态）
  /pixiv dl <编号>              发图（插画）或下载（小说，私聊限定）

端点（全部官方，2026-08-03 实测）：
  /ajax/search/artworks/{kw}     插画搜索（mode=safe，aiType/xRestrict/宽高都有）
  /ajax/illust/{id}              作品详情（urls.regular）
  /ajax/illust/{id}/pages        多页（body 是 list）
  /ranking.php?mode=&content=&format=json  排行榜（顶层 JSON 无 body 包装）
  /ajax/follow_latest/illust     关注新图（thumbnails.illust）
  /ajax/user/{uid}/profile/*     作者插画+小说

管控：R18 写死过滤；群聊可用（全年龄图），但小说下载仍然私聊限定；
每用户冷却 5s；列表缓存 10 分钟。
"""

import re
import time

from junjun_agent.commands import register_command
from junjun_core.contracts import ReplySegment
from junjun_core.observability import get_logger

from . import novel as novel_mod
from .client import (_cookie, _fetch_json, _fetch_raw, BASE_URL,
                     has_r18_tag, images_to_b64, is_safe_item, pximg_proxy,
                     quality_tiers, search_artworks)
from .setu import _ok_item, _ranking_sexual

logger = get_logger("plugin.pixiv.illust")

_COOLDOWN = 5.0
_CACHE_TTL = 600
_LIST_MAX = 10
_ILLUST_IMG_MAX = 3          # 单作品多页最多发 3 张（防刷屏）
_AUTHOR_ILLUST_MAX = 10      # 作者页插画条数

_last_use: dict = {}          # user_id -> ts
_list_cache: dict = {}        # user_id -> {"ts": float, "items": [...]}

_ARTWORKS_URL_RE = re.compile(r"pixiv\.net(?:/en)?/artworks/(?P<id>\d+)", re.IGNORECASE)
_NO_COOKIE = "P 站功能还没配置 Pixiv Cookie，暂时不可用（让主人在 .env 里设置 PIXIV_COOKIE 吧）。"

_HELP = """P 站内容用法：
/pixiv search <关键词> - 搜插画，编号列表
/pixiv illust <作品ID或URL> - 直接发图（多页最多 3 张）
/pixiv rank [daily|weekly|monthly|rookie] [illust|manga] - 排行榜
/pixiv author <作者URL或UID> - 作者作品（插画+小说）
/pixiv new - 关注画师的新图
/pixiv dl <编号> - 发图/下载列表里对应编号
小说系列/单篇全文还是走 /novel（私聊）"""

_RANK_MODES = {"daily", "weekly", "monthly", "rookie"}
_RANK_CONTENTS = {"illust", "manga"}


# ------------------------------------------------------------------ 抓取

async def _search_illusts(keyword: str, group: bool = True) -> list:
    """插画搜索 -> 统一条目列表（kind=illust）。group=True 走群聊严格过滤。

    收藏分层池（免会员质量方案，见 client.quality_tiers）：先搜高收藏层，
    没结果逐级放宽到裸关键词——推荐列表质量接近人気順。
    """
    for tier in quality_tiers():
        data = await search_artworks(f"{keyword.strip()} {tier}".strip(), 1)
        items = []
        for d in data:
            if not _ok_item(d, exclude_ai=False, square=False, group=group):
                continue
            items.append({"kind": "illust", "id": str(d.get("id") or ""),
                          "title": d.get("title") or "(无标题)",
                          "author": d.get("userName") or "",
                          "pages": d.get("pageCount") or 1})
            if len(items) >= _LIST_MAX:
                break
        if items:
            return items
    return []


async def _ranking(mode: str, content: str, group: bool = True) -> list:
    """排行榜 -> 统一条目列表。群聊 sexual==0 才收（轻度擦边也是雷区）。"""
    body = await _fetch_raw(
        BASE_URL + f"/ranking.php?mode={mode}&content={content}&p=1&format=json",
        BASE_URL + "/ranking.php")
    if body.get("error"):
        return []
    items = []
    for c in (body.get("contents") or []):
        # 2026-08-03 实测：illust_content_type 是标签分级 dict
        # （{"sexual": 0|1|2, ...}），不是 0/1 整型——按整型比较会把
        # 所有条目滤掉（排行榜永远空）。sexual>=2 才算 R18 级；
        # 兼容旧的整型形态（0=全年龄 / 1=R18）。
        ceiling = 1 if group else 2  # sexual < ceiling
        if _ranking_sexual(c) >= ceiling or has_r18_tag(c):
            continue
        items.append({"kind": "illust", "id": str(c.get("illust_id") or ""),
                      "title": c.get("title") or "(无标题)",
                      "author": c.get("user_name") or "",
                      "rank": c.get("rank"),
                      "pages": c.get("illust_page_count") or "1"})
        if len(items) >= _LIST_MAX:
            break
    return items


async def _follow_latest(group: bool = True) -> list:
    """关注画师新图 -> 统一条目列表。"""
    body = await _fetch_json(BASE_URL + "/ajax/follow_latest/illust?p=1&mode=all",
                             BASE_URL + "/bookmark_new_illust.php")
    if body.get("error"):
        return []
    ill = ((body.get("thumbnails") or {}).get("illust")) or []
    items = []
    for d in ill:
        if not _ok_item(d, exclude_ai=False, square=False, group=group):
            continue
        items.append({"kind": "illust", "id": str(d.get("id") or ""),
                      "title": d.get("title") or "(无标题)",
                      "author": d.get("userName") or "",
                      "pages": d.get("pageCount") or 1})
        if len(items) >= _LIST_MAX:
            break
    return items


async def _author_items(uid: str, group: bool = True) -> tuple:
    """作者作品合并（插画 + 小说）。返回 (author_name, items)。"""
    profile = await _fetch_json(BASE_URL + f"/ajax/user/{uid}/profile/all",
                                BASE_URL + f"/users/{uid}")
    if profile.get("error"):
        return "", []

    # 插画：profile/all 的 illusts 只有 ID 表，批量取标题
    illusts_map = profile.get("illusts") or {}
    ids = sorted((str(i) for i in illusts_map.keys() if str(i).isdigit()),
                 key=int, reverse=True)[:_AUTHOR_ILLUST_MAX]
    items = []
    author = ""
    if ids:
        query = "&".join(f"ids[]={i}" for i in ids)
        works = await _fetch_json(
            BASE_URL + f"/ajax/user/{uid}/profile/illusts?{query}"
            "&work_category=illustManga&is_first_page=1",
            BASE_URL + f"/users/{uid}")
        wmap = works.get("works") or {}
        for iid in ids:
            w = wmap.get(iid)
            if not w:
                continue
            if not _ok_item(w, exclude_ai=False, square=False, group=group):
                continue
            author = author or (w.get("userName") or "")
            items.append({"kind": "illust", "id": iid,
                          "title": w.get("title") or "(无标题)",
                          "author": w.get("userName") or author,
                          "pages": w.get("pageCount") or 1})

    # 小说：复用 novel 模块的作者逻辑
    nworks = await novel_mod._fetch_author_works(uid)
    if not nworks.get("error"):
        author = author or nworks.get("author") or ""
        for s in nworks.get("series") or []:
            if s.get("r18"):
                continue
            items.append({"kind": "series", "id": s["series_id"],
                          "title": s["title"], "author": s.get("author") or author,
                          "pages": f"{s.get('chapters', 0)}章"})
        for n in nworks.get("novels") or []:
            if n.get("r18"):
                continue
            items.append({"kind": "novel", "id": n["id"],
                          "title": n["title"], "author": n.get("author") or author,
                          "pages": "单篇"})
    return author, items[:_LIST_MAX * 2]


async def _illust_detail(illust_id: str) -> dict:
    return await _fetch_json(BASE_URL + f"/ajax/illust/{illust_id}",
                             BASE_URL + f"/artworks/{illust_id}")


async def _illust_page_urls(illust_id: str, page_count: int) -> list:
    """多页作品图片 URL（regular，代理改写，最多 _ILLUST_IMG_MAX 张）。"""
    body = await _fetch_json(BASE_URL + f"/ajax/illust/{illust_id}/pages",
                             BASE_URL + f"/artworks/{illust_id}")
    pages = body if isinstance(body, list) else []
    urls = []
    for p in pages[:_ILLUST_IMG_MAX]:
        u = ((p.get("urls") or {}).get("regular"))
        if u:
            urls.append(pximg_proxy(u))
    return urls


# ------------------------------------------------------------------ 发送与列表

async def _send_illust(ctx, illust_id: str) -> str:
    """发一个作品的图（详情 + 多页最多 3 张）。返回提示文本或 None（已发送）。"""
    detail = await _illust_detail(illust_id)
    if detail.get("error"):
        return f"获取作品失败：{detail['error']}"
    title = detail.get("title") or "(无标题)"
    author = detail.get("userName") or ""
    if not is_safe_item(detail, bool(ctx.session.is_group)):
        return "这个作品是 R18/擦边内容，发不了哦。"
    pages = detail.get("pageCount") or 1
    urls = await _illust_page_urls(illust_id, pages)
    if not urls:
        return "图地址解析失败了，稍后再试试吧。"
    # NapCat 拉不到图床（被墙无代理），本侧代下转 base64
    b64s = await images_to_b64(urls)
    if not b64s:
        return "图下载失败了（图床得走代理），稍后再试试吧。"
    cap = f"（共 {pages} 页，发前 {len(b64s)} 页）" if pages > _ILLUST_IMG_MAX else ""
    segs = [ReplySegment(type="text", data=f"「{title}」by {author}{cap}")]
    segs += [ReplySegment(type="image", data=b) for b in b64s]
    await ctx.send(segs)
    return None


def _format_list(title: str, items: list, foot: str) -> str:
    lines = [title]
    kind_name = {"illust": "插画", "novel": "小说", "series": "小说系列"}
    for i, it in enumerate(items, 1):
        extra = []
        if it.get("rank"):
            extra.append(f"第{it['rank']}名")
        extra.append(kind_name.get(it["kind"], it["kind"]))
        if it.get("pages") and it["pages"] != 1:
            extra.append(f"{it['pages']}页" if isinstance(it["pages"], int) else str(it["pages"]))
        lines.append(f"{i}. {it['title']} [{'/'.join(extra)}]")
        if it.get("author"):
            lines.append(f"   作者: {it['author']}")
    lines.append(foot)
    return "\n".join(lines)


def _extract_illust_id(target: str) -> str:
    m = _ARTWORKS_URL_RE.search((target or "").strip())
    if m:
        return m.group("id")
    return re.sub(r"\D", "", target or "")


# ------------------------------------------------------------------ 命令入口

@register_command("pixiv", aliases=["p站", "P站"], plugin="pixiv",
                  description="P 站内容：/pixiv search|illust|rank|author|new|dl")
async def pixiv_cmd(ctx):
    user_id = str(ctx.meta.user_id or "")
    args = (ctx.args or "").strip()
    if not args or args.lower() in ("help", "帮助", "?", "？"):
        return _HELP
    if not _cookie():
        return _NO_COOKIE

    now = time.time()
    left = _COOLDOWN - (now - _last_use.get(user_id, 0))
    if left > 0:
        return f"冷却中，{int(left) + 1} 秒后再来吧。"
    _last_use[user_id] = now

    tokens = args.split()
    sub = tokens[0].lower()
    rest = " ".join(tokens[1:]).strip()

    if sub in ("search", "搜索", "搜"):
        if not rest:
            return "关键词呢？用法：/pixiv search <关键词>"
        items = await _search_illusts(rest, bool(ctx.session.is_group))
        if not items:
            return f"没找到和「{rest}」相关的插画，换个关键词试试？"
        _list_cache[user_id] = {"ts": now, "items": items}
        return _format_list(f"插画搜索「{rest}」（{len(items)} 条）：", items,
                            "输入 /pixiv dl <编号> 发对应作品的图")

    if sub in ("illust", "图", "作品"):
        iid = _extract_illust_id(rest)
        if not iid:
            return "没识别到作品 ID，用法：/pixiv illust <作品ID或URL>"
        out = await _send_illust(ctx, iid)
        return out  # None = 已发送

    if sub in ("rank", "排行", "榜"):
        mode = next((t for t in tokens[1:] if t.lower() in _RANK_MODES), "daily")
        content = next((t for t in tokens[1:] if t.lower() in _RANK_CONTENTS), "illust")
        items = await _ranking(mode, content, bool(ctx.session.is_group))
        if not items:
            return "排行榜拉取失败了，稍后再试试吧。"
        _list_cache[user_id] = {"ts": now, "items": items}
        cname = "漫画" if content == "manga" else "插画"
        mname = {"daily": "每日", "weekly": "每周", "monthly": "每月", "rookie": "新人"}[mode]
        return _format_list(f"{mname}{cname}榜 TOP{len(items)}：", items,
                            "输入 /pixiv dl <编号> 发对应作品的图")

    if sub in ("author", "作者"):
        uid = novel_mod._extract_user_id(rest)
        if not uid:
            return "没识别到作者，用法：/pixiv author <作者URL或UID>"
        author, items = await _author_items(uid, bool(ctx.session.is_group))
        if not items:
            return f"这位作者（uid:{uid}）没有公开作品（或主页不可见）。"
        _list_cache[user_id] = {"ts": now, "items": items}
        return _format_list(f"作者「{author or uid}」的作品：", items,
                            "输入 /pixiv dl <编号> 发图（插画）或下载（小说，私聊）")

    if sub in ("new", "新图", "关注"):
        items = await _follow_latest(bool(ctx.session.is_group))
        if not items:
            return "关注的画师最近没新图（或登录态失效了）。"
        _list_cache[user_id] = {"ts": now, "items": items}
        return _format_list(f"关注画师的新图（{len(items)} 条）：", items,
                            "输入 /pixiv dl <编号> 发对应作品的图")

    if sub in ("dl", "download", "下载", "下"):
        return await _do_dl(ctx, user_id, rest)

    return "没看懂的子命令。\n\n" + _HELP


async def _do_dl(ctx, user_id: str, number_text: str) -> str:
    if not number_text.isdigit():
        return "请输入有效的编号，用法：/pixiv dl <编号>"
    entry = _list_cache.get(user_id)
    if not entry or time.time() - entry["ts"] > _CACHE_TTL:
        _list_cache.pop(user_id, None)
        return "列表过期了，重新搜一下吧。"
    items = entry["items"]
    n = int(number_text)
    if not (1 <= n <= len(items)):
        return f"编号超出范围（1-{len(items)}）。"
    it = items[n - 1]
    if it["kind"] == "illust":
        out = await _send_illust(ctx, it["id"])
        return out
    # 小说/系列：私聊限定（对齐 /novel 管控）
    if ctx.session.is_group:
        return "小说下载只能在私聊用哦～"
    if it["kind"] == "series":
        return await novel_mod._do_series(ctx, it["id"])
    return await novel_mod._do_single(ctx, it["id"])
