"""pixiv 插件的 LLM 工具面：Agent 主动搜索/推荐/下载 P 站内容。

策略（2026-08-03 用户定）：
- 群聊：可以帮忙搜索和推荐（文字清单 + 链接），但不能下载发出——
  群聊调下载类工具会收到解释性拒绝（模型据此改给链接/引导私聊）
- 私聊：可以下载并发送（图片直接发，小说 txt 走 NapCat 私聊文件）

搜索/推荐类工具群私通用；发送经 gateway（与 send_message 同一出口），
不直接碰接入层。
"""

import time
from types import SimpleNamespace

from langchain_core.tools import tool

from junjun_core.contracts import ReplySegment, ReplySet
from junjun_skills.builtin.memory_skills import current_chat_id

from . import illust as illust_mod
from . import novel as novel_mod
from .client import _cookie, images_to_b64, passes_policy

_NO_COOKIE = "P 站功能还没配置 Pixiv Cookie，暂时不可用（让主人在 .env 里设置 PIXIV_COOKIE 吧）。"
_DL_COOLDOWN = 60.0
_dl_last: dict = {}  # chat_id -> ts


def _chat() -> tuple:
    """(platform, cid, kind)；kind = group/private。"""
    parts = (current_chat_id.get() or "qq::private").split(":")
    return parts[0], parts[1] if len(parts) > 1 else "", \
        parts[2] if len(parts) > 2 else "private"


class _ToolCtx:
    """给 novel 模块的 _do_single/_do_series 用的轻量 ctx（私聊语义）。"""

    def __init__(self, platform: str, user_id: str):
        self.session = SimpleNamespace(is_group=False, platform=platform,
                                       group_id=None, chat_id=f"{platform}:{user_id}:private")
        self.meta = SimpleNamespace(user_id=user_id)

    async def reply(self, text: str) -> None:
        from junjun_core.gateway.router import get_gateway
        await get_gateway().send_reply(ReplySet(
            platform=self.session.platform, target_user_id=self.meta.user_id,
            segments=[ReplySegment(type="text", data=text)], should_reply=True))


# ---------------------------------------------------------------- 推荐类（群私通用）

@tool
async def pixiv_search_illusts(keyword: str) -> str:
    """搜索 P 站插画并给出推荐清单（标题/作者/页数/链接）。对方想找图、壁纸、头像、
    某角色某作品的同人图，或让你推荐好看的图时使用。群聊私聊都能推荐；
    但直接把图发出来只能在私聊（用 pixiv_send_illust），群聊里给链接让对方自己看。

    Args:
        keyword: 搜索关键词（角色名/作品名/画风标签，日文更准，中文也行）
    """
    if not _cookie():
        return _NO_COOKIE
    _, _, kind = _chat()
    items = await illust_mod._search_illusts(keyword, group=(kind == "group"))
    if not items:
        return f"没找到和「{keyword}」相关的插画，换个关键词（比如换日文）试试。"
    lines = [f"P 站插画搜索「{keyword}」推荐（{len(items)} 条，按收藏排序）："]
    for i, it in enumerate(items, 1):
        pages = f"，{it['pages']}页" if (it.get("pages") or 1) > 1 else ""
        bm = it.get("bookmarks") or 0
        bm_text = f"，{bm / 10000:.1f}万收藏" if bm >= 10000 else (f"，{bm}收藏" if bm else "")
        r18 = "【R18】" if it.get("r18") else ""
        lines.append(f"{i}. {r18}「{it['title']}」by {it['author']}{pages}{bm_text}")
        lines.append(f"   https://www.pixiv.net/artworks/{it['id']}")
    lines.append("（群聊里发链接推荐即可；私聊里对方想要哪张，用 pixiv_send_illust 发图）")
    return "\n".join(lines)


@tool
async def pixiv_search_novels(keyword: str) -> str:
    """搜索 P 站小说并给出推荐清单（标题/作者/类型/链接）。对方想找小说看、
    让推荐某 CP 某题材的文时使用。群聊私聊都能推荐；下载 txt 只能在私聊
    （用 pixiv_download_novel），群聊里给链接让对方自己看。

    Args:
        keyword: 搜索关键词（作品名/CP/题材 tag，日文更准）
    """
    if not _cookie():
        return _NO_COOKIE
    result = await novel_mod._search_novels(keyword)
    if result.get("error"):
        return f"搜索失败了：{result['error']}，稍后再试。"
    data_list = (result.get("novel") or {}).get("data") or []
    if not data_list:
        return f"没找到和「{keyword}」相关的小说，换个关键词试试。"
    _, _, kind = _chat()
    items = [novel_mod._extract_search_item(it) for it in data_list[:6]]
    if kind == "group":
        items = [it for it in items if not it["r18"]]  # 群聊推荐清单不含 R18
    if not items:
        return f"「{keyword}」的结果全是 R18，群里不推荐发出来，私聊我给你找。"
    lines = [f"P 站小说搜索「{keyword}」推荐（{len(items)} 条）："]
    for i, it in enumerate(items, 1):
        r18 = "【R18】" if it["r18"] else ""
        if it["series_id"]:
            lines.append(f"{i}. {r18}「{it['display_title']}」（系列）by {it['author']}")
            lines.append(f"   https://www.pixiv.net/novel/series/{it['series_id']}")
        else:
            lines.append(f"{i}. {r18}「{it['display_title']}」（单篇）by {it['author']}")
            lines.append(f"   https://www.pixiv.net/novel/show.php?id={it['id']}")
    lines.append("（群聊里发链接推荐即可；私聊里对方想要全文，用 pixiv_download_novel 下载 txt）")
    return "\n".join(lines)


# ---------------------------------------------------------------- 下载类（私聊限定）

@tool
async def pixiv_send_illust(illust_id: str) -> str:
    """把指定 P 站作品的图直接发出来（多页最多 3 张）。【仅私聊可用】——
    群聊里不要调用本工具：群聊想要图时用 pixiv_search_illusts 给链接推荐。
    私聊可以发 R18（R-18G/グロ除外）。

    Args:
        illust_id: 作品 ID（pixiv_search_illusts 的结果里有，或作品链接里的数字）
    """
    platform, cid, kind = _chat()
    if kind != "private":
        return ("群聊里不能直接把图发出来（怕刷屏+内容不可控）。"
                "改用 pixiv_search_illusts 给对方链接推荐，或让 ta 私聊我要图。")
    if not _cookie():
        return _NO_COOKIE
    iid = "".join(c for c in str(illust_id) if c.isdigit())
    if not iid:
        return "没识别到作品 ID。"
    detail = await illust_mod._illust_detail(iid)
    if detail.get("error"):
        return f"获取作品失败：{detail['error']}"
    if not passes_policy(detail, group=False):
        return "这个作品是 R-18G/グロ，发不了。可以推荐给 ta 别的作品。"
    title = detail.get("title") or "(无标题)"
    author = detail.get("userName") or ""
    pages = detail.get("pageCount") or 1
    urls = await illust_mod._illust_page_urls(iid, pages)
    if not urls:
        return "图地址解析失败了，稍后再试。"
    # NapCat 拉不到图床（被墙无代理），本侧代下转 base64
    b64s = await images_to_b64(urls)
    if not b64s:
        return "图下载失败了（图床得走代理），稍后再试。"
    cap = f"（共 {pages} 页，发前 {len(b64s)} 页）" if pages > 3 else ""
    from junjun_core.gateway.router import get_gateway
    await get_gateway().send_reply(ReplySet(
        platform=platform, target_user_id=cid,
        segments=[ReplySegment(type="text", data=f"「{title}」by {author}{cap}")]
        + [ReplySegment(type="image", data=b) for b in b64s],
        should_reply=True))
    return f"已把「{title}」的图发出去了。"


@tool
async def pixiv_download_novel(target: str) -> str:
    """下载 P 站小说发 txt 文件（单篇直接发；系列整部后台抓取，抓完发文件）。
    【仅私聊可用】——群聊里不要调用本工具：群聊想要小说时用 pixiv_search_novels
    给链接推荐。每会话 60 秒冷却。

    Args:
        target: 单篇 ID（如 12345678）/ "series <系列ID>" / 小说或系列链接
    """
    platform, cid, kind = _chat()
    if kind != "private":
        return ("小说 txt 文件只能私聊发（内容风险+文件只能发私聊）。"
                "改用 pixiv_search_novels 给对方链接推荐，或让 ta 私聊我要全文。")
    if not _cookie():
        return _NO_COOKIE
    chat_id = current_chat_id.get()
    now = time.time()
    left = _DL_COOLDOWN - (now - _dl_last.get(chat_id, 0))
    if left > 0:
        return f"刚下载过，{int(left)} 秒后再试。"
    _dl_last[chat_id] = now

    ctx = _ToolCtx(platform, cid)
    t = (target or "").strip()
    low = t.lower()
    if low.startswith("series"):
        sid = "".join(c for c in t if c.isdigit())
        if not sid:
            return "没识别到系列 ID（用法：series <系列ID>）。"
        return await novel_mod._do_series(ctx, sid)
    if low.startswith("novel"):
        t = t[5:].strip()
    if t.isdigit():
        # 裸数字 = 单篇 ID（搜索结果给的就是单篇 ID；系列要带 "series" 前缀）
        return await novel_mod._do_single(ctx, t)
    kind_id, nid = novel_mod._extract_id(t)
    if not nid:
        return "没识别到小说 ID 或链接。"
    if kind_id == "novel":
        return await novel_mod._do_single(ctx, nid)
    return await novel_mod._do_series(ctx, nid)


TOOLS = [pixiv_search_illusts, pixiv_send_illust,
         pixiv_search_novels, pixiv_download_novel]
