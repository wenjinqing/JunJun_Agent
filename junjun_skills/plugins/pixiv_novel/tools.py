"""pixiv_novel 插件：Pixiv 小说抓取（迁移自旧 MaiBot pixiv_novel_plugin，新架构重写）。

仅命令执行（不注册任何 LLM 工具，TOOLS=[]）：/novel 命令族全部走命令总线。
仅提取旧插件的 API/协议知识用 httpx 重写：
- AJAX 端点：
    /ajax/novel/{id}                单篇元信息 + 正文（body.content 即全文）
    /ajax/novel/series/{id}         系列元信息
    /ajax/novel/series_content/{id}?limit=&last_order=&order_by=asc   系列分页章节
    /ajax/search/novels/{kw}?word=&order=date_d&mode=all&p=&type=all&s_mode=s_tag&r18=off
- 请求头：User-Agent + Referer + Cookie（PIXIV_COOKIE env）+ x-user-id（从 PHPSESSID 提取）
- 系列合成：逐章抓目录+正文，合成一个 UTF-8 txt（章节分隔），NapCat 私聊发文件

管控（对齐旧插件语义）：
- 仅私聊可用（小说文件涉及内容风险，群聊统一拒绝）
- 插件内部 config.toml 的 [auth] allow_qq_list 白名单，非白名单友好拒绝（不上报 security）
- 每用户冷却 cooldown_seconds（内存 dict）

配置：插件目录 config.toml（真实值，git 忽略）/ config.toml.example（入库模板）。
"""

import asyncio
import os
import re
import time
import tomllib
import urllib.parse
from pathlib import Path

import httpx

from junjun_agent.commands import register_command
from junjun_core import napcat_client
from junjun_core.observability import get_logger

logger = get_logger("plugin.pixiv_novel")

# ------------------------------------------------------------------ 常量

_CONFIG_PATH = Path(__file__).resolve().parent / "config.toml"

_BASE_URL = "https://www.pixiv.net"
_AJAX_NOVEL = _BASE_URL + "/ajax/novel/{}"
_AJAX_SERIES = _BASE_URL + "/ajax/novel/series/{}"
_AJAX_SERIES_CONTENT = _BASE_URL + "/ajax/novel/series_content/{}"
_AJAX_SEARCH = _BASE_URL + "/ajax/search/novels/{}"

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")

_SERIES_PAGE_LIMIT = 30        # 系列目录单页条数（旧插件同款）
_SEARCH_CACHE_TTL = 600        # 搜索结果缓存 10 分钟
_SEARCH_RESULT_MAX = 10        # 搜索展示条数
_CHAPTER_DELAY = 1.0           # 逐章抓取间隔（秒，礼貌爬取；测试中可调 0）

# URL / ID 识别（旧插件同款正则）
_SERIES_URL_RE = re.compile(r"pixiv\.net/novel/series/(?P<id>\d+)", re.IGNORECASE)
_NOVEL_URL_RE = re.compile(r"pixiv\.net/novel/show\.php\?id=(?P<id>\d+)", re.IGNORECASE)
_NOVEL_SHORT_RE = re.compile(r"pixiv\.net/n/(?P<id>\d+)", re.IGNORECASE)
_ILLEGAL_FILENAME_RE = re.compile(r'[\\/:*?"<>|]')

# 群聊统一拒绝（对齐旧插件：小说文件涉及内容风险，仅私聊可用）
_GROUP_REJECT = "这个命令只能在私聊用哦～"
# 非白名单友好拒绝（功能白名单不是管理员面，不提白名单细节，不上报 security）
_NOT_ALLOWED = "这个插件暂时没有对你开放哦～"
_NO_COOKIE = "小说功能还没配置 Pixiv Cookie，暂时不可用（让主人在 .env 里设置 PIXIV_COOKIE 吧）。"

_HELP = """Pixiv 小说下载用法：
/novel <系列URL或ID> - 抓取整个系列，合成 txt 发文件
/novel read <单篇URL或ID> - 抓取单篇小说发 txt
/novel list <系列URL或ID> - 只列出系列章节目录
/novel search <关键词> - 搜索小说，返回编号列表
/novel dl <编号> - 下载搜索结果中对应编号的小说
示例：
  /novel 14998441
  /novel read 12345678
  /novel search 異世界転生
  /novel dl 3"""

# 每用户冷却时间戳（user_id -> ts）
_last_use: dict = {}
# 搜索结果缓存（user_id -> {"ts": float, "items": [...]）
_search_cache: dict = {}


# ------------------------------------------------------------------ 插件内部配置

def _load_config() -> dict:
    """读取插件目录下的 config.toml；缺失/解析失败返回 {}（不炸）。"""
    try:
        with open(_CONFIG_PATH, "rb") as f:
            return tomllib.load(f)
    except FileNotFoundError:
        logger.warning("pixiv_novel/config.toml 不存在，使用默认值")
        return {}
    except Exception as e:
        logger.warning(f"pixiv_novel/config.toml 解析失败: {type(e).__name__}: {e}")
        return {}


def _cfg_section(name: str) -> dict:
    return _load_config().get(name, {}) or {}


def _allow_qq_list() -> list:
    return [str(x) for x in (_cfg_section("auth").get("allow_qq_list") or [])]


def _cooldown_seconds() -> float:
    try:
        return float(_cfg_section("features").get("cooldown_seconds", 60))
    except (TypeError, ValueError):
        return 60.0


def _api_timeout() -> float:
    try:
        return float(_cfg_section("features").get("api_timeout", 30))
    except (TypeError, ValueError):
        return 30.0


def _max_chapters() -> int:
    try:
        return max(1, int(_cfg_section("features").get("max_chapters_per_series", 50)))
    except (TypeError, ValueError):
        return 50


def _save_dir() -> Path:
    raw = str(_cfg_section("features").get("save_dir", "data/pixiv_novel"))
    return Path(raw)


def _proxy() -> str:
    return str(_cfg_section("network").get("proxy", "") or "").strip()


def _is_allowed(user_id: str) -> bool:
    """白名单校验；白名单为空时所有人可用（对齐旧插件语义）。"""
    allow = _allow_qq_list()
    return not allow or str(user_id) in allow


def _cookie() -> str:
    """从 env 读 Pixiv Cookie；接受 PHPSESSID=xxx / 整串 cookie / 裸 session 值。"""
    raw = os.environ.get("PIXIV_COOKIE", "").strip()
    if raw and "=" not in raw:
        raw = "PHPSESSID=" + raw
    return raw


# ------------------------------------------------------------------ 网络层（独立 helper，便于 monkeypatch）

def _headers(referer: str = "") -> dict:
    """构造 Pixiv AJAX 请求头（UA + Referer + Cookie + x-user-id）。"""
    headers = {
        "User-Agent": _UA,
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "ja-JP,ja;q=0.9,zh-CN;q=0.8,zh;q=0.7,en;q=0.6",
        "Referer": referer or (_BASE_URL + "/"),
    }
    cookie = _cookie()
    if cookie:
        headers["Cookie"] = cookie
        # 从 PHPSESSID=<uid>_xxx 提取用户 id（旧插件同款）
        m = re.search(r"PHPSESSID=(\d+)_", cookie)
        if m:
            headers["x-user-id"] = m.group(1)
    return headers


async def _fetch_json(url: str, referer: str = "") -> dict:
    """请求 Pixiv AJAX JSON。成功返回 body 字典；任何失败返回 {"error": ...}，绝不抛异常。"""
    proxy = _proxy()
    try:
        async with httpx.AsyncClient(timeout=_api_timeout(),
                                     proxy=proxy or None) as client:
            resp = await client.get(url, headers=_headers(referer))
            if resp.status_code != 200:
                logger.warning(f"Pixiv 请求失败 HTTP {resp.status_code}: {url}")
                return {"error": f"HTTP {resp.status_code}"}
            data = resp.json()
    except Exception as e:
        logger.warning(f"Pixiv 请求异常: {type(e).__name__}: {e} ({url})")
        return {"error": f"网络请求失败（{type(e).__name__}）"}
    if data.get("error"):
        return {"error": str(data.get("message") or "Pixiv 返回错误")}
    return data.get("body", {}) or {}


async def _fetch_novel_full(novel_id: str) -> dict:
    """单篇元信息 + 正文（正文直接在 body.content，无需单独 content 接口）。"""
    url = _AJAX_NOVEL.format(novel_id)
    referer = _BASE_URL + "/novel/show.php?id=" + str(novel_id)
    meta = await _fetch_json(url, referer)
    if meta.get("error"):
        return meta
    meta["text"] = meta.get("content") or ""
    return meta


async def _fetch_series_meta(series_id: str) -> dict:
    url = _AJAX_SERIES.format(series_id)
    return await _fetch_json(url, _BASE_URL + "/novel/series/" + str(series_id))


async def _fetch_series_all_novels(series_id: str, max_count: int) -> tuple:
    """分页抓取整个系列章节目录。返回 (series_meta, novels)。"""
    series_meta = await _fetch_series_meta(series_id)
    if series_meta.get("error"):
        return series_meta, []
    try:
        total = int(series_meta.get("total") or series_meta.get("novel_count") or 0)
    except (TypeError, ValueError):
        total = 0
    cap = min(total, max_count) if total > 0 else max_count

    novels: list = []
    last_order = 0
    while len(novels) < cap:
        url = (_AJAX_SERIES_CONTENT.format(series_id)
               + f"?limit={_SERIES_PAGE_LIMIT}&last_order={last_order}&order_by=asc")
        batch = await _fetch_json(url, _BASE_URL + "/novel/series/" + str(series_id))
        if batch.get("error"):
            return series_meta, novels
        page = batch.get("page") or {}
        contents = page.get("seriesContents") or batch.get("series_contents") or []
        if not contents:
            break
        for item in contents:
            if item.get("id") is None:
                continue
            novels.append(item)
            if len(novels) >= cap:
                break
        if len(contents) < _SERIES_PAGE_LIMIT:
            break
        last_id = str(contents[-1].get("id") or "")
        if not last_id.isdigit():
            break
        last_order = int(last_id)
        if last_order == 0:
            break
    return series_meta, novels


async def _search_novels(keyword: str, page: int = 1) -> dict:
    """搜索小说；结果列表在 body['novel']['data']。"""
    kw = (keyword or "").strip()
    if not kw:
        return {"error": "关键词为空"}
    enc = urllib.parse.quote(kw)
    url = (_AJAX_SEARCH.format(enc) + f"?word={enc}&order=date_d&mode=all&p={page}"
           + "&type=all&s_mode=s_tag&r18=off")
    return await _fetch_json(url, _BASE_URL + "/tags/")


# ------------------------------------------------------------------ 工具函数

def _extract_id(target: str) -> tuple:
    """识别目标类型与 ID。返回 (kind, id)，kind ∈ series/novel/''。"""
    target = (target or "").strip()
    m = _SERIES_URL_RE.search(target)
    if m:
        return "series", m.group("id")
    m = _NOVEL_URL_RE.search(target)
    if m:
        return "novel", m.group("id")
    m = _NOVEL_SHORT_RE.search(target)
    if m:
        return "novel", m.group("id")
    digits = re.sub(r"\D", "", target)
    if digits:
        return "series", digits
    return "", ""


def _safe_filename(title: str, nid: str) -> str:
    """文件名：标题清洗非法字符 + ID。"""
    safe = _ILLEGAL_FILENAME_RE.sub("_", title or "小说")[:50].strip(" ._") or "小说"
    return f"{safe}_{nid}.txt"


def _extract_search_item(item: dict) -> dict:
    """从搜索结果条目提取统一字段。"""
    nid = str(item.get("id") or item.get("novelId") or "")
    series_id = item.get("seriesId") or item.get("series_id")
    series_id = None if series_id in (None, "", "null") else str(series_id)
    series_title = item.get("seriesTitle") or item.get("series_title")
    series_title = None if series_title in (None, "", "null") else str(series_title).strip()
    title = item.get("title") or "(无标题)"
    try:
        r18 = bool(int(item.get("xRestrict") or 0) >= 1)
    except (TypeError, ValueError):
        r18 = False
    return {
        "id": nid,
        "series_id": series_id,
        "series_title": series_title,
        "title": title,
        "display_title": series_title or title,
        "chapter_title": title if series_id else None,
        "author": item.get("userName") or item.get("user_name") or "",
        "r18": r18,
    }


def _format_chapter(idx: int, total: int, title: str, nid: str, text: str) -> str:
    """单章 txt 片段。"""
    lines = [f"第 {idx}/{total} 章  {title}", "-" * 40,
             f"链接: {_BASE_URL}/novel/show.php?id={nid}", "-" * 40, "", text]
    return "\n".join(lines)


def _build_series_txt(title: str, author: str, sid: str, chapters: list,
                      total: int, success: int) -> str:
    """系列全本合成：头部信息 + 章节分隔。"""
    header = "\n".join([
        "=" * 60, f"  {title}", "=" * 60,
        f"作者: {author}" if author else "作者: (未知)",
        f"系列ID: {sid}",
        f"章节数: {success}/{total}",
        f"来源: {_BASE_URL}/novel/series/{sid}",
        f"抓取时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
    ])
    return header + "\n\n" + ("=" * 60 + "\n\n").join(chapters)


def _build_single_txt(title: str, author: str, nid: str, text: str) -> str:
    """单篇合成。"""
    header = "\n".join([
        "=" * 60, f"  {title}", "=" * 60,
        f"作者: {author}" if author else "作者: (未知)",
        f"小说ID: {nid}",
        f"来源: {_BASE_URL}/novel/show.php?id={nid}",
        f"抓取时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
    ])
    return header + "\n\n" + "=" * 60 + "\n\n" + _format_chapter(1, 1, title, nid, text)


def _save_txt(title: str, nid: str, content: str) -> Path:
    """写入 save_dir（自动建目录），返回文件路径。"""
    save_dir = _save_dir()
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / _safe_filename(title, nid)
    path.write_text(content, encoding="utf-8")
    return path


async def _send_txt(ctx, path: Path) -> bool:
    """通过 NapCat 私聊发文件。"""
    try:
        return bool(await napcat_client.upload_private_file(
            str(ctx.meta.user_id), str(path), path.name))
    except Exception as e:
        logger.warning(f"上传私聊文件失败: {type(e).__name__}: {e}")
        return False


# ------------------------------------------------------------------ 命令子流程

async def _do_search(keyword: str, user_id: str) -> str:
    """搜索并缓存结果，返回编号列表文本。"""
    result = await _search_novels(keyword)
    if result.get("error"):
        return f"搜索失败了：{result['error']}，稍后再试试吧。"
    data_list = (result.get("novel") or {}).get("data") or []
    if not data_list:
        return f"没找到和「{keyword}」相关的小说，换个关键词试试？"
    items = [_extract_search_item(it) for it in data_list[:_SEARCH_RESULT_MAX]]
    _search_cache[user_id] = {"ts": time.time(), "items": items}
    lines = [f"搜索「{keyword}」结果（共 {len(items)} 条）："]
    for i, it in enumerate(items, 1):
        mark = " [R18]" if it["r18"] else ""
        kind = f"系列 {it['series_id']}" if it["series_id"] else "单篇"
        lines.append(f"{i}. {it['display_title']}{mark} [{kind}]")
        if it["author"]:
            lines.append(f"   作者: {it['author']}")
    lines.append("输入 /novel dl <编号> 下载对应小说（有系列的下载整部）")
    return "\n".join(lines)


async def _do_download_by_number(ctx, number_text: str) -> str:
    """按搜索缓存编号下载。"""
    user_id = str(ctx.meta.user_id or "")
    if not number_text.isdigit():
        return "请输入有效的编号，用法：/novel dl <编号>"
    entry = _search_cache.get(user_id)
    if not entry or time.time() - entry["ts"] > _SEARCH_CACHE_TTL:
        _search_cache.pop(user_id, None)
        return "没有搜索记录（或已过期），请先用 /novel search <关键词> 搜索。"
    items = entry["items"]
    number = int(number_text)
    if not (1 <= number <= len(items)):
        return f"编号超出范围（1-{len(items)}）。"
    it = items[number - 1]
    if it.get("series_id"):
        return await _do_series(ctx, it["series_id"])
    if it.get("id"):
        return await _do_single(ctx, it["id"])
    return "该条目缺少小说 ID，下载不了。"


async def _do_series(ctx, series_id: str) -> str:
    """抓取整个系列，合成一个 txt 发私聊文件。"""
    sid = str(series_id)
    await ctx.reply(f"开始抓取系列 {sid} 的全部章节，完成后发你文件～")
    series_meta, novels = await _fetch_series_all_novels(sid, _max_chapters())
    if series_meta.get("error"):
        return f"获取系列失败：{series_meta['error']}"
    if not novels:
        return "这个系列没有抓到任何章节，可能需要检查 Cookie 或该系列不可见。"

    title = series_meta.get("title") or "未知系列"
    author = series_meta.get("userName") or series_meta.get("user_name") or ""
    total = len(novels)
    await ctx.reply(f"《{title}》共 {total} 章，逐章抓取中，请稍候...")

    chapters: list = []
    success = 0
    for idx, item in enumerate(novels, 1):
        nid = str(item.get("id"))
        ctitle = item.get("title") or "(无标题)"
        data = await _fetch_novel_full(nid)
        if data.get("error"):
            logger.warning(f"第 {idx} 章抓取失败: {data['error']}")
            chapters.append(f"第 {idx}/{total} 章  {ctitle}\n[抓取失败: {data['error']}]")
        else:
            text = data.get("text") or ""
            chapters.append(_format_chapter(idx, total, ctitle, nid, text))
            success += 1
        if _CHAPTER_DELAY > 0:
            await asyncio.sleep(_CHAPTER_DELAY)

    full_text = _build_series_txt(title, author, sid, chapters, total, success)
    path = _save_txt(title, sid, full_text)
    logger.info(f"系列《{title}》({sid}) 已保存: {path}（{success}/{total} 章）")
    if await _send_txt(ctx, path):
        return f"《{title}》抓取完成（{success}/{total} 章），txt 已发你～"
    return f"《{title}》已保存到 {path.name}，但文件发送失败，稍后再试试吧。"


async def _do_single(ctx, novel_id: str) -> str:
    """抓取单篇小说发 txt。"""
    nid = str(novel_id)
    data = await _fetch_novel_full(nid)
    if data.get("error"):
        return f"获取小说失败：{data['error']}"
    title = data.get("title") or "(无标题)"
    text = data.get("text") or ""
    if not text:
        return f"《{title}》正文为空，可能需要检查 Cookie 或该作品不可见。"
    author = data.get("userName") or data.get("user_name") or ""
    path = _save_txt(title, nid, _build_single_txt(title, author, nid, text))
    logger.info(f"单篇《{title}》({nid}) 已保存: {path}")
    if await _send_txt(ctx, path):
        return f"《{title}》抓取完成，txt 已发你～"
    return f"《{title}》已保存到 {path.name}，但文件发送失败，稍后再试试吧。"


async def _do_list(series_id: str) -> str:
    """列出系列章节目录文本（不下载正文）。"""
    sid = str(series_id)
    series_meta, novels = await _fetch_series_all_novels(sid, _max_chapters())
    if series_meta.get("error"):
        return f"获取系列失败：{series_meta['error']}"
    if not novels:
        return "这个系列没有抓到任何章节。"
    title = series_meta.get("title") or "未知系列"
    author = series_meta.get("userName") or series_meta.get("user_name") or ""
    lines = [f"系列: {title}" + (f"（作者: {author}）" if author else ""),
             f"共 {len(novels)} 章，目录："]
    for i, item in enumerate(novels, 1):
        lines.append(f"{i}. {item.get('title') or '(无标题)'} (id:{item.get('id')})")
    lines.append(f"链接: {_BASE_URL}/novel/series/{sid}")
    lines.append("（仅列出目录，下载全文用 /novel <系列ID>）")
    return "\n".join(lines)


# ------------------------------------------------------------------ 命令入口

@register_command("novel", plugin="pixiv_novel",
                  description="Pixiv 小说下载：/novel <系列ID> | read <单篇ID> | list <系列ID> | search <关键词> | dl <编号>")
async def novel_cmd(ctx):
    """/novel 命令总入口：群聊拒绝 -> 白名单 -> 帮助 -> Cookie -> 冷却 -> 子命令。"""
    # 群聊统一拒绝（对齐旧插件：小说文件涉及内容风险）
    if ctx.session.is_group:
        return _GROUP_REJECT
    user_id = str(ctx.meta.user_id or "")
    # 功能白名单（不上报 security）
    if not _is_allowed(user_id):
        return _NOT_ALLOWED

    args = (ctx.args or "").strip()
    if not args or args.lower() in ("help", "帮助", "?", "？"):
        return _HELP

    # 无 Cookie 时功能不可用（不消耗冷却）
    if not _cookie():
        return _NO_COOKIE

    # 每用户冷却
    now = time.time()
    left = _cooldown_seconds() - (now - _last_use.get(user_id, 0))
    if left > 0:
        return f"冷却中，{int(left) + 1} 秒后再来吧。"

    tokens = args.split()
    first = tokens[0].lower()
    sub, target = "series", args
    if first in ("read", "读", "正文"):
        sub, target = "read", " ".join(tokens[1:]).strip()
    elif first in ("list", "目录", "列表"):
        sub, target = "list", " ".join(tokens[1:]).strip()
    elif first in ("search", "搜索", "搜", "find"):
        sub, target = "search", " ".join(tokens[1:]).strip()
    elif first in ("dl", "download", "下载", "下"):
        sub, target = "dl", " ".join(tokens[1:]).strip()

    _last_use[user_id] = now

    if sub == "search":
        if not target:
            return "请输入搜索关键词，用法：/novel search <关键词>"
        return await _do_search(target, user_id)
    if sub == "dl":
        return await _do_download_by_number(ctx, target)

    if not target:
        return "没识别到有效的 URL 或 ID。\n\n" + _HELP
    kind, nid = _extract_id(target)
    if not nid:
        return "没识别到有效的 URL 或 ID。\n\n" + _HELP

    if sub == "read":
        # 裸数字 ID 在 read 语境按单篇处理（旧插件此处误拒，属修正）
        if kind == "series" and not target.strip().isdigit():
            return "read 子命令只支持单篇小说 URL/ID。"
        return await _do_single(ctx, nid)
    if sub == "list":
        if kind == "novel":
            return "list 子命令只支持小说系列 URL/ID。"
        return await _do_list(nid)
    # 默认：系列；识别为单篇 URL 时降级按单篇处理（对齐旧插件）
    if kind == "novel":
        return await _do_single(ctx, nid)
    return await _do_series(ctx, nid)


# 仅命令执行，不注册任何 LLM 工具
TOOLS = []
