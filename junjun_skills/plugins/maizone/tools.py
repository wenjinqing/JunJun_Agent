"""maizone 插件：QQ空间发说说 / 看空间 / 定时监控点赞评论。

迁移自旧 MaiBot maizone_plugin，仅提取其协议知识用新架构重写：
- g_tk/bkn 哈希算法（对 p_skey 的 5381 哈希，旧代码原样提取）
- cookie 获取链：NapCat get_cookies（qzone.qq.com / user.qzone.qq.com 多域合并
  确保拿到 p_skey）→ 本地缓存文件兜底；登录态失效时强制重取一次
- Qzone 端点：emotion_cgi_publish_v6（发说说）、feeds3_html_more（好友说说列表）、
  internal_dolike_app（点赞）、emotion_cgi_re_feeds（评论）
- JSONP 剥壳：_Callback(...); / _preloadCallback(...) 用正则处理，不引新依赖

命令（全部 admin_only，bot 身份操作）：
- /send_feed [主题]（/发说说）  LLM 写一条说说并发布，回执文本
- /read_feed [数量]（/看空间）  拉好友说说列表做文本摘要
- /qzone_status                cookie 状态 / 今日已评论数 / 各开关

定时任务 maizone_monitor（10 分钟）：monitor_enable 时刷好友空间，
对未处理说说点赞（like_enable）和/或评论（comment_enable，每日上限
max_reply_per_day），处理记录落 data/maizone/processed_list.json。
"""

import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx

from junjun_agent.commands import register_command
from junjun_agent.loop.scheduler import ScheduledTask, scheduler
from junjun_core import napcat_client
from junjun_core.config import get_global_config
from junjun_core.observability import get_logger

logger = get_logger("plugin.maizone")

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "maizone"

# ---------------------------------------------------------------- Qzone 端点
# （提取自旧插件 qzone_api.py，纯文本说说，裁剪掉图片上传）

EMOTION_PUBLISH_URL = ("https://user.qzone.qq.com/proxy/domain/"
                       "taotao.qzone.qq.com/cgi-bin/emotion_cgi_publish_v6")
DOLIKE_URL = ("https://user.qzone.qq.com/proxy/domain/"
              "w.qzone.qq.com/cgi-bin/likes/internal_dolike_app")
COMMENT_URL = ("https://user.qzone.qq.com/proxy/domain/"
               "taotao.qzone.qq.com/cgi-bin/emotion_cgi_re_feeds")
ZONE_LIST_URL = ("https://user.qzone.qq.com/proxy/domain/"
                 "ic2.qzone.qq.com/cgi-bin/feeds/feeds3_html_more")

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36")

_MONITOR_INTERVAL = 600          # 刷空间间隔（秒）
_PROCESSED_CACHE_SIZE = 200      # 已处理记录上限，防无限增长


class _AuthError(Exception):
    """Qzone 登录态失效（需要强制刷新 cookie 重试）。"""


# ---------------------------------------------------------------- 配置

def _cfg() -> dict:
    """读取 [maizone] 配置节（热改生效，每次现读）。"""
    try:
        return get_global_config().raw.get("maizone", {}) or {}
    except Exception:
        return {}


def _switch(key: str, default: bool = False) -> bool:
    """读取布尔开关。"""
    return bool(_cfg().get(key, default))


def _bot_uin() -> str:
    return str(get_global_config().bot.qq_account or "")


# ---------------------------------------------------------------- g_tk

def generate_gtk(skey: str) -> str:
    """QQ空间 g_tk/bkn 哈希（旧插件算法原样提取，对 p_skey 计算）。"""
    hash_val = 5381
    for ch in skey:
        hash_val += (hash_val << 5) + ord(ch)
    return str(hash_val & 2147483647)


# ---------------------------------------------------------------- cookie 管理

def _cookie_path(uin: str) -> Path:
    """cookie 缓存文件路径（对齐旧插件命名 cookies-<uin>.json）。"""
    return DATA_DIR / f"cookies-{uin.lstrip('0')}.json"


def _parse_cookie_string(cookie_str: str) -> dict:
    """把 'k=v; k=v' 形式的 cookie 串解析成字典。"""
    result = {}
    for pair in cookie_str.split(";"):
        pair = pair.strip()
        if "=" in pair:
            k, v = pair.split("=", 1)
            result[k.strip()] = v.strip()
    return result


def _valid_cookies(cookies: Optional[dict]) -> bool:
    """关键登录态齐全（skey + p_skey）才算可用。"""
    return bool(cookies) and bool(cookies.get("skey")) and bool(cookies.get("p_skey"))


def _load_cached_cookies(uin: str) -> Optional[dict]:
    """读本地缓存 cookie 文件，失败返回 None。"""
    path = _cookie_path(uin)
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"读取 cookie 缓存失败: {e}")
        return None


def _save_cookies(uin: str, cookies: dict) -> None:
    """cookie 落盘缓存。"""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(_cookie_path(uin), "w", encoding="utf-8") as f:
            json.dump(cookies, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"保存 cookie 缓存失败: {e}")


async def _fetch_cookies_via_napcat() -> Optional[dict]:
    """通过 NapCat get_cookies 多域合并取 cookie（p_skey 在 qzone.qq.com 域）。"""
    merged = {}
    for domain in ("qzone.qq.com", "user.qzone.qq.com"):
        data = await napcat_client.call("get_cookies", {"domain": domain})
        if data and data.get("cookies"):
            merged.update(_parse_cookie_string(data["cookies"]))
    if not merged:
        return None
    return merged


async def ensure_cookies(force_refresh: bool = False) -> Optional[dict]:
    """确保拿到可用 cookie。三层：有效缓存 → NapCat 重取（成功则落盘）→ 旧缓存兜底。"""
    uin = _bot_uin()
    if not uin:
        logger.warning("未配置 bot.qq_account，无法获取空间登录态")
        return None

    if not force_refresh:
        cached = _load_cached_cookies(uin)
        if _valid_cookies(cached):
            return cached

    fresh = await _fetch_cookies_via_napcat()
    if _valid_cookies(fresh):
        _save_cookies(uin, fresh)
        return fresh

    # NapCat 不可用/缺关键键：退回旧缓存（可能已过期，让 API 层去验证）
    cached = _load_cached_cookies(uin)
    if _valid_cookies(cached):
        logger.info("NapCat 取 cookie 失败，退回本地旧缓存")
        return cached
    return None


def _cookie_header(cookies: dict) -> str:
    """拼接 Cookie 请求头。"""
    return "; ".join(f"{k}={v}" for k, v in cookies.items())


# ---------------------------------------------------------------- 响应解析

def _strip_jsonp(text: str) -> str:
    """剥掉 Qzone JSONP 外壳（_Callback(...); 等），并替换 undefined 为 null。"""
    text = text.strip()
    m = re.match(r"^[\w$]+\s*\((.*)\)\s*;?\s*$", text, re.S)
    if m:
        text = m.group(1)
    return text.replace("undefined", "null")


def _check_code(payload: dict, action: str) -> None:
    """检查 Qzone 响应 code；登录态类错误抛 _AuthError，其余抛 RuntimeError。"""
    code = payload.get("code")
    if code == 0:
        return
    msg = str(payload.get("message", ""))
    if code in (-3000, 4001, 4002, 4003) or "登录" in msg or "登陆" in msg:
        raise _AuthError(f"{action}登录态失效: code={code} {msg}")
    raise RuntimeError(f"{action}失败: code={code} {msg}")


def _html_to_text(html: str) -> str:
    """粗略剥离 HTML 标签取纯文本（feeds3_html_more 的说说正文，避免引 bs4）。"""
    txt = re.sub(r"<script.*?</script>", "", html, flags=re.S)
    txt = re.sub(r"<style.*?</style>", "", txt, flags=re.S)
    txt = re.sub(r"<[^>]+>", "", txt)
    return re.sub(r"\s+", " ", txt).strip()


# ---------------------------------------------------------------- Qzone API
# 全部隔离为独立 async helper；签名统一 (cookies, uin, ...)，供 _with_auth_retry 注入

async def publish_feed(cookies: dict, uin: str, content: str) -> str:
    """发表纯文本说说，返回 tid。"""
    gtk = generate_gtk(cookies["p_skey"])
    post_data = {
        "syn_tweet_verson": "1",
        "paramstr": "1",
        "who": "1",
        "con": content,
        "feedversion": "1",
        "ver": "1",
        "ugc_right": "1",
        "to_sign": "0",
        "hostuin": uin,
        "code_version": "1",
        "format": "json",
        "qzreferrer": f"https://user.qzone.qq.com/{uin}",
    }
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            EMOTION_PUBLISH_URL,
            params={"g_tk": gtk, "uin": uin},
            data=post_data,
            headers={
                "Cookie": _cookie_header(cookies),
                "User-Agent": _UA,
                "referer": f"https://user.qzone.qq.com/{uin}",
                "origin": "https://user.qzone.qq.com",
            },
        )
    payload = json.loads(_strip_jsonp(resp.text))
    _check_code(payload, "发表说说")
    return str(payload.get("tid", ""))


async def fetch_friend_feeds(cookies: dict, uin: str, num: int = 10) -> list:
    """拉好友空间说说列表（feeds3_html_more，appid=311 为说说）。

    返回 [{target_qq, tid, nickname, content, created_time}]，按页序（最新在前）。
    """
    gtk = generate_gtk(cookies["p_skey"])
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            ZONE_LIST_URL,
            params={
                "uin": uin,
                "scope": 0,
                "view": 1,
                "filter": "all",
                "flag": 1,
                "applist": "all",
                "pagenum": 1,
                "aisortEndTime": 0,
                "aisortOffset": 0,
                "aisortBeginTime": 0,
                "begintime": 0,
                "format": "json",
                "g_tk": gtk,
                "useutf8": 1,
                "outputhtmlfeed": 1,
            },
            headers={
                "Cookie": _cookie_header(cookies),
                "User-Agent": _UA,
                "Referer": f"https://user.qzone.qq.com/{uin}",
            },
        )
    payload = json.loads(_strip_jsonp(resp.text))
    if isinstance(payload, dict) and payload.get("code") not in (None, 0):
        _check_code(payload, "获取说说列表")
    raw_list = (payload.get("data") or {}).get("data") or []

    feeds = []
    for feed in raw_list:
        if not feed or str(feed.get("appid", "")) != "311":
            continue  # 只看说说，过滤广告/其他动态
        target_qq = str(feed.get("uin", ""))
        tid = str(feed.get("key", ""))
        html = feed.get("html", "")
        if not target_qq or not tid or not html:
            continue
        nick_m = re.search(r'class="f-name[^"]*"[^>]*>([^<]+)<', html)
        feeds.append({
            "target_qq": target_qq,
            "tid": tid,
            "nickname": nick_m.group(1).strip() if nick_m else "",
            "content": _html_to_text(html),
            "created_time": str(feed.get("feedstime", "")).strip(),  # 相对时间，如「昨天17:50」
        })
        if len(feeds) >= num:
            break
    return feeds


async def like_feed(cookies: dict, uin: str, target_qq: str, fid: str) -> bool:
    """点赞指定说说，成功 True。"""
    gtk = generate_gtk(cookies["p_skey"])
    post_data = {
        "qzreferrer": f"https://user.qzone.qq.com/{uin}",
        "opuin": uin,
        "unikey": f"http://user.qzone.qq.com/{target_qq}/mood/{fid}",
        "curkey": f"http://user.qzone.qq.com/{target_qq}/mood/{fid}",
        "appid": 311,
        "from": 1,
        "typeid": 0,
        "abstime": int(time.time()),
        "fid": fid,
        "active": 0,
        "format": "json",
        "fupdate": 1,
    }
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            DOLIKE_URL,
            params={"g_tk": gtk},
            data=post_data,
            headers={
                "Cookie": _cookie_header(cookies),
                "User-Agent": _UA,
                "referer": f"https://user.qzone.qq.com/{uin}",
                "origin": "https://user.qzone.qq.com",
            },
        )
    payload = json.loads(_strip_jsonp(resp.text))
    _check_code(payload, "点赞")
    return True


async def comment_feed(cookies: dict, uin: str, target_qq: str, fid: str, content: str) -> bool:
    """评论指定说说，成功 True（该接口响应是 HTML 包裹 frameElement.callback(json)）。"""
    gtk = generate_gtk(cookies["p_skey"])
    post_data = {
        "topicId": f"{target_qq}_{fid}__1",
        "uin": uin,
        "hostUin": target_qq,
        "feedsType": 100,
        "inCharset": "utf-8",
        "outCharset": "utf-8",
        "plat": "qzone",
        "source": "ic",
        "platformid": 52,
        "format": "fs",
        "ref": "feeds",
        "content": content,
    }
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            COMMENT_URL,
            params={"g_tk": gtk},
            data=post_data,
            headers={
                "Cookie": _cookie_header(cookies),
                "User-Agent": _UA,
                "referer": f"https://user.qzone.qq.com/{uin}",
                "origin": "https://user.qzone.qq.com",
            },
        )
    m = re.search(r"frameElement\.callback\((.*?)\)\s*;?\s*(?:</script>)?", resp.text, re.S)
    if not m:
        raise RuntimeError(f"评论失败: 无法解析响应 {resp.text[:100]}")
    payload = json.loads(m.group(1).replace("undefined", "null"))
    _check_code(payload, "评论")
    return True


async def _with_auth_retry(fn, *args):
    """携带登录态执行 Qzone 操作；登录态失效时强制重取 cookie 重试一次。

    cookie 三层都拿不到时返回 None（调用方回「空间登录态获取失败」）。
    """
    cookies = await ensure_cookies()
    if not cookies:
        return None
    try:
        return await fn(cookies, *args)
    except _AuthError:
        logger.info("登录态失效，强制刷新 cookie 重试")
        cookies = await ensure_cookies(force_refresh=True)
        if not cookies:
            return None
        return await fn(cookies, *args)


# ---------------------------------------------------------------- LLM 文案

async def _ask_llm(prompt: str) -> Optional[str]:
    """调用 utils 任务槽模型；任何失败返回 None（由调用方降级模板文本）。"""
    try:
        from langchain_core.messages import HumanMessage

        from junjun_llm import get_chat_model
        model = get_chat_model("utils")
        resp = await model.ainvoke([HumanMessage(content=prompt)])
        content = resp.content
        if isinstance(content, list):  # 兼容多段 content
            content = "".join(
                str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in content)
        return (content or "").strip() or None
    except Exception as e:
        logger.warning(f"maizone LLM 调用失败（将降级模板文本）: {type(e).__name__}: {e}")
        return None


def _persona() -> tuple:
    """取人设与回复风格（[personality] 节）。"""
    p = get_global_config().raw.get("personality", {}) or {}
    return p.get("personality", "一个 AI 助手"), p.get("reply_style", "")


async def _generate_feed_content(topic: str) -> str:
    """LLM 按人设写一条说说；失败降级模板文本。"""
    personality, style = _persona()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    topic = (topic or "").strip()
    theme_part = f"主题是「{topic}」的" if topic else "记录日常生活的"
    prompt = (
        f"你是'{personality}'，现在是'{now}'，你想写一条{theme_part}说说发表在 QQ 空间上，"
        f"{style}，不要浮夸，不要夸张修辞，可以适当使用颜文字，"
        "只输出一条说说正文的内容，不要输出多余内容"
        "（包括前后缀、冒号、引号、括号()、表情包、at 或 @ 等）。"
    )
    text = await _ask_llm(prompt)
    if text:
        return text
    return f"今天也想记录一下：{topic}。" if topic else "今天也要好好生活呀。"


async def _generate_comment(feed: dict) -> str:
    """LLM 按人设给好友说说写一条评论；失败降级模板文本。"""
    personality, style = _persona()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    name = feed.get("nickname") or feed.get("target_qq", "好友")
    prompt = (
        f"你是'{personality}'，你正在浏览好友'{name}'的 QQ 空间，"
        f"看到 ta 在'{feed.get('created_time', '')}'发了一条内容是"
        f"「{feed.get('content', '')[:200]}」的说说，现在是'{now}'，"
        f"你想发表一条评论，{style}，回复平淡一些、简短一些，说中文，"
        "不要浮夸，不要夸张修辞，不要输出多余内容"
        "（包括前后缀、冒号、引号、括号()、表情包、at 或 @ 等）。只输出评论内容。"
    )
    text = await _ask_llm(prompt)
    return text or "写得真好呀~"


# ---------------------------------------------------------------- 监控状态

def _load_json(path: Path, default):
    try:
        if path.exists():
            with open(path, encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"读取 {path.name} 失败: {e}")
    return default


def _save_json(path: Path, data) -> None:
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"写入 {path.name} 失败: {e}")


def _load_processed() -> dict:
    """已处理说说记录 {target_qq_tid: 时间戳}。"""
    data = _load_json(DATA_DIR / "processed_list.json", {})
    return data if isinstance(data, dict) else {}


def _save_processed(processed: dict) -> None:
    while len(processed) > _PROCESSED_CACHE_SIZE:  # 防无限增长，淘汰最旧
        processed.pop(next(iter(processed)))
    _save_json(DATA_DIR / "processed_list.json", processed)


def _daily_comment_count() -> int:
    """今日已评论数（跨天自动清零）。"""
    state = _load_json(DATA_DIR / "processed_comments.json", {})
    if not isinstance(state, dict) or state.get("date") != datetime.now().strftime("%Y-%m-%d"):
        return 0
    return int(state.get("count", 0))


def _incr_daily_comment() -> None:
    """今日评论数 +1 并落盘。"""
    state = {"date": datetime.now().strftime("%Y-%m-%d"),
             "count": _daily_comment_count() + 1}
    _save_json(DATA_DIR / "processed_comments.json", state)


# ---------------------------------------------------------------- 命令

@register_command("send_feed", aliases=["发说说"], plugin="maizone",
                  admin_only=True, description="发一条 QQ 空间说说")
async def send_feed_cmd(ctx) -> str:
    """/send_feed [主题]：LLM 写说说 → 发布 → 回执。"""
    if not (_switch("enable") and _switch("send_enable")):
        return "QQ空间发说说功能没开哦（config 里 maizone 的 enable / send_enable）。"
    content = await _generate_feed_content(ctx.args)
    try:
        tid = await _with_auth_retry(publish_feed, _bot_uin(), content)
    except Exception as e:
        logger.warning(f"发表说说失败: {type(e).__name__}: {e}")
        return "发说说失败了，空间接口暂时不给力，稍后再试吧。"
    if tid is None:
        return "空间登录态获取失败，发不了说说（检查 NapCat 配置或重新登录）。"
    logger.info(f"说说已发布 tid={tid}: {content[:50]}")
    return f"说说发出去啦：{content}"


@register_command("read_feed", aliases=["看空间"], plugin="maizone",
                  admin_only=True, description="看好友 QQ 空间说说")
async def read_feed_cmd(ctx) -> str:
    """/read_feed [数量]：拉好友说说列表做文本摘要（作者/内容/时间）。"""
    if not (_switch("enable") and _switch("read_enable")):
        return "QQ空间看空间功能没开哦（config 里 maizone 的 enable / read_enable）。"
    try:
        num = max(1, min(20, int((ctx.args or "").strip() or 5)))
    except ValueError:
        num = 5
    try:
        feeds = await _with_auth_retry(fetch_friend_feeds, _bot_uin(), num)
    except Exception as e:
        logger.warning(f"读取说说列表失败: {type(e).__name__}: {e}")
        return "看空间失败了，空间接口暂时不给力，稍后再试吧。"
    if feeds is None:
        return "空间登录态获取失败，看不了空间（检查 NapCat 配置或重新登录）。"
    if not feeds:
        return "好友空间最近静悄悄的，没有新说说。"
    lines = [f"好友最近的说说（{len(feeds)} 条）："]
    for i, f in enumerate(feeds, 1):
        name = f.get("nickname") or f.get("target_qq", "?")
        content = (f.get("content") or "")[:80] or "（无文字内容）"
        lines.append(f"{i}. {name}（{f.get('created_time', '未知时间')}）：{content}")
    return "\n".join(lines)


@register_command("qzone_status", plugin="maizone",
                  admin_only=True, description="QQ空间插件状态")
async def qzone_status_cmd(ctx) -> str:
    """/qzone_status：cookie 状态 / 今日已评论数 / 各开关状态。"""
    cfg = _cfg()
    max_reply = int(cfg.get("max_reply_per_day", 5))

    uin = _bot_uin()
    cached = _load_cached_cookies(uin) if uin else None
    if _valid_cookies(cached):
        mtime = datetime.fromtimestamp(
            _cookie_path(uin).stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        cookie_line = f"cookie 缓存有效（skey/p_skey 齐全，更新于 {mtime}）"
    elif cached:
        cookie_line = "cookie 缓存存在但缺关键键（skey/p_skey），需要重取"
    else:
        cookie_line = "无本地 cookie 缓存（NapCat 可用时会自动获取）"

    switches = "、".join(
        f"{k}={'开' if _switch(k) else '关'}"
        for k in ("enable", "send_enable", "read_enable",
                  "monitor_enable", "like_enable", "comment_enable"))
    return (f"QQ空间状态：\n{cookie_line}\n"
            f"今日已评论：{_daily_comment_count()}/{max_reply}\n"
            f"开关：{switches}")


# ---------------------------------------------------------------- 定时监控

async def maizone_monitor() -> None:
    """定时刷好友空间：对未处理说说点赞/评论，处理记录落盘（各开关热读）。"""
    cfg = _cfg()
    if not (bool(cfg.get("enable", False)) and bool(cfg.get("monitor_enable", False))):
        return
    like_on = bool(cfg.get("like_enable", False))
    comment_on = bool(cfg.get("comment_enable", False))
    if not (like_on or comment_on):
        return
    max_reply = int(cfg.get("max_reply_per_day", 5))

    uin = _bot_uin()
    try:
        feeds = await _with_auth_retry(fetch_friend_feeds, uin, 10)
    except Exception as e:
        err_name = type(e).__name__
        if "JSONDecodeError" in err_name or "Expecting" in str(e):
            logger.debug(f"maizone 监控: Qzone 返回非标准 JSON（cookie 过期/登录态失效），跳过本轮: {e}")
        else:
            logger.warning(f"maizone 监控拉取说说失败: {err_name}: {e}")
        return
    if not feeds:
        return

    processed = _load_processed()
    changed = False
    for feed in feeds:
        target_qq = str(feed.get("target_qq", ""))
        if target_qq == str(uin):
            continue  # 跳过自己的说说
        key = f"{target_qq}_{feed.get('tid', '')}"
        if key in processed:
            continue  # 已处理过，去重

        if like_on:
            try:
                ok = await _with_auth_retry(like_feed, uin, target_qq, feed["tid"])
                if ok:
                    logger.info(f"已点赞 {target_qq} 的说说 {feed['tid']}")
            except Exception as e:
                logger.warning(f"点赞失败: {type(e).__name__}: {e}")

        if comment_on:
            if _daily_comment_count() >= max_reply:
                logger.info(f"今日评论已达上限 {max_reply}，跳过评论")
            else:
                text = await _generate_comment(feed)
                try:
                    ok = await _with_auth_retry(
                        comment_feed, uin, target_qq, feed["tid"], text)
                    if ok:
                        _incr_daily_comment()
                        logger.info(f"已评论 {target_qq} 的说说 {feed['tid']}: {text[:30]}")
                except Exception as e:
                    logger.warning(f"评论失败: {type(e).__name__}: {e}")

        processed[key] = int(time.time())
        changed = True

    if changed:
        _save_processed(processed)


# ---------------------------------------------------------------- 注册

scheduler.add(ScheduledTask("maizone_monitor", maizone_monitor, interval=_MONITOR_INTERVAL))

TOOLS = []
