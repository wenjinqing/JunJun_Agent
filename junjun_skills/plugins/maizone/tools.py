"""maizone 插件：QQ空间发说说 / 看空间 / 定时监控点赞评论 / 回复评论 / 定时自动发说说。

迁移自旧 MaiBot maizone_plugin，仅提取其协议知识用新架构重写：
- g_tk/bkn 哈希算法（对 p_skey 的 5381 哈希，旧代码原样提取）
- cookie 获取链：NapCat get_cookies（qzone.qq.com / user.qzone.qq.com 多域合并
  确保拿到 p_skey）→ 本地缓存文件兜底；登录态失效时强制重取一次
- Qzone 端点：emotion_cgi_publish_v6（发说说）、feeds3_html_more（好友说说列表）、
  internal_dolike_app（点赞）、emotion_cgi_re_feeds（评论/回复评论）、
  cgi_upload_image（说说配图上传）、emotion_cgi_msglist_v6（自己的说说+评论列表）
- JSONP 剥壳：_Callback(...); / _preloadCallback(...) 用正则处理，不引新依赖

LLM 工具（空间 = 第三聊天场景，Agent 自主决策）：
- send_feed   发说说（可带 AI 配图；空间不支持语音/视频）
- read_feed   看好友空间说说

命令（全部 admin_only，bot 身份操作）：
- /send_feed [主题]（/发说说）  LLM 写一条说说并发布，回执文本
- /read_feed [数量]（/看空间）  拉好友说说列表做文本摘要
- /qzone_status                cookie 状态 / 今日已评论数 / 各开关

定时任务：
- maizone_monitor（10 分钟）：monitor_enable 时刷好友空间，
  对未处理说说点赞（like_enable）和/或评论（comment_enable，每日上限
  max_reply_per_day），处理记录落 data/maizone/processed_list.json；
  reply_comment_enable 时回复好友对自己说说的评论（首日基线不回旧评论）。
- maizone_auto_post（1 分钟检查）：schedule_enable 时按当日波动时间表
  自动发说说（schedule_times ± fluctuation_minutes，每日发送概率
  schedule_probability，配图概率 schedule_image_probability），
  与手动发共享每日上限 max_feed_per_day。
"""

import base64
import json
import random
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
from langchain_core.tools import tool

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
DELETE_FEED_URL = ("https://user.qzone.qq.com/proxy/domain/"
                   "taotao.qq.com/cgi-bin/emotion_cgi_delete_v6")
# 说说配图上传 / 自己的说说列表（带评论）/ 回复评论（h5 域）
UPLOAD_IMAGE_URL = "https://up.qzone.qq.com/cgi-bin/upload/cgi_upload_image"
LIST_URL = ("https://user.qzone.qq.com/proxy/domain/"
            "taotao.qq.com/cgi-bin/emotion_cgi_msglist_v6")
REPLY_URL = ("https://h5.qzone.qq.com/proxy/domain/"
             "taotao.qzone.qq.com/cgi-bin/emotion_cgi_re_feeds")

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36")

_MONITOR_INTERVAL = 600          # 刷空间间隔（秒）
_PROCESSED_CACHE_SIZE = 200      # 已处理记录上限，防无限增长
_REPLIED_CACHE_SIZE = 500        # 已回复评论记录上限
_MAX_FEED_IMAGES = 3             # 单条说说配图上限


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


def _parse_qzone_json(text: str) -> dict:
    """解析 Qzone 非标准 JSON（data 段是 JS 对象字面量，key 无引号，单引号字符串）。

    Qzone feeds3_html_more 的 format=json 实际返回 JSON5 变体：
    - 外层是标准 JSON（code/subcode/message）
    - data 段是 JS 对象（{main:{...}}，key 无引号，值用单引号）
    用 json5 解析（对齐原插件实现）。
    """
    import json5
    return json5.loads(_strip_jsonp(text))


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


def _parse_frame_callback(text: str, action: str) -> dict:
    """解析 frameElement.callback({...}) 响应。

    re_feeds/delete_v6 的 payload 可长达上万字符且内含大量英文括号
    （说说 HTML 数据），非贪婪正则 (.*?) 会在第一个 ) 处截断导致
    "Unexpected end of input"（评论其实已发出，纯解析假报警——2026-07-29）。
    从 callback( 起取到全文最后一个 ) 为止。
    """
    import json5
    marker = "frameElement.callback("
    start = text.find(marker)
    if start == -1:
        raise RuntimeError(f"{action}失败: 无法解析响应 {text[:100]}")
    inner = text[start + len(marker):]
    end = inner.rfind(")")
    if end <= 0:
        raise RuntimeError(f"{action}失败: 响应不完整 {text[:100]}")
    try:
        return json5.loads(inner[:end].replace("undefined", "null"))
    except ValueError as e:
        raise RuntimeError(f"{action}失败: 响应解析异常 {e}")


# ---------------------------------------------------------------- Qzone API
# 全部隔离为独立 async helper；签名统一 (cookies, uin, ...)，供 _with_auth_retry 注入

async def upload_image(cookies: dict, uin: str, image: bytes) -> tuple:
    """上传图片到 QQ 空间（base64 表单），返回 (picbo, richval)。

    响应是非标准 JSON 文本（旧插件用 eval，这里切 {...} 后 json5 解析）。
    """
    gtk = generate_gtk(cookies["p_skey"])
    post_data = {
        "filename": "filename",
        "zzpanelkey": "",
        "uploadtype": "1",
        "albumtype": "7",
        "exttype": "0",
        "skey": cookies["skey"],
        "zzpaneluin": uin,
        "p_uin": uin,
        "uin": uin,
        "p_skey": cookies["p_skey"],
        "output_type": "json",
        "qzonetoken": "",
        "refer": "shuoshuo",
        "charset": "utf-8",
        "output_charset": "utf-8",
        "upload_hd": "1",
        "hd_width": "2048",
        "hd_height": "10000",
        "hd_quality": "96",
        "backUrls": ("http://upbak.photo.qzone.qq.com/cgi-bin/upload/cgi_upload_image,"
                     "http://119.147.64.75/cgi-bin/upload/cgi_upload_image"),
        "url": f"{UPLOAD_IMAGE_URL}?g_tk={gtk}",
        "base64": "1",
        "picfile": base64.b64encode(image).decode("ascii"),
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            UPLOAD_IMAGE_URL,
            data=post_data,
            headers={
                "Cookie": _cookie_header(cookies),
                "User-Agent": _UA,
                "referer": f"https://user.qzone.qq.com/{uin}",
                "origin": "https://user.qzone.qq.com",
            },
        )
    import json5
    text = resp.text
    result = json5.loads(text[text.find("{"):text.rfind("}") + 1])
    if result.get("ret") != 0:
        raise RuntimeError(f"上传图片失败: ret={result.get('ret')}")
    data = result["data"]
    picbo = data["url"].split("&bo=")[1]
    richval = ",{},{},{},{},{},{},,{},{}".format(
        data["albumid"], data["lloc"], data["sloc"], data["type"],
        data["height"], data["width"], data["height"], data["width"])
    return picbo, richval


async def publish_feed(cookies: dict, uin: str, content: str,
                       images: Optional[list] = None) -> str:
    """发表说说（可带图片 bytes 列表，先逐张上传），返回 tid。"""
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
    images = (images or [])[:_MAX_FEED_IMAGES]
    if images:
        pic_bos, richvals = [], []
        for img in images:
            picbo, richval = await upload_image(cookies, uin, img)
            pic_bos.append(picbo)
            richvals.append(richval)
        post_data["pic_bo"] = ",".join(pic_bos)
        post_data["richtype"] = "1"
        post_data["richval"] = "\t".join(richvals)
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
    payload = _parse_qzone_json(resp.text)
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
    payload = _parse_qzone_json(resp.text)
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
    payload = _parse_qzone_json(resp.text)
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
    # 评论接口也返回 JSON5（key 无引号/单引号），用 json5 解析（对齐原插件）
    # 响应可能被截断（大说说/网络问题），json5 解析失败降级为警告不炸
    import json5
    try:
        payload = json5.loads(m.group(1).replace("undefined", "null"))
    except ValueError as e:
        raise RuntimeError(f"评论失败: 响应截断或格式异常 {e}")
    _check_code(payload, "评论")
    return True


async def get_own_feeds(cookies: dict, uin: str, num: int = 3) -> list:
    """拉自己的说说列表（emotion_cgi_msglist_v6，need_comment=1 带评论）。

    返回 [{tid, content, created_time, comments:[{nickname, qq_account,
    content, comment_tid, created_time, parent_tid}]}]；标准 JSON（_preloadCallback 壳）。
    """
    gtk = generate_gtk(cookies["p_skey"])
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            LIST_URL,
            params={
                "g_tk": gtk,
                "uin": uin,
                "ftype": 0,
                "sort": 0,
                "pos": 0,
                "num": num,
                "replynum": 50,
                "callback": "_preloadCallback",
                "code_version": 1,
                "format": "jsonp",
                "need_comment": 1,
                "need_private_comment": 1,
            },
            headers={
                "Cookie": _cookie_header(cookies),
                "User-Agent": _UA,
                "Referer": f"https://user.qzone.qq.com/{uin}",
            },
        )
    try:
        payload = json.loads(_strip_jsonp(resp.text))
    except ValueError as e:
        # 空响应/HTML 错误页：多为登录态失效，抛 _AuthError 触发 cookie 刷新重试
        raise _AuthError(f"获取自己的说说返回非 JSON（疑似登录态失效）: {e}")
    _check_code(payload, "获取自己的说说")

    feeds = []
    for msg in (payload.get("msglist") or []):
        tid = str(msg.get("tid", ""))
        if not tid:
            continue
        ts = msg.get("created_time", 0)
        created = (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
                   if ts else str(msg.get("createTime", "")))
        comments = []
        for c in (msg.get("commentlist") or []):
            comments.append({
                "nickname": str(c.get("name", "")),
                "qq_account": str(c.get("uin", "")),
                "content": str(c.get("content", "")),
                "comment_tid": str(c.get("tid", "")),
                "created_time": str(c.get("createTime", "") or c.get("createTime2", "")),
                "parent_tid": None,
            })
            for sub in (c.get("list_3") or []):
                comments.append({
                    "nickname": str(sub.get("name", "")),
                    "qq_account": str(sub.get("uin", "")),
                    "content": str(sub.get("content", "")),
                    "comment_tid": str(sub.get("tid", "")),
                    "created_time": str(sub.get("createTime", "") or sub.get("createTime2", "")),
                    "parent_tid": str(c.get("tid", "")),
                })
        feeds.append({"tid": tid,
                      "content": str(msg.get("content", "")),
                      "created_time": created,
                      "comments": comments})
    return feeds


async def delete_feed(cookies: dict, uin: str, tid: str) -> bool:
    """删除自己的一条说说（emotion_cgi_delete_v6）。

    format=fs 时响应是 frameElement.callback(...) HTML 壳；请求 format=json
    拿标准 JSON，两种壳都兼容解析。
    """
    gtk = generate_gtk(cookies["p_skey"])
    post_data = {
        "hostuin": uin,
        "tid": tid,
        "t1_source": "1",
        "code_version": "1",
        "format": "json",
        "qzreferrer": f"https://user.qzone.qq.com/{uin}",
    }
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            DELETE_FEED_URL,
            params={"g_tk": gtk},
            data=post_data,
            headers={
                "Cookie": _cookie_header(cookies),
                "User-Agent": _UA,
                "referer": f"https://user.qzone.qq.com/{uin}",
                "origin": "https://user.qzone.qq.com",
            },
        )
    m = re.search(r"frameElement\.callback\(", resp.text)
    if m:
        payload = _parse_frame_callback(resp.text, "删除说说")
    else:
        try:
            payload = json.loads(_strip_jsonp(resp.text))
        except ValueError as e:
            raise _AuthError(f"删除说说返回非 JSON（疑似登录态失效）: {e}")
    _check_code(payload, "删除说说")
    return True


async def reply_comment(cookies: dict, uin: str, fid: str,
                        target_nickname: str, content: str) -> bool:
    """回复自己说说下的评论（旧插件验证：子评论接口不可用，
    用标准评论格式 + 内容里 @目标昵称 + paramstr 触发提醒）。"""
    gtk = generate_gtk(cookies["p_skey"])
    post_data = {
        "topicId": f"{uin}_{fid}__1",
        "uin": uin,
        "hostUin": uin,
        "content": f"回复@ {target_nickname} ：{content}",
        "format": "fs",
        "plat": "qzone",
        "source": "ic",
        "platformid": 52,
        "ref": "feeds",
        "richtype": "",
        "richval": "",
        "paramstr": f"@{target_nickname}",
    }
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            REPLY_URL,
            params={"g_tk": gtk},
            data=post_data,
            headers={
                "Cookie": _cookie_header(cookies),
                "User-Agent": _UA,
                "referer": f"https://user.qzone.qq.com/{uin}",
                "origin": "https://user.qzone.qq.com",
            },
        )
    payload = _parse_frame_callback(resp.text, "回复评论")
    _check_code(payload, "回复评论")
    return True


async def _with_auth_retry(fn, *args):
    """携带登录态执行 Qzone 操作；登录态失效时强制重取 cookie 重试一次，
    网络瞬态错误（连接重置/超时）重试 3 次。

    cookie 三层都拿不到时返回 None（调用方回「空间登录态获取失败」）。
    """
    from junjun_core.retry import retry_async
    cookies = await ensure_cookies()
    if not cookies:
        return None
    try:
        return await retry_async(lambda: fn(cookies, *args), attempts=3,
                                 base_delay=1.0, label=f"qzone.{fn.__name__}",
                                 retry_on=(httpx.TransportError,))
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


async def _generate_feed_content(topic: str, history: str = "") -> str:
    """LLM 按人设写一条说说；失败降级模板文本。history 为近期说说（防重复主题）。"""
    personality, style = _persona()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    topic = (topic or "").strip()
    theme_part = f"主题是「{topic}」的" if topic else "记录日常生活的"
    history_part = (f"\n以下是你以前发过的说说，新说说不许和它们主题重复：\n{history}\n"
                    if history else "")
    prompt = (
        f"你是'{personality}'，现在是'{now}'，你想写一条{theme_part}说说发表在 QQ 空间上，"
        f"{style}，不要浮夸，不要夸张修辞，可以适当使用颜文字，{history_part}"
        "只输出一条说说正文的内容，不要输出多余内容"
        "（包括前后缀、冒号、引号、括号()、表情包、at 或 @ 等）。"
    )
    text = await _ask_llm(prompt)
    if text:
        return text
    return f"今天也想记录一下：{topic}。" if topic else "今天也要好好生活呀。"


async def _generate_diary_feed(chat_log: str, history: str) -> str:
    """日记体说说：把最近几天的聊天写成日记（素材已匿名化）。失败降级模板。"""
    personality, style = _persona()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    history_part = (f"\n以下是你以前发过的说说，新说说不许和它们内容重复：\n{history}\n"
                    if history else "")
    prompt = (
        f"你是'{personality}'，现在是'{now}'。以下是你这几天在群聊/私聊里的聊天记录"
        f"（已匿名，「某人A」「某人B」是不同的人，「我」是你自己）：\n{chat_log}\n"
        "把这几天的生活写成一条日记风的说说发在 QQ 空间：记录和谁聊了什么、"
        "发生了什么有趣的事、还有你自己的想法和心情。"
        f"{style}，第一人称，像真的日记，不要浮夸，可以适当使用颜文字。"
        "隐私红线：绝不出现真实昵称、QQ 号、群号，一律沿用「某人A」这样的代称；"
        "不要复述聊天里的隐私信息（地址、电话、账号、密码等）。"
        f"{history_part}"
        "只输出说说正文，不要输出多余内容（包括前后缀、冒号、引号、括号()、表情包、at 或 @ 等）。"
    )
    text = await _ask_llm(prompt)
    return text or "今天也在好好生活呀。"


def _recent_chat_log(limit: int = 120) -> str:
    """最近聊天记录渲染为匿名化日记素材：昵称→某人A/B/C，去 QQ 号/@，bot→我。"""
    try:
        from junjun_core.database import Messages
        rows = list(Messages.select()
                    .where(Messages.processed_plain_text != "")
                    .order_by(Messages.time.desc()).limit(limit))
        rows.reverse()
        alias: dict = {}
        lines = []
        for r in rows:
            text = (r.processed_plain_text or "").replace("\n", " ")
            text = re.sub(r"@\S+", "", text)
            text = re.sub(r"\d{5,}", "", text).strip()[:120]
            if not text:
                continue
            if r.is_bot:
                name = "我"
            else:
                nick = r.user_nickname or "群友"
                if nick not in alias:
                    alias[nick] = (f"某人{chr(65 + len(alias))}" if len(alias) < 26
                                   else f"某人{len(alias) + 1}")
                name = alias[nick]
            lines.append(f"{name}: {text}")
        return "\n".join(lines)[-3000:]
    except Exception as e:
        logger.debug(f"拉取日记素材失败: {e}")
        return ""


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


async def _generate_comment_reply(feed: dict, comment: dict) -> str:
    """LLM 按人设回复好友对自己说说的评论；失败降级模板文本。"""
    personality, style = _persona()
    prompt = (
        f"你是'{personality}'，你在 QQ 空间发了一条说说「{feed.get('content', '')[:150]}」，"
        f"好友'{comment.get('nickname', '好友')}'评论说「{comment.get('content', '')[:150]}」，"
        f"你想回复这条评论，{style}，回复要短、自然、像朋友聊天，说中文，"
        "不要浮夸，不要输出多余内容"
        "（包括前后缀、冒号、引号、括号()、表情包、at 或 @ 等——@前缀系统会自动加）。"
        "只输出回复内容。"
    )
    text = await _ask_llm(prompt)
    return text or "嘿嘿，谢谢你的评论呀~"


# ---------------------------------------------------------------- 配图（AI 生成 → bytes）

async def _download_image_bytes(url: str) -> Optional[bytes]:
    """下载图片 URL 为 bytes；失败返回 None。"""
    try:
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            resp = await client.get(url)
        if resp.status_code == 200 and len(resp.content) > 1000:
            return resp.content
        logger.warning(f"说说配图下载失败: HTTP {resp.status_code}")
    except Exception as e:
        logger.warning(f"说说配图下载异常: {type(e).__name__}: {e}")
    return None


async def _feed_image_bytes(prompt: str) -> Optional[bytes]:
    """说说配图：优先复用本会话正在画/刚画好的图（「画图发空间」只画一张），
    没有才走 ai_draw 管线自己生成。任何失败返回 None（降级纯文字说说）。"""
    try:
        from junjun_skills.builtin.memory_skills import current_chat_id
        from junjun_skills.plugins.ai_draw.tools import _draw_pipeline, wait_recent_drawn_url
        url = await wait_recent_drawn_url(current_chat_id.get(""))
        if url:
            logger.info("说说配图复用本会话刚画的图，不再重复生成")
        else:
            url, _final = await _draw_pipeline(prompt)
        if not url:
            return None
        return await _download_image_bytes(url)
    except Exception as e:
        logger.warning(f"说说配图生成失败: {type(e).__name__}: {e}")
    return None


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


def _daily_feed_count() -> int:
    """今日已发说说数（手动 + 自动共享，跨天清零）。"""
    state = _load_json(DATA_DIR / "feeds_sent.json", {})
    if not isinstance(state, dict) or state.get("date") != datetime.now().strftime("%Y-%m-%d"):
        return 0
    return int(state.get("count", 0))


def _incr_daily_feed() -> None:
    """今日说说数 +1 并落盘。"""
    state = {"date": datetime.now().strftime("%Y-%m-%d"),
             "count": _daily_feed_count() + 1}
    _save_json(DATA_DIR / "feeds_sent.json", state)


def _feed_quota_ok() -> bool:
    """今日说说额度是否还有（max_feed_per_day）。"""
    return _daily_feed_count() < int(_cfg().get("max_feed_per_day", 4))


def _load_replied() -> Optional[set]:
    """已回复评论集合 {"tid:comment_tid"}；文件不存在返回 None（首日基线用）。"""
    path = DATA_DIR / "replied_comments.json"
    if not path.exists():
        return None
    data = _load_json(path, [])
    return set(data) if isinstance(data, list) else set()


def _save_replied(replied: set) -> None:
    keys = list(replied)[-_REPLIED_CACHE_SIZE:]
    _save_json(DATA_DIR / "replied_comments.json", keys)


def _daily_reply_count() -> int:
    """今日已回复评论数（跨天清零）。"""
    state = _load_json(DATA_DIR / "replied_comments_daily.json", {})
    if not isinstance(state, dict) or state.get("date") != datetime.now().strftime("%Y-%m-%d"):
        return 0
    return int(state.get("count", 0))


def _incr_daily_reply() -> None:
    state = {"date": datetime.now().strftime("%Y-%m-%d"),
             "count": _daily_reply_count() + 1}
    _save_json(DATA_DIR / "replied_comments_daily.json", state)


# ---------------------------------------------------------------- LLM 工具
# 空间 = 第三聊天场景：Agent 自主发说说/刷空间；空间不支持语音/视频（不发语音说说）

@tool("send_feed")
async def send_feed_tool(content: str, with_image: bool = False) -> str:
    """在自己的 QQ 空间发一条说说（类似朋友圈）。想记录心情/分享见闻/用户要求发说说时用。
    content 就是说说正文（口语自然、可适当颜文字，不要 @ 和引号包裹）。
    空间不支持语音和视频，不要承诺发语音说说；with_image=True 会根据正文自动生成
    一张 AI 配图一起发（较慢，适合风景/心情/二次元主题）。
    要配图时直接 with_image=True 即可——本工具内部会自己画图，不要再单独调 ai_draw，
    否则会画出两张图。

    Args:
        content: 说说正文（100 字内效果最好）
        with_image: 是否自动配一张 AI 图（默认 False 纯文字）
    """
    if not (_switch("enable") and _switch("send_enable")):
        return "QQ空间发说说功能没开（config maizone enable/send_enable）。"
    content = (content or "").strip()
    if not content:
        return "说说正文是空的，没发。"
    if not _feed_quota_ok():
        return f"今天说说已经发了 {int(_cfg().get('max_feed_per_day', 4))} 条（到上限了），明天再发吧。"

    images = []
    if with_image:
        img = await _feed_image_bytes(content)
        if img:
            images.append(img)
        else:
            logger.info("说说配图生成失败，降级纯文字发布")

    try:
        tid = await _with_auth_retry(publish_feed, _bot_uin(), content, images)
    except Exception as e:
        logger.warning(f"工具发说说失败: {type(e).__name__}: {e}")
        return "发说说失败了，空间接口暂时不给力，稍后再试吧。"
    if tid is None:
        return "空间登录态获取失败，发不了说说（需要管理员重新登录 NapCat）。"
    _incr_daily_feed()
    logger.info(f"[tool] 说说已发布 tid={tid} 配图={len(images)}: {content[:50]}")
    return f"说说发出去了（{'带配图' if images else '纯文字'}）：{content}"


@tool("read_feed")
async def read_feed_tool(num: int = 5) -> str:
    """刷一下好友的 QQ 空间，看大家最近发的说说。想了解好友动态/找聊天话题/用户让你看空间时用。

    Args:
        num: 看几条（1-20，默认 5）
    """
    if not (_switch("enable") and _switch("read_enable")):
        return "QQ空间看空间功能没开（config maizone enable/read_enable）。"
    num = max(1, min(20, int(num or 5)))
    try:
        feeds = await _with_auth_retry(fetch_friend_feeds, _bot_uin(), num)
    except Exception as e:
        logger.warning(f"工具看空间失败: {type(e).__name__}: {e}")
        return "看空间失败了，空间接口暂时不给力，稍后再试吧。"
    if feeds is None:
        return "空间登录态获取失败，看不了空间（需要管理员重新登录 NapCat）。"
    if not feeds:
        return "好友空间最近静悄悄的，没有新说说。"
    lines = [f"好友最近的说说（{len(feeds)} 条）："]
    for i, f in enumerate(feeds, 1):
        name = f.get("nickname") or f.get("target_qq", "?")
        content = (f.get("content") or "")[:80] or "（无文字内容）"
        lines.append(f"{i}. {name}（{f.get('created_time', '未知时间')}）：{content}")
    return "\n".join(lines)


@tool("delete_feed")
async def delete_feed_tool(tid: str = "") -> str:
    """删除自己 QQ 空间已经发出去的说说。发错了/内容不妥/用户或管理员要求删说说时用。
    只能删自己发的说说；不知道 tid 就留空（默认删自己最新一条），
    或先 read_feed 查看最近说说的 tid。

    Args:
        tid: 要删的说说 ID（留空删自己最新一条）
    """
    if not (_switch("enable") and _switch("send_enable")):
        return "QQ空间功能没开（config maizone enable/send_enable）。"
    tid = (tid or "").strip()
    uin = _bot_uin()
    try:
        feeds = await _with_auth_retry(get_own_feeds, uin, 10)
    except Exception as e:
        logger.warning(f"工具删说说前查询失败: {type(e).__name__}: {e}")
        return "查自己的说说失败了，空间接口暂时不给力，稍后再试吧。"
    if feeds is None:
        return "空间登录态获取失败，删不了说说（需要管理员重新登录 NapCat）。"
    if not feeds:
        return "你还没有发过说说，没什么好删的。"
    if tid:
        target = next((f for f in feeds if f["tid"] == tid), None)
        if target is None:
            return (f"最近 10 条说说里没有 tid={tid}，只能删自己发的说说。"
                    "先 read_feed 确认一下 tid 吧。")
    else:
        target = feeds[0]
    try:
        ok = await _with_auth_retry(delete_feed, uin, target["tid"])
    except Exception as e:
        logger.warning(f"工具删说说失败: {type(e).__name__}: {e}")
        return "删说说失败了，空间接口暂时不给力，稍后再试吧。"
    if ok is None:
        return "空间登录态获取失败，删不了说说（需要管理员重新登录 NapCat）。"
    preview = (target.get("content") or "")[:30]
    logger.info(f"[tool] 说说已删除 tid={target['tid']}: {preview}")
    return f"说说删掉了（{target.get('created_time', '')} 发的那条）：{preview}"


# ---------------------------------------------------------------- 命令

@register_command("send_feed", aliases=["发说说"], plugin="maizone",
                  admin_only=True, description="发一条 QQ 空间说说")
async def send_feed_cmd(ctx) -> str:
    """/send_feed [主题]：LLM 写说说 → 发布 → 回执。"""
    if not (_switch("enable") and _switch("send_enable")):
        return "QQ空间发说说功能没开哦（config 里 maizone 的 enable / send_enable）。"
    if not _feed_quota_ok():
        return f"今天说说已经发了 {_daily_feed_count()} 条（到上限了），明天再发吧。"
    content = await _generate_feed_content(ctx.args)
    try:
        tid = await _with_auth_retry(publish_feed, _bot_uin(), content)
    except Exception as e:
        logger.warning(f"发表说说失败: {type(e).__name__}: {e}")
        return "发说说失败了，空间接口暂时不给力，稍后再试吧。"
    if tid is None:
        return "空间登录态获取失败，发不了说说（检查 NapCat 配置或重新登录）。"
    _incr_daily_feed()
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
    max_feed = int(cfg.get("max_feed_per_day", 4))

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
                  "monitor_enable", "like_enable", "comment_enable",
                  "schedule_enable", "reply_comment_enable"))
    return (f"QQ空间状态：\n{cookie_line}\n"
            f"今日已发说说：{_daily_feed_count()}/{max_feed}\n"
            f"今日已评论：{_daily_comment_count()}/{max_reply}\n"
            f"今日已回复评论：{_daily_reply_count()}/{int(cfg.get('max_comment_reply_per_day', 10))}\n"
            f"开关：{switches}")


# ---------------------------------------------------------------- 定时监控

async def maizone_monitor() -> None:
    """定时刷好友空间：对未处理说说点赞/评论，处理记录落盘（各开关热读）；
    随后回复好友对自己说说的评论（独立开关）。"""
    cfg = _cfg()
    if not (bool(cfg.get("enable", False)) and bool(cfg.get("monitor_enable", False))):
        return
    like_on = bool(cfg.get("like_enable", False))
    comment_on = bool(cfg.get("comment_enable", False))
    if not (like_on or comment_on):
        await _reply_own_feed_comments(cfg)
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
        await _reply_own_feed_comments(cfg)
        return
    if not feeds:
        await _reply_own_feed_comments(cfg)
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

    await _reply_own_feed_comments(cfg)


async def _reply_own_feed_comments(cfg: dict) -> None:
    """回复好友对自己说说的评论（空间 = 聊天场景的闭环）。

    首日运行只建基线（把存量评论标记为已回复），不轰炸旧评论。
    """
    if not (bool(cfg.get("enable", False)) and bool(cfg.get("reply_comment_enable", False))):
        return
    uin = _bot_uin()
    max_reply = int(cfg.get("max_comment_reply_per_day", 10))

    try:
        feeds = await _with_auth_retry(get_own_feeds, uin, 3)
    except Exception as e:
        logger.warning(f"maizone 拉自己的说说失败: {type(e).__name__}: {e}")
        return
    if not feeds:
        return

    replied = _load_replied()
    if replied is None:  # 首日：存量评论全部标记为已回复，不回
        replied = {f"{f['tid']}:{c['comment_tid'] or c['nickname']}"
                   for f in feeds for c in f["comments"]}
        _save_replied(replied)
        logger.info(f"maizone 评论回复基线已建立（{len(replied)} 条存量评论跳过）")
        return

    changed = False
    for feed in feeds:
        for c in feed["comments"]:
            if not c.get("qq_account") or c["qq_account"] == str(uin):
                continue  # 跳过自己的评论/回复
            key = f"{feed['tid']}:{c['comment_tid'] or c['nickname']}"
            if key in replied:
                continue
            if _daily_reply_count() >= max_reply:
                logger.info(f"今日评论回复已达上限 {max_reply}")
                _save_replied(replied)
                return
            text = await _generate_comment_reply(feed, c)
            try:
                ok = await _with_auth_retry(
                    reply_comment, uin, feed["tid"], c["nickname"], text)
                if ok:
                    _incr_daily_reply()
                    logger.info(f"已回复 {c['nickname']} 对说说 {feed['tid']} 的评论: {text[:30]}")
            except Exception as e:
                logger.warning(f"回复评论失败: {type(e).__name__}: {e}")
            replied.add(key)  # 失败也标记，避免每 10 分钟轰炸同一条
            changed = True
    if changed:
        _save_replied(replied)


# ---------------------------------------------------------------- 定时自动发说说

def _make_fluctuate_table(cfg: dict) -> list:
    """生成当日发送时间表：schedule_times 每个点 ± fluctuation_minutes 随机波动。"""
    times = cfg.get("schedule_times", ["09:30", "15:00", "21:00"])
    fluct = int(cfg.get("fluctuation_minutes", 45))
    table = []
    for base in times:
        try:
            h, m = map(int, str(base).split(":"))
        except ValueError:
            continue
        total = (h * 60 + m + (random.randint(-fluct, fluct) if fluct else 0)) % (24 * 60)
        table.append(f"{total // 60:02d}:{total % 60:02d}")
    return sorted(set(table))


async def _send_scheduled_feed(cfg: dict) -> None:
    """自动发一条说说：日记体（最近聊天记录匿名化）+ 历史去重 + 按概率配图。"""
    if not _feed_quota_ok():
        logger.info("今日说说已达上限，自动发送跳过")
        return

    history = ""
    try:
        own = await _with_auth_retry(get_own_feeds, _bot_uin(), 5)
        if own:
            history = "\n".join(f"- {f['content'][:60]}" for f in own if f.get("content"))
    except Exception as e:
        logger.warning(f"拉历史说说失败（不影响发送）: {type(e).__name__}: {e}")

    chat_log = _recent_chat_log()
    topic = ""
    if chat_log:
        content = await _generate_diary_feed(chat_log, history)
    else:  # 没有聊天素材（刚启动/冷场）退回主题模式
        topics = cfg.get("schedule_topics",
                         ["日常生活", "心情分享", "有趣见闻", "天气", "动漫游戏"])
        topic = random.choice(topics) if topics else ""
        content = await _generate_feed_content(topic, history)

    images = []
    if random.random() < float(cfg.get("schedule_image_probability", 0.4)):
        img = await _feed_image_bytes(topic or content[:40])
        if img:
            images.append(img)

    try:
        tid = await _with_auth_retry(publish_feed, _bot_uin(), content, images)
    except Exception as e:
        logger.warning(f"定时说说发布失败: {type(e).__name__}: {e}")
        return
    if tid is None:
        logger.warning("定时说说: 登录态获取失败，跳过本次")
        return
    _incr_daily_feed()
    mode = "日记" if chat_log else f"主题={topic}"
    logger.info(f"[auto] 定时说说已发布 tid={tid} {mode} 配图={len(images)}: {content[:50]}")


async def maizone_auto_post() -> None:
    """每分钟检查：到达当日波动时间表的时间点就自动发说说（每天按概率决定发不发）。"""
    cfg = _cfg()
    if not (bool(cfg.get("enable", False)) and bool(cfg.get("send_enable", False))
            and bool(cfg.get("schedule_enable", False))):
        return
    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    state = _load_json(DATA_DIR / "schedule_state.json", {})
    if not isinstance(state, dict) or state.get("date") != today:
        state = {
            "date": today,
            "times": _make_fluctuate_table(cfg),
            "allowed": random.random() < float(cfg.get("schedule_probability", 0.75)),
        }
        logger.info(f"maizone 今日发送时间表: {state['times']}（{'发' if state['allowed'] else '今天休息'}）")

    hhmm = now.strftime("%H:%M")
    fired = hhmm in state.get("times", [])
    if fired:
        state["times"].remove(hhmm)
    _save_json(DATA_DIR / "schedule_state.json", state)
    if fired and state.get("allowed"):
        await _send_scheduled_feed(cfg)


# ---------------------------------------------------------------- 注册

scheduler.add(ScheduledTask("maizone_monitor", maizone_monitor, interval=_MONITOR_INTERVAL,
                            plugin="maizone"))
scheduler.add(ScheduledTask("maizone_auto_post", maizone_auto_post, interval=60,
                            plugin="maizone"))

TOOLS = [send_feed_tool, read_feed_tool, delete_feed_tool]
