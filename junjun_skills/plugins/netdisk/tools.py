"""netdisk 插件：网盘分享链接转直链（迁移自 netdisk_parser_plugin，新架构重写）。

命令：/netdisk /直链 /网盘解析 <链接> [密码]
拦截器：消息含网盘分享链接（蓝奏云/123/奶牛/百度等）自动解析，群聊不要求 @；
        缺提取码时记录待补状态，该用户补发一条 4-8 位提取码即自动重试（5 分钟有效）
API：本机 netdisk-fast-download 服务，GET /json/parser?url=<分享链接>&pwd=<密码>
     （默认 http://127.0.0.1:6400，可用 NETDISK_API_BASE 覆盖）
"""

import os
import re
import time

import httpx

from junjun_agent.commands import register_command
from junjun_agent.interceptors import register_interceptor
from junjun_core.observability import get_logger

logger = get_logger("plugin.netdisk")

_API_BASE = os.environ.get("NETDISK_API_BASE", "http://127.0.0.1:6400")
_TIMEOUT = 60.0       # 解析服务超时（秒）
_PENDING_TTL = 300.0  # 待补密码状态有效期（秒）
_RATE_LIMIT = 20.0    # 每会话自动解析最小间隔（秒）
_LINK_PREFIX = "🔗 直链（有效期有限，尽快下载）："

# 支持的网盘分享链接（与旧插件正则集、netdisk-fast-download 覆盖范围对齐）
NETDISK_URL_RE = re.compile(
    r"https?://(?:"
    r"[a-zA-Z0-9-]+\.)?(?:"
    r"lanzou[a-z]?\.com|ilanzou\.com|"          # 蓝奏云 / 蓝奏云优享
    r"feijipan\.com|feijix\.com|"               # 小飞机
    r"cowtransfer\.com|"                        # 奶牛快传
    r"123pan\.com|123865\.com|123684\.com|"     # 123网盘
    r"wenshushu\.cn|ws\d{2}\.cn|"               # 文叔叔
    r"fangcloud\.(?:com|cn)|"                   # 亿方云
    r"lecloud\.lenovo\.com|"                    # 联想乐云
    r"ctfile\.com|474b\.com|ct\.ghpym\.com|"    # 城通
    r"ecpan\.cn|"                               # 移动云空间
    r"118pan\.com|vyuyun\.com|"                 # 118 / 微雨
    r"115\.com|115cdn\.com|anxia\.com|"         # 115
    r"pan\.baidu\.com|yun\.baidu\.com|"         # 百度
    r"drive\.google\.com|onedrive\.live\.com|"  # 海外盘
    r"dropbox\.com|icloud\.com\.cn"
    r")/[^\s，。、；：！？（）()【】「」]+",  # 链接在中文标点处截断
    re.IGNORECASE,
)

# 待补密码状态：{(chat_id, user_id): {"url": 分享链接, "ts": 记录时间}}
_pending_pwd: dict = {}
# 每会话上次自动解析时间（限流用）：{chat_id: ts}
_last_parse: dict = {}

# 解析失败信息里「缺密码/密码错」的特征关键词（不同网盘措辞不一，宽松匹配）
_NEED_PWD_HINTS = ("密码", "提取码", "pwd", "password", "verify", "需要验证", "校验")


def _first_netdisk_url(text: str) -> str | None:
    """取文本中第一个网盘分享链接，去掉尾随的中文标点。"""
    m = NETDISK_URL_RE.search(text or "")
    return m.group(0).rstrip("，。、）)】」』") if m else None


def _extract_pwd(text: str, url: str) -> str:
    """从触发文本提取分享密码（密码:xxxx / 提取码 xxxx / pwd=xxxx / 链接后 @xxxx）。"""
    if not text:
        return ""
    tail = text.split(url, 1)[-1] if url in text else text
    m = re.search(r"@([A-Za-z0-9]{2,8})\b", tail)
    if m:
        return m.group(1)
    m = re.search(r"(?:密码|提取码|访问码|pwd|password)\s*[:：=\s]\s*([A-Za-z0-9]{2,8})",
                  text, re.IGNORECASE)
    return m.group(1) if m else ""


async def fetch_direct_link(share_url: str, pwd: str = "") -> dict:
    """调用 netdisk-fast-download 的 /json/parser 接口。

    返回统一结果 {"ok", "link", "err", "server_error"}；任何异常都降级为 err 文本，绝不抛出。
    server_error=True 表示服务端异常（非 JSON / 格式异常），常见于加密链接缺密码时服务端崩溃。
    """
    params = {"url": share_url}
    if pwd:
        params["pwd"] = pwd
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(f"{_API_BASE}/json/parser", params=params)
        try:
            raw = resp.json()
        except Exception:
            return {"ok": False, "link": "",
                    "err": f"解析服务异常（HTTP {resp.status_code}）", "server_error": True}
    except Exception as e:
        logger.warning(f"netdisk 解析请求失败: {type(e).__name__}: {e}")
        return {"ok": False, "link": "", "err": "无法连接解析服务", "server_error": False}

    if not isinstance(raw, dict):
        return {"ok": False, "link": "", "err": "解析服务返回格式异常", "server_error": True}
    data = raw.get("data") or {}
    direct = ""
    if isinstance(data, dict):
        direct = data.get("directLink") or data.get("direct_link") or ""
    if raw.get("success") and direct:
        return {"ok": True, "link": direct, "err": "", "server_error": False}
    err = str(raw.get("msg") or raw.get("message") or "解析失败")
    return {"ok": False, "link": "", "err": err, "server_error": False}


def _err_needs_pwd(result: dict) -> bool:
    """粗略判断失败是否疑似缺密码/密码错。认不准就按普通失败处理。"""
    if result.get("server_error"):
        return True
    low = (result.get("err") or "").lower()
    return any(h.lower() in low for h in _NEED_PWD_HINTS)


async def _parse_and_reply(ctx, share_url: str, pwd: str = "") -> bool:
    """核心流程：调接口 → 发直链 / 记待补密码 / 失败降级。返回是否解析成功。"""
    key = (ctx.session.chat_id, str(ctx.meta.user_id or ""))
    result = await fetch_direct_link(share_url, pwd)
    if result["ok"]:
        _pending_pwd.pop(key, None)  # 成功即清掉该用户在本会话的待补状态
        await ctx.reply(f"{_LINK_PREFIX}\n{result['link']}")
        return True
    if not pwd and ctx.meta.user_id and _err_needs_pwd(result):
        # 没带密码且疑似缺密码 → 记下待补状态，提示用户补发提取码
        _pending_pwd[key] = {"url": share_url, "ts": time.time()}
        logger.info(f"网盘解析: 记录待补密码 chat={ctx.session.chat_id} url={share_url[:60]}")
        await ctx.reply("这个链接要提取码才能打开喵~ "
                        "把提取码单独回我一条就行，我马上帮你重拆 (｡･ω･｡)")
        return False
    await ctx.reply(f"呜…这条链接君君没扒动喵，可能是失效了或者密码不对~\n（{result['err']}）")
    return False


@register_interceptor(NETDISK_URL_RE.pattern, name="netdisk_link", plugin="netdisk")
async def netdisk_link_hit(ctx) -> bool:
    """自动识别消息中的网盘分享链接并解析（群聊不要求 @）。"""
    text = ctx.meta.text or ""
    share = _first_netdisk_url(text)
    if not share:
        return False
    # 每会话限流：间隔内静默消费，避免刷链接进 LLM
    now = time.time()
    chat_id = ctx.session.chat_id
    if now - _last_parse.get(chat_id, 0.0) < _RATE_LIMIT:
        return True
    _last_parse[chat_id] = now
    pwd = _extract_pwd(text, share)
    logger.info(f"网盘解析(自动): url={share[:80]} pwd={'有' if pwd else '无'}")
    await _parse_and_reply(ctx, share, pwd)
    return True


@register_interceptor(r"^\w{4,8}$", name="netdisk_pwd", plugin="netdisk")
async def netdisk_pwd_hit(ctx) -> bool:
    """补提取码：有待补状态且整条消息像提取码（4-8 位）时自动重试。"""
    key = (ctx.session.chat_id, str(ctx.meta.user_id or ""))
    pending = _pending_pwd.get(key)
    if not pending:
        return False  # 没有待补密码状态，放行给正常决策
    if time.time() - pending.get("ts", 0) > _PENDING_TTL:
        _pending_pwd.pop(key, None)  # 过期作废，放行
        return False
    _pending_pwd.pop(key, None)
    pwd = (ctx.meta.text or "").strip()
    logger.info(f"网盘解析(补密码): url={pending['url'][:60]}")
    await _parse_and_reply(ctx, pending["url"], pwd)
    return True


@register_command("netdisk", aliases=["直链", "网盘解析"], plugin="netdisk",
                  description="网盘分享链接转直链")
async def netdisk_cmd(ctx):
    parts = ctx.args.split()
    url = parts[0] if parts else ""
    pwd = parts[1] if len(parts) > 1 else ""
    if not url:
        return "用法：/netdisk <网盘分享链接> [密码]（或 /直链、/网盘解析）"
    if not NETDISK_URL_RE.search(url):
        return "请提供有效的网盘分享链接（蓝奏云/123/奶牛/百度等）"
    if not pwd:
        # 命令未显式带密码时，尝试从原文里提取（如「提取码: xxxx」）
        pwd = _extract_pwd(ctx.meta.text or "", url)
    logger.info(f"网盘解析(命令): url={url[:80]} pwd={'有' if pwd else '无'}")
    await _parse_and_reply(ctx, url, pwd)
    return None


TOOLS = []
