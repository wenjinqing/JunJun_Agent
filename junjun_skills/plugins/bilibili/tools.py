"""bilibili 插件：B 站视频链接自动/手动解析、下载与发送（迁移自 bilibili_video_sender_plugin，新架构重写）。

拦截器：消息含 bilibili.com/video/BVxxx 或 b23.tv 短链时自动解析
        （BILI_GROUP_AT_ONLY=true 时群聊仅 @ 触发；该值为静态注册，改 env 需重启）
命令：/bilibili <链接>（/b站 同）
流程：提取 BV 号（b23 短链先跟随重定向）-> view API（WBI 签名 + SESSDATA Cookie）
     取标题/简介/时长/封面 -> playurl 取流（优先 durl 单文件免合并；DASH 用 ffmpeg
     合并音视频）-> 下载到 data/bili_tmp/（发送后删除）-> 超时长/超大小用 ffmpeg
     压缩（压不进限制则降级信息卡）-> 发 video 段 + 标题文本
降级：任何失败或无 ffmpeg 时发视频信息卡（标题/UP主/简介/时长/封面 image/链接）
限流：每会话 60 秒最小间隔（内存 dict）
"""

import asyncio
import hashlib
import os
import re
import shutil
import time
import urllib.parse
from pathlib import Path

import httpx

from junjun_agent.commands import register_command
from junjun_agent.interceptors import register_interceptor
from junjun_core.contracts import ReplySegment
from junjun_core.observability import get_logger

logger = get_logger("plugin.bilibili")

# B 站视频链接：标准 BV/av 链接 + b23.tv 短链（允许携带查询参数，如 ?p= 分P）
BILI_LINK_RE = re.compile(
    r"https?://(?:(?:www|m)\.)?bilibili\.com/video/(?:BV\w+|av\d+)[^ \t\n\r\f\v]*|"
    r"https?://b23\.tv/\w+[^ \t\n\r\f\v]*",
    re.IGNORECASE,
)
# 命中串尾部误匹配的中英文标点
_TRAILING_PUNCT = ".,;!?，。；！？）】>》」"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36"
)

# 视频下载临时目录（项目根 data/bili_tmp，自动建，发送后清理）
TMP_DIR = Path(__file__).resolve().parents[3] / "data" / "bili_tmp"

_MIN_INTERVAL = 60.0  # 每会话最小解析间隔（秒）
_HTTP_TIMEOUT = 15.0  # API 请求超时（秒）
_DL_TIMEOUT = 120.0   # 视频下载超时（秒）

# 群聊仅 @ 触发（静态注册值：import 时读取，env 变化需重启）
_GROUP_AT_ONLY = os.environ.get("BILI_GROUP_AT_ONLY", "false").strip().lower() in ("1", "true", "yes")

# 每会话上次解析时间戳（chat_id -> ts）
_last_use: dict = {}

# WBI mixin key 索引表（B 站官方 WBI 签名，长度 64）
_MIXIN_KEY_INDICES = [
    46, 47, 18, 2, 53, 8, 23, 32, 15, 50, 10, 31, 58, 3, 45, 35,
    27, 43, 5, 49, 33, 9, 42, 19, 29, 28, 14, 39, 12, 38, 41, 13,
    37, 48, 7, 16, 24, 55, 40, 61, 26, 17, 0, 1, 60, 51, 30, 4,
    22, 25, 54, 21, 56, 59, 6, 63, 57, 62, 11, 36, 20, 34, 44, 52,
]
# mixin key 缓存（1 小时有效期，与旧插件一致）
_wbi_cache: dict = {"key": None, "at": 0.0}
_WBI_CACHE_TTL = 3600.0


# ---------------------------------------------------------------- 配置读取（运行时读 env，便于测试 monkeypatch）

def _sessdata() -> str:
    """B 站登录 Cookie 的 SESSDATA（提画质/过风控，可为空=游客模式）。"""
    return os.environ.get("BILI_SESSDATA", "").strip()


def _max_duration() -> float:
    """视频最大时长限制（秒），默认 600。"""
    try:
        return float(os.environ.get("BILI_MAX_DURATION", "600"))
    except ValueError:
        return 600.0


def _max_size_mb() -> float:
    """视频文件大小限制（MB），默认 100。"""
    try:
        return float(os.environ.get("BILI_MAX_SIZE_MB", "100"))
    except ValueError:
        return 100.0


# ---------------------------------------------------------------- 基础工具

def _first_bili_url(text: str) -> str | None:
    """从消息文本提取第一条 B 站链接，剥掉尾部误匹配的标点。"""
    if not text:
        return None
    m = BILI_LINK_RE.search(text)
    return m.group(0).rstrip(_TRAILING_PUNCT) if m else None


def _ffmpeg_path() -> str | None:
    """ffmpeg 可执行文件路径（PATH 探测）；None 表示不可用（降级信息卡模式）。"""
    return shutil.which("ffmpeg")


def probe_available() -> bool:
    """探测 ffmpeg 是否在 PATH；没有也返回 True（插件可降级为信息卡模式），但记 WARN。"""
    if _ffmpeg_path() is None:
        logger.warning("未在 PATH 检测到 ffmpeg：bilibili 插件将降级为「信息卡」模式（无法合并/压缩视频）")
    return True


def _check_rate_limit(chat_id: str) -> int:
    """返回剩余冷却秒数；0 表示可解析并记录本次时间。"""
    now = time.time()
    remain = _MIN_INTERVAL - (now - _last_use.get(chat_id, 0.0))
    if remain > 0:
        return int(remain) + 1
    _last_use[chat_id] = now
    return 0


def _spawn_bg(coro):
    """后台任务入口（独立函数便于测试替换为同步等待）。"""
    return asyncio.create_task(coro)


# ---------------------------------------------------------------- HTTP / WBI 签名

async def _fetch_json(url: str, params: dict | None = None) -> dict | None:
    """GET JSON（带 UA/Referer/SESSDATA Cookie）；任何失败返回 None（独立 helper 便于测试 mock）。"""
    headers = {"User-Agent": USER_AGENT, "Referer": "https://www.bilibili.com/"}
    if _sessdata():
        headers["Cookie"] = f"SESSDATA={_sessdata()}"
    try:
        async with httpx.AsyncClient(headers=headers, timeout=_HTTP_TIMEOUT) as client:
            resp = await client.get(url, params=params)
            return resp.json()
    except Exception as e:
        logger.warning(f"B站 API 请求失败 {url}: {type(e).__name__}: {e}")
        return None


async def _mixin_key() -> str | None:
    """从 nav 接口拉 wbi img/sub key 并按索引表混排成 mixin key（带 1 小时缓存）。"""
    now = time.time()
    if _wbi_cache["key"] and (now - _wbi_cache["at"]) < _WBI_CACHE_TTL:
        return _wbi_cache["key"]
    data = await _fetch_json("https://api.bilibili.com/x/web-interface/nav")
    wbi_img = (((data or {}).get("data") or {}).get("wbi_img")) or {}
    raw = ""
    for u in (wbi_img.get("img_url", ""), wbi_img.get("sub_url", "")):
        raw += u.rsplit("/", 1)[-1].split(".")[0] if u else ""
    if len(raw) < 64:
        logger.warning(f"WBI key 长度不足（{len(raw)}），本次放弃签名")
        return None
    mixed = "".join(raw[i] for i in _MIXIN_KEY_INDICES)[:32]
    _wbi_cache.update(key=mixed, at=now)
    return mixed


async def _wbi_sign(params: dict) -> dict:
    """WBI 签名：清洗参数 -> 加 wts -> 排序 urlencode -> md5(query+mixin_key)=w_rid。

    签名失败时原样返回（降级为非签名请求，与旧插件一致）。
    """
    key = await _mixin_key()
    if not key:
        return params
    safe = {k: (re.sub(r"[!'()*]", "", v) if isinstance(v, str) else v) for k, v in params.items()}
    safe["wts"] = int(time.time())
    query = urllib.parse.urlencode(sorted(safe.items(), key=lambda x: x[0]))
    safe["w_rid"] = hashlib.md5((query + key).encode("utf-8")).hexdigest()
    return safe


# ---------------------------------------------------------------- B 站 API（全部独立 async helper，便于 monkeypatch）

async def _follow_redirect(url: str) -> str:
    """跟随 b23.tv 短链重定向取落地 URL（curl UA 可避 412 风控）；失败返回原链接。"""
    try:
        async with httpx.AsyncClient(
            headers={"User-Agent": "curl/8.0"}, timeout=_HTTP_TIMEOUT, follow_redirects=True
        ) as client:
            resp = await client.get(url)
            return str(resp.url)
    except Exception as e:
        logger.warning(f"b23 短链跳转失败，仍用原链: {type(e).__name__}: {e}")
        return url


async def extract_bvid(url: str) -> str | None:
    """从链接提取 BV 号；b23.tv 短链先跟随重定向。失败返回 None。"""
    if "b23.tv" in url.lower():
        url = await _follow_redirect(url)
    m = re.search(r"/video/(BV\w+)", url, re.IGNORECASE)
    return m.group(1) if m else None


async def _fetch_view(bvid: str) -> dict | None:
    """view API 取视频信息（WBI 签名 + SESSDATA）：标题/简介/时长/封面/UP主/aid/cid。"""
    params = await _wbi_sign({"bvid": bvid})
    payload = await _fetch_json("https://api.bilibili.com/x/web-interface/view", params=params)
    if not payload or payload.get("code") != 0:
        logger.warning(f"view API 返回异常: code={payload.get('code') if payload else '无响应'}")
        return None
    data = payload.get("data") or {}
    pages = data.get("pages") or []
    if not pages:
        return None
    owner = data.get("owner") or {}
    return {
        "bvid": data.get("bvid") or bvid,
        "aid": data.get("aid"),
        "cid": pages[0].get("cid"),          # 默认取 P1
        "title": str(data.get("title", "")).strip() or "B站视频",
        "desc": str(data.get("desc", "")).strip(),
        "duration": int(data.get("duration") or pages[0].get("duration") or 0),
        "pic": data.get("pic") or "",
        "owner": str(owner.get("name", "")).strip(),
    }


def _codec_rank(codecs: str) -> int:
    """编码偏好：avc/h264 最优先（QQ 兼容性最好），其次 hevc，再次 av1。"""
    c = (codecs or "").lower()
    if "avc" in c or "h264" in c:
        return 0
    if "hev" in c or "hvc" in c:
        return 1
    if "av01" in c:
        return 2
    return 3


async def _fetch_playurl(aid: int, cid: int) -> dict | None:
    """playurl API 取播放流（WBI 签名）。

    返回 {"type":"durl","url":...}（单文件免合并，优先）或
         {"type":"dash","video":...,"audio":...|None}（需 ffmpeg 合并）；失败返回 None。
    """
    qn = 64 if _sessdata() else 32  # 登录默认 720P，游客默认 480P（与旧插件一致）
    params = {
        "avid": str(aid), "cid": str(cid), "otype": "json",
        "fnver": "0", "fnval": "4048", "fourk": "0", "platform": "pc", "qn": str(qn),
    }
    params = await _wbi_sign(params)
    payload = await _fetch_json("https://api.bilibili.com/x/player/wbi/playurl", params=params)
    if not payload or payload.get("code") != 0:
        logger.warning(f"playurl API 返回异常: code={payload.get('code') if payload else '无响应'}")
        return None
    data = payload.get("data") or {}

    # 优先 durl：单文件直链，无需 ffmpeg 合并
    durl = data.get("durl") or []
    if durl:
        url = durl[0].get("url") or durl[0].get("baseUrl") or durl[0].get("base_url")
        if url:
            return {"type": "durl", "url": url.replace("http:", "https:")}

    # DASH：视频流 + 音频流分离，需 ffmpeg 合并
    dash = data.get("dash") or {}
    videos = dash.get("video") or []
    audios = dash.get("audio") or []
    if not videos:
        return None
    # 清晰度不超过 qn 的流里取最高档，同档按编码偏好 + 码率排序
    eligible = [v for v in videos if int(v.get("id") or 0) <= qn] or list(videos)
    best_id = max(int(v.get("id") or 0) for v in eligible)
    candidates = [v for v in eligible if int(v.get("id") or 0) == best_id]
    candidates.sort(key=lambda v: (_codec_rank(str(v.get("codecs", ""))), -int(v.get("bandwidth") or 0)))
    video_url = candidates[0].get("baseUrl") or candidates[0].get("base_url")
    audio_url = None
    if audios:
        audios.sort(key=lambda a: int(a.get("bandwidth") or 0), reverse=True)
        audio_url = audios[0].get("baseUrl") or audios[0].get("base_url")
    if not video_url:
        return None
    return {
        "type": "dash",
        "video": video_url.replace("http:", "https:"),
        "audio": audio_url.replace("http:", "https:") if audio_url else None,
    }


async def _download(url: str, path: Path) -> bool:
    """流式下载到本地（带 Referer/Cookie 防盗链）；失败清理残文件并返回 False。"""
    headers = {
        "User-Agent": USER_AGENT,
        "Referer": "https://www.bilibili.com/",
        "Origin": "https://www.bilibili.com",
    }
    if _sessdata():
        headers["Cookie"] = f"SESSDATA={_sessdata()}"
    try:
        async with httpx.AsyncClient(headers=headers, timeout=_DL_TIMEOUT, follow_redirects=True) as client:
            async with client.stream("GET", url) as resp:
                resp.raise_for_status()
                with open(path, "wb") as f:
                    async for chunk in resp.aiter_bytes(1024 * 256):
                        f.write(chunk)
        return True
    except Exception as e:
        logger.warning(f"视频下载失败: {type(e).__name__}: {e}")
        path.unlink(missing_ok=True)
        return False


async def _run_ffmpeg(args: list) -> bool:
    """执行 ffmpeg 子进程（async，不阻塞事件循环）；返回是否成功。"""
    ffmpeg = _ffmpeg_path() or "ffmpeg"
    try:
        proc = await asyncio.create_subprocess_exec(
            ffmpeg, *args,
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
        )
        return (await proc.wait()) == 0
    except Exception as e:
        logger.warning(f"ffmpeg 执行失败: {type(e).__name__}: {e}")
        return False


async def _ffmpeg_merge(video: Path, audio: Path | None, out: Path) -> bool:
    """DASH 音视频合并（视频流 copy 不重编码；音频转 aac 保证兼容性）。"""
    if audio is not None:
        args = ["-i", str(video), "-i", str(audio), "-c:v", "copy", "-c:a", "aac",
                "-b:a", "192k", "-y", str(out)]
    else:
        args = ["-i", str(video), "-c", "copy", "-y", str(out)]
    return await _run_ffmpeg(args)


async def _ffmpeg_compress(src: Path, out: Path) -> bool:
    """压缩视频（libx264 crf28，兼顾体积与画质）。"""
    args = ["-i", str(src), "-c:v", "libx264", "-crf", "28", "-preset", "veryfast",
            "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", "-y", str(out)]
    return await _run_ffmpeg(args)


# ---------------------------------------------------------------- 发送

def _fmt_duration(seconds: int) -> str:
    """秒数格式化为「X分Y秒」。"""
    minutes, sec = divmod(int(seconds), 60)
    return f"{minutes}分{sec}秒" if minutes else f"{sec}秒"


async def _send_info_card(ctx, info: dict, page_url: str, note: str = "") -> None:
    """降级信息卡：标题/UP主/时长/简介/链接文本 + 封面 image 段。"""
    lines = [f"📺 {info['title']}"]
    if info.get("owner"):
        lines.append(f"UP主：{info['owner']}")
    if info.get("duration"):
        lines.append(f"时长：{_fmt_duration(info['duration'])}")
    if info.get("desc"):
        lines.append(f"简介：{info['desc'][:100]}")
    if note:
        lines.append(note)
    lines.append(f"🔗 {page_url}")
    segs = [ReplySegment(type="text", data="\n".join(lines))]
    pic = info.get("pic") or ""
    if pic.startswith("http"):
        segs.append(ReplySegment(type="image", data=pic))
    await ctx.send(segs)


# ---------------------------------------------------------------- 主流程

async def _process(ctx, url: str) -> None:
    """解析 -> 下载 -> （必要时压缩）-> 发送；所有失败路径降级为信息卡/友好文本，绝不抛异常。"""
    touched: list[Path] = []  # 本次产生的临时文件，finally 统一清理
    try:
        bvid = await extract_bvid(url)
        if not bvid:
            await ctx.reply("没认出这个 B 站链接，发个 bilibili.com/video/BVxxx 或 b23.tv 短链试试？")
            return

        info = await _fetch_view(bvid)
        if not info or not info.get("aid") or not info.get("cid"):
            await ctx.reply("视频信息获取失败了，可能是链接失效或被风控，稍后再试试吧。")
            return
        page_url = f"https://www.bilibili.com/video/{info['bvid']}"

        # 无 ffmpeg：只发信息卡
        if _ffmpeg_path() is None:
            logger.info("无 ffmpeg，降级发送信息卡")
            await _send_info_card(ctx, info, page_url, note="（当前环境无 ffmpeg，仅提供信息卡）")
            return

        sources = await _fetch_playurl(info["aid"], info["cid"])
        if not sources:
            await _send_info_card(ctx, info, page_url, note="（播放地址获取失败，请戳链接观看）")
            return

        TMP_DIR.mkdir(parents=True, exist_ok=True)
        base = TMP_DIR / f"{info['bvid']}_{int(time.time() * 1000)}"
        final: Path | None = None

        if sources["type"] == "durl":
            # 单文件直链：直接下载，免合并
            target = base.with_suffix(".mp4")
            touched.append(target)
            if await _download(sources["url"], target):
                final = target
        else:
            # DASH：分别下载音视频流后 ffmpeg 合并
            v_path = base.with_name(base.name + "_v.m4s")
            a_path = base.with_name(base.name + "_a.m4s")
            out_path = base.with_suffix(".mp4")
            touched += [v_path, a_path, out_path]
            if await _download(sources["video"], v_path):
                audio_ok = bool(sources.get("audio")) and await _download(sources["audio"], a_path)
                if not audio_ok:
                    logger.warning("音频流下载失败，合并为无声视频")
                if await _ffmpeg_merge(v_path, a_path if audio_ok else None, out_path):
                    final = out_path

        if final is None or not final.exists():
            await _send_info_card(ctx, info, page_url, note="（视频下载/合并失败，请戳链接观看）")
            return

        # 时长/大小限制决策：超限则尝试 ffmpeg 压缩
        size_mb = final.stat().st_size / (1024 * 1024)
        over_duration = _max_duration() > 0 and info.get("duration", 0) > _max_duration()
        over_size = size_mb > _max_size_mb()
        if over_duration or over_size:
            logger.info(f"视频超限（时长 {info.get('duration')}s / {size_mb:.1f}MB），尝试压缩")
            compressed = base.with_name(base.name + "_c.mp4")
            touched.append(compressed)
            if (await _ffmpeg_compress(final, compressed) and compressed.exists()
                    and compressed.stat().st_size <= _max_size_mb() * 1024 * 1024):
                final = compressed
            else:
                reason = f"视频时长 {_fmt_duration(info['duration'])} 超过限制" if over_duration \
                    else f"视频大小 {size_mb:.0f}MB 超过限制"
                await _send_info_card(ctx, info, page_url, note=f"（{reason}且压缩失败，请戳链接观看）")
                return

        await ctx.send([
            ReplySegment(type="text", data=f"📺 {info['title']}"),
            ReplySegment(type="video", data=str(final)),
        ])
        logger.info(f"B站视频已发送: {info['bvid']} {info['title'][:30]}")
    except Exception as e:
        logger.error(f"B站视频处理异常: {type(e).__name__}: {e}")
        try:
            await ctx.reply("解析过程中出了点问题，稍后再试试吧。")
        except Exception:
            pass
    finally:
        for p in touched:
            try:
                p.unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"临时文件清理失败 {p}: {e}")


# ---------------------------------------------------------------- 命令 / 拦截器注册

@register_command("bilibili", aliases=["b站"], plugin="bilibili",
                  description="解析 B 站视频链接并发送视频（BV 链接 / b23.tv 短链）")
async def bilibili_cmd(ctx):
    """手动解析：/bilibili <链接> 或 /b站 <链接>。"""
    url = (ctx.args or "").strip()
    if not url:
        return "用法：/bilibili <B站链接>  或  /b站 <链接>（支持 BV 链接和 b23.tv 短链）"
    url = _first_bili_url(url) or ""
    if not url:
        return "请提供有效的 B 站视频链接（bilibili.com/video/BVxxx 或 b23.tv 短链）。"

    remain = _check_rate_limit(ctx.session.chat_id)
    if remain > 0:
        return f"B站解析太频繁啦，{remain} 秒后再试。"

    logger.info(f"B站解析(命令): url={url[:80]}")
    await ctx.reply("开始解析 B 站视频，请稍候～")
    _spawn_bg(_process(ctx, url))
    return None


@register_interceptor(BILI_LINK_RE.pattern, name="bilibili_link", plugin="bilibili",
                      group_at_only=_GROUP_AT_ONLY)
async def bilibili_hit(ctx) -> bool:
    """自动识别消息中的 B 站链接并解析；消费消息不再进 LLM 决策。"""
    url = (ctx.args or "").rstrip(_TRAILING_PUNCT)
    if not url:
        return False

    remain = _check_rate_limit(ctx.session.chat_id)
    if remain > 0:
        await ctx.reply(f"B站解析太频繁啦，{remain} 秒后再试。")
        return True

    logger.info(f"B站解析(自动): url={url[:80]}")
    await ctx.reply("开始解析 B 站视频，请稍候～")
    _spawn_bg(_process(ctx, url))
    return True


TOOLS = []
