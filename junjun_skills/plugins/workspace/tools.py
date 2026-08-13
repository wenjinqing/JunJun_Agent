"""workspace 插件：TaskKernel Phase 2 通用工具域——沙箱跑代码 + 会话工作区 + 网页深读。

LLM 工具：
- run_code        在隔离沙箱里跑 Python（数据统计 / 文件处理 / 图表 / 文档产出）
- workspace_read  读当前会话工作区里的文件
- workspace_write 写文件到当前会话工作区
- workspace_list  列出工作区文件
- workspace_send  把工作区文件发到当前聊天（图片直发，其他传群文件/私聊文件）
- workspace_delete 删除工作区文件
- workspace_save_file 把聊天里刚收到的文件（表格/PDF/文本等）存进工作区
- fetch_page      深读指定网址正文（区别于 web_search 的关键词检索）

安全模型（真正的隔离在容器，工具侧只是门禁）：
- run_code 门禁：管理员（ADMIN_QQ 信任根）直跑；非管理员只有内核人审批准
  的步骤能放行（executor._kernel_step_approved 放行位）。manifest 不配
  admin_only——框架级 admin 门会把「已批准的非管理员步骤」也挡掉。
- 静态预检：ast 扫描禁 import os/sys/subprocess/socket/ctypes/importlib
  与 __import__——挡手滑不挡黑客，隔离靠容器（无网络/只读根fs/限额/一次性）。
- 工作区按会话隔离：data/workspace/<chat_id>/，路径穿越在解析层挡死。
- fetch_page SSRF 防护：解析域名后拒私网/回环/保留地址。
"""

import ast
import re
from pathlib import Path

import httpx
from langchain_core.tools import tool

from junjun_core.format import fmt_size
from junjun_core.http_client import make_async_client
from junjun_core.observability import get_logger

logger = get_logger("plugin.workspace")

_ROOT = Path("data/workspace")

_MAX_WRITE_CHARS = 256_000       # workspace_write 上限（约 256KB 文本）
_READ_DEFAULT = 6000             # workspace_read 默认返回字符数
_READ_MAX = 20_000
_LIST_MAX = 200
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
_SEND_MAX_BYTES = 50 * 1024 * 1024   # workspace_send 上限
_FETCH_MAX_CHARS = 8000          # fetch_page 返回截断（深读上下文预算）
_FETCH_MAX_BYTES = 10 * 1024 * 1024
_FETCH_SAVE_CHARS = 512_000      # save_as 落盘上限
_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36")


# ---------------------------------------------------------------- 工作区路径

def _safe_name(s: str) -> str:
    """chat_id 等任意串 -> 目录名片段（Windows 文件名禁冒号，必须清洗）。"""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", s)[:64] or "default"


def _session_dir(create: bool = False) -> Path:
    from junjun_skills.builtin.memory_skills import current_chat_id
    d = _ROOT / _safe_name(current_chat_id.get("") or "unknown")
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


_WIN_ILLEGAL = re.compile(r'[<>:"|?*\x00-\x1f]')
_WIN_RESERVED = ({"con", "prn", "aux", "nul"}
                 | {f"com{i}" for i in range(1, 10)}
                 | {f"lpt{i}" for i in range(1, 10)})


def _sanitize_segment(seg: str) -> str:
    """单段文件名消毒（2026-08-13 P2）：Windows 非法字符替换、保留设备名
    （CON/PRN/AUX/NUL/COM1-9/LPT1-9，不分大小写、看扩展名前的词干）加前缀、
    尾点/尾空格去掉（Windows 静默吞掉会致 round-trip 对不上）。
    注意：「..」不在此处理——消毒先行会把它洗成无害段名，等于放行穿越；
    穿越段在 _resolve 里消毒前显式拒绝。"""
    seg = _WIN_ILLEGAL.sub("_", seg).rstrip(" .")
    if seg.split(".", 1)[0].lower() in _WIN_RESERVED:
        seg = "_" + seg
    return seg or "_"


def _resolve(user_path: str) -> Path:
    """用户相对路径 -> 工作区内绝对路径；越界（../、绝对路径）一律 ValueError。"""
    p = (user_path or "").strip().replace("\\", "/")
    if not p or p.startswith(("/", "~")) or (len(p) > 1 and p[1] == ":"):
        raise ValueError("路径必须是工作区内的相对路径（如 report.md 或 sub/a.csv）")
    raw_parts = p.split("/")
    if ".." in raw_parts:
        raise ValueError("路径越出工作区，已拒绝")
    parts = [_sanitize_segment(seg) for seg in raw_parts if seg not in ("", ".")]
    if not parts:
        raise ValueError("文件名消毒后为空，换个名字")
    base = _session_dir(create=False).resolve()
    target = base.joinpath(*parts).resolve()
    if target != base and base not in target.parents:
        raise ValueError("路径越出工作区，已拒绝")
    return target


# ---------------------------------------------------------------- run_code 门禁与预检

# 挡手滑不挡黑客（真正的隔离是容器）：这些模块在沙箱里没有合法用途——
# 无网络所以 socket 无用，无子进程需求所以 subprocess 无用。
_BLOCKED_IMPORTS = frozenset({"os", "sys", "subprocess", "socket", "ctypes", "importlib"})


def _static_scan(code: str) -> str:
    """代码静态预检。返回问题描述（空串=通过）。ast 解析而非正则——
    正则挡不住 `import json, os` 这类逗号并列绕过。"""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"语法错误（第 {e.lineno} 行：{e.msg}）"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mods = [(node.module or "").split(".")[0]]
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == "__import__":
            mods = ["__import__"]
        else:
            continue
        bad = [m for m in mods if m in _BLOCKED_IMPORTS or m == "__import__"]
        if bad:
            return (f"禁用的模块/用法：{'、'.join(bad)}"
                    f"（沙箱里用不到它们；文件读写直接用 open/pathlib，限当前工作区）")
    return ""


def _run_code_permitted() -> bool:
    """run_code 门禁：管理员直跑；非管理员只有内核人审批准的步骤放行。"""
    from junjun_core.security import current_user_id, is_admin, is_admin_privileged
    if is_admin_privileged() or is_admin(current_user_id.get()):
        return True
    try:
        from junjun_agent.task_kernel.executor import kernel_step_approved
        return kernel_step_approved()
    except Exception:
        return False


def _sandbox_url() -> str:
    try:
        from junjun_core.config import get_global_config
        url = get_global_config().raw.get("sandbox", {}).get("base_url")
        if url:
            return str(url).rstrip("/")
    except Exception:
        pass
    return "http://127.0.0.1:8100"


@tool
async def run_code(code: str, timeout: int = 30) -> str:
    """在隔离沙箱里运行一段 Python 代码，返回输出和产生的文件列表。何时使用：需要做数据
    计算/统计、处理工作区里的文件（csv/excel/文本）、用代码画数据图表存图、做 Word/PDF
    文档、词云、二维码、图片处理时。沙箱预装 pandas/numpy/openpyxl/matplotlib/seaborn/
    pillow/python-docx/pdfplumber/reportlab/jieba/wordcloud/qrcode/requests，无网络，
    单文件读写限定 /workspace（即当前会话工作区），最长跑 30 秒。图表中文已自动适配
    （matplotlib 默认 Noto Sans CJK SC）；wordcloud 要传 font_path=
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc" 才不出豆腐块。
    区别于 ai_draw（AI 画插画）：run_code 画的是数据图表。
    非管理员使用需管理员事先批准（任务通道的人审步）。"""
    if not _run_code_permitted():
        return ("跑代码这事我得管理员点头才行——让管理员跟我说一声，"
                "或者把需求当成任务交给我（会走审批）。")
    bad = _static_scan(code)
    if bad:
        return f"代码没通过预检：{bad}。改掉再交给我。"
    timeout = max(1, min(int(timeout), 30))
    from junjun_skills.builtin.memory_skills import current_chat_id
    workdir = _safe_name(current_chat_id.get("") or "unknown")
    import os
    headers = {}
    if os.environ.get("SANDBOX_TOKEN"):
        headers["X-Sandbox-Token"] = os.environ["SANDBOX_TOKEN"]
    try:
        async with make_async_client(timeout=timeout + 20) as client:  # 沙箱是本机服务，绝不走系统代理
            resp = await client.post(f"{_sandbox_url()}/run",
                                     json={"code": code, "timeout": timeout,
                                           "workdir": workdir},
                                     headers=headers)
    except httpx.HTTPError as e:
        err = RuntimeError(f"沙箱服务不可达: {type(e).__name__}: {e}")
        err.tool_suggestion = ("沙箱服务没启动，重试无意义；向用户说明跑代码的功能暂时不可用，"
                               "管理员启动 sandbox 服务后才能用")
        raise err
    if resp.status_code != 200:
        raise RuntimeError(f"沙箱服务 HTTP {resp.status_code}: {resp.text[:120]}")
    data = resp.json()
    parts = []
    if data.get("killed"):
        parts.append(f"执行被强制终止（超过 {timeout}s 上限）。")
    else:
        parts.append(f"执行完成（退出码 {data.get('returncode', '?')}，"
                     f"耗时 {data.get('duration_ms', 0) / 1000:.1f}s）。")
    out = (data.get("stdout") or "").strip()
    if out:
        parts.append("输出：\n" + out[:3000])
    err_text = (data.get("stderr") or "").strip()
    if err_text:
        parts.append("错误输出：\n" + err_text[:1500])
    files = data.get("files") or []
    if files:
        parts.append("产生的文件（已存到工作区）："
                     + "、".join(f"{f['path']}（{fmt_size(int(f.get('size', 0)))}）"
                                for f in files[:20]))
    if data.get("returncode") not in (0, None) and not out and not err_text:
        parts.append("（无输出）")
    return "\n".join(parts)


# ---------------------------------------------------------------- 工作区三工具

@tool
async def workspace_read(path: str, max_chars: int = _READ_DEFAULT) -> str:
    """读取当前会话工作区里的文件内容（文本）。何时使用：要查看之前存下的文件、或任务里
    提到「工作区里的 xx 文件」时。不知道有什么文件就先调 workspace_list。太长会截断。
    区别于 query_chat_history（翻聊天记录）：这里读的是文件不是聊天。"""
    target = _resolve(path)
    if not target.is_file():
        raise ValueError(f"工作区里没有「{path}」这个文件；先调 workspace_list 看看有什么")
    text = target.read_text(encoding="utf-8", errors="replace")
    cap = max(200, min(int(max_chars), _READ_MAX))
    if len(text) > cap:
        return text[:cap] + f"\n……（已截断，全文 {len(text)} 字，max_chars 调大能看更多）"
    return text


@tool
async def workspace_write(path: str, content: str) -> str:
    """把文本内容写进当前会话工作区（已存在则覆盖，子目录自动创建）。何时使用：产出需要
    落盘备查（报告、整理后的表格、汇总文档）或给 run_code 准备输入数据时。
    只写文本；画图/二进制文件用 run_code 在沙箱里生成。"""
    if len(content) > _MAX_WRITE_CHARS:
        raise ValueError(f"内容太长（{len(content)} 字，上限 {_MAX_WRITE_CHARS}），"
                         f"请分段写或精简")
    target = _resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    logger.info(f"工作区写入: {target} ({len(content)} 字)")
    return f"已存到工作区：{path}（{len(content)} 字）。"


@tool
async def workspace_list(subdir: str = "") -> str:
    """列出当前会话工作区里的文件（含大小）。何时使用：不确定工作区里有什么文件、
    或读/写之前想确认文件名时。subdir 可看子目录。"""
    base = _resolve(subdir) if subdir.strip() else _session_dir(create=False).resolve()
    root = _session_dir(create=False).resolve()
    if not base.exists():
        return "工作区还是空的。"
    entries = []
    for p in sorted(base.rglob("*")):
        rel = p.relative_to(root).as_posix()
        if p.is_dir():
            entries.append(f"{rel}/")
        else:
            entries.append(f"{rel}（{fmt_size(p.stat().st_size)}）")
        if len(entries) >= _LIST_MAX:
            entries.append(f"……（超过 {_LIST_MAX} 条，截断）")
            break
    return "工作区文件：\n" + "\n".join(entries) if entries else "工作区还是空的。"


@tool
async def workspace_send(path: str) -> str:
    """把工作区里的文件发到当前聊天。何时使用：run_code 产出了图表/文档、或对方想要
    工作区里已存的某个文件时。图片（png/jpg/gif/webp/bmp）以图片消息直接发出；
    其他类型（docx/pdf/csv/txt/zip…）上传为群文件/私聊文件。区别于 send_message
    （发文字消息）和 send_emoji（发表情包库里的贴纸）。"""
    target = _resolve(path)
    if not target.is_file():
        raise ValueError(f"工作区里没有「{path}」这个文件；先调 workspace_list 看看")
    size = target.stat().st_size
    if size > _SEND_MAX_BYTES:
        raise ValueError(f"文件太大（{fmt_size(size)}，上限 50MB），发不出去")
    kind, cid = _chat_target()
    if not kind:
        raise RuntimeError("拿不到当前会话，发送失败")
    if target.suffix.lower() in _IMAGE_EXTS:
        from junjun_agent.outbound import send_proactive
        from junjun_core.contracts import ReplySegment
        from junjun_skills.builtin.memory_skills import current_chat_id
        ok = await send_proactive(current_chat_id.get(""),
                                  [ReplySegment(type="image", data=str(target))],
                                  source="workspace", remember=False)
        if not ok:
            raise RuntimeError("图片发送失败（网关未送达）")
        return f"图片 {path} 已发到当前聊天。"
    from junjun_core import napcat_client
    if kind == "group":
        ok = await napcat_client.upload_group_file(cid, str(target), name=target.name)
        if not ok:
            raise RuntimeError("群文件上传失败（NapCat 未确认）")
        return f"文件 {path}（{fmt_size(size)}）已上传到群文件。"
    ok = await napcat_client.upload_private_file(cid, str(target), name=target.name)
    if not ok:
        raise RuntimeError("私聊文件发送失败（NapCat 未确认）")
    return f"文件 {path}（{fmt_size(size)}）已发给对方。"


@tool
async def workspace_delete(path: str) -> str:
    """删除工作区里的某个文件（只删文件，目录删不掉）。何时使用：对方明确要求删掉
    某个已存文件、或清理过期产物腾地方时。"""
    target = _resolve(path)
    if not target.exists():
        raise ValueError(f"工作区里没有「{path}」；先调 workspace_list 看看")
    if not target.is_file():
        raise ValueError("只能删文件；目录请先把里面的文件删完")
    target.unlink()
    logger.info(f"工作区删除: {target}")
    return f"已删除工作区文件：{path}。"


@tool
async def workspace_save_file(save_as: str = "") -> str:
    """把当前聊天里最近收到的文件（别人刚发来的表格/PDF/文本/压缩包等）下载存进
    工作区。何时使用：对方发来文件让你处理（「看看这个表格」「把这个 PDF 总结一下」）、
    或任务需要用到刚收到的文件时——先存进来，再用 run_code 处理或 workspace_send 发回
    结果。save_as 可改名（可带子目录），留空用原文件名。只认最近 10 分钟内收到的文件，
    超过 50MB 存不下。区别于 workspace_write（写文本内容）和 fetch_page（抓网页）。"""
    from junjun_skills.builtin.memory_skills import current_chat_id
    from junjun_memory.recent_files import recent_file
    ref = recent_file(current_chat_id.get("") or "")
    if not ref:
        return ("最近 10 分钟这个聊天里没收到过文件。让对方先把文件发出来，"
                "发完跟我说一声我就能存。")
    url = str(ref.get("url") or "")
    if not url.startswith(("http://", "https://")):
        raise RuntimeError("文件下载地址不合法（只认 http/https）")
    name = (save_as or "").strip() or str(ref.get("name") or "未命名文件")
    target = _resolve(name)
    try:
        data = await _download_capped(url, _SEND_MAX_BYTES)
    except OverflowError:
        raise RuntimeError(f"文件超过 50MB 上限，存不下（声明大小 "
                           f"{fmt_size(int(ref.get('size') or 0))}）")
    except httpx.HTTPError as e:
        raise RuntimeError(f"文件下载失败: {type(e).__name__}（链接可能已过期，"
                           f"让对方重发一次）") from e
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    logger.info(f"工作区存文件: {target} ({len(data)}B, 原名 {ref.get('name')})")
    return (f"已把「{ref.get('name')}」存到工作区：{name}"
            f"（{fmt_size(len(data))}）。接下来可以用 run_code 处理它。")


async def _download_capped(url: str, cap: int) -> bytes:
    """流式下载，超 cap 立即中止抛 OverflowError（不读完，防拖死连接）。
    手工跟随重定向 + 每跳 SSRF 检查：URL 信任链是 NapCat 解析的 QQ 文件
    服务器，但中间跳转可能打回环/内网（2026-08-13 审查 P2-7）。"""
    from urllib.parse import urljoin
    chunks, total = [], 0
    current = url
    async with make_async_client(timeout=60.0) as client:  # QQ 文件服务器：国内直连，不走系统代理
        for _ in range(5):
            bad = await _ssrf_check_async(current)
            if bad:
                raise RuntimeError(f"文件地址不可访问：{bad}")
            async with client.stream("GET", current) as resp:
                if resp.status_code in (301, 302, 303, 307, 308):
                    loc = resp.headers.get("location") or ""
                    if not loc:
                        raise RuntimeError(f"HTTP {resp.status_code} 但没有跳转地址")
                    current = urljoin(current, loc)
                    continue
                resp.raise_for_status()
                length = int(resp.headers.get("content-length") or 0)
                if length > cap:
                    raise OverflowError(url)
                async for chunk in resp.aiter_bytes(65536):
                    total += len(chunk)
                    if total > cap:
                        raise OverflowError(url)
                    chunks.append(chunk)
                return b"".join(chunks)
    raise RuntimeError("重定向超过 5 次，放弃")


def _chat_target() -> tuple:
    """当前会话 -> (kind, id)：kind=group/private；解析失败返回 (\"\", \"\")。"""
    from junjun_skills.builtin.memory_skills import current_chat_id
    parts = (current_chat_id.get("") or "").split(":")
    if len(parts) == 3 and parts[2] in ("group", "private") and parts[1]:
        return parts[2], parts[1]
    return "", ""


# ---------------------------------------------------------------- fetch_page 网页深读

def _ssrf_check(url: str) -> str:
    """拒私网/回环/保留地址。返回问题描述（空串=通过）。
    务实级防护：string + gethostbyname 一次解析（不防 DNS rebinding，
    生产部署在以 bot 身份运行的内网时这是必要的最低线）。"""
    from urllib.parse import urlparse
    p = urlparse((url or "").strip())
    if p.scheme not in ("http", "https") or not p.hostname:
        return "不是合法的 http(s) 网址"
    host = p.hostname
    if host.lower() in ("localhost",) or host.lower().endswith((".local", ".internal")):
        return "这个地址不能访问（本机/内网地址）"
    import ipaddress
    import socket
    try:
        ip = ipaddress.ip_address(socket.gethostbyname(host))
    except (socket.gaierror, ValueError):
        return "域名解析失败，拿不到地址"
    if ip.is_private or ip.is_loopback or ip.is_link_local \
            or ip.is_reserved or ip.is_multicast or ip.is_unspecified:
        return "这个地址不能访问（内网/保留地址）"
    return ""


@tool
async def fetch_page(url: str, save_as: str = "") -> str:
    """抓取指定网址的正文转成纯文本（深读长文）。何时使用：对方发来具体链接想看里面写了
    什么、或搜索结果里某条需要细读全文时。区别于 web_search（关键词检索返回结果列表）和
    watch_video（看视频内容）。正文过长会截断，save_as 给个文件名可把全文存进工作区。
    需要登录/强 JS 渲染的页面可能抓不到正文。"""
    bad = await _ssrf_check_async(url)
    if bad:
        return f"这个网址抓不了：{bad}。"
    try:
        raw, ctype = await _fetch_bytes(url)
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"页面返回 HTTP {e.response.status_code}") from e
    if "html" in ctype:
        title, text = _html_to_text(raw)
    elif ctype.startswith("text/") or "json" in ctype:
        title, text = "", raw.decode("utf-8", errors="replace")
    else:
        return f"这个链接是 {ctype or '未知'} 类型，不是网页/文本，我读不了正文。"
    text = text.strip()
    if not text:
        return "页面抓下来了但没有可读正文（可能是强 JS 渲染或纯图片页面）。"
    saved = ""
    if save_as.strip():
        target = _resolve(save_as.strip())
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text[:_FETCH_SAVE_CHARS], encoding="utf-8")
        saved = f"\n（全文已存到工作区：{save_as.strip()}，{len(text)} 字）"
    head = f"【{title}】\n" if title else ""
    if len(text) > _FETCH_MAX_CHARS:
        return (head + text[:_FETCH_MAX_CHARS]
                + f"\n……（已截断，全文 {len(text)} 字" +
                ("，完整内容在工作区）" if saved else "，要全文可用 save_as 存工作区）")
                + saved)
    return head + text + saved


async def _ssrf_check_async(url: str) -> str:
    import asyncio
    return await asyncio.to_thread(_ssrf_check, url)


async def _fetch_bytes(url: str) -> tuple:
    """流式下载，10MB 封顶。返回 (bytes, content_type)。
    手工跟随重定向：每一跳重跑 SSRF 检查（2026-08-13 审查 P1 实锤——
    follow_redirects=True 时公网 URL 302 到 127.0.0.1/169.254.169.254
    就绕过了只对原始 URL 做的检查）。"""
    from urllib.parse import urljoin
    chunks, total = [], 0
    current = url
    # 公网任意 URL（fetch_page）：显式 trust_env=True 穿系统代理——用户分享的
    # 链接可能是 GFW 封锁站点；本机 Clash 规则代理国内直连不亏。
    async with httpx.AsyncClient(timeout=20.0, headers={"User-Agent": _UA},
                                 trust_env=True) as client:
        for _ in range(5):
            bad = await _ssrf_check_async(current)
            if bad:
                raise RuntimeError(f"重定向目标不可访问：{bad}")
            async with client.stream("GET", current) as resp:
                if resp.status_code in (301, 302, 303, 307, 308):
                    loc = resp.headers.get("location") or ""
                    if not loc:
                        raise RuntimeError(f"HTTP {resp.status_code} 但没有跳转地址")
                    current = urljoin(current, loc)
                    continue
                resp.raise_for_status()
                ctype = resp.headers.get("content-type", "").lower()
                async for chunk in resp.aiter_bytes(65536):
                    chunks.append(chunk)
                    total += len(chunk)
                    if total > _FETCH_MAX_BYTES:
                        break
                return b"".join(chunks), ctype
    raise RuntimeError("重定向超过 5 次，放弃")


def _html_to_text(raw: bytes) -> tuple:
    """HTML -> (标题, 正文文本)。bs4 轻量抽取：剥脚本样式导航页脚，按行压实。"""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(raw, "html.parser")
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    for tag in soup(["script", "style", "noscript", "iframe",
                     "header", "footer", "nav", "aside", "form"]):
        tag.decompose()
    lines = [ln.strip() for ln in soup.get_text("\n").splitlines() if ln.strip()]
    return title, "\n".join(lines)


TOOLS = [run_code, workspace_read, workspace_write, workspace_list, workspace_send,
         workspace_delete, workspace_save_file, fetch_page]
