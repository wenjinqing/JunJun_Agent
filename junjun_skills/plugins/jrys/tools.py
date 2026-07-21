"""jrys 插件：今日运势（迁移自 MaiBot jrys_prpr_maimbot，新架构重写）。

命令（raw 关键词）：
- 今日运势          简版：总签文字 + 运势卡片图
- 今日运势详/详细   详版：附加分项星级（综合/桃花/工作/财运/学业）
- 今日桃花/今日工作/今日财运/今日学业  单项运势卡

核心玩法：
- jrrp(user_id, date_str)：hashlib 确定性伪随机 0~100 人品值，同日同人固定
- 每日签：data/jrys/{user_id}_{date}.json 记录当天总签与签文，同日重抽复用（不重复调 LLM）
- 总签文案由 LLM 生成（签诗 VERSE + 白话 LINE），失败降级本地 fortune_quotes
- 详版分项星级由 LLM 摇出，失败降级确定性伪随机星级（1~5）
- Pillow 渲染卡片 PNG：render_card(record, out_path)
"""

import hashlib
import json
import random
import re
from datetime import date
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from junjun_agent.commands import register_command
from junjun_core.contracts import ReplySegment
from junjun_core.observability import get_logger

logger = get_logger("plugin.jrys")

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "jrys"

_FONT_PATH = "C:/Windows/Fonts/msyh.ttc"
_DISCLAIMER = "仅供娱乐 | 相信科学 | 请勿迷信"

# 本地签文库（内嵌自旧插件 fortune_quotes.json，LLM 失败时降级用）
FORTUNE_QUOTES = [
    {"title": "大凶", "line": "今天宜静不宜动，把力气留给吃饭睡觉就好。", "stars": 0},
    {"title": "凶", "line": "今天宜休息、忌硬刚；把难题留明天也不错。", "stars": 1},
    {"title": "小凶", "line": "说话前多停一秒，能省很多解释成本。", "stars": 2},
    {"title": "末吉", "line": "别急，慢一点反而少踩坑。", "stars": 2},
    {"title": "微末", "line": "小磕绊难免，当作提醒就好，不必放大。", "stars": 2},
    {"title": "半吉", "line": "偶有插曲，笑一下当彩蛋就过去了。", "stars": 3},
    {"title": "吉", "line": "小确幸在线，喝杯喜欢的饮料会更开心。", "stars": 3},
    {"title": "小吉", "line": "平顺的一天，适合整理桌面和心情。", "stars": 4},
    {"title": "中吉", "line": "稳步推进就好，别贪快，细节里藏着小惊喜。", "stars": 4},
    {"title": "大吉", "line": "今天适合把想做的事开个头，行动力会回报你。", "stars": 5},
    {"title": "上吉", "line": "势头不错，重要的事可以往前排一排。", "stars": 5},
    {"title": "特吉", "line": "难得的好彩头，认真接住，别浪费运气。", "stars": 5},
]

# 总签档位：人品值 0~100 线性映射（title, 总运星级）
_LOT_TIERS = tuple((e["title"], e["stars"]) for e in FORTUNE_QUOTES)

# 详版分项（综合 + 四个单项）
_SUB_DIMS = ("综合", "桃花", "工作", "财运", "学业")

# 分项命令白名单：关键词 -> 维度名
_SINGLE_COMMANDS = {
    "今日桃花": "桃花",
    "今日工作": "工作",
    "今日财运": "财运",
    "今日学业": "学业",
}


# ---------------------------------------------------------------- jrrp 与抽签

def jrrp(user_id: str, date_str: str) -> int:
    """每日人品值 0~100：user_id + 日期 确定性伪随机（核心玩法：同日同人结果固定）。"""
    d = hashlib.sha256(f"jrrp|v1|{date_str}|{user_id}".encode("utf-8")).digest()
    return int.from_bytes(d[:4], "big") % 101


def _det_int(seed: str, lo: int, hi: int) -> int:
    """确定性伪随机整数 [lo, hi]（降级分项星级用）。"""
    d = hashlib.sha256(seed.encode("utf-8")).digest()
    return lo + int.from_bytes(d[:4], "big") % (hi - lo + 1)


def _tier_for(rp: int) -> tuple:
    """人品值线性映射到总签档位。"""
    idx = min(len(_LOT_TIERS) - 1, max(0, rp) * len(_LOT_TIERS) // 101)
    return _LOT_TIERS[idx]


def _star_bar(n: int) -> str:
    n = max(0, min(5, int(n)))
    return "★" * n + "☆" * (5 - n)


def _local_fallback_line(title: str, user_id: str, date_str: str) -> str:
    """本地签文降级：优先同签档条目，按 user+date 确定性挑选。"""
    subs = [e for e in FORTUNE_QUOTES if e["title"] == title] or FORTUNE_QUOTES
    rng = random.Random(f"jrys-quote|{date_str}|{user_id}")
    return rng.choice(subs)["line"]


# ---------------------------------------------------------------- LLM 调用与解析

async def _ask_llm(prompt: str) -> str | None:
    """调用 utils 任务槽模型；任何失败返回 None（由调用方降级）。"""
    try:
        from langchain_core.messages import HumanMessage

        from junjun_llm import get_chat_model
        model = get_chat_model("utils")
        resp = await model.ainvoke([HumanMessage(content=prompt)])
        content = resp.content
        if isinstance(content, list):  # 兼容多段 content
            content = "".join(str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in content)
        return (content or "").strip() or None
    except Exception as e:
        logger.warning(f"jrys LLM 调用失败（将降级本地签文）: {type(e).__name__}: {e}")
        return None


def _parse_verse_line(raw: str) -> tuple:
    """解析签诗 VERSE + 白话 LINE；LINE 缺失视为不可用。"""
    verse, line = None, None
    for ln in (raw or "").splitlines():
        s = ln.strip()
        m = re.match(r"(?i)^VERSE\s*[:：]\s*(.+)$", s)
        if m:
            verse = m.group(1).strip()[:60]
        m2 = re.match(r"(?i)^LINE\s*[:：]\s*(.+)$", s)
        if m2:
            line = m2.group(1).strip()[:200]
    if not line:
        body = [ln.strip() for ln in (raw or "").splitlines()
                if ln.strip() and not re.match(r"(?i)^VERSE\s*[:：]", ln.strip())]
        joined = "\n".join(body).strip()
        if len(joined) >= 12:
            line = joined[:200]
    return verse, line


async def _generate_fortune_text(title: str, stars: int, nickname: str,
                                 user_id: str, date_str: str) -> tuple:
    """LLM 生成总签签诗+白话；失败降级本地签文库。返回 (verse, line)。"""
    prompt = (
        "你是解签助手，为「今日运势」娱乐签写签文（仅供娱乐）。\n"
        f"今日总签：{title}  总运：{stars}/5（须按此吉凶轻重来写，勿改判签档）\n"
        f"用户昵称：{nickname}  日期：{date_str}\n"
        "要求：\n"
        "1) VERSE：签诗式短句，16~24 个汉字，文言、对仗或顿号分节，不要昵称、不要 markdown。\n"
        "2) LINE：白话解签，50~100 个汉字，温暖或略带俏皮，自然带上用户昵称一次。\n"
        "严格只输出两行（键名大写英文，冒号后一个空格）：\n"
        "VERSE: （签诗）\n"
        "LINE: （白话解签）"
    )
    raw = await _ask_llm(prompt)
    if raw:
        verse, line = _parse_verse_line(raw)
        if line:
            return verse or "", line
        logger.warning(f"jrys 解签无法解析，降级本地签文。原始片段: {raw[:200]!r}")
    return "", _local_fallback_line(title, user_id, date_str)


async def _roll_sub_stars(title: str, stars: int, user_id: str, date_str: str) -> dict:
    """详版分项星级：LLM 摇 5 项（1~5），失败/缺项降级确定性伪随机。"""
    result = {d: _det_int(f"jrys-sub|{date_str}|{user_id}|{d}", 1, 5) for d in _SUB_DIMS}
    dims = "\n".join(f"{d}:" for d in _SUB_DIMS)
    prompt = (
        "你是运势签辅助，只输出分项星级（每项 1~5 的整数），与总签略协调，勿写正文。\n"
        f"总签：{title}  总运：{stars}/5  日期：{date_str}\n"
        "请输出且仅输出下面 5 行，行名完全一致，冒号后只跟一个 1~5 的整数，不要解释、不要 markdown：\n"
        f"{dims}"
    )
    raw = await _ask_llm(prompt)
    if raw:
        for ln in raw.splitlines():
            s = ln.strip()
            for d in _SUB_DIMS:
                m = re.match(rf"^{re.escape(d)}\s*[:：]\s*(\d)\s*$", s)
                if m:
                    result[d] = max(1, min(5, int(m.group(1))))
                    break
    return result


async def _generate_single_dim(dim: str, title: str, stars: int, nickname: str,
                               user_id: str, date_str: str) -> tuple:
    """单项运势：LLM 给该项星级+短解签；失败降级确定性星级+本地模板。返回 (star, line)。"""
    prompt = (
        f"你是分项运势助手。只针对「{dim}」这一项给出星级与短解签（仅供娱乐）。\n"
        f"用户今日总签（语气参考）：「{title}」，总运 {stars}/5。用户昵称：{nickname}  日期：{date_str}\n"
        "要求：\n"
        f"1) STAR：只表示「{dim}」这一项，1~5 的整数。\n"
        f"2) LINE：只写「{dim}」白话，30~90 个汉字，自然带上昵称一次，禁止 markdown，禁止列举其他分项。\n"
        "严格只输出两行：\n"
        "STAR: （1~5 的整数）\n"
        "LINE: （白话解签）"
    )
    raw = await _ask_llm(prompt)
    if raw:
        star, line = None, None
        for ln in raw.splitlines():
            s = ln.strip()
            m = re.match(r"(?i)^STAR\s*[:：]\s*(\d)\s*$", s)
            if m:
                star = max(1, min(5, int(m.group(1))))
            m2 = re.match(r"(?i)^LINE\s*[:：]\s*(.+)$", s)
            if m2:
                line = m2.group(1).strip()[:200]
        if line:
            return (star or 3), line
        logger.warning(f"jrys 单项「{dim}」解签无法解析，降级。原始片段: {raw[:200]!r}")
    star = _det_int(f"jrys-single|{date_str}|{user_id}|{dim}", 1, 5)
    line = (f"{nickname}，今日「{dim}」先给个 {star} 星档～模型开小差了，"
            "这句是本地兜底，娱乐向别当真。")
    return star, line


# ---------------------------------------------------------------- 每日签记录

def _safe_id(user_id: str) -> str:
    """用户标识转文件名安全串。"""
    return re.sub(r"[^0-9A-Za-z_\-]", "_", str(user_id or "anonymous"))


def _record_path(user_id: str, date_str: str) -> Path:
    return DATA_DIR / f"{_safe_id(user_id)}_{date_str}.json"


def _load_record(user_id: str, date_str: str) -> dict | None:
    p = _record_path(user_id, date_str)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _save_record(record: dict) -> None:
    p = _record_path(record["user_id"], record["date"])
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")


async def _get_or_create_record(user_id: str, nickname: str) -> dict:
    """取当天签；没有则本地抽签 + LLM 写签文后落盘（同日重抽复用，不重复调 LLM）。"""
    date_str = date.today().isoformat()
    rec = _load_record(user_id, date_str)
    if rec:
        return rec
    rp = jrrp(user_id, date_str)
    title, stars = _tier_for(rp)
    verse, line = await _generate_fortune_text(title, stars, nickname, user_id, date_str)
    rec = {
        "user_id": str(user_id), "date": date_str, "nickname": nickname,
        "jrrp": rp, "title": title, "stars": stars,
        "verse": verse, "line": line, "sub_stars": None,
    }
    try:
        _save_record(rec)
    except Exception as e:
        logger.warning(f"jrys 签文记录写入失败（不影响本次结果）: {e}")
    return rec


# ---------------------------------------------------------------- 卡片渲染

def _load_font(size: int):
    """加载微软雅黑；取不到用 Pillow 默认字体降级（不炸）。"""
    try:
        return ImageFont.truetype(_FONT_PATH, size)
    except OSError:
        return ImageFont.load_default()


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font, max_w: int) -> list:
    """按像素宽度逐字换行（适配中文）。"""
    lines, buf = [], ""
    for ch in (text or ""):
        if ch == "\n":
            lines.append(buf)
            buf = ""
            continue
        if draw.textlength(buf + ch, font=font) <= max_w:
            buf += ch
        else:
            lines.append(buf)
            buf = ch
    if buf:
        lines.append(buf)
    return lines or [""]


def _palette(stars: int) -> dict:
    """按总运星级选配色：>=4 暖金，3 青蓝，<=2 灰紫。"""
    if stars >= 4:
        return {"top": (255, 248, 232), "bottom": (242, 176, 120), "accent": (188, 72, 38),
                "title": (118, 52, 28), "body": (62, 40, 32), "muted": (155, 108, 78)}
    if stars == 3:
        return {"top": (228, 246, 255), "bottom": (138, 198, 242), "accent": (42, 118, 148),
                "title": (28, 92, 108), "body": (36, 62, 82), "muted": (88, 125, 142)}
    return {"top": (248, 238, 255), "bottom": (198, 172, 235), "accent": (108, 62, 155),
            "title": (78, 38, 92), "body": (58, 40, 92), "muted": (125, 98, 148)}


def render_card(record: dict, out_path) -> Path:
    """渲染运势卡 PNG（宽 600：日期/昵称、签档大字、星级、人品进度条、签诗、白话、可选分项）。"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    W, MARGIN = 600, 32
    stars = max(0, min(5, int(record.get("stars", 3))))
    pal = _palette(stars)
    title = str(record.get("title") or "吉")
    verse = str(record.get("verse") or "").strip()
    line = str(record.get("line") or "").strip()
    rp = int(record.get("jrrp", 0))
    sub_stars = record.get("sub_stars") if isinstance(record.get("sub_stars"), dict) else None

    f_head = _load_font(30)
    f_small = _load_font(16)
    f_title = _load_font(52)
    f_star = _load_font(26)
    f_body = _load_font(20)
    f_verse = _load_font(22)

    scratch = Image.new("RGB", (W, 100))
    sd = ImageDraw.Draw(scratch)
    max_w = W - MARGIN * 2
    verse_lines = _wrap_text(sd, verse, f_verse, max_w) if verse else []
    body_lines = _wrap_text(sd, line, f_body, max_w)

    # 布局量高：头部 -> 签档 -> 星级 -> 人品条 -> 签诗 -> 白话 -> 分项 -> 页脚
    y = MARGIN
    y += 40 + 26          # 标题 + 日期/昵称
    y += 70               # 签档大字
    y += 40               # 星级
    y += 46               # 人品进度条
    y += len(verse_lines) * 32 + (12 if verse_lines else 0)
    y += len(body_lines) * 30 + 16
    if sub_stars:
        y += len(_SUB_DIMS) * 30 + 16
    y += 40               # 页脚
    H = y + MARGIN

    img = Image.new("RGB", (W, H))
    draw = ImageDraw.Draw(img)
    for yy in range(H):  # 纵向渐变背景
        t = yy / max(H - 1, 1)
        c = tuple(int(pal["top"][i] + (pal["bottom"][i] - pal["top"][i]) * t) for i in range(3))
        draw.line([(0, yy), (W - 1, yy)], fill=c)

    y = MARGIN
    draw.text((MARGIN, y), str(record.get("header") or "今日运势"), fill=pal["title"], font=f_head)
    y += 44
    draw.text((MARGIN, y), f"{record.get('date', '')} ｜ {record.get('nickname') or '你'}",
              fill=pal["muted"], font=f_small)
    y += 30
    draw.text((MARGIN + 2, y + 2), title, fill=(255, 255, 255), font=f_title)  # 简单阴影
    draw.text((MARGIN, y), title, fill=pal["title"], font=f_title)
    y += 70
    draw.text((MARGIN, y), _star_bar(stars), fill=pal["accent"], font=f_star)
    y += 42

    # 人品值进度条
    bar_w, bar_h = max_w, 18
    draw.rounded_rectangle((MARGIN, y, MARGIN + bar_w, y + bar_h), radius=bar_h // 2,
                           fill=(255, 255, 255))
    fill_w = max(bar_h, int(bar_w * max(0, min(100, rp)) / 100))
    draw.rounded_rectangle((MARGIN, y, MARGIN + fill_w, y + bar_h), radius=bar_h // 2,
                           fill=pal["accent"])
    draw.text((MARGIN, y + bar_h + 4), f"今日人品 {rp}/100", fill=pal["muted"], font=f_small)
    y += bar_h + 28

    for vl in verse_lines:
        draw.text((MARGIN, y), vl, fill=pal["title"], font=f_verse)
        y += 32
    if verse_lines:
        y += 12
    for bl in body_lines:
        draw.text((MARGIN, y), bl, fill=pal["body"], font=f_body)
        y += 30
    y += 16

    if sub_stars:
        for d in _SUB_DIMS:
            n = max(1, min(5, int(sub_stars.get(d, 3))))
            draw.text((MARGIN, y), f"{d}  {_star_bar(n)}", fill=pal["body"], font=f_body)
            y += 30
        y += 16

    draw.text((MARGIN, y), _DISCLAIMER, fill=pal["muted"], font=f_small)

    img.save(out_path, format="PNG")
    return out_path


def _card_path(user_id: str, date_str: str, suffix: str = "") -> Path:
    d = DATA_DIR / "cards"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"card_{_safe_id(user_id)}_{date_str}{suffix}.png"


# ---------------------------------------------------------------- 命令入口

def _summary_text(rec: dict) -> str:
    verse = f"{rec['verse']} " if rec.get("verse") else ""
    return f"【{rec['title']}】{_star_bar(rec['stars'])} 人品 {rec['jrrp']}/100\n{verse}{rec['line']}"


async def _send_card(ctx, rec: dict, card: Path) -> None:
    await ctx.send([
        ReplySegment(type="text", data=_summary_text(rec)),
        ReplySegment(type="image", data=str(card)),
    ])


@register_command("今日运势", raw=True, plugin="jrys",
                  description="今日运势：总签 + 运势卡片图")
async def jrys_today_cmd(ctx):
    rec = await _get_or_create_record(ctx.meta.user_id, ctx.meta.nickname or "你")
    card = render_card(rec, _card_path(rec["user_id"], rec["date"]))
    await _send_card(ctx, rec, card)
    return None


@register_command("今日运势详", aliases=["今日运势详细"], raw=True, plugin="jrys",
                  description="今日运势详版：附加分项星级（综合/桃花/工作/财运/学业）")
async def jrys_detail_cmd(ctx):
    rec = await _get_or_create_record(ctx.meta.user_id, ctx.meta.nickname or "你")
    if not rec.get("sub_stars"):
        rec["sub_stars"] = await _roll_sub_stars(
            rec["title"], int(rec["stars"]), rec["user_id"], rec["date"])
        try:
            _save_record(rec)
        except Exception as e:
            logger.warning(f"jrys 分项星级写入失败（不影响本次结果）: {e}")
    card = render_card(rec, _card_path(rec["user_id"], rec["date"], "_detail"))
    await _send_card(ctx, rec, card)
    return None


async def _run_single_dim(ctx, dim: str):
    """分项命令公共逻辑：单项星级 + 解签 + 卡片图。"""
    user_id = str(ctx.meta.user_id)
    nickname = ctx.meta.nickname or "你"
    base = await _get_or_create_record(user_id, nickname)
    star, line = await _generate_single_dim(
        dim, base["title"], int(base["stars"]), nickname, user_id, base["date"])
    rec = {
        "user_id": user_id, "date": base["date"], "nickname": nickname,
        "jrrp": base["jrrp"], "title": dim, "stars": star,
        "verse": "", "line": line, "sub_stars": None,
        "header": f"今日{dim}",
    }
    card = render_card(rec, _card_path(user_id, base["date"], f"_{dim}"))
    await _send_card(ctx, rec, card)
    return None


@register_command("今日桃花", raw=True, plugin="jrys", description="今日桃花运（单项运势卡）")
async def jrys_taohua_cmd(ctx):
    return await _run_single_dim(ctx, "桃花")


@register_command("今日工作", raw=True, plugin="jrys", description="今日工作运（单项运势卡）")
async def jrys_work_cmd(ctx):
    return await _run_single_dim(ctx, "工作")


@register_command("今日财运", raw=True, plugin="jrys", description="今日财运（单项运势卡）")
async def jrys_fortune_cmd(ctx):
    return await _run_single_dim(ctx, "财运")


@register_command("今日学业", raw=True, plugin="jrys", description="今日学业运（单项运势卡）")
async def jrys_study_cmd(ctx):
    return await _run_single_dim(ctx, "学业")


TOOLS = []
