"""topic_finder 插件：定时/静默触发，自动生成群聊开场话题（迁移自旧 MaiBot topic_finder_plugin，新架构重写）。

触发方式（scheduler 每 60s 调 topic_finder_tick，内部自行判断到点）：
- 定时：每天 daily_times 命中当前分钟且当日该时刻未发过 -> 发到全部 target_groups
- 静默：某目标群 silence_minutes 无人说话、距本插件上次发言 >= min_interval_hours、
  且不在 02:00-06:00 深夜时段 -> 只发该群

素材：RSS 标题（feedparser，可选依赖，缺失时自动跳过）+ 可选联网大模型热点
（OpenAI 兼容 /chat/completions，key 走 env TOPIC_WEB_LLM_API_KEY）。
生成：utils 模型按 [personality] 人设写 2-3 句口语开场话题；LLM 失败降级用素材标题改写；
素材全无则本轮跳过不发。

状态（DATA_DIR 下，测试可 monkeypatch）：
- recent_topics.json  最近话题（20 条内不重复选题）
- last_send.json      {"daily": {"日期 时刻": ts}, "groups": {群号: ts}}

命令：/topic_test 立即在当前会话生成并发一条（调试用）。
"""

import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path

from junjun_agent.commands import register_command
from junjun_agent.loop.scheduler import ScheduledTask, scheduler
from junjun_core.config import get_global_config
from junjun_core.contracts import ReplySegment, ReplySet
from junjun_core.observability import get_logger

logger = get_logger("plugin.topic_finder")

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "topic_finder"

RECENT_LIMIT = 20            # 最近话题去重窗口（条）
_DUP_EXCLUDE_SECONDS = 600   # 同一轮群发（10 分钟内）刚记录的话题不参与去重（多群同题正常）
_RSS_PER_FEED = 10           # 每个 RSS 源最多取几条标题
_QUIET_START, _QUIET_END = 2, 6   # 深夜静默期：02:00-06:00 不做静默触发

# 话题归一化时剔除的字符（去重比较用，对齐旧插件）
_NORM_CHARS = " \t\n-_.,!?:;，。！？：；·—~\"'“”‘’"


def probe_available() -> bool:
    """依赖探测：feedparser 缺失不阻断加载（RSS 源自动跳过，联网热点仍可用）。"""
    return True


def _cfg() -> dict:
    """读 [topic_finder] 配置（每轮重读，热改生效）。"""
    return get_global_config().raw.get("topic_finder", {}) or {}


# ---------------------------------------------------------------- 状态读写

def _read_json(name: str, default):
    """读 DATA_DIR 下的 JSON 状态文件，损坏/不存在返回 default。"""
    path = DATA_DIR / name
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"读取状态文件失败 {name}: {type(e).__name__}: {e}")
    return default


def _write_json(name: str, data) -> None:
    """写 DATA_DIR 下的 JSON 状态文件。"""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        (DATA_DIR / name).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        logger.warning(f"写入状态文件失败 {name}: {type(e).__name__}: {e}")


def _load_recent() -> list:
    """最近话题列表 [{"content": str, "ts": float}]。"""
    data = _read_json("recent_topics.json", [])
    return data if isinstance(data, list) else []


def _record_recent(topic: str) -> None:
    """记录一条已发话题，仅保留最近 RECENT_LIMIT 条。"""
    recent = _load_recent()
    recent.append({"content": topic, "ts": time.time()})
    _write_json("recent_topics.json", recent[-RECENT_LIMIT:])


def _load_last_send() -> dict:
    """上次发送状态 {"daily": {slot: ts}, "groups": {gid: ts}}。"""
    data = _read_json("last_send.json", {})
    return data if isinstance(data, dict) else {}


def _norm(text: str) -> str:
    """话题归一化：小写 + 去空白标点，用于去重比较。"""
    s = (text or "").strip().lower()
    for ch in _NORM_CHARS:
        s = s.replace(ch, "")
    return s


def _is_dup(topic: str, recent: list) -> bool:
    """话题是否与最近 RECENT_LIMIT 条重复选题（同一轮群发的记录除外）。"""
    key = _norm(topic)
    if not key:
        return False
    cutoff = time.time() - _DUP_EXCLUDE_SECONDS
    return any(_norm(it.get("content", "")) == key
               for it in recent if float(it.get("ts", 0) or 0) < cutoff)


# ---------------------------------------------------------------- 素材获取

async def fetch_rss(cfg: dict) -> list:
    """拉取全部 RSS 源标题。feedparser 缺失/单源失败不阻断，返回标题列表。"""
    feeds = cfg.get("rss_feeds") or []
    if not feeds:
        return []
    try:
        import feedparser
    except ImportError:
        logger.warning("feedparser 未安装，RSS 素材跳过")
        return []
    titles: list[str] = []
    try:
        import httpx
        async with httpx.AsyncClient(timeout=15) as client:
            for url in feeds:
                try:
                    resp = await client.get(str(url))
                    feed = feedparser.parse(resp.text)
                    for entry in feed.entries[:_RSS_PER_FEED]:
                        title = (entry.get("title") or "").strip()
                        if title:
                            titles.append(title)
                except Exception as e:
                    logger.warning(f"RSS 源获取失败 {url}: {type(e).__name__}: {e}")
    except Exception as e:
        logger.warning(f"RSS 获取异常: {type(e).__name__}: {e}")
    return titles


def _parse_hot_lines(content: str) -> list:
    """解析联网大模型返回的热点列表：按行拆分，去编号/列表符。"""
    import re
    out = []
    for line in (content or "").splitlines():
        line = re.sub(r"^\s*(?:\d+[.、)]|[-*•])\s*", "", line).strip()
        if len(line) >= 4:
            out.append(line)
    return out


async def fetch_web_hot(cfg: dict) -> list:
    """调用联网大模型拉热点标题（OpenAI 兼容接口）。未启用/配置不全/失败返回 []。"""
    if not cfg.get("web_llm_enable", False):
        return []
    base_url = str(cfg.get("web_llm_base_url", "") or "").rstrip("/")
    model_name = str(cfg.get("web_llm_model", "") or "")
    api_key = os.environ.get("TOPIC_WEB_LLM_API_KEY", "")
    if not base_url or not model_name or not api_key:
        logger.warning("联网热点配置不完整（base_url/model/TOPIC_WEB_LLM_API_KEY），跳过")
        return []
    now = datetime.now()
    prompt = (
        f"今天是{now.strftime('%Y年%m月%d日')}。请列出 5 条当下中文互联网的热门话题或新闻标题，"
        "每行一条，只输出标题本身，不要解释。"
    )
    try:
        import httpx
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.8,
                    "max_tokens": 500,
                },
            )
            data = resp.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return _parse_hot_lines(content)
    except Exception as e:
        logger.warning(f"联网热点获取失败: {type(e).__name__}: {e}")
        return []


async def _gather_materials(cfg: dict) -> list:
    """并发拉 RSS + 联网热点，合并去重保序；单路失败不影响另一路。"""
    rss, web = await asyncio.gather(
        fetch_rss(cfg), fetch_web_hot(cfg), return_exceptions=True)
    materials: list[str] = []
    for result in (rss, web):
        if isinstance(result, Exception):
            logger.warning(f"素材获取异常: {type(result).__name__}: {result}")
            continue
        for title in result or []:
            if title and _norm(title) not in {_norm(m) for m in materials}:
                materials.append(title)
    return materials


# ---------------------------------------------------------------- 话题生成

def _persona_text() -> str:
    """从全局配置 [personality] 取人设 + 回复风格，拼成 system 提示。"""
    p = get_global_config().raw.get("personality", {}) or {}
    persona = (p.get("personality") or "").strip()
    style = (p.get("reply_style") or "").strip()
    if style:
        persona = f"{persona}\n说话风格：{style}" if persona else f"说话风格：{style}"
    return persona


async def generate_topic(cfg: dict, materials: list, recent: list | None = None) -> str | None:
    """用 utils 模型按人设把素材改写成一条群聊开场话题；失败返回 None。"""
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        from junjun_llm import get_chat_model

        material_lines = "\n".join(f"- {m}" for m in materials[:12])
        recent_items = (recent or [])[-RECENT_LIMIT:]
        recent_lines = "\n".join(f"- {it.get('content', '')}" for it in recent_items) or "（无）"
        now_str = datetime.now().strftime("%Y年%m月%d日 %H:%M")
        human = (
            f"现在是 {now_str}。参考素材（RSS/热点标题）：\n{material_lines}\n\n"
            f"最近已经发过的话题（不要重复选题、不要换皮重发）：\n{recent_lines}\n\n"
            "请结合你的人设，从素材里挑一个点，写一条发在 QQ 群里的开场话题：\n"
            "- 2~3 句口语，像随手水群抛砖引玉，能引发闲聊，不超过 80 字\n"
            "- 只输出正文，不要引号/标签/链接/emoji\n"
            "- 避免「大家好」「来聊天吧」这类万能开场套话"
        )
        model = get_chat_model("utils")
        resp = await model.ainvoke([
            SystemMessage(content=_persona_text() or "你是一个群聊话题助手。"),
            HumanMessage(content=human),
        ])
        content = resp.content
        if isinstance(content, list):  # 兼容多段 content
            content = "".join(
                str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in content)
        text = str(content or "").strip().strip('"“”「」')
        return text or None
    except Exception as e:
        logger.warning(f"LLM 话题生成失败（将尝试素材标题降级）: {type(e).__name__}: {e}")
        return None


def _fallback_from_title(materials: list) -> str | None:
    """LLM 失败时的降级：直接把第一条素材标题改写成问句。"""
    if not materials:
        return None
    return f"刚刷到一条——{materials[0]}，你们怎么看？"


# ---------------------------------------------------------------- 发送

async def _send_to_groups(topic: str, groups: list) -> None:
    """把话题文本发到每个目标群。"""
    from junjun_core.gateway.router import get_gateway
    gw = get_gateway()
    for gid in groups:
        await gw.send_reply(ReplySet(
            platform="qq",
            target_group_id=str(gid),
            segments=[ReplySegment(type="text", data=topic)],
            should_reply=True,
        ))


def _last_message_ts(group_id: str) -> float | None:
    """从数据库取该群最新一条消息的时间戳；无记录返回 None（视为已长期静默）。"""
    try:
        from junjun_core.database.models import Messages
        row = (Messages.select(Messages.time)
               .where(Messages.chat_id == f"qq:{group_id}:group")
               .order_by(Messages.time.desc()).limit(1).first())
        return float(row.time) if row else None
    except Exception as e:
        logger.warning(f"读取群 {group_id} 最近消息时间失败: {type(e).__name__}: {e}")
        return None


def _time_hit(now: datetime, time_str: str) -> bool:
    """daily_times 时刻（HH:MM）是否命中当前分钟。"""
    try:
        hour, minute = map(int, str(time_str).split(":"))
        return now.hour == hour and now.minute == minute
    except ValueError:
        logger.warning(f"无效的 daily_times 时刻: {time_str}")
        return False


async def run_round(groups: list, reason: str, cfg: dict | None = None,
                    state: dict | None = None) -> bool:
    """完整一轮：拉素材 -> 生成（去重）-> 发送到目标群 -> 记录状态。返回是否实际发送。"""
    cfg = cfg if cfg is not None else _cfg()
    materials = await _gather_materials(cfg)
    if not materials:
        logger.info(f"本轮无任何素材（RSS/联网热点全失败），跳过发送（{reason}）")
        return False

    recent = _load_recent()
    topic = await generate_topic(cfg, materials, recent)
    if not topic:
        topic = _fallback_from_title(materials)
    if not topic:
        return False

    if _is_dup(topic, recent):
        # 与近期话题撞车：重试一次，仍重复则本轮放弃（20 条内不重复选题）
        retry = await generate_topic(cfg, materials, recent)
        if retry and not _is_dup(retry, recent):
            topic = retry
        else:
            logger.info(f"话题与近期重复且重试未果，本轮跳过（{reason}）")
            return False

    await _send_to_groups(topic, groups)
    _record_recent(topic)

    state = state if state is not None else _load_last_send()
    group_state = state.setdefault("groups", {})
    ts = time.time()
    for gid in groups:
        group_state[str(gid)] = ts
    _write_json("last_send.json", state)
    logger.info(f"话题已发送（{reason}）-> {groups}: {topic[:50]}")
    return True


# ---------------------------------------------------------------- 调度入口

async def topic_finder_tick() -> None:
    """scheduler 每 60s 调用：内部判断定时/静默是否到点（配置每轮重读，热改生效）。"""
    cfg = _cfg()
    if not cfg.get("enable", False):
        return
    groups = [str(g) for g in cfg.get("target_groups", []) or []]
    if not groups:
        return

    now = datetime.now()
    now_ts = time.time()
    state = _load_last_send()
    daily_state = state.setdefault("daily", {})
    today = now.strftime("%Y-%m-%d")

    # 1) 定时触发：命中 daily_times 当前分钟且当日该时刻未发
    for t in cfg.get("daily_times", []) or []:
        slot = f"{today} {t}"
        if slot not in daily_state and _time_hit(now, str(t)):
            daily_state[slot] = now_ts  # 无论成败当日只试一次，避免分钟级重试风暴
            _write_json("last_send.json", state)
            await run_round(groups, reason=f"定时 {t}", cfg=cfg, state=state)
            break

    # 2) 静默触发：深夜 02:00-06:00 不打扰
    if _QUIET_START <= now.hour < _QUIET_END:
        return
    silence_sec = float(cfg.get("silence_minutes", 60)) * 60
    min_interval_sec = float(cfg.get("min_interval_hours", 3)) * 3600
    group_state = state.setdefault("groups", {})
    for gid in groups:
        if now_ts - float(group_state.get(gid, 0) or 0) < min_interval_sec:
            continue  # 距本插件上次发言不足最小间隔
        last_msg = _last_message_ts(gid)
        if last_msg is not None and now_ts - last_msg < silence_sec:
            continue  # 群内还有人在说话
        await run_round([gid], reason="群聊静默", cfg=cfg, state=state)


# import 时自注册：scheduler 在 load_plugins 之后启动
scheduler.add(ScheduledTask("topic_finder", topic_finder_tick, interval=60))


# ---------------------------------------------------------------- 调试命令

@register_command("topic_test", plugin="topic_finder", admin_only=True,
                  description="立即生成并发一条话题（调试用，触发 LLM 生成，管理员专用）")
async def topic_test_cmd(ctx):
    cfg = _cfg()
    materials = await _gather_materials(cfg)
    if not materials:
        return "没拉到任何素材（RSS/联网热点都失败了），这轮生成不了话题。"
    recent = _load_recent()
    topic = await generate_topic(cfg, materials, recent)
    if not topic:
        topic = _fallback_from_title(materials)
    if not topic:
        return "话题生成失败了，稍后再试试吧。"
    await ctx.send([ReplySegment(type="text", data=topic)])
    _record_recent(topic)
    return None


TOOLS = []
