"""主动私聊：对齐原 proactive_system/proactive_chat_manager 语义。

- 调度器定期扫描会话空闲（min_idle_minutes）
- LLM 生成话题（proactive_topics.json 去重存最近 10 条）-> LLM 二判 -> 发送
- silent_hours + max_daily_proactive 限额；默认仅私聊（enable_in_groups=false）
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from junjun_core.config import get_global_config
from junjun_core.observability import get_logger

logger = get_logger("loop.proactive")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOPICS_FILE = PROJECT_ROOT / "data" / "proactive_topics.json"

_TOPIC_PROMPT = """你是"{nickname}"。你想主动找朋友聊天。
你们最近的对话：
{recent}

最近已经主动聊过的话题（避免重复）：
{used_topics}

生成一条自然的开场消息（像朋友突然想起对方发的微信，简短口语化，可以基于之前聊过的内容延续）。
只输出消息本身。"""

_JUDGE_PROMPT = """你是聊天机器人的把关员。机器人想主动给朋友发这条消息：
「{message}」
现在是 {now}。判断这条消息是否合适主动发出（不打扰、不突兀、内容自然）。
只输出：合适 / 不合适"""


def _cfg() -> dict:
    return get_global_config().raw.get("proactive_chat", {})


def _in_silent_hours(now: Optional[datetime] = None) -> bool:
    from junjun_agent.funnel.frequency import _in_range
    spec = str(_cfg().get("silent_hours", "23:00-09:00"))
    now = now or datetime.now()
    return _in_range(now.hour * 60 + now.minute, spec)


class ProactiveChatManager:
    def __init__(self):
        self._daily_count: Dict[str, int] = {}
        self._count_date = ""

    def _reset_daily(self) -> None:
        today = datetime.now().strftime("%Y-%m-%d")
        if self._count_date != today:
            self._count_date = today
            self._daily_count = {}

    def _load_topics(self) -> list:
        try:
            return json.loads(TOPICS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []

    def _save_topic(self, topic: str) -> None:
        topics = self._load_topics()
        topics.append({"time": time.time(), "topic": topic})
        TOPICS_FILE.parent.mkdir(parents=True, exist_ok=True)
        TOPICS_FILE.write_text(json.dumps(topics[-10:], ensure_ascii=False, indent=1), encoding="utf-8")

    def eligible(self, session, *, now: Optional[float] = None) -> bool:
        """会话是否满足主动条件（不含 LLM 判断）。"""
        cfg = _cfg()
        if not cfg.get("enable", False):
            return False
        if session.is_group and not cfg.get("enable_in_groups", False):
            return False
        if not session.is_group and not cfg.get("enable_in_private", True):
            return False
        if _in_silent_hours():
            return False
        self._reset_daily()
        if self._daily_count.get(session.chat_id, 0) >= int(cfg.get("max_daily_proactive", 2)):
            return False
        # 空闲判定
        if session.memory is None or not session.memory.entries:
            return False  # 从没聊过的不主动（没有上下文可延续）
        idle_min = float(cfg.get("min_idle_minutes", 120))
        last = getattr(session, "last_active_ts", 0)
        now = now if now is not None else time.time()
        return (now - last) >= idle_min * 60 if last else False

    async def try_proactive(self, session, *, model=None, callbacks=None) -> bool:
        """生成话题 -> 二判 -> 发送。返回是否发出。"""
        cfg = get_global_config()
        if model is None:
            from junjun_llm import get_chat_model
            model = get_chat_model("utils")
        from langchain_core.messages import HumanMessage

        used = "\n".join(f"- {t['topic'][:40]}" for t in self._load_topics()) or "（无）"
        recent = session.memory.render(limit=10) if session.memory else "（无）"
        try:
            resp = await model.ainvoke(
                [HumanMessage(content=_TOPIC_PROMPT.format(
                    nickname=cfg.bot.nickname, recent=recent, used_topics=used))],
                config={"callbacks": callbacks or []},
            )
            message = str(resp.content).strip().splitlines()[0][:200]
            if not message:
                return False
            judge = await model.ainvoke(
                [HumanMessage(content=_JUDGE_PROMPT.format(
                    message=message, now=datetime.now().strftime("%H:%M")))],
                config={"callbacks": callbacks or []},
            )
            if "不合适" in str(judge.content):
                logger.info(f"[{session.chat_id}] 主动话题被二判拦截: {message[:30]}")
                return False
        except Exception as e:
            logger.warning(f"[{session.chat_id}] 主动话题生成失败: {e}")
            return False

        from junjun_core.contracts import ReplySet, ReplySegment
        from junjun_core.gateway.router import get_gateway
        await get_gateway().send_reply(ReplySet(
            platform=session.platform,
            target_group_id=session.group_id,
            target_user_id=session.user_id if not session.is_group else None,
            segments=[ReplySegment(type="text", data=message)],
            should_reply=True,
        ))
        session.memory.add_bot(message)
        self._daily_count[session.chat_id] = self._daily_count.get(session.chat_id, 0) + 1
        self._save_topic(message)
        logger.info(f"[{session.chat_id}] 主动发起: {message[:40]}")
        return True

    async def scan(self) -> None:
        """调度器任务：扫全部会话。"""
        from junjun_core.gateway.session_manager import get_session_manager
        for session in list(get_session_manager().all_sessions().values()):
            try:
                if self.eligible(session):
                    await self.try_proactive(session)
            except Exception as e:
                logger.warning(f"[{session.chat_id}] 主动扫描异常: {e}")


proactive_manager = ProactiveChatManager()
