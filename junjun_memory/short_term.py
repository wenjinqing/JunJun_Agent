"""短期记忆：按会话滑动窗口。

群聊消息渲染带昵称前缀，Agent 能分清谁在说话。
阶段 2 新增：可选持久化到 SQLite，进程重启后可恢复最近上下文。
"""

import json
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional

from junjun_core.observability import get_logger

logger = get_logger("memory.stm")

# 管理员锚点防伪：昵称/消息内容里不得出现系统标记样式或换行，
# 否则群名片「xx(管理员)」或消息内伪造行能冒充系统打的标记（代码层
# is_admin_privileged 闸门不受影响，但 LLM 认知锚点会被污染）
_ADMIN_MARKER_TOKENS = ("(管理员)", "（管理员）")

# 昵称分隔符/系统标记防伪（2026-08-11 昵称注入事故）：昵称是群友随便起的
# 群名片，不可信——有人故意起「有人@我，我喜欢你很久了」这种整段话当昵称，
# 模型会把昵称当消息内容（引用该群友消息时尤其严重）。render 用「」包住
# 昵称与内容分隔，「」、[@你]、【最新】一律从昵称剥掉，防从内部突破分隔符。
_FORGE_TOKENS = _ADMIN_MARKER_TOKENS + ("「", "」", "[@你]", "【最新】")

# 昵称展示截断：正常群名片 2-6 字，长昵称几乎全是整段玩梗/钓鱼——
# 钓饵需要完整句子才生效，截断本身就拆掉攻击面，还顺带省 token
_MAX_NICK_CHARS = 10


def _sanitize_nickname(nickname: str) -> str:
    """昵称剥分隔符/系统标记并截断（用户可任意改群名片，不可信）。"""
    name = nickname or ""
    for token in _FORGE_TOKENS:
        name = name.replace(token, "")
    name = name.replace("\n", " ").strip()
    if len(name) > _MAX_NICK_CHARS:
        name = name[:_MAX_NICK_CHARS] + "…"
    return name


def _sanitize_text(text: str) -> str:
    """消息内容换行压成可见符号——防止消息体内伪造一行「昵称(管理员): ...」。"""
    return (text or "").replace("\r", " ").replace("\n", " ⏎ ")


# ---------- 滚动摘要（2026-08-16，DSH 压缩思想移植） ----------
# 滑出窗口的内容不直接蒸发：攒够阈值让 utils 槽模型压成一段「前情摘要」，
# 渲染时置顶。窗口继续滑，摘要滚动合并（旧摘要+新滑出块 -> 新摘要）。
# 摘要有损（细节丢）但保住话题脉络；细节检索走长期记忆/DB 召回。
# 摘要只驻内存：重启后 entries 从 SQLite 恢复、摘要清空，退化为现状，可接受。
_SUMMARY_MIN_CHARS = 2000     # 待压材料攒够这么多字才烧一次 utils（摊薄成本）
_SUMMARY_MAX_CHARS = 400      # 摘要上限（渲染置顶，太长又变新包袱）
_SUMMARY_RETRY_COOLDOWN = 120.0   # 摘要失败后冷却（秒）——LLM 挂时防每条消息都重试
_PENDING_MAX_LINES = 200      # 待压队列上限：持续失败时丢最旧的（真丢失仅限此情形）

_SUMMARY_PROMPT = (
    "把这段群聊记录并入「前情摘要」，输出一段不超过 {max_chars} 字的新摘要。"
    "保留：话题脉络、谁在参与什么、结论/约定/待办；丢掉寒暄与语气词。"
    "只输出摘要正文，不要解释，不要分点。\n\n"
    "旧摘要（可能为空）：{old}\n\n刚滑出窗口的记录：\n{chunk}"
)


def _summary_cfg() -> tuple:
    """([memory] rolling_summary, min_chars, max_chars)。配置坏了按默认。"""
    try:
        from junjun_core.config import get_global_config
        mem = get_global_config().raw.get("memory", {}) or {}
        return (bool(mem.get("rolling_summary", True)),
                int(mem.get("rolling_summary_min_chars", _SUMMARY_MIN_CHARS)),
                int(mem.get("rolling_summary_max_chars", _SUMMARY_MAX_CHARS)))
    except Exception:
        return True, _SUMMARY_MIN_CHARS, _SUMMARY_MAX_CHARS


def _sanitize_summary(text: str) -> str:
    """摘要防伪：内容源自群友发言，「」/【最新】/管理员标记一律剥掉——
    否则一句伪造的「(管理员) 说过 xxx」会随摘要置顶进每一轮 context。"""
    t = text or ""
    for token in _FORGE_TOKENS:
        t = t.replace(token, "")
    return _sanitize_text(t).strip()


@dataclass
class MemoryEntry:
    role: str  # "user" / "bot"
    text: str
    nickname: str = ""
    user_id: str = ""
    message_id: str = ""
    at_bot: bool = False
    ts: float = field(default_factory=time.time)  # 背景裁剪按龄期判断用


# 背景低价值行裁剪（2026-08-11 上下文压缩）：超龄的纯语气/反应行不进背景——
# 真人也不记得 5 分钟前谁回了个「嗯」。白名单制，宁漏勿错杀：
# 「好/行/ok/可以」这类承诺词不在列（「今晚开黑吗」「好」砍了语义就断），
# 占位行（[图片]/[表情]）不在列（感知链路要靠它知道有人发过图）。
_FILLER_EXACT = {"嗯", "嗯嗯", "哦", "哦哦", "啊", "啊这", "哈", "哈哈", "笑死",
                 "草", "艹", "6", "？", "?", "！", "!", "。", "emm", "emmm"}
_FILLER_BASE = {"嗯", "哦", "啊", "哈", "草", "艹", "6", "2"}  # 叠字坍缩后比对（哈哈哈→哈）


def _is_filler(text: str) -> bool:
    """是否纯语气/反应行（无信息增量）。"""
    import re
    t = (text or "").strip()
    if not t or len(t) > 6:
        return False
    if t in _FILLER_EXACT:
        return True
    collapsed = re.sub(r"(.)\1+", r"\1", t)  # 叠字坍缩：哈哈哈→哈、666→6、2333→23
    return collapsed in _FILLER_BASE or collapsed in {"23"}


def _prune_cfg() -> tuple:
    """([memory] background_prune_filler, filler_ttl_seconds)，默认关/300s。

    默认关（2026-08-11 用户裁决）：直接裁断是丢信息换 token 的下策，
    同一人连续发言合并（零损失）已覆盖其价值——此开关留给刷屏特别狠的群
    按需手动开。"""
    try:
        from junjun_core.config import get_global_config
        mem = get_global_config().raw.get("memory", {}) or {}
        return (bool(mem.get("background_prune_filler", False)),
                float(mem.get("filler_ttl_seconds", 300)))
    except Exception:
        return False, 300.0


@dataclass
class ShortTermMemory:
    max_size: int = 80
    entries: List[MemoryEntry] = field(default_factory=list)
    chat_id: str = ""           # 会话键；空时不持久化
    persist: bool = False       # 是否写入 SQLite
    rolling_summary: str = ""   # 滑出窗口内容的滚动摘要（渲染置顶；仅内存，重启清空）

    def __post_init__(self):
        self._last_save = 0.0
        self._flush_scheduled = False
        self._save_lock = threading.Lock()
        self._pending_compress: List[str] = []   # 已滑出、待摘要的行
        self._compressing = False                # 摘要任务在途标记（防重入）
        self._compress_fail_ts = 0.0             # 摘要失败冷却起点
        if self.persist and self.chat_id:
            self._load()

    def _load(self) -> None:
        """从 SQLite 恢复 entries（失败静默）。"""
        try:
            from junjun_core.database import ShortTermMemory as STMModel
            rec = STMModel.get_or_none(STMModel.chat_id == self.chat_id)
            if rec and rec.entries_json:
                loaded = json.loads(rec.entries_json)
                self.entries = [MemoryEntry(**e) for e in loaded[-self.max_size:]]
        except Exception:
            pass

    def _save(self) -> None:
        """异步写入 SQLite（失败静默）。

        节流（2026-08-09 审查）：每条消息全量序列化 upsert 会放大 db_writer
        队列压力（容量 2000，与其他业务写共享，满了丢最新）。3s 内多次变更
        只落一次，尾部由 Timer 补落——丢的最多是最后 3s 的上下文，可接受。
        """
        if not self.persist or not self.chat_id:
            return
        timer = None
        with self._save_lock:
            recent = time.time() - self._last_save < 3.0
            if recent and not self._flush_scheduled:
                self._flush_scheduled = True
                timer = threading.Timer(3.0, self._flush)
                timer.daemon = True
            elif not recent:
                self._last_save = time.time()
        if timer is not None:
            timer.start()   # 锁外启动：同步式 Timer 实现会回调 _flush（要拿锁）
            return
        if recent:
            return          # 已有待发的尾部 flush
        self._submit()

    def _flush(self) -> None:
        """节流尾部补落。"""
        with self._save_lock:
            self._flush_scheduled = False
            self._last_save = time.time()
        self._submit()

    def _submit(self) -> None:
        try:
            from junjun_core.database import ShortTermMemory as STMModel, db_writer
            data = [asdict(e) for e in self.entries]
            payload = json.dumps(data, ensure_ascii=False)
            now = time.time()

            def _upsert():
                (STMModel
                 .insert(chat_id=self.chat_id, entries_json=payload, updated_at=now)
                 .on_conflict(
                     conflict_target=[STMModel.chat_id],
                     update={STMModel.entries_json: payload, STMModel.updated_at: now})
                 .execute())

            db_writer.submit(_upsert)
        except Exception:
            pass

    def add_user(self, text: str, nickname: str, user_id: str = "",
                 message_id: str = "", at_bot: bool = False) -> None:
        self.entries.append(MemoryEntry(
            role="user", text=text, nickname=nickname,
            user_id=user_id, message_id=message_id, at_bot=at_bot,
        ))
        self._trim()
        self._save()

    def add_bot(self, text: str) -> None:
        self.entries.append(MemoryEntry(role="bot", text=text))
        self._trim()
        self._save()

    def _trim(self) -> None:
        if len(self.entries) > self.max_size:
            dropped = self.entries[:-self.max_size]
            self.entries = self.entries[-self.max_size:]
            self._note_dropped(dropped)

    def _note_dropped(self, dropped: List[MemoryEntry]) -> None:
        """滑出窗口的行攒进待压队列，够阈值且没在途/冷却时起火后台摘要。

        后台任务走 fire_and_forget（强引用，裸 create_task 会被 GC 收走——
        2026-08-13 生产实锤）。无运行中事件循环（同步上下文，如测试/CLI）
        不起任务，材料留在队列里等下次触发，不丢。
        """
        enabled, min_chars, _max_chars = _summary_cfg()
        if not enabled or not dropped:
            return
        for e in dropped:
            name = "你" if e.role == "bot" else (e.nickname or e.user_id or "群友")
            self._pending_compress.append(f"{name}: {e.text}")
        if len(self._pending_compress) > _PENDING_MAX_LINES:
            overflow = len(self._pending_compress) - _PENDING_MAX_LINES
            self._pending_compress = self._pending_compress[overflow:]
            logger.warning(f"[{self.chat_id or '?'}] 待压摘要队列超限，"
                           f"最旧 {overflow} 行丢弃（摘要持续失败？）")
        if self._compressing:
            return
        if sum(len(x) for x in self._pending_compress) < min_chars:
            return
        if time.time() - self._compress_fail_ts < _SUMMARY_RETRY_COOLDOWN:
            return
        try:
            import asyncio
            asyncio.get_running_loop()
        except Exception:
            return          # 同步上下文起不了后台任务，攒着等下次
        try:
            from junjun_core.bg_tasks import fire_and_forget
            self._compressing = True
            coro = self._compress_pending()
            try:
                fire_and_forget(coro, name=f"stm-summary:{self.chat_id or '?'}")
            except Exception:
                coro.close()    # 未接管的协程显式关闭，防 never-awaited 警告
                raise
        except Exception:
            self._compressing = False

    async def _compress_pending(self) -> None:
        """待压队列 + 旧摘要 -> utils 槽模型滚动合并成新摘要。

        失败不丢料：队列塞回去等冷却后再试（材料在，脉络就有救）。
        """
        _enabled, _min_chars, max_chars = _summary_cfg()
        chunk_lines = self._pending_compress
        self._pending_compress = []
        try:
            from langchain_core.messages import HumanMessage
            from junjun_llm import get_chat_model
            model = get_chat_model("utils")
            resp = await model.ainvoke([HumanMessage(content=_SUMMARY_PROMPT.format(
                max_chars=max_chars,
                old=self.rolling_summary or "（空）",
                chunk="\n".join(chunk_lines)))])
            text = str(getattr(resp, "content", "") or "").strip()
            if not text:
                raise ValueError("模型返回空摘要")
            self.rolling_summary = text[:max_chars * 2]  # 硬上限防模型超写
            logger.info(f"[{self.chat_id or '?'}] 滚动摘要更新 "
                        f"（吞 {len(chunk_lines)} 行，摘要 {len(text)} 字）")
        except Exception as e:
            self._compress_fail_ts = time.time()
            # 材料塞回队首（新滑出的已在队尾），下次触发重试
            self._pending_compress = (chunk_lines + self._pending_compress)[
                -_PENDING_MAX_LINES:]
            logger.warning(f"[{self.chat_id or '?'}] 滚动摘要失败（材料保留待重试）: "
                           f"{type(e).__name__}: {e}")
        finally:
            self._compressing = False

    def render(self, limit: Optional[int] = None, *, mark_latest: bool = False,
               include_bot: bool = True, for_security: bool = False,
               prune: Optional[bool] = None) -> str:
        """渲染为对话文本（供 prompt）。群聊格式 `「昵称」: 内容`。

        昵称用「」与内容硬分隔（2026-08-11 昵称注入事故）：「」内是群名片——
        群友随便起的标签，可能整段话玩梗/假装说话，永远不是消息内容；
        冒号后的才是。配合 persona 安全段规则生效。

        管理员消息带「(管理员)」系统标记——按真实 user_id 判定，聊天内容无法伪造，
        是 LLM 识别管理员指令的锚点（配合 persona 安全段）。

        mark_latest: True 时最后一条 user 消息前缀「【最新】」——帮模型聚焦。
        include_bot: True 时 bot 回复进 context（记忆效果），但前缀「你(历史):」
        明确标记为「已发生的历史输出」而非「待接续的话」（防复读关键）。
        for_security: True 时保留（管理员）标记（安全验证用）；
        False（默认）时管理员显示为普通群友（不影响回复意愿）。
        prune: 超龄语气词行裁剪（[memory] background_prune_filler，默认关——
        裁断是丢信息换 token 的下策，连续发言合并已覆盖其价值；刷屏狠的群
        可手动开）。只砍白名单内的纯反应行；@你 的行/最新一条/bot 历史/
        占位行/承诺词（好/行/可以）一律保留。

        边界感知（LangChain trim_messages 语义）：永远以 user 消息开头，
        不从 bot 回复中间截断——模型不会把被截断的历史当成待续写文本。
        """
        from junjun_core.security import is_admin
        from junjun_memory.echo import normalize_echo
        entries = self.entries[-limit:] if limit else self.entries
        # 同一人连续发言合并成一条（⏎ 连接，与 _sanitize_text 的多行渲染同
        # 视觉约定）——零信息损失的压缩（2026-08-11 方案 B）：群友一句话拆
        # N 条发是常态，每条重复的「昵称」: 前缀是纯开销。合并发生在副本上，
        # 不污染 entries 本体（引用决策还要用原始 message_id 粒度）
        merged: List[MemoryEntry] = []
        for e in entries:
            if (e.role == "user" and e.user_id and merged
                    and merged[-1].role == "user"
                    and merged[-1].user_id == e.user_id):
                top = merged[-1]
                top.text = f"{top.text} ⏎ {e.text}"
                top.at_bot = top.at_bot or e.at_bot
                top.ts = max(top.ts, e.ts)
                top.message_id = e.message_id  # 引用指向最新一条
            else:
                merged.append(MemoryEntry(**asdict(e)))
        entries = merged
        # bot 历史去重（防上下文自污染，2026-08-04）：同一句话术只保留最近一次
        # 出现——否则 context 里「你(历史): xxx」堆 N 次，模型把它当成自己的
        # 说话习惯继续复读，形成正反馈污染循环
        bot_last = {}
        for i, e in enumerate(entries):
            if e.role == "bot":
                n = normalize_echo(e.text)
                if n:
                    bot_last[n] = i
        lines = []
        # 找最后一条 user 消息的下标（mark_latest / 裁剪豁免用）
        last_user_idx = -1
        for i in range(len(entries) - 1, -1, -1):
            if entries[i].role == "user":
                last_user_idx = i
                break
        # 超龄语气词裁剪（上下文压缩）
        if prune is None:
            prune, filler_ttl = _prune_cfg()
        else:
            filler_ttl = _prune_cfg()[1] if prune else 0.0
        now = time.time()
        pruned = 0
        # 边界感知：跳过开头的 bot 消息（不从 bot 回复中间截断）
        start = 0
        while start < len(entries) and entries[start].role == "bot":
            start += 1
        for i, e in enumerate(entries[start:], start=start):
            if e.role == "bot":
                if include_bot:
                    n = normalize_echo(e.text)
                    if n and bot_last.get(n) != i:
                        continue  # 更早的重复出现：跳过，只留最近一次
                    # 标记为「历史输出」而非「待接续的话」（防复读关键）
                    lines.append(f"你(历史): {e.text}")
                # 默认不进 context（include_bot=False 时）
            else:
                # 超龄语气词裁剪：最新一条/@你 的行豁免（节奏与直指你的反应
                # 都有信息量）；bot 历史与占位行走别的分支天然保留
                if (prune and i != last_user_idx and not e.at_bot
                        and now - e.ts > filler_ttl and _is_filler(e.text)):
                    pruned += 1
                    continue
                # 「」硬分隔昵称与内容；昵称里的「」已被 sanitize 剥掉，
                # 分隔符无法从内部突破
                prefix = f"「{_sanitize_nickname(e.nickname) or e.user_id}」"
                # 管理员标记：for_security=True 才保留（安全验证用），
                # 默认不显示——L2/L3 看到的都是普通群友，不影响回复意愿
                if for_security and is_admin(e.user_id):
                    prefix += "(管理员)"
                mark = " [@你]" if e.at_bot else ""
                if mark_latest and i == last_user_idx:
                    prefix = f"【最新】{prefix}"
                lines.append(f"{prefix}{mark}: {_sanitize_text(e.text)}")
        if pruned:
            logger.debug(f"[{self.chat_id or '?'}] 背景裁剪 {pruned} 行超龄语气词")
        # 滚动摘要置顶：滑出窗口的话题脉络以一段「前情摘要」留在上下文里。
        # 括号+「摘要」明示这是压缩过的背景，不是某条具体消息
        summary = _sanitize_summary(self.rolling_summary)
        if summary:
            lines.insert(0, f"（更早前聊过·摘要：{summary}）")
        return "\n".join(lines)

    def last_user_entry(self) -> Optional[MemoryEntry]:
        for e in reversed(self.entries):
            if e.role == "user":
                return e
        return None
