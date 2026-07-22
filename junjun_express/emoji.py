"""表情包系统：对齐原 emoji_system/emoji_manager 语义。

- steal_emoji: 群里出现的图片按 hash 去重下载入库 data/emoji/
- 注册循环: 未注册图片 -> VLM 描述+情感分类 -> data/emoji_registed/ + Emoji 表
- 发送选择: 情感/语境关键词 -> 候选池 -> 按使用次数加权随机
- max_reg_num 上限 + do_replace 换血；skill 独立冷却 60s/会话
"""

import hashlib
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

from junjun_core.config import get_global_config
from junjun_core.observability import get_logger

logger = get_logger("express.emoji")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EMOJI_RAW_DIR = PROJECT_ROOT / "data" / "emoji"
EMOJI_REG_DIR = PROJECT_ROOT / "data" / "emoji_registed"

_SEND_COOLDOWN = 60.0

_DESCRIBE_PROMPT = (
    "这是一张 QQ 群聊天表情包。用一句话描述它的画面与含义，"
    "再给出 1-3 个情感标签（如 开心/嘲讽/无语/可爱/愤怒/悲伤/疑惑/得意）。\n"
    '只输出 JSON：{"description": "...", "emotions": ["...", "..."]}'
)


def _cfg() -> dict:
    return get_global_config().raw.get("emoji", {})


class EmojiManager:
    def __init__(self):
        EMOJI_RAW_DIR.mkdir(parents=True, exist_ok=True)
        EMOJI_REG_DIR.mkdir(parents=True, exist_ok=True)
        self._last_sent: Dict[str, float] = {}  # chat_id -> ts

    # ---------- 偷图 ----------

    async def steal(self, image_urls: List[str]) -> int:
        """下载群图片入待注册池（hash 去重）。返回新增数。"""
        if not _cfg().get("steal_emoji", True) or not image_urls:
            return 0
        from junjun_core.database import Emoji
        count = 0
        for url in image_urls[:3]:  # 单条消息最多偷 3 张
            try:
                data = await self._download(url)
                if data is None or len(data) > 2 * 1024 * 1024:  # >2MB 不是表情包
                    continue
                h = hashlib.md5(data).hexdigest()
                if Emoji.get_or_none(Emoji.emoji_hash == h) or (EMOJI_RAW_DIR / f"{h}.img").exists():
                    continue
                # 按实际格式存正确扩展名（NapCat/QQ 按扩展名识别类型，.img 显示为未知文件）
                ext = self._detect_ext(data)
                (EMOJI_RAW_DIR / f"{h}{ext}").write_bytes(data)
                count += 1
            except Exception as e:
                logger.debug(f"偷图失败（忽略）: {e}")
        if count:
            logger.info(f"偷到 {count} 张新表情包（待注册）")
        return count

    @staticmethod
    def _detect_ext(data: bytes) -> str:
        """按 magic bytes 检测真实图片格式。"""
        if data[:3] == b"\xff\xd8\xff":
            return ".jpg"
        if data[:6] in (b"GIF87a", b"GIF89a"):
            return ".gif"
        if data[:8] == b"\x89PNG\r\n\x1a\n":
            return ".png"
        if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            return ".webp"
        return ".jpg"  # 默认兜底

    async def _download(self, url: str) -> Optional[bytes]:
        import aiohttp
        async with aiohttp.ClientSession() as s:
            async with s.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    return None
                return await resp.read()

    # ---------- 注册循环（调度器任务）----------

    async def register_pending(self, *, limit: int = 5, model=None) -> int:
        """把待注册池的图片 VLM 描述后注册。每轮限量防烧 token。返回注册数。"""
        from junjun_core.database import Emoji
        cfg = _cfg()
        pending = sorted(EMOJI_RAW_DIR.glob("*.*"))[:limit]  # 任意扩展名（.jpg/.gif/.png/.webp）
        pending = [p for p in pending if p.suffix.lower() in (".jpg", ".jpeg", ".gif", ".png", ".webp")]
        if not pending:
            return 0

        max_reg = int(cfg.get("max_reg_num", 2000))
        total = Emoji.select().count()
        registered = 0
        for p in pending:
            h = p.stem
            if total + registered >= max_reg:
                if not cfg.get("do_replace", True):
                    logger.info("表情包已达上限且 do_replace=false，停止注册")
                    break
                self._evict_one()
            try:
                desc, emotions = await self._describe(p.read_bytes(), model=model)
            except Exception as e:
                logger.debug(f"表情包描述失败（保留待注册）: {e}")
                continue
            if desc is None:
                p.unlink(missing_ok=True)  # VLM 明确拒绝/无效图，丢弃
                continue
            target = EMOJI_REG_DIR / p.name
            p.replace(target)
            Emoji.create(full_path=str(target), emoji_hash=h,
                         description=desc, emotion=json.dumps(emotions, ensure_ascii=False))
            registered += 1
        if registered:
            logger.info(f"注册 {registered} 张表情包（库存 {total + registered}）")
        return registered

    def _evict_one(self) -> None:
        """换血：删使用次数最低的一张。"""
        from junjun_core.database import Emoji
        victim = Emoji.select().order_by(Emoji.usage_count).first()
        if victim:
            Path(victim.full_path).unlink(missing_ok=True)
            victim.delete_instance()

    async def _describe(self, data: bytes, *, model=None):
        """VLM 描述。当前 model_config 无 vlm 槽时用 utils 文本模型不可行——
        降级：无 VLM 可用时返回通用描述（仍可按随机情感使用）。"""
        import base64
        if model is None:
            model = self._get_vlm()
        if model is None:
            return "一张群友表情包", ["通用"]
        from langchain_core.messages import HumanMessage
        b64 = base64.b64encode(data).decode()
        resp = await model.ainvoke([HumanMessage(content=[
            {"type": "text", "text": _DESCRIBE_PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ])])
        import re
        m = re.search(r"\{.*\}", str(resp.content), re.S)
        if not m:
            return None, []
        try:
            obj = json.loads(m.group(0))
            return obj.get("description", ""), list(obj.get("emotions", []))[:3]
        except json.JSONDecodeError:
            return None, []

    def _get_vlm(self):
        try:
            from junjun_llm import get_chat_model
            return get_chat_model("vlm")
        except Exception:
            return None

    # ---------- 发送 ----------

    def cooldown_ok(self, chat_id: str) -> bool:
        return (time.time() - self._last_sent.get(chat_id, 0)) >= _SEND_COOLDOWN

    def pick(self, mood_or_context: str, chat_id: str) -> Optional[dict]:
        """按情感/语境选一张。冷却中或无库存返回 None。"""
        from junjun_core.database import Emoji
        if not self.cooldown_ok(chat_id):
            return None
        rows = list(Emoji.select())
        if not rows:
            return None
        # 情感标签或描述包含语境关键词的进入候选池
        candidates = []
        for r in rows:
            try:
                emotions = json.loads(r.emotion or "[]")
            except json.JSONDecodeError:
                emotions = []
            if any(e and e in mood_or_context for e in emotions) or \
               any(w in (r.description or "") for w in mood_or_context.split()[:3] if len(w) > 1):
                candidates.append(r)
        if not candidates:
            candidates = rows  # 无匹配随机全库
        chosen = random.choice(candidates)
        chosen.usage_count += 1
        chosen.save()
        self._last_sent[chat_id] = time.time()
        return {"path": chosen.full_path, "description": chosen.description}


emoji_manager = EmojiManager()
