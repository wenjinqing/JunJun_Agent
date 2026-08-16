"""回复后处理流水线：agent 原始文本 -> 多条待发消息。

顺序：去 <think> 残留 -> 去 markdown 标记 -> 去语气词/emoji -> 分割多条 ->
错别字 -> （引用决策在 processor 层）。
纯函数，配置从 bot_config [response_post_process]/[response_splitter]/[chinese_typo] 读取。
"""

import random
import re
from dataclasses import dataclass
from typing import List, Optional

from junjun_core.config import get_global_config
from junjun_agent.postprocess.cleaner import clean_markdown, clean_tone
from junjun_agent.postprocess.splitter import split_response, typing_delay
from junjun_agent.postprocess.typo import ChineseTypoGenerator

# 完整 <think>...</think> 块剥离
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.S)
# 无闭合标签的尾部思考链（LLM 没吐 </think> 的情况）——从 <think> 到文本末尾全砍
_THINK_TAIL_RE = re.compile(r"<think>.*$", re.S)
_NICKNAME_PREFIX_RE = re.compile(r"^\s*(你|君君)\s*[:：]\s*")

_typo_gen: Optional[ChineseTypoGenerator] = None


@dataclass
class OutboundMessage:
    text: str
    delay: float = 0.0  # 发送前延迟（秒）


def _get_typo_gen() -> ChineseTypoGenerator:
    global _typo_gen
    if _typo_gen is None:
        cfg = get_global_config().raw.get("chinese_typo", {})
        _typo_gen = ChineseTypoGenerator(
            error_rate=float(cfg.get("error_rate", 0.01)),
            min_freq=int(cfg.get("min_freq", 9)),
            tone_error_rate=float(cfg.get("tone_error_rate", 0.1)),
            word_replace_rate=float(cfg.get("word_replace_rate", 0.006)),
        )
    return _typo_gen


def process_response(text: str, *, rand: Optional[random.Random] = None,
                     incoming: str = "") -> List[OutboundMessage]:
    """agent 文本 -> 待发消息列表（含阅读延迟 + 逐条打字延迟）。

    incoming：触发回复的用户消息原文——用于模拟「看完消息再打字」的阅读时间，
    避免永远秒回的机器人感（[response_timing] 可关）。
    """
    raw = get_global_config().raw
    pp = raw.get("response_post_process", {})
    sp = raw.get("response_splitter", {})
    typo_cfg = raw.get("chinese_typo", {})
    timing_cfg = raw.get("response_timing", {})
    rng = rand or random

    text = _THINK_BLOCK_RE.sub("", text or "")
    text = _THINK_TAIL_RE.sub("", text)
    text = _NICKNAME_PREFIX_RE.sub("", text.strip())
    if pp.get("clean_markdown", True):
        text = clean_markdown(text)
    # 语气词/emoji 结构层清理（2026-08-17 用户拍板）：reply_style 早写了
    # 「不用 emoji」弱模型照吐——prompt 是劝告，这里是闸门
    if pp.get("strip_emoji", True) or pp.get("strip_tone_particles", True):
        text = clean_tone(text,
                          strip_emoji=bool(pp.get("strip_emoji", True)),
                          strip_particles=bool(pp.get("strip_tone_particles", True)))
    if not text:
        return []

    if not pp.get("enable_response_post_process", True):
        return [OutboundMessage(text=text)]

    pieces = split_response(
        text,
        enable=bool(sp.get("enable", True)),
        max_sentence_num=int(sp.get("max_sentence_num", 5)),
        max_chars_per_message=int(sp.get("max_chars_per_message", 120)),
        enable_kaomoji_protection=bool(sp.get("enable_kaomoji_protection", False)),
        enable_overflow_return_all=bool(sp.get("enable_overflow_return_all", True)),
        strip_split_punct=bool(sp.get("strip_split_punct", True)),
        rand=rng,
    )

    if typo_cfg.get("enable", True):
        gen = _get_typo_gen()
        pieces = [gen.create_typo_sentence(p, rand=rng) for p in pieces]

    # 阅读延迟：首条前的「看消息」时间（字数越多看得越久，带抖动）
    first_delay = 0.0
    if timing_cfg.get("enable", True) and incoming:
        first_delay = min(
            float(timing_cfg.get("base", 0.6)) + len(incoming) * float(timing_cfg.get("per_char", 0.02)),
            float(timing_cfg.get("cap", 4.0)),
        ) * rng.uniform(0.7, 1.3)

    out: List[OutboundMessage] = []
    for i, p in enumerate(pieces):
        # 首条带阅读延迟，后续按前一条字数模拟打字
        delay = first_delay if i == 0 else typing_delay(pieces[i - 1], rand=rng)
        out.append(OutboundMessage(text=p, delay=delay))
    return out
