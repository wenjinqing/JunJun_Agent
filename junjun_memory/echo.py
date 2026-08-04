"""复读检测：归一化 + 相似度判定（agent 出口拦截与短期记忆去重共用）。

背景（2026-08-04 用户反馈）：bot 反复用同一话术，每条复读落进短期记忆后，
下一轮 context 里同一句话出现 N 次，模型把「自己老说这句」当成说话习惯
继续复读——上下文自我污染的正反馈循环。prompt 层劝不住，必须代码层
确定性处理：输入端渲染去重 + 出口端撞车拦截（agent.py echo guard）。

本模块放 junjun_memory 而非 junjun_agent/loop：short_term 要用它，
而 junjun_agent 依赖 junjun_memory，反向放会循环导入。
"""

import re
from difflib import SequenceMatcher
from typing import List, Optional

_WS_PUNCT_RE = re.compile(r"[\s　，。！？!?、~…—\-·．.\"'“”‘’（）()【】\[\]<>《》:；;]+")
_EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F02F\U0001F900-\U0001F9FF]+",
    flags=re.UNICODE,
)

_MIN_LEN = 4       # 归一化后短于 4 字不参与复读判定（「嗯嗯」「好耶」天然会重复）
_CONTAIN_LEN = 6   # 较短一方达到 6 字且被对方完整包含即判复读


def normalize_echo(text: str) -> str:
    """比较用归一化：去 emoji/标点/空白，小写。"""
    t = _EMOJI_RE.sub("", text or "")
    t = _WS_PUNCT_RE.sub("", t)
    return t.lower()


def is_echo(new_text: str, recent_texts: List[str], *,
            similarity: float = 0.85) -> Optional[str]:
    """新文本与近期发言撞车 -> 返回撞车的历史原文；否则 None。

    判定三档：归一化全等 / 6 字以上一方被另一方完整包含 / 相似度过阈。
    """
    new = normalize_echo(new_text)
    if len(new) < _MIN_LEN:
        return None
    for old_text in recent_texts:
        old = normalize_echo(old_text)
        if len(old) < _MIN_LEN:
            continue
        if new == old:
            return old_text
        if min(len(new), len(old)) >= _CONTAIN_LEN and (new in old or old in new):
            return old_text
        if SequenceMatcher(None, new, old).ratio() >= similarity:
            return old_text
    return None
