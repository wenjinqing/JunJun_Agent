"""TTS 情绪映射：君君的真实心情 / LLM 指定情绪 -> 各后端情绪参数。

设计（2026-08-02 实测定案）：
- 豆包 Seed-TTS 2.0（vv/uranus 音色）实测支持 emotion + emotion_scale：
  happy/sad/angry 等值显著改变韵律（sad 拖长、happy 加快），非法值静默忽略
- GSV2P other_params.emotion 收中文词（旧配置「默认」），零成本接入
- 情绪来源优先级：LLM 显式指定（unified_tts emotion 参数）> 会话心情
  （mood_manager）> 全局自我心境；都没有 -> 不带情绪参数（默认语气）
"""

from typing import Optional, Tuple

from junjun_core.config import get_global_config

# 心情词 -> 豆包 emotion 码（多情感音色文档词表：
# 开心/悲伤/生气/惊讶/恐惧/厌恶/激动/冷漠/中性）
_MOOD_TO_DOUBAO: Tuple[Tuple[tuple, str], ...] = (
    (("开心", "高兴", "快乐", "甜", "满足", "幸福"), "happy"),
    (("兴奋", "激动", "期待", "惊喜", "得意"), "excited"),
    (("生气", "怒", "发火", "气死", "烦死"), "angry"),
    (("烦", "无语", "嫌弃", "冷漠"), "coldness"),
    (("难过", "伤心", "丧", "低落", "emo", "委屈", "郁闷", "哭"), "sad"),
    (("累", "疲", "困"), "tired"),
    (("惊讶", "震惊", "吓到"), "surprise"),
    (("害怕", "恐惧"), "fear"),
    (("厌恶", "恶心"), "hate"),
)

# LLM 显式情绪（中文自然词）-> (豆包码, 强度提升)——「发疯/气炸」这类拉满 scale
_LLM_EMOTION = {
    "开心": ("happy", 0), "高兴": ("happy", 0), "快乐": ("happy", 0),
    "兴奋": ("excited", 0), "激动": ("excited", 0), "得意": ("excited", 0),
    "生气": ("angry", 0), "愤怒": ("angry", 0), "发火": ("angry", 0),
    "发疯": ("angry", 1), "气炸": ("angry", 1), "暴走": ("angry", 1),
    "难过": ("sad", 0), "伤心": ("sad", 0), "委屈": ("sad", 0), "哭": ("sad", 0),
    "惊讶": ("surprise", 0), "震惊": ("surprise", 0),
    "害怕": ("fear", 0), "恐惧": ("fear", 0),
    "冷漠": ("coldness", 0), "无语": ("coldness", 0), "嫌弃": ("hate", 0),
    "温柔": ("tender", 0), "撒娇": ("tender", 0),
}

# 豆包码 -> GSV2P 中文情绪词（other_params.emotion；不认识的回「默认」）
_DOUBAO_TO_GSV2P = {
    "happy": "开心", "excited": "兴奋", "angry": "生气",
    "sad": "难过", "surprise": "惊讶", "fear": "害怕",
    "coldness": "冷漠", "hate": "嫌弃", "tender": "温柔",
}


def _cfg() -> dict:
    try:
        return get_global_config().raw.get("tts", {}) or {}
    except Exception:
        return {}


def emotion_enabled() -> bool:
    return bool(_cfg().get("enable_emotion", True))


def _base_scale() -> int:
    try:
        return max(1, min(5, int(_cfg().get("emotion_scale", 4))))
    except (TypeError, ValueError):
        return 4


def parse_llm_emotion(text: str) -> Optional[Tuple[str, int]]:
    """LLM 指定的中文情绪词 -> (豆包码, scale 提升)；不认识返回 None。"""
    text = (text or "").strip()
    if not text:
        return None
    for word, (code, boost) in _LLM_EMOTION.items():
        if word in text:
            return code, boost
    return None


def mood_to_doubao(mood: str) -> str:
    """心情文本（如「有点生气」「被夸了很得意」）-> 豆包 emotion 码；无匹配 ""。"""
    mood = (mood or "").strip()
    if not mood or mood == "平静":
        return ""
    for words, code in _MOOD_TO_DOUBAO:
        if any(w in mood for w in words):
            return code
    return ""


def doubao_to_gsv2p(code: str) -> str:
    return _DOUBAO_TO_GSV2P.get(code, "默认")


def resolve_emotion(chat_id: str = "", llm_emotion: str = "") -> Tuple[str, int]:
    """定案本次合成的 (豆包 emotion 码, scale)；不带情绪返回 ("", 0)。

    优先级：LLM 显式 > 会话心情 > 全局自我心境。功能关闭时返回 ("", 0)。
    """
    if not emotion_enabled():
        return "", 0
    scale = _base_scale()
    explicit = parse_llm_emotion(llm_emotion)
    if explicit:
        code, boost = explicit
        return code, min(5, scale + boost)
    mood = ""
    try:
        from junjun_express.mood import mood_manager
        mood = mood_manager.get_mood(chat_id) if chat_id else ""
        if not mood or mood == "平静":
            mood = mood_manager.get_self_mood()
    except Exception:
        pass
    code = mood_to_doubao(mood)
    return (code, scale) if code else ("", 0)
