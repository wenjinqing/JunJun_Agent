"""TTS 情绪风格：君君的真实心情 / LLM 指定情绪 -> 各后端情绪参数。

2026-08-02 实测定案（F0 音高追踪 + 时长矩阵，5 音色 × 4 情绪）：
- 账号只有 seed-tts-2.0 资源（1.0/ICL 均 403），真·多情感音色（_emo_v2_mars）
  在此资源下报 resource ID mismatched，不可用
- vv 等 uranus 音色对 audio_params.emotion 有响应但不按类别区分
  （angry/happy/sad 音高同向漂移）——单靠 emotion 参数「听不出情绪」
- 可区分情绪 = 参数组合：emotion（音高色彩）+ speech_rate（快慢）+
  loudness_rate（轻重）。愤怒=快+响+飘，难过=慢+轻+平——速率/音量是
  全音色通用的文档参数，叠加后 categorical 可听
- 情绪来源优先级：LLM 显式指定 > 会话心情 > 全局自我心境；都没有 -> 默认语气
"""

from typing import Dict, Optional, Tuple

from junjun_core.config import get_global_config

# 官方 9 个 emotion 码：happy sad angry surprised fear hate excited coldness neutral
# （非法值服务端静默忽略；scale 1-5，默认 4，超界会空输出——ja_tts 层已钳制）
#
# 风格预设：emotion 码 + 语速/音量微调（speech_rate/loudness_rate 范围 [-50,100]）
_STYLES: Dict[str, dict] = {
    "happy":     {"emotion": "happy",     "speech_rate": 10,  "loudness_rate": 5},
    "excited":   {"emotion": "excited",   "speech_rate": 15,  "loudness_rate": 5},
    "angry":     {"emotion": "angry",     "speech_rate": 20,  "loudness_rate": 10},
    "crazy":     {"emotion": "angry",     "speech_rate": 30,  "loudness_rate": 15},  # 发疯：拉满
    "sad":       {"emotion": "sad",       "speech_rate": -15, "loudness_rate": -10},
    "surprised": {"emotion": "surprised", "speech_rate": 10,  "loudness_rate": 5},
    "fear":      {"emotion": "fear",      "speech_rate": 10,  "loudness_rate": -5},
    "hate":      {"emotion": "hate",      "speech_rate": -5,  "loudness_rate": -5},
    "coldness":  {"emotion": "coldness",  "speech_rate": -5,  "loudness_rate": -5},
    "tender":    {"emotion": "",          "speech_rate": -10, "loudness_rate": -5},  # 温柔：慢而轻
}

# 心情词 -> 风格名
_MOOD_TO_STYLE: Tuple[Tuple[tuple, str], ...] = (
    (("开心", "高兴", "快乐", "甜", "满足", "幸福"), "happy"),
    (("兴奋", "激动", "期待", "惊喜", "得意"), "excited"),
    (("气炸", "发疯", "暴走"), "crazy"),
    (("生气", "怒", "发火", "气死", "烦死"), "angry"),
    (("烦", "无语", "嫌弃", "冷漠"), "coldness"),
    (("难过", "伤心", "丧", "低落", "emo", "委屈", "郁闷", "哭"), "sad"),
    (("累", "疲", "困"), "sad"),
    (("惊讶", "震惊", "吓到"), "surprised"),
    (("害怕", "恐惧"), "fear"),
    (("厌恶", "恶心"), "hate"),
    (("温柔", "撒娇", "软糯"), "tender"),
)

# LLM 显式情绪（中文自然词）-> 风格名
_LLM_EMOTION = {
    "开心": "happy", "高兴": "happy", "快乐": "happy",
    "兴奋": "excited", "激动": "excited", "得意": "excited",
    "发疯": "crazy", "气炸": "crazy", "暴走": "crazy",
    "生气": "angry", "愤怒": "angry", "发火": "angry",
    "难过": "sad", "伤心": "sad", "委屈": "sad", "哭": "sad",
    "惊讶": "surprised", "震惊": "surprised",
    "害怕": "fear", "恐惧": "fear",
    "冷漠": "coldness", "无语": "coldness", "嫌弃": "hate",
    "温柔": "tender", "撒娇": "tender",
}

# 风格名 -> GSV2P 中文情绪词（other_params.emotion；不认识的回「默认」）
_STYLE_TO_GSV2P = {
    "happy": "开心", "excited": "兴奋", "angry": "生气", "crazy": "生气",
    "sad": "难过", "surprised": "惊讶", "fear": "害怕",
    "coldness": "冷漠", "hate": "嫌弃", "tender": "温柔",
}

# 参数硬边界（文档：speech_rate/loudness_rate [-50,100]，emotion_scale [1,5]）
_RATE_MIN, _RATE_MAX = -50, 100


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


def _clamp_rate(v: int) -> int:
    return max(_RATE_MIN, min(_RATE_MAX, int(v)))


def _style_dict(name: str) -> dict:
    """风格名 -> 完整参数 dict（emotion/emotion_scale/speech_rate/loudness_rate）。"""
    preset = _STYLES[name]
    return {
        "style": name,
        "emotion": preset["emotion"],
        "emotion_scale": 5 if name == "crazy" else _base_scale(),
        "speech_rate": _clamp_rate(preset["speech_rate"]),
        "loudness_rate": _clamp_rate(preset["loudness_rate"]),
    }


def parse_llm_emotion(text: str) -> Optional[str]:
    """LLM 指定的中文情绪词 -> 风格名；不认识返回 None。"""
    text = (text or "").strip()
    if not text:
        return None
    for word, style in _LLM_EMOTION.items():
        if word in text:
            return style
    return None


def mood_to_style(mood: str) -> str:
    """心情文本（如「有点生气」「被夸了很得意」）-> 风格名；无匹配 ""。"""
    mood = (mood or "").strip()
    if not mood or mood == "平静":
        return ""
    for words, style in _MOOD_TO_STYLE:
        if any(w in mood for w in words):
            return style
    return ""


def style_to_gsv2p(style: str) -> str:
    return _STYLE_TO_GSV2P.get(style, "默认")


def resolve_emotion(chat_id: str = "", llm_emotion: str = "") -> Optional[dict]:
    """定案本次合成的情绪风格 dict；不带情绪返回 None（默认语气）。

    返回 {"style","emotion","emotion_scale","speech_rate","loudness_rate"}。
    优先级：LLM 显式 > 会话心情 > 全局自我心境。功能关闭时返回 None。
    """
    if not emotion_enabled():
        return None
    style = parse_llm_emotion(llm_emotion)
    if not style:
        mood = ""
        try:
            from junjun_express.mood import mood_manager
            mood = mood_manager.get_mood(chat_id) if chat_id else ""
            if not mood or mood == "平静":
                mood = mood_manager.get_self_mood()
        except Exception:
            pass
        style = mood_to_style(mood)
    return _style_dict(style) if style else None
