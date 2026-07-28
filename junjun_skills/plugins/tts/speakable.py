"""口播文本工程：把聊天文本清洗成「适合念出来」的口播稿。

TTS 合成器的输入质量直接决定听感。LLM/用户的聊天文本含有大量
「视觉上好看、念出来灾难」的成分：emoji、markdown、URL、占位符、
括号舞台指示、重复标点、颜文字。本模块统一清洗（tts/ja_tts 共用）：

- 占位符 [图片]/[表情]/[视频]/[语音]/[文件] -> 删（念出来出戏）
- URL/链接 -> 删（逐字符念网址是灾难）
- markdown 符号（` * # > ~）-> 删
- emoji / 颜文字 -> 删（TTS 读不出或乱读）
- 括号舞台指示（笑）（叹气）-> 删动作保留语气：中文全角括号整段删除
- 重复标点压缩（！！！->！），删不掉的孤立符号转停顿
"""

import re

# 占位符 [图片]/[表情]/[视频] 等方括号段
_PLACEHOLDER_RE = re.compile(r"\[[^\]]{0,20}\]")
# URL
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
# emoji（与 persona.strip_emoji 同范围）
_EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F02F\U0001F900-\U0001F9FF]+",
    flags=re.UNICODE,
)
# 常见颜文字（kaomoji）：括号包裹的符号表情，如 (´･ω･`)、(≧▽≦)、(T_T)
_KAOMOJI_RE = re.compile(
    r"[（(][^（）()一-鿿\w]{1,12}[´`｀°˚•·≧≦＞＜>_<ω∀▽△○●◎★☆♪〜~^\-_=;:'Tπ×]\S{0,12}[）)]"
)
# 全角/半角括号舞台指示：（笑）(点头)（叹气地说）——整段删除
_STAGE_RE = re.compile(r"[（(][^（）()]{1,10}[）)]")
# markdown 结构符号
_MD_RE = re.compile(r"[`#>*_~]{1,}")
# 重复标点压缩
_REPEAT_PUNCT_RE = re.compile(r"([！？!?。．…—~～])\1+")
# 连续空白/换行
_WS_RE = re.compile(r"\s+")


def make_speakable(text: str, max_len: int = 300) -> str:
    """清洗为口播稿。返回空串表示没有可念内容（调用方应放弃合成）。"""
    if not text:
        return ""
    t = text
    t = _URL_RE.sub("", t)
    t = _PLACEHOLDER_RE.sub("", t)
    t = _KAOMOJI_RE.sub("", t)
    t = _STAGE_RE.sub("", t)       # 舞台指示整段删（在颜文字之后，避免误删颜文字残留）
    t = _EMOJI_RE.sub("", t)
    t = _MD_RE.sub("", t)
    t = _REPEAT_PUNCT_RE.sub(r"\1", t)
    t = _WS_RE.sub("，", t)        # 换行/连续空白转停顿
    # 清理压缩后产生的粘连标点（，。/，！等取后者）
    t = re.sub(r"[，、]([。！？])", r"\1", t)
    t = re.sub(r"^[\W_]+", "", t)  # 开头孤立符号
    t = re.sub(r"[^\w。！？!?，、…—~～一-鿿぀-ヿ가-힯]+$", "", t)  # 结尾孤立符号（保留句读）
    t = t.strip("，")
    if not t:
        return ""
    return t[:max_len]
