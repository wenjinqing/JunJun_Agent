"""Agent 技能手册（md skills）：给君君本人看的场景化操作指南。

与 Claude Code 的 skill 同一模式：
- 手册是 md 文件，放 junjun_skills/agent_skills/<name>.md，frontmatter 带 name/when
- system prompt 里只放索引（名字 + 何时用），不占每轮 context
- 模型判断场景命中时调 use_skill 工具取回全文，照着手册做

与工具 docstring 的分工：docstring 告诉模型「什么时候选这个工具」，
手册告诉模型「一整个场景从头到尾怎么办」（多个工具怎么串、话怎么说、
坑在哪）。手册写的是流程与分寸，不是台词——具体句子照样会被复读。
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SKILLS_DIR = Path(__file__).resolve().parent / "agent_skills"

_FM_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)

# (mtime_scan_key, skills) 简单缓存：目录内容变化才重扫
_cache: Optional[Tuple[int, Dict[str, dict]]] = None


def _parse(path: Path) -> Optional[dict]:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return None
    m = _FM_RE.match(text)
    if not m:
        return None
    meta = {}
    for line in m.group(1).splitlines():
        k, sep, v = line.partition(":")
        if sep:
            meta[k.strip()] = v.strip()
    name = meta.get("name") or path.stem
    when = meta.get("when") or ""
    body = text[m.end():].strip()
    if not body:
        return None
    return {"name": name, "when": when, "body": body}


def load_skills() -> Dict[str, dict]:
    """扫描手册目录，{name: {name, when, body}}。目录不存在/为空返回 {}。"""
    global _cache
    if not SKILLS_DIR.exists():
        return {}
    files = sorted(SKILLS_DIR.glob("*.md"))
    key = hash(tuple((f.name, f.stat().st_mtime_ns) for f in files))
    if _cache and _cache[0] == key:
        return _cache[1]
    skills = {}
    for f in files:
        s = _parse(f)
        if s:
            skills[s["name"]] = s
    _cache = (key, skills)
    return skills


def skill_index() -> str:
    """system prompt 用的索引块（名字 + 何时用），无手册时返回空串。"""
    skills = load_skills()
    if not skills:
        return ""
    lines = ["【技能包】这些场景你有写好的经验手册，命中时先调 use_skill 看一遍再动手："]
    for s in skills.values():
        lines.append(f"- {s['name']}：{s['when']}")
    return "\n".join(lines)


def get_skill(name: str) -> Optional[str]:
    """取手册全文（frontmatter 之外的正文）。不存在返回 None。"""
    s = load_skills().get((name or "").strip())
    return s["body"] if s else None


def skill_names() -> List[str]:
    return sorted(load_skills())
