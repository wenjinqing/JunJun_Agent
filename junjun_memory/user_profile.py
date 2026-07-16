"""用户画像：对齐原 person_info/Person 语义。

- person_id = MD5(platform + user_id)
- memory_points: JSON 列表，元素 "分类:内容:权重"（如 "喜好:爱吃火锅:0.9"）
- 字段级 merge：同分类同内容更新权重，不覆盖全量（防并发写丢失）
- build_relation_block(): 拼 prompt 关系块
"""

import hashlib
import json
import time
from typing import Dict, List, Optional

from junjun_core.observability import get_logger

logger = get_logger("memory.profile")

MAX_POINTS = 50


def make_person_id(platform: str, user_id: str) -> str:
    return hashlib.md5(f"{platform}{user_id}".encode()).hexdigest()


def _parse_point(raw: str) -> Optional[Dict]:
    parts = raw.split(":", 2)
    if len(parts) != 3:
        return None
    try:
        return {"category": parts[0], "content": parts[1], "weight": float(parts[2])}
    except ValueError:
        return None


def _format_point(category: str, content: str, weight: float) -> str:
    return f"{category}:{content}:{weight:.2f}"


class UserProfileStore:
    """画像读写（peewee PersonInfo 表）。"""

    def get_or_create(self, platform: str, user_id: str, nickname: str = ""):
        from junjun_core.database import PersonInfo
        pid = make_person_id(platform, user_id)
        person = PersonInfo.get_or_none(PersonInfo.person_id == pid)
        if person is None:
            person = PersonInfo.create(
                person_id=pid, platform=platform, user_id=user_id,
                person_name=nickname, memory_points="[]",
            )
        return person

    def add_point(self, platform: str, user_id: str, category: str,
                  content: str, weight: float = 0.8, nickname: str = "") -> None:
        """字段级 merge：同分类同内容更新权重，否则追加；超上限淘汰最低权重。"""
        from junjun_core.database import PersonInfo, db
        pid = make_person_id(platform, user_id)
        with db.atomic():
            person = self.get_or_create(platform, user_id, nickname)
            try:
                points = json.loads(person.memory_points or "[]")
            except json.JSONDecodeError:
                points = []
            parsed = [p for p in (_parse_point(x) for x in points) if p]

            for p in parsed:
                if p["category"] == category and p["content"] == content:
                    p["weight"] = min(1.0, max(p["weight"], weight) + 0.05)  # 重复提及强化
                    break
            else:
                parsed.append({"category": category, "content": content, "weight": weight})

            if len(parsed) > MAX_POINTS:
                parsed.sort(key=lambda x: -x["weight"])
                parsed = parsed[:MAX_POINTS]

            person.memory_points = json.dumps(
                [_format_point(p["category"], p["content"], p["weight"]) for p in parsed],
                ensure_ascii=False,
            )
            if nickname and not person.person_name:
                person.person_name = nickname
            person.save()

    def set_name(self, platform: str, user_id: str, name: str) -> None:
        from junjun_core.database import db
        with db.atomic():
            person = self.get_or_create(platform, user_id)
            person.person_name = name
            person.save()

    def get_points(self, platform: str, user_id: str, *, top_k: int = 8) -> List[Dict]:
        from junjun_core.database import PersonInfo
        person = PersonInfo.get_or_none(PersonInfo.person_id == make_person_id(platform, user_id))
        if person is None:
            return []
        try:
            points = json.loads(person.memory_points or "[]")
        except json.JSONDecodeError:
            return []
        parsed = [p for p in (_parse_point(x) for x in points) if p]
        parsed.sort(key=lambda x: -x["weight"])
        return parsed[:top_k]

    def build_relation_block(self, platform: str, user_id: str, nickname: str = "") -> str:
        """拼 prompt 关系块；无画像返回空串。"""
        from junjun_core.database import PersonInfo
        person = PersonInfo.get_or_none(PersonInfo.person_id == make_person_id(platform, user_id))
        if person is None:
            return ""
        points = self.get_points(platform, user_id)
        if not points and not person.person_name:
            return ""
        name = person.person_name or nickname or user_id
        lines = [f"关于「{name}」你记得："]
        for p in points:
            lines.append(f"- {p['category']}: {p['content']}")
        return "\n".join(lines)


_store: Optional[UserProfileStore] = None


def get_profile_store() -> UserProfileStore:
    global _store
    if _store is None:
        _store = UserProfileStore()
    return _store
