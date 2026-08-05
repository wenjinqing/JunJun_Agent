"""TaskKernel 数据结构：步骤图是代码侧状态机，不是注入文本（方案 §4.3）。

与旧 PlanMiddleware 的本质区别：步骤的完成度由代码按验证结果推进，
失败由代码决定 retry/replan/abort——不靠模型自己数 ToolMessage。
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# 步骤动作白名单外的特殊动作：LLM 直接合成文本（不调用工具）
SYNTH_ACTION = "llm_synthesize"

_VERIFY_KINDS = ("tool_ok", "schema", "llm_judge", "none")


@dataclass
class Step:
    id: str
    action: str                      # 注册表工具名 或 llm_synthesize
    desc: str                        # 这一步要做什么（给执行器的自然语言指令）
    args_hint: dict = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)
    verify: str = "tool_ok"          # tool_ok / llm_judge / none
    status: str = "pending"          # pending/running/done/failed/skipped
    result: str = ""                 # 产出摘要（大产出的全文不落这里）
    error: str = ""


@dataclass
class TaskPlan:
    goal: str
    chat_id: str
    user_id: str = ""
    steps: List[Step] = field(default_factory=list)
    plan_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    state: str = "planning"          # planning/running/reporting/done/failed
    attempts: Dict[str, int] = field(default_factory=dict)   # step_id -> 已试次数
    replans: int = 0                 # 已重规划次数（上限见 [task_kernel] max_replans）
    created_ts: float = field(default_factory=time.time)
    deadline_ts: float = 0.0
    note: str = ""                   # 失败原因等备注

    # ---------- 持久化 ----------
    def to_dict(self) -> dict:
        return {
            "plan_id": self.plan_id, "goal": self.goal, "chat_id": self.chat_id,
            "user_id": self.user_id, "state": self.state, "attempts": self.attempts,
            "replans": self.replans, "created_ts": self.created_ts,
            "deadline_ts": self.deadline_ts, "note": self.note,
            "steps": [vars(s) for s in self.steps],
        }

    @staticmethod
    def from_dict(d: dict) -> "TaskPlan":
        p = TaskPlan(goal=d["goal"], chat_id=d["chat_id"], user_id=d.get("user_id", ""))
        p.plan_id = d.get("plan_id", p.plan_id)
        p.state = d.get("state", "planning")
        p.attempts = d.get("attempts", {})
        p.replans = int(d.get("replans", 0))
        p.created_ts = float(d.get("created_ts", time.time()))
        p.deadline_ts = float(d.get("deadline_ts", 0.0))
        p.note = d.get("note", "")
        p.steps = [Step(**s) for s in d.get("steps", [])]
        return p

    # ---------- 状态查询 ----------
    def ready_steps(self) -> List[Step]:
        done = {s.id for s in self.steps if s.status == "done"}
        return [s for s in self.steps
                if s.status == "pending" and all(d in done for d in s.depends_on)]

    def summary_lines(self) -> List[str]:
        mark = {"done": "✓", "failed": "✗", "running": "…"}.get
        return [f"{mark(s.status, '·')} {s.desc}" for s in self.steps]


def parse_plan(payload: dict, *, goal: str, chat_id: str, user_id: str,
               valid_actions: set, max_steps: int = 6) -> Optional[TaskPlan]:
    """规划器 JSON 产出 -> TaskPlan；非法步骤丢弃，全废则 None（回退对话通道）。

    防御点：LLM 会编不存在的工具、会漏 depends_on、verify 会乱写——
    全部在入口清洗，执行器只面对合法计划。
    """
    raw_steps = payload.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        return None
    steps: List[Step] = []
    known_ids = set()
    for i, rs in enumerate(raw_steps[:max_steps]):
        if not isinstance(rs, dict):
            continue
        action = str(rs.get("action") or "").strip()
        if action != SYNTH_ACTION and action not in valid_actions:
            continue  # 编出来的工具，丢
        sid = str(rs.get("id") or f"s{i + 1}")
        if sid in known_ids:
            continue
        known_ids.add(sid)
        verify = str(rs.get("verify") or "tool_ok")
        if verify not in _VERIFY_KINDS:
            verify = "tool_ok"
        deps = [d for d in (rs.get("depends_on") or []) if d in known_ids]  # 只允许前向依赖
        steps.append(Step(
            id=sid, action=action,
            desc=str(rs.get("desc") or rs.get("description") or action)[:120],
            args_hint=rs.get("args_hint") if isinstance(rs.get("args_hint"), dict) else {},
            depends_on=deps, verify=verify,
        ))
    if not steps:
        return None
    return TaskPlan(goal=goal[:200], chat_id=chat_id, user_id=user_id, steps=steps)
