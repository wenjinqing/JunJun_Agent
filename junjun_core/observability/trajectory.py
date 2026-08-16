"""追加式轨迹日志：本地结构化 run 轨迹（DSH 追记式会话日志/Trajectory 移植）。

data/trajectory/<YYYYMMDD>.jsonl 一日一文件（本机时间），每行一个事件：
{"ts": epoch, "kind": ..., "chat_id": ..., ...}。Langfuse 之外的本地兜底——
Langfuse 没配/挂掉时，排障不靠散文日志捞针（2026-08-15 幽灵事件/视频丢失/
身份幻觉三连排障催生；对照 DSH：一切运行信息进仅追加日志，按来源看轨迹）。

事件种类（v1）：inbound（收到消息）/ outbound（网关发出）/ agent_round
（决策轮结局：tier/工具名单/回复长度/沉默）/ tk_plan / tk_step / tk_done
（TaskKernel 计划/步骤/终态）。

铁律：emit 绝不抛异常（全路径 try 包住），绝不动生产库；data/ 天然
gitignored。开关 [observability] trajectory（默认开——每事件一行 JSONL，
文本字段调用方先截断，这里再兜底截 500）。
"""

import json
import time
from pathlib import Path

_DEFAULT_DIR = "data/trajectory"


def _enabled() -> bool:
    try:
        from junjun_core.config import get_global_config
        return bool(get_global_config().raw.get("observability", {})
                    .get("trajectory", True))
    except Exception:
        return True


def _dir() -> Path:
    try:
        from junjun_core.config import get_global_config
        d = get_global_config().raw.get("observability", {}).get("trajectory_dir")
        if d:
            return Path(str(d))
    except Exception:
        pass
    return Path(_DEFAULT_DIR)


def _cut(v, n: int = 500):
    """字符串字段兜底截断（调用方已按语义截过，这里防失控）。"""
    return v[:n] + "…" if isinstance(v, str) and len(v) > n else v


def emit(kind: str, chat_id: str = "", **fields) -> None:
    """追加一条事件。任何失败静默吞掉——轨迹是观测件，绝不许炸主流程。"""
    try:
        if not _enabled():
            return
        d = _dir()
        d.mkdir(parents=True, exist_ok=True)
        day = time.strftime("%Y%m%d", time.localtime())
        rec = {"ts": round(time.time(), 3), "kind": str(kind), "chat_id": chat_id}
        rec.update({k: _cut(v) for k, v in fields.items()})
        with open(d / f"{day}.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass
