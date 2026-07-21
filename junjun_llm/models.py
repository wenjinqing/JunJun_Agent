"""任务槽模型工厂：按任务构造 ChatOpenAI。

对齐原 MaiBot model_task_config 语义：
- config/model_config.toml 声明任务槽（gate/agent/utils/utils_small/vlm...）
- 每槽从 env 读 base_url / model / api_key（env 名按槽可配，默认同组 LLM_*）
- 每槽支持 [[task.X.models]] 多条目，用 LangChain 原生 with_fallbacks 顺序故障转移
  （对齐原 LLMRequest 故障转移语义：主模型挂了自动切下一个）
"""

import os
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import tomlkit
from langchain_openai import ChatOpenAI

from junjun_core.observability import get_logger

logger = get_logger("llm.models")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_CONFIG_PATH = PROJECT_ROOT / "config" / "model_config.toml"

# TCP keepalive：长连接防代理断链（原项目踩过 idle 断连坑）
_SOCKET_OPTIONS = [
    (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),
    (socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 60),
    (socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 120),
]


@dataclass
class ModelSpec:
    base_url: str
    model: str
    api_key: str
    temperature: float = 0.7
    max_tokens: int = 1024


@dataclass
class TaskSlot:
    name: str
    specs: List[ModelSpec]  # 顺序即 fallback 优先级；空列表 = 该槽未配置


_slots = None


def _spec_from(cfg: dict, defaults: dict) -> Optional[ModelSpec]:
    """从一条配置（槽级默认 + 条目级覆盖）构造 ModelSpec；env 不全返回 None。"""
    merged = {**defaults, **cfg}
    base_url = os.environ.get(merged.get("base_url_env", "LLM_BASE_URL"), "")
    model = os.environ.get(merged.get("model_env", "LLM_MODEL"), "")
    api_key = os.environ.get(merged.get("api_key_env", "LLM_API_KEY"), "")
    if not (base_url and model and api_key):
        return None
    return ModelSpec(
        base_url=base_url,
        model=model,
        api_key=api_key,
        temperature=float(merged.get("temperature", 0.7)),
        max_tokens=int(merged.get("max_tokens", 1024)),
    )


def _load_slots():
    global _slots
    if _slots is not None:
        return _slots
    if not MODEL_CONFIG_PATH.exists():
        raise FileNotFoundError(f"模型配置缺失: {MODEL_CONFIG_PATH}")
    with open(MODEL_CONFIG_PATH, "r", encoding="utf-8") as f:
        data = tomlkit.parse(f.read()).unwrap()
    slots = {}
    for name, cfg in data.get("task", {}).items():
        entries = cfg.pop("models", None) or [{}]  # 无 models 列表则槽级配置自身即唯一条目
        specs = [s for s in (_spec_from(e, cfg) for e in entries) if s]
        if not specs:
            logger.warning(f"任务槽 [{name}] 未配置完整（检查对应 env）")
        slots[name] = TaskSlot(name=name, specs=specs)
    _slots = slots
    return slots


def _build_chat(spec: ModelSpec) -> ChatOpenAI:
    return ChatOpenAI(
        model=spec.model,
        base_url=spec.base_url,
        api_key=spec.api_key,
        temperature=spec.temperature,
        max_tokens=spec.max_tokens,
        timeout=60,
        max_retries=1,
        http_socket_options=_SOCKET_OPTIONS,
    )


def get_chat_model(task: str):
    """取任务槽模型；多条目时返回带 with_fallbacks 的链（调用失败自动切下一个）。"""
    slot = _load_slots().get(task)
    if slot is None or not slot.specs:
        raise ValueError(f"任务槽未配置或不可用: {task}（检查 model_config.toml 与对应 env）")
    chat = _build_chat(slot.specs[0])
    if len(slot.specs) > 1:
        chat = chat.with_fallbacks([_build_chat(s) for s in slot.specs[1:]])
    suffix = f"（fallback ×{len(slot.specs) - 1}）" if len(slot.specs) > 1 else ""
    logger.debug(f"模型 [{task}] -> {slot.specs[0].base_url[:40]}... / {slot.specs[0].model}{suffix}")
    return chat


def reset_slots() -> None:
    """测试/热更配置用：清空槽缓存强制下次重读。"""
    global _slots
    _slots = None
