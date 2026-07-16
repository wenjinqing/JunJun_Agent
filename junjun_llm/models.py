"""任务模型配置：按任务槽构造 ChatOpenAI。

对齐原 MaiBot model_task_config 语义：不同任务用不同模型/provider。
阶段 2 先落 gate / agent / utils 三槽，后续阶段补齐（vlm/embedding/lpmm_* 等）。
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import tomlkit
from langchain_openai import ChatOpenAI

from junjun_core.observability import get_logger

logger = get_logger("llm.models")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_CONFIG_PATH = PROJECT_ROOT / "config" / "model_config.toml"

# provider -> (base_url, api_key_env)
PROVIDERS: Dict[str, tuple] = {
    "deepseek": ("https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"),
    "siliconflow": ("https://api.siliconflow.cn/v1", "SILICONFLOW_API_KEY"),
    "bailian": ("https://dashscope.aliyuncs.com/compatible-mode/v1", "BAILIAN_API_KEY"),
}


@dataclass
class ModelSpec:
    provider: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 1024


@dataclass
class TaskSlot:
    """一个任务槽：按序 fallback 的模型列表。"""
    name: str
    specs: List[ModelSpec] = field(default_factory=list)


_slots: Optional[Dict[str, TaskSlot]] = None


def _load_slots() -> Dict[str, TaskSlot]:
    global _slots
    if _slots is not None:
        return _slots
    if not MODEL_CONFIG_PATH.exists():
        raise FileNotFoundError(f"缺少模型配置: {MODEL_CONFIG_PATH}")
    with open(MODEL_CONFIG_PATH, "r", encoding="utf-8") as f:
        data = tomlkit.parse(f.read()).unwrap()
    slots: Dict[str, TaskSlot] = {}
    for name, cfg in data.get("task", {}).items():
        specs = [
            ModelSpec(
                provider=m["provider"],
                model=m["model"],
                temperature=float(m.get("temperature", cfg.get("temperature", 0.7))),
                max_tokens=int(m.get("max_tokens", cfg.get("max_tokens", 1024))),
            )
            for m in cfg.get("models", [])
        ]
        slots[name] = TaskSlot(name=name, specs=specs)
    _slots = slots
    return slots


def _build_chat(spec: ModelSpec) -> ChatOpenAI:
    if spec.provider not in PROVIDERS:
        raise ValueError(f"未知 provider: {spec.provider}（可选 {list(PROVIDERS)}）")
    base_url, key_env = PROVIDERS[spec.provider]
    api_key = os.environ.get(key_env, "")
    if not api_key:
        raise ValueError(f"provider {spec.provider} 的 {key_env} 未设置")
    return ChatOpenAI(
        model=spec.model,
        base_url=base_url,
        api_key=api_key,
        temperature=spec.temperature,
        max_tokens=spec.max_tokens,
        timeout=60,
        max_retries=1,
    )


def get_chat_model(task: str) -> ChatOpenAI:
    """按任务槽取模型；首选不可用（key 缺失）时按序 fallback。"""
    slot = _load_slots().get(task)
    if slot is None or not slot.specs:
        raise ValueError(f"任务槽未配置: {task}")
    last_err: Optional[Exception] = None
    for spec in slot.specs:
        try:
            chat = _build_chat(spec)
            logger.debug(f"任务槽 [{task}] -> {spec.provider}/{spec.model}")
            return chat
        except ValueError as e:
            last_err = e
            logger.warning(f"任务槽 [{task}] 候选 {spec.provider}/{spec.model} 不可用: {e}")
    raise RuntimeError(f"任务槽 [{task}] 无可用模型") from last_err
