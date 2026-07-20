"""????????????? ChatOpenAI?

??? MaiBot model_task_config ???
? model_config.toml ? base_url_env / model_env / api_key_env?
???? .env ??????????????? .env?
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import tomlkit
from langchain_openai import ChatOpenAI

from junjun_core.observability import get_logger

logger = get_logger("llm.models")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_CONFIG_PATH = PROJECT_ROOT / "config" / "model_config.toml"


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
    spec: Optional[ModelSpec] = None


_slots = None


def _load_slots():
    global _slots
    if _slots is not None:
        return _slots
    if not MODEL_CONFIG_PATH.exists():
        raise FileNotFoundError(f"??????: {MODEL_CONFIG_PATH}")
    with open(MODEL_CONFIG_PATH, "r", encoding="utf-8") as f:
        data = tomlkit.parse(f.read()).unwrap()
    slots = {}
    for name, cfg in data.get("task", {}).items():
        base_url_env = cfg.get("base_url_env", "LLM_BASE_URL")
        model_env = cfg.get("model_env", "LLM_MODEL")
        api_key_env = cfg.get("api_key_env", "LLM_API_KEY")
        base_url = os.environ.get(base_url_env, "")
        model = os.environ.get(model_env, "")
        api_key = os.environ.get(api_key_env, "")
        if not base_url or not model or not api_key:
            logger.warning(f"??? [{name}] ?????")
        slots[name] = TaskSlot(
            name=name,
            spec=ModelSpec(
                base_url=base_url,
                model=model,
                api_key=api_key,
                temperature=float(cfg.get("temperature", 0.7)),
                max_tokens=int(cfg.get("max_tokens", 1024)),
            ) if base_url and model and api_key else None,
        )
    _slots = slots
    return slots


def _build_chat(spec):
    return ChatOpenAI(
        model=spec.model,
        base_url=spec.base_url,
        api_key=spec.api_key,
        temperature=spec.temperature,
        max_tokens=spec.max_tokens,
        timeout=60,
        max_retries=1,
    )


def get_chat_model(task: str):
    slot = _load_slots().get(task)
    if slot is None or slot.spec is None:
        raise ValueError(f"????????????: {task}???? .env ?? LLM_BASE_URL / LLM_MODEL / LLM_API_KEY")
    chat = _build_chat(slot.spec)
    logger.debug(f"??? [{task}] -> {slot.spec.base_url[:40]}... / {slot.spec.model}")
    return chat
