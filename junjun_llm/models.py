"""任务槽模型工厂：按任务构造 ChatOpenAI。

任务槽语义：
- config/model_config.toml 声明任务槽（gate/agent/utils/utils_small/vlm...）
- 每槽从 env 读 base_url / model / api_key（env 名按槽可配，默认同组 LLM_*）
- 每槽支持 [[task.X.models]] 多条目，用 LangChain 原生 with_fallbacks 顺序故障转移
  （主模型挂了自动切下一个）
"""

import os
import socket
from dataclasses import dataclass, field
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
    # 厂商特定请求参数（如硅基 ***REMOVED*** 关思考 thinking={"type":"disabled"}——
    # 2026-08-09 实测：默认开思考一句 hi 白烧 ~180 reasoning tokens（计费），
    # 延迟 1.0s->3.8s；高频闲聊槽纯浪费。enable_thinking=false 对 GLM 无效，
    # Qwen 系才认那个字段；思考 token 不占 max_tokens 配额（实测 max_tokens=50
    # 仍 reasoning=213 + 正文正常），所以关思考图的是费用和延迟）
    extra_body: dict = field(default_factory=dict)
    # 单跳请求超时/重试：默认 60s×4 次适合闲聊槽；思考型槽（thinker 规划）
    # 单次生成轻松超 60s——2026-08-13 生产实锤：thinker 两条腿 4×60s 全超时
    # 烧穿，后台规划整单超时蒸发。槽级/条目级配置 timeout/max_retries 覆盖。
    timeout: float = 60.0
    max_retries: int = 3


@dataclass
class TaskSlot:
    name: str
    specs: List[ModelSpec]  # 顺序即 fallback 优先级；空列表 = 该槽未配置


_slots = None
_slots_pool_gen = -1  # 槽缓存展开时的号池代数（-1=无关/未知）


def _maybe_rebuild_slots() -> None:
    """号池健康集合变了（巡检杀/复活 key、文件热更）-> 重建槽缓存剔除死腿。

    2026-08-06 实锤：fallback 链是启动时一次性展开的，巡检把欠费 key
    标死之后没人重建链——死 key 永远留在链上，每次调用先白撞 3 次重试
    才 fallback。同时给号池打 tick 心跳：主链从此不再依赖 ASR 等偶尔
    路径顺带触发懒巡检。
    """
    if _slots is None:
        return
    try:
        from junjun_llm.key_pool import sf_pool
    except Exception:
        return
    sf_pool.tick()
    if _slots_pool_gen != sf_pool.generation:
        logger.info(f"号池状态变化（gen {_slots_pool_gen} -> {sf_pool.generation}），"
                    f"重建模型 fallback 链")
        reset_slots()


def _spec_from(cfg: dict, defaults: dict) -> Optional[ModelSpec]:
    """从一条配置（槽级默认 + 条目级覆盖）构造 ModelSpec；env 不全返回 None。"""
    merged = {**defaults, **cfg}
    base_url = os.environ.get(merged.get("base_url_env", "DS_BASE_URL"), "")
    model = os.environ.get(merged.get("model_env", "DS_MODEL"), "")
    api_key = os.environ.get(merged.get("api_key_env", "DEEPSEEK_API_KEY"), "")
    if not (base_url and model and api_key):
        return None
    return ModelSpec(
        base_url=base_url,
        model=model,
        api_key=api_key,
        temperature=float(merged.get("temperature", 0.7)),
        max_tokens=int(merged.get("max_tokens", 1024)),
        extra_body=dict(merged.get("extra_body") or {}),
        timeout=float(merged.get("timeout", 60.0)),
        max_retries=int(merged.get("max_retries", 3)),
    )


# api_key_env 写这个值 = 走硅基流动号池（junjun_llm/key_pool.py）
_POOL_MARKER = "SF_POOL"


def _pool_specs(cfg: dict, defaults: dict) -> List[ModelSpec]:
    """号池条目展开：每个健康 key 一条 ModelSpec（fallback 链的腿）。

    不同槽调用时起始 key 轮转（负载摊开）；腿数上限防巨型链。
    池空时返回空列表（该条目缺席，其余静态条目照常兜底）。
    """
    merged = {**defaults, **cfg}
    base_url = os.environ.get(merged.get("base_url_env", "SF_LLM_BASE_URL"), "")
    model = os.environ.get(merged.get("model_env", "SF_LLM_MODEL"), "")
    if not (base_url and model):
        logger.warning("SF_POOL 条目缺 base_url/model env，跳过")
        return []
    from junjun_llm.key_pool import sf_pool, _cfg as _pool_cfg
    keys = sf_pool.healthy_keys()
    if not keys:
        logger.warning("SF_POOL 号池为空（检查 data/sf_keys.txt 与 key 余额）")
        try:
            from junjun_core.alerting import note_pool_empty
            note_pool_empty()
        except Exception:
            pass
        return []
    max_legs = int(_pool_cfg().get("max_legs", 10))
    keys = keys[:max_legs]
    logger.info(f"号池展开: {len(keys)} 条腿 -> {model}")
    return [ModelSpec(
        base_url=base_url, model=model, api_key=k,
        temperature=float(merged.get("temperature", 0.7)),
        max_tokens=int(merged.get("max_tokens", 1024)),
        extra_body=dict(merged.get("extra_body") or {}),
        timeout=float(merged.get("timeout", 60.0)),
        max_retries=int(merged.get("max_retries", 3)),
    ) for k in keys]


def _load_slots():
    global _slots, _slots_pool_gen
    if _slots is not None:
        return _slots
    if not MODEL_CONFIG_PATH.exists():
        raise FileNotFoundError(f"模型配置缺失: {MODEL_CONFIG_PATH}")
    with open(MODEL_CONFIG_PATH, "r", encoding="utf-8") as f:
        data = tomlkit.parse(f.read()).unwrap()
    slots = {}
    for name, cfg in data.get("task", {}).items():
        entries = cfg.pop("models", None) or [{}]  # 无 models 列表则槽级配置自身即唯一条目
        specs = []
        for e in entries:
            if {**cfg, **e}.get("api_key_env") == _POOL_MARKER:
                specs.extend(_pool_specs(e, cfg))  # 号池条目 -> N 条腿
            else:
                s = _spec_from(e, cfg)
                if s:
                    specs.append(s)
        if not specs:
            logger.warning(f"任务槽 [{name}] 未配置完整（检查对应 env）")
        slots[name] = TaskSlot(name=name, specs=specs)
    _slots = slots
    try:
        from junjun_llm.key_pool import sf_pool
        _slots_pool_gen = sf_pool.generation   # 记录展开时的号池代数
    except Exception:
        _slots_pool_gen = -1
    return slots


def _build_chat(spec: ModelSpec) -> ChatOpenAI:
    return ChatOpenAI(
        model=spec.model,
        base_url=spec.base_url,
        api_key=spec.api_key,
        temperature=spec.temperature,
        max_tokens=spec.max_tokens,
        timeout=spec.timeout,
        max_retries=spec.max_retries,  # 503 限流时自动重试（指数退避），避免 L2 gate 直接兜底 no_reply
        http_socket_options=_SOCKET_OPTIONS,
        extra_body=spec.extra_body or None,
    )


def get_chat_model(task: str):
    """取任务槽模型；多条目时返回带 with_fallbacks 的链（调用失败自动切下一个）。"""
    _maybe_rebuild_slots()
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
