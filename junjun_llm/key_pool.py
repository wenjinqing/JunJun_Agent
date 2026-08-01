"""硅基流动 key 号池：多 key 轮用 + 余额不足自动丢弃。

用法：
- key 放 data/sf_keys.txt（一行一个，# 开头注释；data/ 已 gitignore，不会进仓库）
- model_config.toml 里条目写 api_key_env = "SF_POOL"，即从池取 key 展开成
  fallback 链（某 key 限流/失效 LangChain 自动换下一条；不同任务槽起始 key
  轮转，负载摊开）
- 文件 mtime 变化热加载，不用重启

余额巡检（/v1/user/info + 探针两阶段）：
- 余额 >= min_balance -> 健康
- 余额不足不直接杀：发 1-token 探针（代金券账号余额恒为 0 但能用，
  2026-08 实测），探针 402/401/余额不足文案才判死，429 算活（有额度）
- 网络故障/超时 -> 保持原状态（宁可用错，不可错杀）
- 每 check_hours 懒复查一次（用到池时才触发，不占调度器）；
  复查覆盖文件里全部 key——手动 top up 的 key 下一轮自动复活
"""

import asyncio
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import httpx

from junjun_core.observability import get_logger

logger = get_logger("llm.key_pool")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
# 探针默认模型（SF_LLM_MODEL 未配时用）：非 Pro 前缀全 tier 可调，付费模型探针才有意义
_DEFAULT_PROBE_MODEL = "Qwen/Qwen3.5-397B-A17B"


def _cfg() -> dict:
    try:
        from junjun_core.config import get_global_config
        return get_global_config().raw.get("key_pool", {}) or {}
    except Exception:
        return {}


class SFKeyPool:
    """号池。测试可独立实例化（传 path），生产用模块级单例 sf_pool。"""

    def __init__(self, path: Optional[Path] = None):
        self._path = path or Path(
            os.environ.get("SF_KEY_POOL_FILE") or PROJECT_ROOT / "data" / "sf_keys.txt")
        self._mtime: float = -1.0
        self._keys: List[str] = []          # 文件里的全部 key（有序）
        self._dead: Dict[str, str] = {}     # key -> 丢弃原因（低余额/无效）
        self._checked_at: float = 0.0       # 上次余额巡检时间（0=从未）
        self._rr = 0                        # 轮转游标
        self._refreshing = False

    # ---------------------------------------------------------------- key 文件

    def _reload_if_changed(self) -> None:
        try:
            mtime = self._path.stat().st_mtime
        except OSError:
            if self._keys:
                logger.warning(f"号池文件消失: {self._path}，按空池处理")
            self._keys, self._mtime = [], 0.0
            return
        if mtime == self._mtime:
            return
        keys = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                keys.append(line)
        self._keys = keys
        self._dead = {k: v for k, v in self._dead.items() if k in keys}  # 文件删掉的同步清
        self._mtime = mtime
        logger.info(f"号池加载: {len(keys)} 个 key（{self._path.name}）")

    # ---------------------------------------------------------------- 对外

    def healthy_keys(self) -> List[str]:
        """健康 key 列表（起始位置轮转，调用方各拿一段不同的链）。空池返回 []。"""
        self._reload_if_changed()
        self._maybe_refresh()
        alive = [k for k in self._keys if k not in self._dead]
        if not alive:
            return []
        rot = self._rr % len(alive)
        self._rr += 1
        return alive[rot:] + alive[:rot]

    def status(self) -> dict:
        """WebUI/排查用：总数/健康/丢弃明细。"""
        self._reload_if_changed()
        return {"total": len(self._keys),
                "healthy": len([k for k in self._keys if k not in self._dead]),
                "dead": {k[:8] + "...": why for k, why in self._dead.items()},
                "checked_at": self._checked_at}

    # ---------------------------------------------------------------- 余额巡检

    def _maybe_refresh(self) -> None:
        hours = float(_cfg().get("check_hours", 2))
        if self._refreshing or (time.time() - self._checked_at) < hours * 3600:
            return
        if not self._keys:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # 同步上下文（测试/脚本）——保持乐观，等异步环境再查
        self._refreshing = True
        loop.create_task(self.refresh(), name="sf-key-pool-refresh")

    async def refresh(self) -> None:
        """全量余额巡检：死 key 丢弃，复活的 key 回池。任何单 key 失败不影响其他。"""
        self._reload_if_changed()
        base_url = os.environ.get("SF_LLM_BASE_URL") or _DEFAULT_BASE_URL
        min_balance = float(_cfg().get("min_balance", 0.5))
        try:
            keys = list(self._keys)  # 快照：巡检期间热加载不改本轮判定
            async with httpx.AsyncClient(timeout=8.0) as client:
                results = await asyncio.gather(
                    *(self._check_key(client, base_url, k) for k in keys),
                    return_exceptions=True)
            dead: Dict[str, str] = {}
            for key, res in zip(keys, results):
                if isinstance(res, Exception):
                    if key in self._dead:  # 网络层炸了：保持原状态
                        dead[key] = self._dead[key]
                    continue
                state, reason = res
                if state == "dead":
                    dead[key] = reason
                elif state == "unknown" and key in self._dead:
                    dead[key] = self._dead[key]
            revived = [k[:8] for k in self._dead if k not in dead]
            dropped = [k[:8] for k in dead if k not in self._dead]
            self._dead = dead
            self._checked_at = time.time()
            if dropped:
                logger.warning(f"号池丢弃 {len(dropped)} 个 key（余额不足/无效）: {dropped}")
            if revived:
                logger.info(f"号池复活 {len(revived)} 个 key: {revived}")
            logger.info(f"号池巡检完成: {len(self._keys) - len(dead)}/{len(self._keys)} 健康"
                        f"（阈值 ¥{min_balance}）")
        except Exception as e:
            logger.warning(f"号池巡检异常（保持原状态）: {type(e).__name__}: {e}")
        finally:
            self._refreshing = False

    async def _check_key(self, client: httpx.AsyncClient, base_url: str, key: str):
        """单 key 检查 -> ("alive"/"dead"/"unknown", 原因)。异常向外抛（gather 兜住）。

        两阶段：先查余额（付费账号）；余额不足不直接杀——发 1-token 探针
        （代金券账号 /user/info 恒为 ¥0 但调用照跑），探针确认没钱才判死。
        """
        base = base_url.rstrip("/")
        headers = {"Authorization": f"Bearer {key}"}
        resp = await client.get(f"{base}/user/info", headers=headers)
        if resp.status_code == 401:
            return "dead", "key 无效(401)"
        if resp.status_code != 200:
            return "unknown", f"http {resp.status_code}"
        data = (resp.json() or {}).get("data") or {}
        raw = data.get("totalBalance", data.get("balance", "0"))
        try:
            balance = float(raw)
        except (TypeError, ValueError):
            return "unknown", "余额解析失败"
        min_balance = float(_cfg().get("min_balance", 0.5))
        if balance >= min_balance:
            return "alive", f"¥{balance:.2f}"

        # 第二阶段：代金券探针（余额为 0 不代表不能调用）
        probe_model = os.environ.get("SF_LLM_MODEL") or _DEFAULT_PROBE_MODEL
        probe = await client.post(
            f"{base}/chat/completions", headers=headers,
            json={"model": probe_model, "max_tokens": 1,
                  "messages": [{"role": "user", "content": "hi"}]})
        if probe.status_code == 200:
            return "alive", "代金券可用"
        if probe.status_code == 429:
            return "alive", "限流中(有额度)"
        text = (probe.text or "")[:200].lower()
        if probe.status_code in (401, 402, 403) or \
                any(w in text for w in ("insufficient", "balance", "余额", "quota")):
            return "dead", f"余额/代金券耗尽({probe.status_code})"
        return "unknown", f"探针 http {probe.status_code}"


sf_pool = SFKeyPool()
