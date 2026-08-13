"""httpx 统一工厂测试（2026-08-13 审查 P2）：默认不穿环境代理，显式才穿。"""

import httpx
import pytest

from junjun_core.http_client import make_async_client, make_client


class TestTrustEnvDefault:
    def test_default_ignores_env_proxy(self, monkeypatch):
        """localhost 杀手场景：环境里有代理变量时默认也不能走
        （2026-08-13 Langfuse 脚本 502 实锤同类）。"""
        monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:7890")
        monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7890")
        assert dict(make_client()._mounts) == {}
        assert dict(make_async_client()._mounts) == {}

    def test_opt_in_reads_env_proxy(self, monkeypatch):
        monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7890")
        assert make_client(trust_env=True)._mounts, "显式 trust_env=True 必须生效"
        assert make_async_client(trust_env=True)._mounts

    def test_kwargs_passthrough(self):
        c = make_async_client(timeout=3.0, headers={"X-Test": "1"})
        assert c.timeout.read == 3.0
        assert c.headers["X-Test"] == "1"
