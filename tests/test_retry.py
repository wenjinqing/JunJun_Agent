"""retry_async 重试助手测试。"""

import asyncio

import pytest

from junjun_core.retry import retry_async


class TestRetryAsync:
    @pytest.mark.asyncio
    async def test_first_try_success(self):
        calls = []

        async def fn():
            calls.append(1)
            return "ok"

        assert await retry_async(fn, label="t") == "ok"
        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_succeeds_on_third(self, monkeypatch):
        monkeypatch.setattr(asyncio, "sleep", _no_sleep(monkeypatch))
        calls = []

        async def fn():
            calls.append(1)
            if len(calls) < 3:
                raise ConnectionError("ECONNRESET")
            return "ok"

        assert await retry_async(fn, attempts=3, label="t") == "ok"
        assert len(calls) == 3

    @pytest.mark.asyncio
    async def test_exhausts_and_raises_last(self, monkeypatch):
        monkeypatch.setattr(asyncio, "sleep", _no_sleep(monkeypatch))
        calls = []

        async def fn():
            calls.append(1)
            raise ConnectionError(f"fail{len(calls)}")

        with pytest.raises(ConnectionError, match="fail3"):
            await retry_async(fn, attempts=3, label="t")
        assert len(calls) == 3

    @pytest.mark.asyncio
    async def test_programming_error_no_retry(self):
        calls = []

        async def fn():
            calls.append(1)
            raise ValueError("bad param")

        with pytest.raises(ValueError):
            await retry_async(fn, attempts=3, label="t")
        assert len(calls) == 1  # 确定性错误不重试

    @pytest.mark.asyncio
    async def test_retry_on_filter(self):
        calls = []

        async def fn():
            calls.append(1)
            raise PermissionError("auth")

        with pytest.raises(PermissionError):
            await retry_async(fn, attempts=3, label="t",
                              retry_on=(ConnectionError,))
        assert len(calls) == 1  # 不在 retry_on 里的异常直接抛

    @pytest.mark.asyncio
    async def test_backoff_delays(self, monkeypatch):
        delays = []

        async def _sleep(d):
            delays.append(d)

        monkeypatch.setattr(asyncio, "sleep", _sleep)

        async def fn():
            raise ConnectionError("x")

        with pytest.raises(ConnectionError):
            await retry_async(fn, attempts=3, base_delay=1.0, label="t")
        assert delays == [1.0, 2.0]  # 指数退避


def _no_sleep(monkeypatch):
    async def _sleep(d):
        pass
    return _sleep
