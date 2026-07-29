"""图片预热识图测试：发图 -> 再 @君君看 场景，Agent 不再看不到图。"""

import asyncio

import pytest

from junjun_memory import vision


@pytest.fixture(autouse=True)
def _clean():
    vision._RECENT.clear()
    vision._PENDING.clear()
    yield
    vision._RECENT.clear()
    vision._PENDING.clear()


def _fake_model(track: dict, desc: str = "一只猫"):
    class _M:
        async def ainvoke(self, msgs):
            track["calls"] = track.get("calls", 0) + 1
            await asyncio.sleep(0.01)  # 模拟 VLM 耗时

            class R:
                content = desc
            return R()
    return _M()


def _unique_bytes(tag: str) -> bytes:
    import uuid
    return b"\x89PNG prewarm-" + tag.encode() + uuid.uuid4().bytes


class TestSharedTask:
    @pytest.mark.asyncio
    async def test_concurrent_describe_shares_task(self, monkeypatch):
        """预热与回复路径并发：同一 url 只调一次 VLM。"""
        track = {}
        monkeypatch.setattr(vision, "_download",
                            lambda url: asyncio.sleep(0, result=_unique_bytes("shared")))
        model = _fake_model(track)
        results = await asyncio.gather(
            vision.describe_images(["http://x/a.png"], model=model),
            vision.describe_images(["http://x/a.png"], model=model),
        )
        assert track["calls"] == 1
        assert results[0]["http://x/a.png"] == "一只猫"
        assert results[1]["http://x/a.png"] == "一只猫"

    @pytest.mark.asyncio
    async def test_multiple_images_parallel(self, monkeypatch):
        """多张图并行识图（不是串行）。"""
        track = {}
        monkeypatch.setattr(vision, "_download",
                            lambda url: asyncio.sleep(0, result=_unique_bytes(url)))
        model = _fake_model(track)
        out = await vision.describe_images(
            ["http://x/1.png", "http://x/2.png", "http://x/3.png"], model=model)
        assert track["calls"] == 3
        assert all(v == "一只猫" for v in out.values())


class TestPrewarm:
    @pytest.mark.asyncio
    async def test_prewarm_records_recent(self, monkeypatch):
        monkeypatch.setattr(vision, "_download",
                            lambda url: asyncio.sleep(0, result=_unique_bytes("pw")))
        vision.prewarm_images("chat1", ["http://x/a.png"], ["http://x/s.png"])
        recent = vision.recent_image_urls("chat1")
        assert ("image", "http://x/a.png") in recent
        assert ("sticker", "http://x/s.png") in recent

    def test_recent_ttl_and_limit(self, monkeypatch):
        old = vision.time.time() - 9999  # 超 TTL
        vision._RECENT["chat2"] = vision.deque(
            [(old, "image", f"http://x/{i}.png") for i in range(10)], maxlen=40)
        assert vision.recent_image_urls("chat2") == []
        now = vision.time.time()
        vision._RECENT["chat3"] = vision.deque(
            [(now, "image", f"http://x/{i}.png") for i in range(10)], maxlen=40)
        assert len(vision.recent_image_urls("chat3")) == vision._RECENT_MAX

    @pytest.mark.asyncio
    async def test_prewarm_then_describe_hits_same_task(self, monkeypatch):
        """预热启动后，回复路径 describe 共享同一在途任务。"""
        track = {}
        monkeypatch.setattr(vision, "_download",
                            lambda url: asyncio.sleep(0, result=_unique_bytes("pw2")))
        model = _fake_model(track)
        monkeypatch.setattr(vision, "_get_vlm", lambda: model)
        vision.prewarm_images("chat1", ["http://x/a.png"], [])
        out = await vision.describe_images(["http://x/a.png"], model=model)
        assert track["calls"] == 1
        assert out["http://x/a.png"] == "一只猫"
