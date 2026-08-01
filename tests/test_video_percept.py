"""视频文件感知测试：下载上限/抽帧 VLM/抽音 ASR/组合描述/md5 缓存/降级/渲染。"""

import asyncio

import pytest

import junjun_core.config.config as cfg_mod
from junjun_memory import video


@pytest.fixture
def env(monkeypatch):
    old = cfg_mod.global_config
    cfg_mod.global_config = cfg_mod.GlobalConfig(
        bot=cfg_mod.BotConfig(platform="qq", qq_account="12345", nickname="君君"),
        raw={"video_percept": {"enable": True, "max_bytes": 10 * 1024 * 1024,
                               "max_frames": 2},
             "perception": {"ready_wait_seconds": 3.0}})
    video._CACHE.clear()
    video._PENDING.clear()
    monkeypatch.setattr(video.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    yield monkeypatch
    video._CACHE.clear()
    video._PENDING.clear()
    cfg_mod.global_config = old


def _stub_pipeline(monkeypatch, *, frames="一只猫在键盘上", transcript="今晚吃什么"):
    """桩掉下载/ffmpeg/VLM/ASR 四件套。"""
    async def _dl(url, max_bytes):
        return b"videobytes"
    monkeypatch.setattr(video, "_download", _dl)

    async def _ffmpeg(args):
        from pathlib import Path
        if "-vn" in args:
            Path(args[-1]).write_bytes(b"a" * 2048)  # 过 1KB 无声轨判定
        elif "-vf" in args:
            out_dir = Path(args[-1]).parent
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "f_01.jpg").write_bytes(b"jpg")
        return True
    monkeypatch.setattr(video, "_run_ffmpeg", _ffmpeg)

    async def _describe(data, *, model, prompt):
        return frames
    import junjun_memory.vision as vision_mod
    monkeypatch.setattr(vision_mod, "_describe", _describe)

    import junjun_llm
    monkeypatch.setattr(junjun_llm, "get_chat_model", lambda slot: object())

    async def _asr(path, **kw):
        return transcript
    import junjun_llm.asr as asr_mod
    monkeypatch.setattr(asr_mod, "transcribe_file", _asr)


class TestPerceive:
    @pytest.mark.asyncio
    async def test_full_pipeline(self, env, monkeypatch):
        _stub_pipeline(monkeypatch)
        out = await video.describe_videos(["http://x/v.mp4"], wait=0)
        desc = out["http://x/v.mp4"]
        assert "一只猫在键盘上" in desc and "今晚吃什么" in desc
        assert video.render_video_block(out).startswith("对方发来一段视频")

    @pytest.mark.asyncio
    async def test_md5_cache(self, env, monkeypatch):
        _stub_pipeline(monkeypatch)
        calls = []

        async def _dl(url, max_bytes):
            calls.append(url)
            return b"videobytes"
        monkeypatch.setattr(video, "_download", _dl)
        await video.describe_videos(["http://x/1"], wait=0)
        await video.describe_videos(["http://x/2"], wait=0)  # 不同 url 同内容
        assert len(calls) == 2  # 下载两次（url 不同）
        # 第二次命中 md5 缓存：描述一致且没再跑 ffmpeg
        assert video._CACHE  # 缓存已写

    @pytest.mark.asyncio
    async def test_download_over_limit(self, env, monkeypatch):
        async def _dl(url, max_bytes):
            return None  # _download 内部超限返回 None
        monkeypatch.setattr(video, "_download", _dl)
        out = await video.describe_videos(["http://x/big.mp4"], wait=0)
        assert out["http://x/big.mp4"] == "[视频]"
        assert video.render_video_block(out) == ""

    @pytest.mark.asyncio
    async def test_no_audio_no_frames_placeholder(self, env, monkeypatch):
        _stub_pipeline(monkeypatch, frames=None, transcript="")
        # frames=None 时 _describe 返回 None -> 无画面描述
        out = await video.describe_videos(["http://x/v"], wait=0)
        assert out["http://x/v"] == "[视频]"

    @pytest.mark.asyncio
    async def test_disabled_without_ffmpeg(self, env, monkeypatch):
        monkeypatch.setattr(video.shutil, "which", lambda name: None)
        assert await video.describe_videos(["http://x/v"]) == {}
        video.prewarm_videos(["http://x/v"])  # 不创建任务
        assert not video._PENDING

    @pytest.mark.asyncio
    async def test_bounded_wait_task_lives(self, env, monkeypatch):
        """3s 内没感知完 -> 本轮占位；在途不取消，完成后直接命中。"""
        async def _dl(url, max_bytes):
            await asyncio.sleep(0.15)
            return b"v"
        monkeypatch.setattr(video, "_download", _dl)
        _stub_pipeline(monkeypatch)
        async def _slow_dl(url, max_bytes):
            await asyncio.sleep(0.15)
            return b"v"
        monkeypatch.setattr(video, "_download", _slow_dl)

        out = await video.describe_videos(["http://x/v"], wait=0.01)
        assert out["http://x/v"] == "[视频]"
        await asyncio.sleep(0.4)
        out2 = await video.describe_videos(["http://x/v"], wait=1.0)
        assert "一只猫在键盘上" in out2["http://x/v"]


class TestAdapterAndGateway:
    def test_video_seg_emits_ref(self):
        from junjun_adapter_napcat.recv_handler.message_handler import MessageHandler
        h = MessageHandler.__new__(MessageHandler)

        async def _run():
            return await h._parse_message_segments([
                {"type": "video", "data": {"file": "v.mp4", "url": "http://x/v.mp4"}},
            ], self_id="1", group_id="2")
        segs, _ = asyncio.run(_run())
        types = [s.type for s in segs]
        assert "text" in types and "video_file" in types
        assert next(s for s in segs if s.type == "video_file").data == "http://x/v.mp4"

    def test_gateway_extract_videos(self):
        from junjun_core.contracts import Seg
        from junjun_core.gateway.router import _extract_videos
        seg = Seg(type="seglist", data=[
            Seg(type="text", data="[视频]"),
            Seg(type="video_file", data="http://x/v.mp4"),
        ])
        assert _extract_videos(seg) == ["http://x/v.mp4"]
