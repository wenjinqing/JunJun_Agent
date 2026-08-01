"""video_watch job 测试：字幕短路/ASR 管线/关键帧/降级路径/时长上限/工具入口。

下载/ASR/VLM/LLM 全部打桩；临时目录用 tmp_path。
"""

import pytest

import junjun_core.config.config as cfg_mod
import junjun_skills.plugins.bilibili.tools as bili_tools
from junjun_skills.plugins.bilibili import content, watch

CHAT = "qq:999:group"

_VIEW = {
    "bvid": "BV1xx411c7mD", "aid": 17001, "cid": 280001,
    "title": "无声游戏实况", "desc": "全程无解说",
    "duration": 300, "pic": "http://i0.hdslb.com/cover.jpg", "owner": "UP主乙",
}


@pytest.fixture
def env(monkeypatch, tmp_path):
    old = cfg_mod.global_config
    cfg_mod.global_config = cfg_mod.GlobalConfig(
        bot=cfg_mod.BotConfig(platform="qq", qq_account="12345", nickname="君君"),
        raw={"video_watch": {"max_duration": 1800, "asr_max_chars": 100,
                             "keyframe_interval": 30, "max_keyframes": 3,
                             "report_max_chars": 500},
             "bilibili": {"enable_understand": True, "subtitle_max_chars": 100}})
    content._MATERIAL_CACHE.clear()
    content._PENDING.clear()
    monkeypatch.setattr(bili_tools, "TMP_DIR", tmp_path)

    async def _extract(url):
        return "BV1xx411c7mD"

    async def _view(bvid):
        return dict(_VIEW)

    monkeypatch.setattr(bili_tools, "extract_bvid", _extract)
    monkeypatch.setattr(bili_tools, "_fetch_view", _view)
    # 默认：无字幕、弹幕热评有一点、ffmpeg 可用
    monkeypatch.setattr(content, "_fetch_subtitle_text", _async_ret(""))
    monkeypatch.setattr(content, "_fetch_danmaku_sample", _async_ret(["666"]))
    monkeypatch.setattr(content, "_fetch_top_replies", _async_ret(["前排"]))
    monkeypatch.setattr(bili_tools, "_ffmpeg_path", lambda: "/usr/bin/ffmpeg")
    yield monkeypatch
    cfg_mod.global_config = old


def _async_ret(value):
    async def _f(*a, **kw):
        return value
    return _f


class _FakeModel:
    def __init__(self, text="观后报告：这个视频讲的是……"):
        self._text = text

    async def ainvoke(self, messages, config=None):
        return type("R", (), {"content": self._text})()


def _job():
    return type("J", (), {"job_id": "w1", "chat_id": CHAT,
                          "title": "看视频", "kind": "video_watch"})()


class TestBranches:
    @pytest.mark.asyncio
    async def test_subtitle_short_circuit(self, env, monkeypatch):
        """字幕命中：直接综述，不下载不 ASR。"""
        monkeypatch.setattr(content, "_fetch_subtitle_text", _async_ret("完整字幕内容"))

        async def _boom(*a, **kw):
            raise AssertionError("不该走到下载")
        monkeypatch.setattr(bili_tools, "_fetch_playurl", _boom)

        out = await watch.video_watch_handler(_job(), {"url": "x"}, synth_model=_FakeModel())
        assert "观后报告" in out

    @pytest.mark.asyncio
    async def test_asr_pipeline(self, env, monkeypatch, tmp_path):
        """无字幕：下载 -> 抽音频 -> ASR 转写进材料 -> 综述。"""
        seen = {}

        async def _play(aid, cid):
            return {"type": "dash", "video": "http://cdn/v.m4s", "audio": "http://cdn/a.m4s"}

        async def _dl(url, path):
            path.write_bytes(b"\x00" * 64)
            return True

        async def _ffmpeg(args):
            if "-vn" in args:
                seen["audio_extract"] = True
                # 抽出 m4a：写点内容
                from pathlib import Path
                Path(args[-1]).write_bytes(b"audio")
            return True

        monkeypatch.setattr(bili_tools, "_fetch_playurl", _play)
        monkeypatch.setattr(bili_tools, "_download", _dl)
        monkeypatch.setattr(bili_tools, "_run_ffmpeg", _ffmpeg)

        async def _fake_asr(path):
            return "这里是语音转写全文"
        monkeypatch.setattr(watch, "_vlm_available", lambda: False)

        out = await watch.video_watch_handler(
            _job(), {"url": "x"}, synth_model=_FakeModel(), asr=_fake_asr)
        assert "观后报告" in out and seen.get("audio_extract")
        # 临时目录已清理
        assert not list(tmp_path.glob("watch_*"))

    @pytest.mark.asyncio
    async def test_keyframes_described(self, env, monkeypatch, tmp_path):
        """有 VLM：抽帧 + 帧描述进材料。"""
        async def _play(aid, cid):
            return {"type": "dash", "video": "http://cdn/v.m4s", "audio": "http://cdn/a.m4s"}

        async def _dl(url, path):
            path.write_bytes(b"\x00" * 64)
            return True

        async def _ffmpeg(args):
            from pathlib import Path
            if "-vn" in args:
                Path(args[-1]).write_bytes(b"audio")
            elif "-vf" in args:
                out_dir = Path(args[-1]).parent
                out_dir.mkdir(parents=True, exist_ok=True)
                for i in range(2):
                    (out_dir / f"f_{i:03d}.jpg").write_bytes(b"jpg")
            return True

        monkeypatch.setattr(bili_tools, "_fetch_playurl", _play)
        monkeypatch.setattr(bili_tools, "_download", _dl)
        monkeypatch.setattr(bili_tools, "_run_ffmpeg", _ffmpeg)

        captured = {}

        class _Synth:
            async def ainvoke(self, messages, config=None):
                captured["material"] = str(messages[-1].content)
                return type("R", (), {"content": "报告"})()

        async def _fake_asr(path):
            return "转写"

        async def _fake_describe(data, *, model, prompt):
            return "一只猫在键盘上"
        import junjun_memory.vision as vision_mod
        monkeypatch.setattr(vision_mod, "_describe", _fake_describe)

        out = await watch.video_watch_handler(
            _job(), {"url": "x"}, synth_model=_Synth(), asr=_fake_asr, vlm=object())
        assert out == "报告"
        assert "一只猫在键盘上" in captured["material"]
        assert "画面关键帧" in captured["material"]

    @pytest.mark.asyncio
    async def test_asr_failure_degrades(self, env, monkeypatch, tmp_path):
        """ASR 失败：用弹幕热评材料综述，不算 job 失败。"""
        async def _play(aid, cid):
            return {"type": "dash", "video": None, "audio": "http://cdn/a.m4s"}

        async def _dl(url, path):
            path.write_bytes(b"\x00" * 64)
            return True

        async def _ffmpeg(args):
            from pathlib import Path
            if "-vn" in args:
                Path(args[-1]).write_bytes(b"audio")
            return True

        monkeypatch.setattr(bili_tools, "_fetch_playurl", _play)
        monkeypatch.setattr(bili_tools, "_download", _dl)
        monkeypatch.setattr(bili_tools, "_run_ffmpeg", _ffmpeg)
        monkeypatch.setattr(watch, "_vlm_available", lambda: False)

        async def _bad_asr(path):
            return ""

        out = await watch.video_watch_handler(
            _job(), {"url": "x"}, synth_model=_FakeModel(), asr=_bad_asr)
        assert "观后报告" in out

    @pytest.mark.asyncio
    async def test_no_ffmpeg_degrades(self, env, monkeypatch):
        """无 ffmpeg：降级弹幕热评综述。"""
        monkeypatch.setattr(bili_tools, "_ffmpeg_path", lambda: None)
        out = await watch.video_watch_handler(_job(), {"url": "x"}, synth_model=_FakeModel())
        assert "观后报告" in out

    @pytest.mark.asyncio
    async def test_too_long_rejected(self, env, monkeypatch):
        async def _view(bvid):
            v = dict(_VIEW)
            v["duration"] = 7200
            return v
        monkeypatch.setattr(bili_tools, "_fetch_view", _view)
        with pytest.raises(RuntimeError, match="太长"):
            await watch.video_watch_handler(_job(), {"url": "x"}, synth_model=_FakeModel())

    @pytest.mark.asyncio
    async def test_no_url_and_no_view(self, env, monkeypatch):
        with pytest.raises(RuntimeError, match="链接"):
            await watch.video_watch_handler(_job(), {"url": ""}, synth_model=_FakeModel())

        async def _view(bvid):
            return None
        monkeypatch.setattr(bili_tools, "_fetch_view", _view)
        with pytest.raises(RuntimeError, match="拿不到"):
            await watch.video_watch_handler(_job(), {"url": "x"}, synth_model=_FakeModel())


class TestTool:
    def test_watch_video_tool_submits(self, env, monkeypatch):
        from peewee import SqliteDatabase
        from junjun_core.database import models as m
        test_db = SqliteDatabase(":memory:")
        cfg_mod.global_config.raw["async_task"] = {"enable": True, "max_pending_per_chat": 5}
        from junjun_skills.builtin.memory_skills import current_chat_id
        from junjun_core.security import current_user_id, current_nickname
        with test_db.bind_ctx([m.AsyncJob]):
            test_db.create_tables([m.AsyncJob])
            t1 = current_chat_id.set(CHAT)
            t2 = current_user_id.set("111")
            t3 = current_nickname.set("甲")
            try:
                out = bili_tools.watch_video.invoke(
                    {"url": "https://www.bilibili.com/video/BV1xx411c7mD"})
                assert "接单成功" in out
                row = m.AsyncJob.get()
                assert row.kind == "video_watch" and "BV1xx411c7mD" in row.payload
            finally:
                current_chat_id.reset(t1)
                current_user_id.reset(t2)
                current_nickname.reset(t3)
