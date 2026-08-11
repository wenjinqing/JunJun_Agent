"""梦境测试：碎片→成梦→发空间/写记忆；隐私（含私聊素材不外发）；一日一梦。

LLM/空间发布/长期记忆全部打桩，不触生产库与真实 QZone。
"""

import pytest

import junjun_core.config.config as cfg_mod
from junjun_express import dream


def _set_config(raw: dict):
    cfg_mod.global_config = cfg_mod.GlobalConfig(
        bot=cfg_mod.BotConfig(platform="qq", qq_account="12345", nickname="君君"),
        raw=raw)


class _FakeModel:
    def __init__(self, content="梦到群友全变成猫，在我的键盘上开运动会……睡醒愣了半天。"):
        self._content = content

    async def ainvoke(self, messages, config=None):
        return type("R", (), {"content": self._content})()


@pytest.fixture
def env(tmp_path, monkeypatch):
    old = cfg_mod.global_config
    _set_config({"dream": {"enable": True, "time": "07:30", "target": "qzone"}})
    monkeypatch.setattr(dream, "_STATE", tmp_path / "dream" / "last.json")
    fragments = ("- 我的日记：今天陪甲聊了很久考试\n- 乙的猫生病了", False)
    monkeypatch.setattr(dream, "_gather_fragments", lambda: fragments)
    published, memories = [], []
    import junjun_skills.plugins.junzone.tools as jz
    async def fake_auth_retry(fn, *args):
        published.append(args[-1])
        return "tid-1"
    monkeypatch.setattr(jz, "_with_auth_retry", fake_auth_retry)
    import junjun_memory.long_term as ltm_mod

    class FakeLTM:
        async def add(self, text, chat_id, **kw):
            memories.append((text, chat_id))
            return True
    monkeypatch.setattr(ltm_mod, "get_long_term_memory", lambda: FakeLTM())
    yield type("E", (), {"published": published, "memories": memories})()
    cfg_mod.global_config = old


class TestWriteDream:
    @pytest.mark.asyncio
    async def test_happy_publishes_and_remembers(self, env):
        out = await dream.write_dream(model=_FakeModel())
        assert "运动会" in out
        assert env.published and "昨晚的梦" in env.published[0]
        assert env.memories and env.memories[0][1] == "self:diary"
        assert "[我的梦" in env.memories[0][0]

    @pytest.mark.asyncio
    async def test_private_fragments_not_published(self, env, monkeypatch):
        """隐私纪律：素材含私聊场景时只写 private 记忆域，不外发空间。"""
        monkeypatch.setattr(dream, "_gather_fragments",
                            lambda: ("- 碎片", True))
        out = await dream.write_dream(model=_FakeModel())
        assert out
        assert not env.published, "私聊素材绝不外发"
        assert env.memories[0][1] == "self:diary:private"

    @pytest.mark.asyncio
    async def test_once_per_day(self, env):
        await dream.write_dream(model=_FakeModel())
        assert await dream.write_dream(model=_FakeModel()) is None
        assert len(env.published) == 1

    @pytest.mark.asyncio
    async def test_disabled_noop(self, env, monkeypatch):
        monkeypatch.setattr(dream, "_cfg", lambda: {"enable": False})
        assert await dream.write_dream(model=_FakeModel()) is None
        assert not env.published and not env.memories

    @pytest.mark.asyncio
    async def test_no_fragments_noop(self, env, monkeypatch):
        monkeypatch.setattr(dream, "_gather_fragments", lambda: ("", False))
        assert await dream.write_dream(model=_FakeModel()) is None
        assert not env.published

    @pytest.mark.asyncio
    async def test_publish_failure_still_remembers(self, env, monkeypatch):
        """发空间失败不丢梦：记忆照写（空间只是橱窗，记忆才是本体）。"""
        import junjun_skills.plugins.junzone.tools as jz
        async def boom(fn, *args):
            raise RuntimeError("qzone down")
        monkeypatch.setattr(jz, "_with_auth_retry", boom)
        out = await dream.write_dream(model=_FakeModel())
        assert out and env.memories

    @pytest.mark.asyncio
    async def test_target_memory_only(self, env, monkeypatch):
        monkeypatch.setattr(dream, "_cfg",
                            lambda: {"enable": True, "target": "memory"})
        out = await dream.write_dream(model=_FakeModel())
        assert out and not env.published and env.memories
