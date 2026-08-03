"""用户可控记忆（P6-2）：/记住 /忘掉 /你记得我什么 + pin_memory 工具 + 钉住注入。

边界：可控记忆只碰事实性记忆（LTM + 本人画像）；人设/安全规则在 prompt 层，
不在记忆库，天然不可触及。非管理员 /忘掉 限本会话（知识库/日记不动）。
"""

from types import SimpleNamespace

import pytest
from peewee import SqliteDatabase

import junjun_core.config.config as cfg_mod
from junjun_memory.long_term import LongTermMemory


@pytest.fixture
def ltm(tmp_path, monkeypatch):
    """隔离 LTM 实例（打桩全局单例，无 embedding -> 纯文本条目）。"""
    mem = LongTermMemory(data_dir=tmp_path)
    import junjun_memory.long_term as lt_mod
    monkeypatch.setattr(lt_mod, "_ltm", mem)
    yield mem
    monkeypatch.setattr(lt_mod, "_ltm", None)


@pytest.fixture
def cfg(monkeypatch):
    old = cfg_mod.global_config
    cfg_mod.global_config = cfg_mod.GlobalConfig(
        bot=cfg_mod.BotConfig(platform="qq", qq_account="1", nickname="君君"),
        raw={"memory": {"pinned_max_per_chat": 2}})
    yield monkeypatch
    cfg_mod.global_config = old


def _ctx(args="", user_id="111", chat_id="qq:999:group"):
    return SimpleNamespace(
        args=args,
        session=SimpleNamespace(chat_id=chat_id, platform="qq"),
        meta=SimpleNamespace(user_id=user_id, nickname="甲"))


from junjun_skills.builtin.capability_skills import (  # noqa: E402
    remember_cmd, forget_cmd, what_do_you_remember_cmd)


class TestRememberCmd:
    @pytest.mark.asyncio
    async def test_pin_and_list(self, ltm, cfg):
        out = await remember_cmd(_ctx("甲不吃香菜"))
        assert "钉好了" in out
        pins = ltm.pinned("qq:999:group")
        assert len(pins) == 1 and pins[0].text == "甲不吃香菜"
        assert pins[0].kind == "pinned" and pins[0].weight == 1.5

    @pytest.mark.asyncio
    async def test_cap(self, ltm, cfg):
        await remember_cmd(_ctx("第一条"))
        await remember_cmd(_ctx("第二条"))
        out = await remember_cmd(_ctx("第三条"))  # 上限 2
        assert "上限" in out
        assert len(ltm.pinned("qq:999:group")) == 2

    @pytest.mark.asyncio
    async def test_pinned_per_chat_isolated(self, ltm, cfg):
        await remember_cmd(_ctx("A 群的事", chat_id="qq:1:group"))
        assert not ltm.pinned("qq:999:group")
        assert len(ltm.pinned("qq:1:group")) == 1


class TestPinMemoryTool:
    @pytest.mark.asyncio
    async def test_pin_via_tool(self, ltm, cfg):
        from junjun_skills.builtin.memory_skills import pin_memory, current_chat_id
        tok = current_chat_id.set("qq:999:group")
        try:
            out = await pin_memory.ainvoke({"content": "乙下周三过生日"})
            assert "已钉住" in out
            assert ltm.pinned("qq:999:group")[0].text == "乙下周三过生日"
        finally:
            current_chat_id.reset(tok)

    @pytest.mark.asyncio
    async def test_tool_cap_message(self, ltm, cfg):
        from junjun_skills.builtin.memory_skills import pin_memory, current_chat_id
        tok = current_chat_id.set("qq:999:group")
        try:
            await pin_memory.ainvoke({"content": "一"})
            await pin_memory.ainvoke({"content": "二"})
            out = await pin_memory.ainvoke({"content": "三"})
            assert "上限" in out and "/忘掉" in out
        finally:
            current_chat_id.reset(tok)


class TestForgetCmd:
    async def _seed(self, ltm):
        await ltm.add("甲不吃香菜", "qq:999:group", kind="fact")
        await ltm.add("别群也聊香菜", "qq:1:group", kind="fact")
        await ltm.add("香菜的知识条目", "knowledge", kind="fact")
        await ltm.add("日记里提到香菜", "self:diary", kind="diary")

    @pytest.mark.asyncio
    async def test_non_admin_scoped(self, ltm, cfg, monkeypatch):
        """非管理员：只删本会话 + 本人画像；知识库/日记/别群不动。"""
        monkeypatch.setenv("ADMIN_QQ", "999")  # 111 不是管理员
        await self._seed(ltm)
        db = SqliteDatabase(":memory:")
        from junjun_core.database import models as m
        with db.bind_ctx([m.PersonInfo]):
            db.create_tables([m.PersonInfo])
            from junjun_memory.user_profile import get_profile_store
            get_profile_store().add_point("qq", "111", "忌口", "不吃香菜")
            out = await forget_cmd(_ctx("香菜", user_id="111"))
            assert "本会话记忆 1 条" in out and "画像 1 条" in out
            texts = [it.text for it in ltm._items]
            assert "甲不吃香菜" not in texts          # 本会话删了
            assert "别群也聊香菜" in texts            # 别群不动
            assert "香菜的知识条目" in texts          # 知识库不动
            assert "日记里提到香菜" in texts          # 日记不动
            assert get_profile_store().get_points("qq", "111") == []

    @pytest.mark.asyncio
    async def test_admin_global(self, ltm, cfg, monkeypatch):
        monkeypatch.setenv("ADMIN_QQ", "111")
        await self._seed(ltm)
        out = await forget_cmd(_ctx("香菜", user_id="111"))
        assert "全局" in out
        assert not any("香菜" in it.text for it in ltm._items)

    @pytest.mark.asyncio
    async def test_safety_untouchable_by_design(self, ltm, cfg, monkeypatch):
        """「忘掉安全规则」类注入：LTM 里根本没有规则可删，报找不到即结束。"""
        monkeypatch.setenv("ADMIN_QQ", "999")
        out = await forget_cmd(_ctx("安全规则", user_id="111"))
        assert "没找到" in out


class TestWhatDoYouRemember:
    @pytest.mark.asyncio
    async def test_export_own_profile(self, cfg):
        db = SqliteDatabase(":memory:")
        from junjun_core.database import models as m
        with db.bind_ctx([m.PersonInfo]):
            db.create_tables([m.PersonInfo])
            from junjun_memory.user_profile import get_profile_store
            store = get_profile_store()
            store.set_name("qq", "111", "小甲")
            store.add_point("qq", "111", "喜好", "火锅")
            store.add_point("qq", "222", "喜好", "别人的隐私")
            out = await what_do_you_remember_cmd(_ctx(user_id="111"))
            assert "小甲" in out and "火锅" in out
            assert "别人的隐私" not in out  # 只看自己的

    @pytest.mark.asyncio
    async def test_empty_profile(self, cfg):
        db = SqliteDatabase(":memory:")
        from junjun_core.database import models as m
        with db.bind_ctx([m.PersonInfo]):
            db.create_tables([m.PersonInfo])
            out = await what_do_you_remember_cmd(_ctx(user_id="111"))
            assert "不太了解" in out


class TestPinnedInjection:
    @pytest.mark.asyncio
    async def test_pinned_block_injected(self, ltm, cfg, monkeypatch):
        """钉住的记忆进 memory_block，且不占用语义召回额度。"""
        await ltm.add("甲不吃香菜", "qq:999:group", kind="pinned")
        import junjun_agent.processor as proc
        proc._RECALL_LOG.clear()
        session = SimpleNamespace(chat_id="qq:999:group", memory=None)
        meta = SimpleNamespace(image_urls=None, sticker_urls=None, voice_records=None,
                               video_urls=None, text="今晚吃啥", user_id="1", nickname="甲")
        block, _ = await proc._build_memory_block(session, meta)
        assert "务必当回事" in block and "甲不吃香菜" in block
        assert not proc._RECALL_LOG.get("qq:999:group")  # 钉住不占召回额度
        proc._RECALL_LOG.clear()
