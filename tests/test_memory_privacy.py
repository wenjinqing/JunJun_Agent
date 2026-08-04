"""严厉审查 P0-4 隐私收口回归：
- recall_memory 不再有全库 fallback（A 群套不出 B 群/私聊记忆）
- 含私聊素材的日记进 self:diary:private 域，群聊召回域不含它
"""

import pytest


class _FakeLTM:
    def __init__(self, items=None):
        self.calls = []
        self._items = items or []

    async def search(self, query, *, top_k=5, chat_id=None):
        self.calls.append(chat_id)
        return [it for it in self._items
                if chat_id is not None and it.chat_id in chat_id][:top_k]

    async def add(self, text, chat_id, *, weight=1.0, kind="chat"):
        from types import SimpleNamespace
        self.calls.append(("add", chat_id))
        self._items.append(SimpleNamespace(text=text, chat_id=chat_id))
        return True


class TestRecallMemoryScope:
    @pytest.mark.asyncio
    async def test_no_global_fallback(self, monkeypatch):
        """本会话+knowledge 搜不到就直接说没有——不再放宽全库。"""
        import junjun_memory.long_term as lt
        from types import SimpleNamespace
        fake = _FakeLTM(items=[SimpleNamespace(text="B群的秘密", chat_id="qq:B:group")])
        monkeypatch.setattr(lt, "get_long_term_memory", lambda: fake)

        from junjun_skills.builtin import memory_skills
        token = memory_skills.current_chat_id.set("qq:A:group")
        try:
            out = await memory_skills.recall_memory.ainvoke({"query": "秘密"})
        finally:
            memory_skills.current_chat_id.reset(token)
        assert out == "没有找到相关记忆。"
        # 只搜了一次，且召回域 = (本会话, knowledge)——无全库 None 调用
        assert fake.calls == [("qq:A:group", "knowledge")]

    @pytest.mark.asyncio
    async def test_knowledge_recallable_anywhere(self, monkeypatch):
        import junjun_memory.long_term as lt
        from types import SimpleNamespace
        fake = _FakeLTM(items=[SimpleNamespace(text="常识条目", chat_id="knowledge")])
        monkeypatch.setattr(lt, "get_long_term_memory", lambda: fake)
        from junjun_skills.builtin import memory_skills
        token = memory_skills.current_chat_id.set("qq:A:group")
        try:
            out = await memory_skills.recall_memory.ainvoke({"query": "常识"})
        finally:
            memory_skills.current_chat_id.reset(token)
        assert "常识条目" in out


class TestDiaryPrivacyScope:
    @pytest.mark.asyncio
    async def test_private_material_goes_to_private_domain(self, monkeypatch):
        import junjun_memory.long_term as lt
        fake = _FakeLTM()
        monkeypatch.setattr(lt, "get_long_term_memory", lambda: fake)
        from junjun_express.diary import _index_to_memory
        await _index_to_memory("2026-08-04", "今天和他私聊了很多", private=True)
        await _index_to_memory("2026-08-03", "今天在群里很热闹", private=False)
        assert ("add", "self:diary:private") in fake.calls
        assert ("add", "self:diary") in fake.calls

    @pytest.mark.asyncio
    async def test_group_recall_excludes_private_diary(self, monkeypatch, tmp_path):
        """群聊会话的「你忽然想起」召回域不含私聊日记域。"""
        import junjun_core.config.config as cfg_mod
        monkeypatch.setattr(cfg_mod, "global_config", cfg_mod.GlobalConfig(
            bot=cfg_mod.BotConfig(platform="qq", qq_account="1", nickname="君君"),
            raw={}))
        import junjun_memory.long_term as lt
        fake = _FakeLTM()
        monkeypatch.setattr(lt, "get_long_term_memory", lambda: fake)

        from types import SimpleNamespace
        from junjun_agent.processor import _build_memory_block
        session = SimpleNamespace(chat_id="qq:789:group", memory=None, is_group=True)
        meta = SimpleNamespace(image_urls=None, sticker_urls=None, voice_records=None,
                               video_urls=None, text="你还记得之前说的事吗",
                               user_id="1", nickname="甲")
        await _build_memory_block(session, meta)
        search_scopes = [c for c in fake.calls if c != ("add",) and isinstance(c, tuple)]
        assert search_scopes, "召回未被触发"
        for scope in search_scopes:
            assert "self:diary:private" not in scope
