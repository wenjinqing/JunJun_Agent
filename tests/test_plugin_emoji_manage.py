"""emoji_manage 插件测试：/emoji add|delete|list + /random_emojis + admin 门。"""

import json
from types import SimpleNamespace

import pytest

from junjun_agent import commands


@pytest.fixture(autouse=True)
def _clean_buses():
    commands.clear_commands()
    yield
    commands.clear_commands()


def _session(is_group=True):
    return SimpleNamespace(platform="qq", group_id="999" if is_group else None,
                           is_group=is_group, chat_id="qq:999:group" if is_group else "qq:1:private")


def _meta(text, user_id="12345", image_urls=None):
    return SimpleNamespace(text=text, user_id=user_id, nickname="甲", at_bot=False,
                           message_id="m1", image_urls=image_urls)


@pytest.fixture
def _fake_gateway(monkeypatch):
    sent = []

    class _FakeGW:
        async def send_reply(self, reply_set):
            sent.append(reply_set)

    import junjun_core.gateway.router as router_mod
    monkeypatch.setattr(router_mod, "get_gateway", lambda: _FakeGW())
    return sent


def _ctx(text, args="", image_urls=None, user_id="12345"):
    return commands.CommandContext(session=_session(),
                                   meta=_meta(text, user_id=user_id, image_urls=image_urls),
                                   args=args)


@pytest.fixture
def _emoji_env(tmp_path, monkeypatch):
    """临时 Emoji 表（bind_ctx，不动 _meta.database）+ 假下载/假 VLM + 假配置。"""
    import peewee
    from junjun_core.database import models as m
    import junjun_core.config.config as cfg_mod
    from junjun_express import emoji as emoji_mod

    # 全局配置：bot 身份信息（random_emojis 合并转发节点用）
    cfg_mod.global_config = cfg_mod.GlobalConfig(
        bot=cfg_mod.BotConfig(platform="qq", qq_account="10000001", nickname="君君"), raw={})

    # 注册落盘目录隔离到 tmp
    reg_dir = tmp_path / "reg"
    reg_dir.mkdir()
    monkeypatch.setattr(emoji_mod, "EMOJI_REG_DIR", reg_dir)

    # 假下载：按 url 尾号生成不同字节流（hash 不同）
    async def _download(url):
        return f"fake-img-{url}".encode()

    async def _describe(data, *, model=None):
        return "一只猫在笑", ["开心"]

    monkeypatch.setattr(emoji_mod.emoji_manager, "_download", _download)
    monkeypatch.setattr(emoji_mod.emoji_manager, "_describe", _describe)

    db = peewee.SqliteDatabase(str(tmp_path / "t.db"))
    with db.bind_ctx([m.Emoji]):
        db.create_tables([m.Emoji])
        yield m


def _tools():
    import junjun_skills.plugins.emoji_manage.tools as em
    return em


class TestEmojiAdd:
    @pytest.mark.asyncio
    async def test_add_success(self, _emoji_env):
        m = _emoji_env
        em = _tools()
        result = await em.emoji_cmd(_ctx("/emoji add", args="add",
                                         image_urls=["http://x/1.jpg"]))
        assert "成功 1 张" in result
        row = m.Emoji.get()
        assert row.description == "一只猫在笑"
        assert json.loads(row.emotion) == ["开心"]
        assert row.emoji_hash and row.full_path

    @pytest.mark.asyncio
    async def test_add_no_image_hint(self, _emoji_env):
        em = _tools()
        result = await em.emoji_cmd(_ctx("/emoji add", args="add", image_urls=None))
        assert "发图" in result

    @pytest.mark.asyncio
    async def test_add_dedup_by_hash(self, _emoji_env):
        m = _emoji_env
        em = _tools()
        await em.emoji_cmd(_ctx("/emoji add", args="add", image_urls=["http://x/1.jpg"]))
        result = await em.emoji_cmd(_ctx("/emoji add", args="add", image_urls=["http://x/1.jpg"]))
        assert "重复" in result
        assert m.Emoji.select().count() == 1


class TestEmojiDelete:
    @pytest.mark.asyncio
    async def test_delete_by_id(self, _emoji_env):
        m = _emoji_env
        em = _tools()
        m.Emoji.create(full_path="E:/nowhere/a.img", emoji_hash="h1", description="图一")
        row_id = m.Emoji.get().id
        result = await em.emoji_cmd(_ctx("/emoji delete 1", args=f"delete {row_id}"))
        assert "已删除" in result
        assert m.Emoji.select().count() == 0
        # 再删一次 -> 找不到
        result = await em.emoji_cmd(_ctx("/emoji delete 1", args=f"delete {row_id}"))
        assert "没找到" in result

    @pytest.mark.asyncio
    async def test_delete_no_args_no_image_hint(self, _emoji_env):
        em = _tools()
        result = await em.emoji_cmd(_ctx("/emoji delete", args="delete"))
        assert "用法" in result or "发图" in result


class TestEmojiList:
    @pytest.mark.asyncio
    async def test_list_output(self, _emoji_env):
        m = _emoji_env
        em = _tools()
        m.Emoji.create(full_path="E:/nowhere/a.img", emoji_hash="h1",
                       description="猫猫", emotion='["开心"]', usage_count=3)
        m.Emoji.create(full_path="E:/nowhere/b.img", emoji_hash="h2",
                       description="狗狗", emotion='["无语"]', usage_count=0)
        result = await em.emoji_cmd(_ctx("/emoji list", args="list"))
        assert "共 2 张" in result
        assert "猫猫" in result and "狗狗" in result
        assert "用过3次" in result
        assert f"#{m.Emoji.get(m.Emoji.emoji_hash == 'h1').id}" in result

    @pytest.mark.asyncio
    async def test_list_empty(self, _emoji_env):
        em = _tools()
        result = await em.emoji_cmd(_ctx("/emoji list", args="list"))
        assert "空的" in result


class TestRandomEmojis:
    @pytest.mark.asyncio
    async def test_forward_segment_json(self, _emoji_env, _fake_gateway):
        m = _emoji_env
        em = _tools()
        for i in range(3):
            m.Emoji.create(full_path=f"E:/nowhere/{i}.img", emoji_hash=f"h{i}",
                           description=f"图{i}", emotion='["开心"]')
        result = await em.random_emojis_cmd(_ctx("/random_emojis 2", args="2"))
        assert result is None  # 走 ctx.send，不返回文本
        assert len(_fake_gateway) == 1
        seg = _fake_gateway[0].segments[0]
        assert seg.type == "forward"
        nodes = json.loads(seg.data)  # JSON 可解析
        assert len(nodes) == 2  # 只要 2 张
        for node in nodes:
            assert node["type"] == "node"
            assert node["data"]["user_id"] == "10000001"
            assert node["data"]["nickname"] == "君君"
            content = node["data"]["content"][0]
            assert content["type"] == "image"
            assert content["data"]["file"].startswith("file:///")
        # 发出的表情 usage_count +1
        used = sum(r.usage_count for r in m.Emoji.select())
        assert used == 2

    @pytest.mark.asyncio
    async def test_empty_library_hint(self, _emoji_env):
        em = _tools()
        result = await em.random_emojis_cmd(_ctx("/random_emojis"))
        assert "空的" in result


class TestAdminGate:
    @pytest.mark.asyncio
    async def test_non_admin_refused_and_reported(self, _emoji_env, _fake_gateway,
                                                  monkeypatch):
        """dispatch 层：/emoji 是 admin_only，非管理员被拒并上报。"""
        import importlib
        em = importlib.reload(_tools())  # 总线已清空，重新注册命令

        monkeypatch.setenv("ADMIN_QQ", "10001")
        reports = []
        monkeypatch.setattr("junjun_core.security.report_violation",
                            lambda *a, **k: reports.append(a))

        session = _session()
        handled = await commands.dispatch(session, _meta("/emoji list", user_id="12345"))
        assert handled is True
        assert "管理员" in _fake_gateway[0].segments[0].data
        assert reports  # 已上报

        # 管理员 @bot（权限激活）可以正常用
        from junjun_core.security import set_caller
        set_caller("10001", at_bot=True, is_group=True)
        _fake_gateway.clear()
        handled = await commands.dispatch(session, _meta("/emoji list", user_id="10001"))
        assert handled is True
        assert "空的" in _fake_gateway[0].segments[0].data

        # /random_emojis 非 admin_only，普通用户可用
        _fake_gateway.clear()
        handled = await commands.dispatch(session, _meta("/random_emojis", user_id="12345"))
        assert handled is True
        assert "空的" in _fake_gateway[0].segments[0].data
        assert em.TOOLS == []
