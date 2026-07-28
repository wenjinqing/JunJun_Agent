"""maizone 空间场景化测试：配图上传 / 自己的说说+评论 / 回复评论 / 自动发说说 / LLM 工具。"""

import json

import pytest

from junjun_skills.plugins.maizone import tools as mz


@pytest.fixture
def _env(monkeypatch, tmp_path):
    """隔离数据目录 + 固定配置 + 假 cookie。"""
    monkeypatch.setattr(mz, "DATA_DIR", tmp_path)
    monkeypatch.setattr(mz, "_cfg", lambda: {
        "enable": True, "send_enable": True, "read_enable": True,
        "monitor_enable": True, "like_enable": False, "comment_enable": False,
        "schedule_enable": True, "reply_comment_enable": True,
        "max_feed_per_day": 2, "max_comment_reply_per_day": 3,
        "schedule_times": ["09:30", "21:00"], "fluctuation_minutes": 30,
        "schedule_probability": 1.0, "schedule_image_probability": 0.0,
        "schedule_topics": ["测试主题"],
    })
    monkeypatch.setattr(mz, "_bot_uin", lambda: "123456")
    cookies = {"skey": "s", "p_skey": "p"}
    monkeypatch.setattr(mz, "ensure_cookies", _async(cookies))
    return cookies


def _async(ret):
    async def _f(*a, **kw):
        return ret
    return _f


def _ctx(text):
    class S: chat_id = "qq:1:group"
    class C:
        args = text
        session = S()
    return C()


# ---------------- 配图上传 ----------------

class TestUploadImage:
    @pytest.mark.asyncio
    async def test_parse_upload_result(self, _env, monkeypatch):
        """上传响应切 {...} json5 解析，提取 picbo/richval。"""
        captured = {}

        class _Resp:
            text = ("_Callback({ret:0, data:{url:'http://x/pic.jpg&bo=ABC123', "
                    "albumid:'a1', lloc:'l1', sloc:'s1', type:1, height:100, width:200}});")

        class _Client:
            def __init__(self, **kw): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): pass
            async def post(self, url, data=None, headers=None):
                captured["data"] = data
                return _Resp()

        monkeypatch.setattr(mz.httpx, "AsyncClient", _Client)
        picbo, richval = await mz.upload_image(_env, "123456", b"\x89PNG fake")
        assert picbo == "ABC123"
        assert richval == ",a1,l1,s1,1,100,200,,100,200"
        # base64 表单上传
        import base64
        assert captured["data"]["picfile"] == base64.b64encode(b"\x89PNG fake").decode()

    @pytest.mark.asyncio
    async def test_upload_ret_nonzero_raises(self, _env, monkeypatch):
        class _Resp:
            text = "_Callback({ret:-1});"

        class _Client:
            def __init__(self, **kw): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): pass
            async def post(self, *a, **kw): return _Resp()

        monkeypatch.setattr(mz.httpx, "AsyncClient", _Client)
        with pytest.raises(RuntimeError):
            await mz.upload_image(_env, "123456", b"x")

    @pytest.mark.asyncio
    async def test_publish_with_images_sets_picbo(self, _env, monkeypatch):
        """带图说说：先上传再带 pic_bo/richtype/richval 发布。"""
        async def _upload(cookies, uin, img):
            return "PICBO", ",a,l,s,1,1,1,,1,1"

        monkeypatch.setattr(mz, "upload_image", _upload)
        captured = {}

        class _Resp:
            text = json.dumps({"code": 0, "tid": "TID1"})

        class _Client:
            def __init__(self, **kw): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): pass
            async def post(self, url, params=None, data=None, headers=None):
                captured["data"] = data
                return _Resp()

        monkeypatch.setattr(mz.httpx, "AsyncClient", _Client)
        tid = await mz.publish_feed(_env, "123456", "带图说说", [b"img1"])
        assert tid == "TID1"
        assert captured["data"]["pic_bo"] == "PICBO"
        assert captured["data"]["richtype"] == "1"
        assert captured["data"]["con"] == "带图说说"


# ---------------- 自己的说说 + 评论 ----------------

_OWN_FEEDS_JSON = {
    "code": 0,
    "msglist": [{
        "tid": "F1",
        "content": "今天天气好",
        "created_time": 1753700000,
        "commentlist": [
            {"name": "白菜兔", "uin": "111", "content": "好看！", "tid": "10",
             "createTime": "2026-07-28 10:00", "list_3": [
                 {"name": "君君", "uin": "123456", "content": "回复@ 白菜兔 ：谢谢",
                  "tid": "11", "createTime": "2026-07-28 10:01"}]},
            {"name": "阿黄", "uin": "222", "content": "666", "tid": "12",
             "createTime": "2026-07-28 10:05"},
        ],
    }],
}


def _mock_get_client(payload):
    class _Resp:
        text = "_preloadCallback(" + json.dumps(payload) + ");"

    class _Client:
        def __init__(self, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): pass
        async def get(self, *a, **kw): return _Resp()

    return _Client


class TestGetOwnFeeds:
    @pytest.mark.asyncio
    async def test_parse_comments_with_sub(self, _env, monkeypatch):
        monkeypatch.setattr(mz.httpx, "AsyncClient", _mock_get_client(_OWN_FEEDS_JSON))
        feeds = await mz.get_own_feeds(_env, "123456", 3)
        assert len(feeds) == 1
        f = feeds[0]
        assert f["tid"] == "F1" and f["content"] == "今天天气好"
        assert f["created_time"].startswith("2025-") or f["created_time"]  # epoch 转换
        # 主评论 + 子评论平铺：白菜兔(主) + 君君(子) + 阿黄(主)
        assert len(f["comments"]) == 3
        main = f["comments"][0]
        assert main["nickname"] == "白菜兔" and main["comment_tid"] == "10"
        sub = f["comments"][1]
        assert sub["parent_tid"] == "10" and sub["qq_account"] == "123456"
        assert f["comments"][2]["nickname"] == "阿黄"


# ---------------- 回复评论（监控闭环） ----------------

class TestReplyComments:
    @pytest.mark.asyncio
    async def test_first_run_seeds_baseline(self, _env, monkeypatch):
        """首日运行：存量评论全部标记，不回复。"""
        monkeypatch.setattr(mz, "get_own_feeds", _async([
            {"tid": "F1", "content": "x",
             "comments": [{"nickname": "白菜兔", "qq_account": "111",
                           "content": "旧评论", "comment_tid": "10"}]},
        ]))
        sent = []

        async def _reply(cookies, uin, fid, nick, text):
            sent.append(nick)
            return True

        monkeypatch.setattr(mz, "reply_comment", _reply)
        await mz._reply_own_feed_comments(mz._cfg())
        assert sent == []  # 旧评论不回
        assert "F1:10" in mz._load_replied()

    @pytest.mark.asyncio
    async def test_replies_new_comment_once(self, _env, monkeypatch):
        """新评论回复一次；自己的评论跳过；重复轮询不重复回。"""
        feed = {"tid": "F1", "content": "今天天气好",
                "comments": [
                    {"nickname": "白菜兔", "qq_account": "111",
                     "content": "好看！", "comment_tid": "10"},
                    {"nickname": "君君", "qq_account": "123456",
                     "content": "自己的评论", "comment_tid": "11"},
                ]}
        monkeypatch.setattr(mz, "get_own_feeds", _async([feed]))
        monkeypatch.setattr(mz, "_generate_comment_reply",
                            _async("谢谢夸夸"))
        # 先建基线（含旧评论），再放入新评论
        mz._save_replied({"F1:old"})
        sent = []

        async def _reply(cookies, uin, fid, nick, text):
            sent.append((nick, text))
            return True

        monkeypatch.setattr(mz, "reply_comment", _reply)
        await mz._reply_own_feed_comments(mz._cfg())
        assert sent == [("白菜兔", "谢谢夸夸")]
        assert mz._daily_reply_count() == 1
        # 第二轮：不重复回
        await mz._reply_own_feed_comments(mz._cfg())
        assert len(sent) == 1

    @pytest.mark.asyncio
    async def test_daily_quota(self, _env, monkeypatch):
        comments = [{"nickname": f"好友{i}", "qq_account": str(100 + i),
                     "content": "评论", "comment_tid": str(i)} for i in range(5)]
        monkeypatch.setattr(mz, "get_own_feeds",
                            _async([{"tid": "F1", "content": "x", "comments": comments}]))
        monkeypatch.setattr(mz, "_generate_comment_reply", _async("回复"))
        mz._save_replied(set())
        sent = []

        async def _reply(cookies, uin, fid, nick, text):
            sent.append(nick)
            return True

        monkeypatch.setattr(mz, "reply_comment", _reply)
        await mz._reply_own_feed_comments(mz._cfg())
        assert len(sent) == 3  # max_comment_reply_per_day = 3


# ---------------- 定时自动发说说 ----------------

class TestAutoPost:
    def test_fluctuate_table(self, _env):
        table = mz._make_fluctuate_table(mz._cfg())
        assert len(table) == 2
        for t in table:
            h, m = map(int, t.split(":"))
            assert 0 <= h < 24 and 0 <= m < 60

    @pytest.mark.asyncio
    async def test_fires_on_matching_time(self, _env, monkeypatch):
        published = []

        async def _publish(cookies, uin, content, images=None):
            published.append((content, images))
            return "TID"

        monkeypatch.setattr(mz, "publish_feed", _publish)
        monkeypatch.setattr(mz, "get_own_feeds", _async([
            {"tid": "H1", "content": "昨天发过的说说", "comments": []}]))
        prompts = []

        async def _llm(prompt):
            prompts.append(prompt)
            return "今天也是元气满满的一天"

        monkeypatch.setattr(mz, "_ask_llm", _llm)

        class _Now:
            def strftime(self, fmt):
                return "2026-07-28" if fmt == "%Y-%m-%d" else "09:30"

        monkeypatch.setattr(mz, "datetime", type("dt", (), {"now": staticmethod(lambda: _Now())}))
        # fluctuation=0 保证时间表恰为基准时间
        cfg = dict(mz._cfg())
        cfg["fluctuation_minutes"] = 0
        monkeypatch.setattr(mz, "_cfg", lambda: cfg)
        await mz.maizone_auto_post()
        assert published and published[0][0] == "今天也是元气满满的一天"
        # 历史说说注入 prompt 防重复
        assert any("昨天发过的说说" in p for p in prompts)
        assert mz._daily_feed_count() == 1
        # 同一分钟不重复发
        await mz.maizone_auto_post()
        assert len(published) == 1

    @pytest.mark.asyncio
    async def test_respects_daily_quota(self, _env, monkeypatch):
        """到达每日说说上限后自动发送跳过。"""
        monkeypatch.setattr(mz, "_daily_feed_count", lambda: 99)
        called = []

        async def _gen(topic, history=""):
            called.append(topic)
            return "x"

        monkeypatch.setattr(mz, "_generate_feed_content", _gen)
        await mz._send_scheduled_feed(mz._cfg())
        assert called == []


# ---------------- LLM 工具 ----------------

class TestTools:
    @pytest.mark.asyncio
    async def test_send_feed_tool_with_image(self, _env, monkeypatch):
        published = []

        async def _publish(cookies, uin, content, images=None):
            published.append((content, images or []))
            return "TID"

        monkeypatch.setattr(mz, "publish_feed", _publish)
        monkeypatch.setattr(mz, "_feed_image_bytes", _async(b"\x89PNG img"))
        out = await mz.send_feed_tool.ainvoke({"content": "看云", "with_image": True})
        assert "带配图" in out
        assert published[0][1] == [b"\x89PNG img"]
        assert mz._daily_feed_count() == 1

    @pytest.mark.asyncio
    async def test_send_feed_tool_image_fail_degrades(self, _env, monkeypatch):
        """配图生成失败降级纯文字发布。"""
        published = []

        async def _publish(cookies, uin, content, images=None):
            published.append((content, images or []))
            return "TID"

        monkeypatch.setattr(mz, "publish_feed", _publish)
        monkeypatch.setattr(mz, "_feed_image_bytes", _async(None))
        out = await mz.send_feed_tool.ainvoke({"content": "看云", "with_image": True})
        assert "纯文字" in out and published[0][1] == []

    @pytest.mark.asyncio
    async def test_send_feed_tool_quota(self, _env, monkeypatch):
        monkeypatch.setattr(mz, "_daily_feed_count", lambda: 2)  # 上限 2
        out = await mz.send_feed_tool.ainvoke({"content": "x"})
        assert "上限" in out

    @pytest.mark.asyncio
    async def test_read_feed_tool(self, _env, monkeypatch):
        monkeypatch.setattr(mz, "fetch_friend_feeds", _async([
            {"nickname": "白菜兔", "content": "新番好看", "created_time": "昨天18:00",
             "target_qq": "111", "tid": "T1"},
        ]))
        out = await mz.read_feed_tool.ainvoke({"num": 5})
        assert "白菜兔" in out and "新番好看" in out

    @pytest.mark.asyncio
    async def test_tools_disabled(self, _env, monkeypatch):
        cfg = dict(mz._cfg())
        cfg["send_enable"] = False
        monkeypatch.setattr(mz, "_cfg", lambda: cfg)
        out = await mz.send_feed_tool.ainvoke({"content": "x"})
        assert "没开" in out
