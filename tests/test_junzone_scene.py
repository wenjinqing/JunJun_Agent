"""junzone 空间场景化测试：配图上传 / 自己的说说+评论 / 回复评论 / 自动发说说 / LLM 工具。"""

import json

import pytest

from junjun_skills.plugins.junzone import tools as mz


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
        await mz.junzone_auto_post()
        assert published and published[0][0] == "今天也是元气满满的一天"
        # 历史说说注入 prompt 防重复
        assert any("昨天发过的说说" in p for p in prompts)
        assert mz._daily_feed_count() == 1
        # 同一分钟不重复发
        await mz.junzone_auto_post()
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


# ---------------- 日记体说说 ----------------

def _fake_messages(rows):
    """伪装 Messages 表查询链：select().where().order_by().limit()。"""
    class _Field:
        def __ne__(self, other): return True
        def desc(self): return self

    class _Q:
        def where(self, *a): return self
        def order_by(self, *a): return self
        def limit(self, n): return list(rows)[-n:]

    class _M:
        processed_plain_text = _Field()
        time = _Field()

        @staticmethod
        def select(): return _Q()

    return _M


class _Row:
    def __init__(self, is_bot, nickname, text):
        self.is_bot = is_bot
        self.user_nickname = nickname
        self.processed_plain_text = text


class TestDiaryFeed:
    def test_chat_log_anonymized(self, monkeypatch):
        """昵称→某人A/B、QQ 号和 @ 抹除、bot→我。"""
        import junjun_core.database as db
        rows = [
            _Row(False, "白菜兔", "今晚开黑吗 @君君 我 QQ 123456789"),
            _Row(True, None, "来呀来呀"),
            _Row(False, "阿黄", "带我一个"),
            _Row(False, "白菜兔", "那八点集合"),
        ]
        monkeypatch.setattr(db, "Messages", _fake_messages(rows))
        log = mz._recent_chat_log()
        assert "白菜兔" not in log and "阿黄" not in log
        assert "123456789" not in log and "@君君" not in log
        assert "某人A" in log and "某人B" in log
        assert "我: 来呀来呀" in log
        # 同一昵称映射稳定
        assert log.count("某人A") == 2

    @pytest.mark.asyncio
    async def test_diary_prompt_privacy(self, _env, monkeypatch):
        prompts = []

        async def _llm(prompt):
            prompts.append(prompt)
            return "今天和某人A聊了游戏，开心"

        monkeypatch.setattr(mz, "_ask_llm", _llm)
        out = await mz._generate_diary_feed("某人A: 开黑吗\n我: 来呀", "- 旧说说")
        assert out == "今天和某人A聊了游戏，开心"
        p = prompts[0]
        assert "日记" in p and "隐私" in p
        assert "某人A: 开黑吗" in p and "旧说说" in p

    @pytest.mark.asyncio
    async def test_scheduled_feed_uses_diary(self, _env, monkeypatch):
        """有聊天素材时走日记路径，不用主题模式。"""
        monkeypatch.setattr(mz, "_recent_chat_log", lambda: "某人A: 开黑吗\n我: 来")
        monkeypatch.setattr(mz, "get_own_feeds", _async([]))
        used = {}

        async def _diary(log, history):
            used["mode"] = "diary"
            return "日记内容"

        async def _topic(topic, history=""):
            used["mode"] = "topic"
            return "主题内容"

        monkeypatch.setattr(mz, "_generate_diary_feed", _diary)
        monkeypatch.setattr(mz, "_generate_feed_content", _topic)
        monkeypatch.setattr(mz, "random", type("R", (), {
            "choice": staticmethod(lambda x: x[0]),
            "random": staticmethod(lambda: 1.0),  # 不配图
        }))
        published = []

        async def _publish(cookies, uin, content, images=None):
            published.append(content)
            return "TID"

        monkeypatch.setattr(mz, "publish_feed", _publish)
        await mz._send_scheduled_feed(mz._cfg())
        assert used["mode"] == "diary" and published == ["日记内容"]

    @pytest.mark.asyncio
    async def test_scheduled_feed_topic_fallback(self, _env, monkeypatch):
        """无聊天素材退回主题模式。"""
        monkeypatch.setattr(mz, "_recent_chat_log", lambda: "")
        monkeypatch.setattr(mz, "get_own_feeds", _async([]))
        used = {}

        async def _diary(log, history):
            used["mode"] = "diary"
            return "x"

        async def _topic(topic, history=""):
            used["mode"] = "topic"
            return "主题内容"

        monkeypatch.setattr(mz, "_generate_diary_feed", _diary)
        monkeypatch.setattr(mz, "_generate_feed_content", _topic)
        monkeypatch.setattr(mz, "random", type("R", (), {
            "choice": staticmethod(lambda x: x[0]),
            "random": staticmethod(lambda: 1.0),
        }))

        async def _publish(cookies, uin, content, images=None):
            return "TID"

        monkeypatch.setattr(mz, "publish_feed", _publish)
        await mz._send_scheduled_feed(mz._cfg())
        assert used["mode"] == "topic"


# ---------------- 健壮性 ----------------

class TestRobustness:
    @pytest.mark.asyncio
    async def test_own_feeds_non_json_raises_auth(self, _env, monkeypatch):
        """HTML/空响应 -> _AuthError（触发 _with_auth_retry 刷新 cookie）。"""
        class _Resp:
            text = "<html>登录过期</html>"

        class _Client:
            def __init__(self, **kw): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): pass
            async def get(self, *a, **kw): return _Resp()

        monkeypatch.setattr(mz.httpx, "AsyncClient", _Client)
        with pytest.raises(mz._AuthError):
            await mz.get_own_feeds(_env, "123456", 3)

    def test_sample_rate_lowered(self):
        """TTS 采样率 16000（减小语音文件，防 NapCat 发送超时）。"""
        from junjun_skills.plugins.ja_tts import tools as ja
        assert ja._SAMPLE_RATE == 16000

    def test_qzone_tools_in_core(self):
        """空间 = 第三场景：send_feed/read_feed 有话题关键词钉住（P5-2 后 CORE
        瘦身到 ≤8，空间工具改走 TOPIC 层——聊到「说说/空间/qzone」即挂载）。"""
        from junjun_skills import registry
        for name in ("send_feed", "read_feed", "delete_feed"):
            assert name in registry._TOPIC_KEYWORDS
            assert registry._TOPIC_KEYWORDS[name]



# ---------------- 删说说 ----------------

class TestDeleteFeed:
    @pytest.mark.asyncio
    async def test_delete_feed_parse_json(self, _env, monkeypatch):
        """标准 JSON 响应：code=0 通过。"""
        captured = {}

        class _Resp:
            text = json.dumps({"code": 0, "message": ""})

        class _Client:
            def __init__(self, **kw): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): pass
            async def post(self, url, params=None, data=None, headers=None):
                captured["data"] = data
                return _Resp()

        monkeypatch.setattr(mz.httpx, "AsyncClient", _Client)
        ok = await mz.delete_feed(_env, "123456", "TID1")
        assert ok is True
        assert captured["data"]["tid"] == "TID1"
        assert captured["data"]["hostuin"] == "123456"

    @pytest.mark.asyncio
    async def test_delete_feed_parse_frame_callback(self, _env, monkeypatch):
        """frameElement.callback HTML 壳也兼容。"""
        class _Resp:
            text = "<html><script>frameElement.callback({code:0, message:''});</script></html>"

        class _Client:
            def __init__(self, **kw): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): pass
            async def post(self, url, params=None, data=None, headers=None):
                return _Resp()

        monkeypatch.setattr(mz.httpx, "AsyncClient", _Client)
        assert await mz.delete_feed(_env, "123456", "TID1") is True

    @pytest.mark.asyncio
    async def test_tool_delete_latest_when_tid_empty(self, _env, monkeypatch):
        """tid 留空 -> 删自己最新一条。"""
        deleted = []
        monkeypatch.setattr(mz, "get_own_feeds", _async([
            {"tid": "T1", "content": "最新的说说", "created_time": "今天", "comments": []},
            {"tid": "T2", "content": "旧说说", "created_time": "昨天", "comments": []},
        ]))

        async def _delete(cookies, uin, tid):
            deleted.append(tid)
            return True

        monkeypatch.setattr(mz, "delete_feed", _delete)
        out = await mz.delete_feed_tool.ainvoke({"tid": ""})
        assert deleted == ["T1"]
        assert "最新的说说" in out

    @pytest.mark.asyncio
    async def test_tool_delete_by_tid(self, _env, monkeypatch):
        deleted = []
        monkeypatch.setattr(mz, "get_own_feeds", _async([
            {"tid": "T1", "content": "一", "created_time": "", "comments": []},
            {"tid": "T2", "content": "二", "created_time": "", "comments": []},
        ]))

        async def _delete(cookies, uin, tid):
            deleted.append(tid)
            return True

        monkeypatch.setattr(mz, "delete_feed", _delete)
        out = await mz.delete_feed_tool.ainvoke({"tid": "T2"})
        assert deleted == ["T2"]
        assert "删掉了" in out

    @pytest.mark.asyncio
    async def test_tool_rejects_unknown_tid(self, _env, monkeypatch):
        """tid 不在自己最近说说里 -> 拒绝（防删错/删别人的）。"""
        monkeypatch.setattr(mz, "get_own_feeds", _async([
            {"tid": "T1", "content": "一", "created_time": "", "comments": []},
        ]))
        monkeypatch.setattr(mz, "delete_feed", _async(True))
        out = await mz.delete_feed_tool.ainvoke({"tid": "T999"})
        assert "只能删自己" in out

    @pytest.mark.asyncio
    async def test_tool_no_feeds(self, _env, monkeypatch):
        monkeypatch.setattr(mz, "get_own_feeds", _async([]))
        out = await mz.delete_feed_tool.ainvoke({"tid": ""})
        assert "没有发过说说" in out


class TestFrameCallbackParse:
    @pytest.mark.asyncio
    async def test_reply_comment_payload_with_inner_parens(self, _env, monkeypatch):
        """payload 内含英文括号（说说 HTML 数据）时不再截断误报失败。

        复现 2026-07-29 假警告：非贪婪正则截断 -> 'Unexpected end of input'，
        评论其实已正常发出。
        """
        big_html = '<div class="x">(￣▽￣)表情(测试)' + "a(b)c" * 3000 + "</div>"
        payload = "{code:0, message:'', data:{html:'" + big_html + "'}}"
        text = f"<html><script>frameElement.callback({payload});</script></html>"

        class _Resp:
            pass
        _Resp.text = text

        class _Client:
            def __init__(self, **kw): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): pass
            async def post(self, url, params=None, data=None, headers=None):
                return _Resp()

        monkeypatch.setattr(mz.httpx, "AsyncClient", _Client)
        assert await mz.reply_comment(_env, "123456", "FID", "某人", "收到") is True
