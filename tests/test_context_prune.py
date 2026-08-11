"""背景低价值行裁剪测试：超龄语气词不进上下文，但绝不误伤有语义的行。

核心断言方向是「不误伤」（白名单制、宁漏勿错杀）：
- 承诺词（好/行/可以）、占位行（[图片]）、@bot 的行、最新一条、bot 历史、
  新鲜语气词——全部保留
- 只有「超龄 + 白名单语气词 + 普通群友行」才被裁
"""

import time

from junjun_memory.short_term import ShortTermMemory, _is_filler

NOW = time.time()
OLD = NOW - 600    # 10 分钟前（超 TTL 300s）
FRESH = NOW - 60   # 1 分钟前（保鲜期内）


def _mem_with(rows):
    """rows: (text, ts, at_bot)。末尾自动补一条正常消息作为最新行。
    行内用户交替（防同一人连续合并把行并掉——合并的测试在下面专职类里）。"""
    m = ShortTermMemory()
    for i, (text, ts, at) in enumerate(rows):
        uid = f"u{i % 2 + 10}"
        m.add_user(text, nickname=f"友{i % 2 + 10}", user_id=uid, at_bot=at)
        m.entries[-1].ts = ts
    m.add_user("正经内容收尾", nickname="乙", user_id="u2")
    m.entries[-1].ts = NOW
    return m


class TestIsFiller:
    def test_filler_variants(self):
        for t in ("嗯", "嗯嗯", "哦", "啊", "哈哈", "哈哈哈哈", "笑死",
                  "草", "6", "666", "2333", "？", "?", "！", "emm"):
            assert _is_filler(t), t

    def test_commitments_not_filler(self):
        """承诺/应答词不是语气词——「今晚开黑吗」「好」里的好有语义。"""
        for t in ("好", "好的", "行", "可以", "ok", "OK", "收到", "来了",
                  "不去", "算了"):
            assert not _is_filler(t), t

    def test_content_not_filler(self):
        for t in ("今晚开黑吗", "哈哈哈哈笑死我了这个梗", "嗯我觉得不太行"):
            assert not _is_filler(t), t


class TestPrune:
    def test_old_filler_pruned(self):
        m = _mem_with([("嗯", OLD, False), ("哈哈哈", OLD, False),
                       ("666", OLD, False), ("？", OLD, False)])
        out = m.render(prune=True)
        for t in ("嗯", "哈", "666", "？"):
            assert t not in out, t
        assert "正经内容收尾" in out

    def test_fresh_filler_kept(self):
        """误判回归：保鲜期内的语气词是会话节奏，不裁。"""
        m = _mem_with([("嗯", FRESH, False), ("哈哈", FRESH, False)])
        out = m.render(prune=True)
        assert "嗯" in out and "哈哈" in out

    def test_at_bot_filler_kept(self):
        """@你 的语气词是直指你的反应，永远不裁。"""
        m = _mem_with([("嗯", OLD, True)])
        assert "嗯" in m.render(prune=True)

    def test_commitments_survive_age(self):
        """超龄的承诺词也不裁——语义不随时间过期。"""
        m = _mem_with([("好", OLD, False), ("可以", OLD, False),
                       ("行", OLD, False)])
        out = m.render(prune=True)
        for t in ("好", "可以", "行"):
            assert t in out, t

    def test_placeholders_survive(self):
        """占位行不裁：感知链路靠它知道有人发过图。"""
        m = _mem_with([("[图片]", OLD, False), ("[表情]", OLD, False)])
        out = m.render(prune=True)
        assert "[图片]" in out and "[表情]" in out

    def test_bot_history_untouched(self):
        m = _mem_with([("嗯", OLD, False)])
        m.add_bot("哈哈")
        m.entries[-1].ts = OLD
        out = m.render(prune=True)
        assert "你(历史): 哈哈" in out

    def test_prune_disabled_keeps_everything(self):
        m = _mem_with([("嗯", OLD, False)])
        assert "嗯" in m.render(prune=False)

    def test_old_content_never_pruned(self):
        """有内容的行无论多旧都保留。"""
        m = _mem_with([("三周前我说的那句很重要的话", OLD, False)])
        assert "三周前" in m.render(prune=True)

    def test_prune_default_off(self):
        """用户裁决：裁断默认关（合并已覆盖其价值，裁断留作刷屏群的手动开关）。"""
        m = _mem_with([("嗯", OLD, False)])
        assert "嗯" in m.render()  # 不传 prune 参数走配置默认


class TestSameSpeakerMerge:
    """同一人连续发言合并（零信息损失压缩）：内容一个字不少，
    只省重复的「昵称」: 前缀。"""

    def _m(self):
        return ShortTermMemory()

    def test_consecutive_same_user_merged(self):
        m = self._m()
        m.add_user("在吗", nickname="甲", user_id="u1")
        m.add_user("今晚开黑吗", nickname="甲", user_id="u1")
        m.add_user("我买了新皮肤", nickname="甲", user_id="u1")
        out = m.render()
        assert "「甲」: 在吗 ⏎ 今晚开黑吗 ⏎ 我买了新皮肤" in out
        assert out.count("「甲」") == 1, "重复昵称前缀必须只出现一次"

    def test_different_users_not_merged(self):
        m = self._m()
        m.add_user("在吗", nickname="甲", user_id="u1")
        m.add_user("在的", nickname="乙", user_id="u2")
        out = m.render()
        assert "「甲」: 在吗" in out and "「乙」: 在的" in out

    def test_bot_breaks_run(self):
        m = self._m()
        m.add_user("第一句", nickname="甲", user_id="u1")
        m.add_bot("我在")
        m.add_user("第二句", nickname="甲", user_id="u1")
        out = m.render()
        assert "「甲」: 第一句" in out and "「甲」: 第二句" in out

    def test_at_bot_marks_whole_run(self):
        """run 里任意一条 @ 了你，合并行带 [@你] 标记。"""
        m = self._m()
        m.add_user("君君", nickname="甲", user_id="u1", at_bot=True)
        m.add_user("帮我查个东西", nickname="甲", user_id="u1")
        out = m.render()
        assert "「甲」 [@你]: 君君 ⏎ 帮我查个东西" in out

    def test_mark_latest_on_merged_run(self):
        m = self._m()
        m.add_user("别人说的", nickname="乙", user_id="u2")
        m.add_user("在吗", nickname="甲", user_id="u1")
        m.add_user("看到没", nickname="甲", user_id="u1")
        out = m.render(mark_latest=True)
        assert "【最新】「甲」: 在吗 ⏎ 看到没" in out
        assert out.count("【最新】") == 1

    def test_placeholders_merge_naturally(self):
        """连发图/表情的占位行同样合并——但内容一条不丢。"""
        m = self._m()
        m.add_user("[图片]", nickname="甲", user_id="u1")
        m.add_user("[图片]", nickname="甲", user_id="u1")
        m.add_user("好看吗", nickname="甲", user_id="u1")
        out = m.render()
        assert "「甲」: [图片] ⏎ [图片] ⏎ 好看吗" in out

    def test_original_entries_not_mutated(self):
        """合并发生在渲染副本上：entries 本体（引用决策要用）不受影响。"""
        m = self._m()
        m.add_user("第一条", nickname="甲", user_id="u1", message_id="m1")
        m.add_user("第二条", nickname="甲", user_id="u1", message_id="m2")
        m.render()
        assert m.entries[0].text == "第一条" and m.entries[1].text == "第二条"
        assert m.entries[0].message_id == "m1" and m.entries[1].message_id == "m2"
