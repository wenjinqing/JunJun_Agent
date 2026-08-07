"""Router 0-token 规则测试（离线）——宁漏勿错：误判（把闲聊派成任务）是重点防守对象。"""

from junjun_agent.router import route_to_task


class TestRouteToTask:
    # ---- 命中：强信号 ----
    def test_research_chain(self):
        assert route_to_task("帮我调研一下最近的AI新闻然后写份报告") is True

    def test_reading_chain(self):
        assert route_to_task("看完这个视频然后整理成笔记给我") is True

    def test_schedule_commitment(self):
        assert route_to_task("每天早上给我推一下AI圈的新鲜事") is True
        assert route_to_task("以后每周帮我汇总一次群里的精华") is True

    def test_explicit_multistep(self):
        assert route_to_task("先搜一下最新的模型对比然后画一张图") is True

    def test_search_then_draw_pair(self):
        assert route_to_task("查一下明日方舟新角色然后给我画一张同人") is True

    # ---- 不命中：闲聊/单步/疑问/否定 ----
    def test_single_action_is_chat(self):
        assert route_to_task("帮我查一下明天天气") is False
        assert route_to_task("给我画一张猫娘") is False

    def test_chatter(self):
        assert route_to_task("今天好累啊不想活了哈哈") is False
        assert route_to_task("在干嘛呢") is False

    def test_short_text(self):
        assert route_to_task("写报告") is False
        assert route_to_task("") is False

    def test_question_is_not_order(self):
        assert route_to_task("调研报告一般怎么写？") is False
        assert route_to_task("你会做视频笔记吗") is False

    def test_negation_is_not_order(self):
        assert route_to_task("别帮我调研了，我自己来") is False
        assert route_to_task("不用写报告了，算了") is False

    def test_join_without_actions_both_sides(self):
        # 连接词两侧都有动作词才算多步
        assert route_to_task("吃了吗然后去散步了") is False

    def test_question_with_please_still_routes(self):
        # 带「帮我/给我」的疑问句仍是委托
        assert route_to_task("能帮我调研一下然后写份报告吗") is True

    def test_report_as_object_not_deliverable(self):
        # 「报告/摘要」作宾语（查询已有之物）不是产出诉求——2026-08-06 审查实锤：
        # 扩词后这类单步查询全被拆步骤
        assert route_to_task("帮我查一下上次的体检报告出来了没") is False
        assert route_to_task("查一下这篇论文的摘要讲的啥") is False
        assert route_to_task("帮我看看这份调研的摘要写得行不行") is False
        # 真产出诉求仍然命中
        assert route_to_task("帮我查资料整理成一份行业报告") is True
