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


class TestRouteWidening20260813:
    """2026-08-13 加宽（「写一份研究笔记」漏路由——疯狂搜索事故第一环）：
    重产出委托 / 查询×提醒链 / 文件加工链 / 发布链扩展。每条都配误判回归。"""

    # ---- 命中：新信号 ----
    def test_heavy_deliverable(self):
        assert route_to_task("写一份研究笔记") is True
        assert route_to_task("帮我做一份塞尔达的攻略") is True
        assert route_to_task("今天有什么科技新闻，给我整一份简报") is True

    def test_query_then_remind_chain(self):
        assert route_to_task("查一下明天北京天气，如果下雨就提醒我带伞") is True
        assert route_to_task("查一下下周三有什么电影上映，上映前一天提醒我买票") is True

    def test_file_process_chain(self):
        assert route_to_task("把工作区的 notes.md 转成表格存回去") is True
        assert route_to_task("我刚发了个表格过来，帮我统计一下里面的销售额") is True
        assert route_to_task("用工作区的数据画个趋势图存起来") is True

    def test_publish_chain_widened(self):
        """发布必须走任务通道才有人审门（对话通道直发没门）。"""
        assert route_to_task("搜最近科技新闻写个综述发到空间") is True
        assert route_to_task("看个 B 站热门视频，把感想发到空间") is True

    def test_produce_noun_completion(self):
        assert route_to_task("查查 2026 年有什么新出的好评游戏，给我个推荐清单") is True
        assert route_to_task("搜一下新能源汽车的最新政策，整理要点给我") is True
        assert route_to_task("翻翻咱们这周都聊过什么，给我个总结") is True

    # ---- 误判回归：日常句子不许派单 ----
    def test_salary_is_not_file(self):
        """「刚发」必须配文件类名词——工资是日常不是派单。"""
        assert route_to_task("我刚发工资了，今晚搓一顿") is False
        assert route_to_task("我刚发工资了，帮我存起来") is False

    def test_plain_reminder_stays_chat(self):
        """单步提醒走对话通道的 set_reminder，不派任务。"""
        assert route_to_task("提醒我明天下午三点开会") is False

    def test_light_produce_stays_chat(self):
        """轻产出（段子/主意/小作文）单轮合成足够，不派任务。"""
        assert route_to_task("写个段子逗我开心") is False
        assert route_to_task("帮我出个主意呗") is False

    def test_single_query_stays_chat(self):
        assert route_to_task("搜一下附近哪家火锅好吃") is False
        assert route_to_task("看看这个视频拍得怎么样") is False

    def test_question_guard_beats_produce(self):
        """疑问句守卫优先于产出信号（「做个表格那么难吗」是吐槽）。"""
        assert route_to_task("做个表格那么难吗") is False
        assert route_to_task("今天有什么新闻吗") is False

    def test_negation_guard_beats_chain(self):
        assert route_to_task("帮我查一下天气，如果热就算了") is False

    def test_subscription_stays_chat(self):
        """盯梢/订阅是对话通道 subscribe_updates 的专属地盘——「盯…更新了
        告诉我」派成一次性任务是误路由（2026-08-13 golden_cases 实锤）。"""
        assert route_to_task("帮我盯一下UP主影视飓风，更新了第一时间告诉我") is False
        assert route_to_task("订阅一下这个作者，出新了叫我") is False

    def test_plain_summary_stays_chat(self):
        """「总结一下聊的」是单步蒸馏（无来源动作），对话通道足够。"""
        assert route_to_task("总结一下我们今天聊了啥") is False
