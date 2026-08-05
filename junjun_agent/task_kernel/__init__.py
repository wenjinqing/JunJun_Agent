"""任务通道：Router 判定的复杂任务在这里规划-执行-验证-汇报（方案 §4.3）。

对外只暴露 kernel 单例与启用判定；processor 只调 try_submit。
"""

from junjun_agent.task_kernel.executor import enabled, enable_persistence, kernel
from junjun_agent.task_kernel.plan import Step, TaskPlan

__all__ = ["kernel", "enabled", "enable_persistence", "Step", "TaskPlan"]
