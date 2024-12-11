# data/task.py

import logging

class Task:
    """
    Task类表示集群中的单个任务，包含任务的各种属性和状态。
    """
    TaskId = 0  # 静态变量，用于生成唯一的任务ID

    def __init__(self, create_time, start_time, cards, duration_time,
                 gpu_time, migration_cost, pre_queue_time, node_num, checkpoint_time=2):
        """
        初始化Task实例。

        参数:
            create_time (int): 任务创建时间（归一化后）。
            start_time (int): 任务开始时间（归一化后）。
            cards (int): 任务需要的GPU数量。
            duration_time (int): 任务持续时间（归一化后）。
            gpu_time (float): GPU运行时间。
            migration_cost (float): 任务迁移成本。
            pre_queue_time (float): 任务在队列中的等待时间。
            node_num (int): 任务需要的节点数量。
            checkpoint_time (int, optional): 检查点周期。默认为2小时。
        """
        self.task_id = Task.TaskId
        self.create_time = create_time
        self.start_time = start_time
        self.cards = cards
        self.gpu_time = gpu_time
        self.duration_time = duration_time
        self.migration_cost = migration_cost
        self.pre_queue_time = pre_queue_time
        self.node_num = node_num
        self.checkpoint_time = checkpoint_time
        self.migration_times = 0
        self.node_id = []
        self.status = "UNARRIVAL"  # 任务状态：UNARRIVAL, WAITING, RUNNING, COMPLETED
        self.release = False

        # 实际参数，用于记录真实的开始和结束时间
        self.real_start_time = None
        self.real_end_time = None

        Task.TaskId += 1  # 更新静态任务ID

    def __repr__(self):
        """
        返回Task对象的字符串表示。
        """
        return (f"Task(task_id={self.task_id}, cards={self.cards}, node_num={self.node_num})")
