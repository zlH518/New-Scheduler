# main.py

import logging
import torch
from data.tasks import Tasks
from scheduler.ppo_scheduler import PPOScheduler
from utils.config import load_config
from utils.monitor import Monitor
import os

# 定义Cluster和Monitor类的占位符（已在 scheduler/ppo_scheduler.py 中定义）
# 在实际项目中，这些类应放在适当的模块中，并具备完整的功能

class Cluster:
    """
    Cluster类管理集群资源和任务分配。
    """

    def __init__(self, num_nodes=10, num_gpus_per_node=8):
        """
        初始化Cluster实例。

        参数:
            num_nodes (int, optional): 集群中的物理节点数量。默认为10。
            num_gpus_per_node (int, optional): 每个物理节点的GPU数量。默认为8。
        """
        self.num_nodes = num_nodes
        self.num_gpus_per_node = num_gpus_per_node
        self.nodes = [Node(node_id=i, num_gpus=num_gpus_per_node) for i in range(num_nodes)]
        self.waiting_queue = []
        self.running_tasks = []
        self.completed_tasks = []
        self.utilization = 0.0
        self.fragmentation_rate = 0.0
        self.throughput = 0.0
        self.avg_waiting_time = 0.0

    def add_to_waiting_queue(self, task):
        """
        将任务添加到等待队列。

        参数:
            task (Task): 任务对象。
        """
        self.waiting_queue.append(task)

    def get_active_tasks(self):
        """
        获取所有正在运行或等待的任务。

        返回:
            list: Task实例的列表。
        """
        return self.running_tasks + self.waiting_queue

    def get_task_by_id(self, task_id):
        """
        根据任务ID获取任务对象。

        参数:
            task_id (int): 任务ID。

        返回:
            Task: 任务对象，若未找到则为None。
        """
        for task in self.running_tasks + self.waiting_queue:
            if task.task_id == task_id:
                return task
        return None

    def can_allocate(self, task, node_id):
        """
        检查是否可以将任务分配到指定节点。

        参数:
            task (Task): 任务对象。
            node_id (int): 节点ID。

        返回:
            bool: 是否可以分配。
        """
        node = self.nodes[node_id]
        return node.can_allocate(task)

    def allocate_task(self, task, node_id):
        """
        将任务分配到指定节点，更新GPU状态。

        参数:
            task (Task): 任务对象。
            node_id (int): 节点ID。
        """
        node = self.nodes[node_id]
        node.allocate(task)
        task.status = "RUNNING"
        self.running_tasks.append(task)

    def can_migrate(self, task, src_node_id, des_node_id):
        """
        检查是否可以迁移任务。

        参数:
            task (Task): 任务对象。
            src_node_id (int): 源节点ID。
            des_node_id (int): 目标节点ID。

        返回:
            bool: 是否可以迁移。
        """
        src_node = self.nodes[src_node_id]
        des_node = self.nodes[des_node_id]
        return src_node.can_migrate(task) and des_node.can_allocate(task)

    def migrate_task(self, task, src_node_id, des_node_id):
        """
        迁移任务到目标节点，释放源节点资源并分配到目标节点。

        参数:
            task (Task): 任务对象。
            src_node_id (int): 源节点ID。
            des_node_id (int): 目标节点ID。
        """
        src_node = self.nodes[src_node_id]
        des_node = self.nodes[des_node_id]
        src_node.migrate_task(task, des_node)
        logging.info(f"Migrated Task {task.task_id} from Node {src_node_id} to Node {des_node_id}.")

    def can_stop(self, task, node_id):
        """
        检查是否可以停止任务。

        参数:
            task (Task): 任务对象。
            node_id (int): 节点ID。

        返回:
            bool: 是否可以停止。
        """
        node = self.nodes[node_id]
        return node.can_stop(task)

    def stop_task(self, task, node_id):
        """
        停止任务，释放节点资源，并将任务状态更新为等待。

        参数:
            task (Task): 任务对象。
            node_id (int): 节点ID。
        """
        node = self.nodes[node_id]
        node.stop_task(task)
        task.status = "WAITING"
        self.running_tasks.remove(task)
        self.waiting_queue.append(task)
        logging.info(f"Stopped Task {task.task_id} on Node {node_id}.")

    def release_completed_tasks(self, current_time):
        """
        释放已完成的任务。

        参数:
            current_time (int): 当前时间。

        返回:
            list: 已完成的Task实例列表。
        """
        completed = []
        for task in self.running_tasks:
            if task.real_end_time and task.real_end_time <= current_time:
                completed.append(task)
        for task in completed:
            self.running_tasks.remove(task)
            self.completed_tasks.append(task)
            logging.info(f"Task {task.task_id} completed at time {current_time}.")
        return completed

    def get_utilization(self):
        """
        计算集群的利用率。

        返回:
            float: 集群利用率。
        """
        total_gpus = self.num_nodes * self.num_gpus_per_node
        used_gpus = sum(node.get_used_gpus() for node in self.nodes)
        self.utilization = used_gpus / total_gpus
        return self.utilization

    def get_fragmentation_rate(self):
        """
        计算集群的碎片率。

        返回:
            float: 集群碎片率。
        """
        # 示例：碎片率 = 空闲GPU的最大连续数 / 总GPU数
        max_continuous_free = 0
        current_free = 0
        for node in self.nodes:
            node_free = node.get_continuous_free_gpus()
            max_continuous_free = max(max_continuous_free, node_free)
        self.fragmentation_rate = max_continuous_free / (self.num_nodes * self.num_gpus_per_node)
        return self.fragmentation_rate

    def get_throughput(self):
        """
        计算集群的吞吐量。

        返回:
            float: 集群吞吐量。
        """
        # 示例：吞吐量 = 完成任务数 / 总时间
        total_time = self.current_time if hasattr(self, 'current_time') else 1
        self.throughput = len(self.completed_tasks) / total_time
        return self.throughput

    def get_avg_waiting_time(self):
        """
        计算集群中任务的平均等待时间。

        返回:
            float: 平均等待时间。
        """
        if not self.running_tasks and not self.waiting_queue:
            self.avg_waiting_time = 0.0
        else:
            total_waiting_time = sum(task.start_time - task.create_time for task in self.waiting_queue)
            num_tasks = len(self.waiting_queue)
            self.avg_waiting_time = total_waiting_time / num_tasks if num_tasks > 0 else 0.0
        return self.avg_waiting_time

class Node:
    """
    Node类表示集群中的单个物理节点，管理其GPU资源。
    """

    def __init__(self, node_id, num_gpus=8):
        """
        初始化Node实例。

        参数:
            node_id (int): 节点ID。
            num_gpus (int, optional): 节点中的GPU数量。默认为8。
        """
        self.node_id = node_id
        self.num_gpus = num_gpus
        self.gpu_status = [0] * num_gpus  # 0表示空闲，1表示占用
        self.tasks = []  # 当前分配到该节点的任务

    def can_allocate(self, task):
        """
        检查是否可以在该节点上分配任务。

        参数:
            task (Task): 任务对象。

        返回:
            bool: 是否可以分配。
        """
        required_gpus = task.cards
        available_gpus = self.gpu_status.count(0)
        return available_gpus >= required_gpus

    def allocate(self, task):
        """
        在该节点上分配任务，更新GPU状态。

        参数:
            task (Task): 任务对象。
        """
        required_gpus = task.cards
        allocated = 0
        for i in range(self.num_gpus):
            if self.gpu_status[i] == 0:
                self.gpu_status[i] = 1
                allocated += 1
                if allocated == required_gpus:
                    break
        task.node_id.append(self.node_id)
        self.tasks.append(task)
        logging.info(f"Allocated Task {task.task_id} to Node {self.node_id}.")

    def can_migrate(self, task):
        """
        检查是否可以从该节点迁移任务。

        参数:
            task (Task): 任务对象。

        返回:
            bool: 是否可以迁移。
        """
        return task in self.tasks

    def migrate_task(self, task, des_node):
        """
        从该节点迁移任务到目标节点，释放资源。

        参数:
            task (Task): 任务对象。
            des_node (Node): 目标节点对象。
        """
        if task in self.tasks:
            # 释放GPU
            for _ in range(task.cards):
                for i in range(self.num_gpus):
                    if self.gpu_status[i] == 1:
                        self.gpu_status[i] = 0
                        break
            # 移除任务
            self.tasks.remove(task)
            task.node_id.remove(self.node_id)
            # 分配到目标节点
            des_node.allocate(task)
            logging.info(f"Migrated Task {task.task_id} from Node {self.node_id} to Node {des_node.node_id}.")

    def can_stop(self, task):
        """
        检查是否可以停止任务。

        参数:
            task (Task): 任务对象。

        返回:
            bool: 是否可以停止。
        """
        return task in self.tasks

    def stop_task(self, task):
        """
        停止任务，释放资源。

        参数:
            task (Task): 任务对象。
        """
        if task in self.tasks:
            # 释放GPU
            for _ in range(task.cards):
                for i in range(self.num_gpus):
                    if self.gpu_status[i] == 1:
                        self.gpu_status[i] = 0
                        break
            # 移除任务
            self.tasks.remove(task)
            task.node_id.remove(self.node_id)
            logging.info(f"Stopped Task {task.task_id} on Node {self.node_id}.")

    def get_used_gpus(self):
        """
        获取节点上已使用的GPU数量。

        返回:
            int: 已使用的GPU数量。
        """
        return self.gpu_status.count(1)

    def get_continuous_free_gpus(self):
        """
        获取节点上连续空闲的GPU最大数量。

        返回:
            int: 最大连续空闲GPU数量。
        """
        max_free = 0
        current_free = 0
        for status in self.gpu_status:
            if status == 0:
                current_free += 1
                if current_free > max_free:
                    max_free = current_free
            else:
                current_free = 0
        return max_free



# main.py

import logging
import torch
from data.tasks import Tasks
from scheduler.ppo_scheduler import PPOScheduler
from utils.config import load_config
from utils.monitor import Monitor
import os

def main():
    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 加载配置
    config_path = 'config.json'
    if not os.path.exists(config_path):
        logging.error(f"Configuration file {config_path} does not exist.")
        return

    config = load_config(config_path)

    # 初始化Tasks
    task_config = config['taskConfig']
    tasks_obj = Tasks(task_config)
    tasks = tasks_obj.tasks

    # 初始化Cluster
    cluster_config = config['clusterConfig']
    cluster = Cluster(num_nodes=cluster_config.get('num_nodes', 10),
                      num_gpus_per_node=cluster_config.get('num_gpus_per_node', 8))

    # 初始化Monitor
    monitor = Monitor()

    # 初始化PPOScheduler
    ppo_config = config['ppoConfig']
    ppo_scheduler = PPOScheduler(config=ppo_config)

    # 运行调度器
    ppo_scheduler.run(cluster, tasks, monitor)

    # 保存模型
    model_save_path = config['ppoConfig']['modelSavePath']
    ppo_scheduler.save_model(model_save_path)

if __name__ == "__main__":
    main()
