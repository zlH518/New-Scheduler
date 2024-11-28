import pulp
import random
import numpy as np
import math



class MigrateSolver:
    def __init__(self, config):
        """
        :param migration_cost_weight: 迁移代价的权重
        :param checkpoint_weight: checkpoint保存时间差的权重
        :param load_balance_weight: 负载均衡的权重
        """
        self.config = config

    def normalize(self, value, min_value, max_value):
        """
        :param value: 需要归一化的值
        :param min_value: 最小值
        :param max_value: 最大值
        :return: 归一化后的值
        """
        if max_value != min_value:
            return (value - min_value) / (max_value - min_value)
        return 0

    def solver(self, nodes, tasks, current_time):
        """
        :param cluster: 集群对象，包含节点信息（实例数据）
        :param tasks: 任务列表（实例数据）
        :param current_time: 当前时间，用于计算任务剩余时间和checkpoint时间
        :return: 迁移方案
        """
        prob = pulp.LpProblem("TaskMigration", pulp.LpMinimize)

        x = {}
        migration_costs = []
        load_balances = []
        checkpoint_gaps = []

        for task in tasks:
            for source_node in nodes:
                for target_node in nodes:
                    if source_node.Id == target_node.Id:
                        continue
                    x[task.task_id, source_node.Id, target_node.Id] = pulp.LpVariable(f"x_{task.task_id}_{source_node.Id}_{target_node.Id}", 0, 1, pulp.LpInteger)

                    migration_cost = self.calculate_migration_cost(task, source_node, target_node, current_time)
                    load_balance = self.calculate_load_balance(task, target_node)
                    checkpoint_gap = self.calculate_checkpoint_gap(task, current_time)

                    migration_costs.append(migration_cost)
                    load_balances.append(load_balance)
                    checkpoint_gaps.append(checkpoint_gap)

        max_migration_cost = max(migration_costs)
        min_migration_cost = min(migration_costs)
        max_load_balance = max(load_balances)
        min_load_balance = min(load_balances)
        max_checkpoint_gap = max(checkpoint_gaps)
        min_checkpoint_gap = min(checkpoint_gaps)

        prob += pulp.lpSum([self.config['migration_cost_weight'] * x[task.task_id, source_node.Id, target_node.Id] * self.normalize(migration_cost, min_migration_cost, max_migration_cost) +
                            self.config['load_balance_weight'] * x[task.task_id, source_node.Id, target_node.Id] * self.normalize(load_balance, min_load_balance, max_load_balance) +
                            self.config['checkpoint_weight'] * x[task.task_id, source_node.Id, target_node.Id] * self.normalize(checkpoint_gap, min_checkpoint_gap, max_checkpoint_gap)
                            for task in tasks for source_node in nodes for target_node in nodes if source_node.Id != target_node.Id])

        # 约束1：只有一个任务迁移
        for task in tasks:
            prob += pulp.lpSum([x[task.task_id, source_node.Id, target_node.Id] for source_node in nodes for target_node in nodes if source_node.Id != target_node.Id]) <= 1

        # 约束2：所有任务中只能有一个任务被迁移
        prob += pulp.lpSum([x[task.task_id, source_node.Id, target_node.Id] for task in tasks for source_node in nodes for target_node in nodes if source_node.Id != target_node.Id]) == 1


        # 约束3：选择迁移的任务的剩余完成时间大于整体完成时间的 2/3
        for task in tasks:
            remaining_time = (task.duration_time - (current_time - task.real_start_time))
            if remaining_time > (2 / 3) * task.duration_time:
                prob += pulp.lpSum([x[task.task_id, source_node.Id, target_node.Id] for source_node in nodes for target_node in nodes if source_node.Id != target_node.Id]) >= 1

        # 求解优化问题
        prob.solve()

        # 输出迁移方案，要求输入source_node和target_node的id和迁移的任务的id
        # 输出迁移方案
        for task in tasks:
            for source_node in nodes:
                for target_node in nodes:
                    if source_node.Id != target_node.Id and x[task.task_id, source_node.Id, target_node.Id].varValue == 1.0:
                        print(f"迁移任务{task.task_id}：源节点{source_node.Id} -> 目标节点{target_node.Id}")
                        return source_node.Id, target_node.Id, task.task_id

    def calculate_migration_cost(self, task, source_node, target_node, current_time):
        """
        计算迁移代价
        :param task: 任务对象
        :param source_node: 源节点
        :param target_node: 目标节点
        :return: 迁移代价

        通信代价根据目标节点和源节点之间的Id进行hash运算当作随机数种子，然后随机0.5~1.5
        cost的比例根据任务的剩余gpu_time进行计算
        """
        task_left_gpu_time = task.cards * (current_time - task.real_start_time)
        seed = hash((source_node.Id, target_node.Id)) 
        random.seed(seed)
        commucation_rate = random.uniform(0.5, 1.5)
        cost = self.config['migration_cost_rate'] * task_left_gpu_time * commucation_rate
        return cost

    def calculate_load_balance(self, task, target_node):
        """
        计算迁移适配度
        这里首先是任务与目标节点的适配程度，如果任务的卡数大于目标节点的空闲卡，那么直接不适配，返回无穷负数
        如果任务的卡数小于目标节点的空闲卡，如果任务的卡数和目标节点的空闲卡距离越小，那么适配值越大，
        以上条件都相等的话，如果任务本身的卡数越大，那么适配度越小(尽量不移动卡数比较大的任务)
        
        :param task: 任务对象
        :param source_node: 源节点
        :param target_node: 目标节点
        :return: 负载均衡值
        """
        if task.cards > target_node.cards:
            return -math.inf  
        card_difference = target_node.cards - task.cards
        adaptation_score = 1 / (card_difference + 1)  
        task_card_penalty = 1 / task.cards
        load_balance_value = adaptation_score + task_card_penalty
        return load_balance_value


    def calculate_checkpoint_gap(self, task, current_time):
        """
        计算任务距离上次 checkpoint 保存的时间长短
        :param task: 任务对象
        :param current_time: 当前时间
        :return: 距离上次 checkpoint 保存的时间
        
        这里每个任务的checkpointtime是按照最佳ck时间来计算的来模拟的
        """
        return 100
        
