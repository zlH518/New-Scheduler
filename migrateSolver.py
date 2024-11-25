import pulp

class MigrateSolver:
    def __init__(self, config):
        """
        :param migration_cost_weight: 迁移代价的权重
        :param checkpoint_weight: checkpoint保存时间差的权重
        :param load_balance_weight: 负载均衡的权重
        """
        self.migration_cost_weight = config['migration_cost_weight']
        self.checkpoint_weight = config['checkpoint_weight']
        self.load_balance_weight = config['load_balance_weight']

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

    def solver(self, cluster, tasks, current_time):
        """
        :param cluster: 集群对象，包含节点信息（实例数据）
        :param tasks: 任务列表（实例数据）
        :param current_time: 当前时间，用于计算任务剩余时间和checkpoint时间
        :return: 迁移方案
        """
        # 创建线性规划问题
        prob = pulp.LpProblem("TaskMigration", pulp.LpMinimize)

        # 创建变量，x[i][j] 表示任务 i 是否迁移到节点 j
        x = pulp.LpVariable.dicts("x", ((task.id, node.id) for task in tasks for node in cluster.nodes),
                                  cat='Binary')

        # 目标函数项：迁移代价，节点负载均衡，checkpoint时间差
        migration_costs = []
        load_balances = []
        checkpoint_gaps = []

        for task in tasks:
            for source_node in cluster.nodes:
                for target_node in cluster.nodes:
                    migration_cost = self.calculate_migration_cost(task, source_node, target_node)
                    load_balance = self.calculate_load_balance(task, source_node, target_node)
                    checkpoint_gap = self.calculate_checkpoint_gap(task, current_time)

                    migration_costs.append(migration_cost)
                    load_balances.append(load_balance)
                    checkpoint_gaps.append(checkpoint_gap)

        # 获取归一化后的最大值和最小值
        max_migration_cost = max(migration_costs)
        min_migration_cost = min(migration_costs)
        max_load_balance = max(load_balances)
        min_load_balance = min(load_balances)
        max_checkpoint_gap = max(checkpoint_gaps)
        min_checkpoint_gap = min(checkpoint_gaps)

        # 归一化目标函数的每一项，并加权
        prob += pulp.lpSum([
            self.migration_cost_weight * self.normalize(self.calculate_migration_cost(task, source_node, target_node), 
                                                       min_migration_cost, max_migration_cost)
            * x[task.id, target_node.id]
            for task in tasks for source_node in cluster.nodes for target_node in cluster.nodes
        ])

        prob += pulp.lpSum([
            self.load_balance_weight * self.normalize(self.calculate_load_balance(task, source_node, target_node),
                                                      min_load_balance, max_load_balance)
            * x[task.id, target_node.id]
            for task in tasks for source_node in cluster.nodes for target_node in cluster.nodes
        ])

        prob += pulp.lpSum([
            self.checkpoint_weight * self.normalize(self.calculate_checkpoint_gap(task, current_time),
                                                    min_checkpoint_gap, max_checkpoint_gap)
            * x[task.id, target_node.id]
            for task in tasks for source_node in cluster.nodes for target_node in cluster.nodes
        ])

        # 约束1：每个任务只能迁移一次
        for task in tasks:
            prob += pulp.lpSum([x[task.id, target_node.id] for target_node in cluster.nodes]) <= 1

        # 约束2：任务剩余完成时间大于整体完成时间的 2/3
        for task in tasks:
            remaining_time = task.end_time - current_time
            total_time = task.end_time - task.start_time
            prob += remaining_time >= (2 / 3) * total_time

        # 约束3：节点负载约束：迁移后，源节点尽量保持空闲，目标节点尽量被占满
        for target_node in cluster.nodes:
            prob += pulp.lpSum([task.cards * x[task.id, target_node.id] for task in tasks]) <= target_node.cards

        # 约束4：每个任务从未迁移过（任务迁移计数为0）
        for task in tasks:
            prob += pulp.lpSum([x[task.id, target_node.id] for target_node in cluster.nodes]) == 1

        # 求解优化问题
        prob.solve()

        # 输出迁移方案
        migration_plan = []
        for task in tasks:
            for target_node in cluster.nodes:
                if pulp.value(x[task.id, target_node.id]) == 1:
                    source_node = next(node for node in cluster.nodes if task in node.task_list)
                    migration_plan.append((task, source_node, target_node))

        return migration_plan

    def calculate_migration_cost(self, task, source_node, target_node):
        """
        计算迁移代价（示例）
        
        :param task: 任务对象
        :param source_node: 源节点
        :param target_node: 目标节点
        :return: 迁移代价
        """
        # 这里返回任务本身的迁移代价，具体的代价计算根据实际情况进行调整
        return task.migration_cost

    def calculate_load_balance(self, task, source_node, target_node):
        """
        计算负载均衡（示例）
        
        :param task: 任务对象
        :param source_node: 源节点
        :param target_node: 目标节点
        :return: 负载均衡值
        """
        # 负载均衡计算逻辑，尽量使目标节点填满，源节点尽量空闲
        return abs(target_node.cards - sum(t.cards for t in target_node.task_list))

    def calculate_checkpoint_gap(self, task, current_time):
        """
        计算任务距离上次 checkpoint 保存的时间（示例）
        
        :param task: 任务对象
        :param current_time: 当前时间
        :return: 距离上次 checkpoint 保存的时间
        """
        return current_time - task.last_checkpoint_time

