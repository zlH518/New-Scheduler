# scheduler/ppo_scheduler.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import logging
from collections import deque
from utils.experience import Experience
from scheduler.gnn_models import TaskGNN, TaskAggregationGNN, ClusterEncoder, PolicyNetwork, ValueNetwork
from torch_geometric.data import Data, Batch

class PPOScheduler:
    """
    PPOScheduler类实现基于PPO算法的调度器，结合GNN来表示和聚合任务与集群状态。
    """

    def __init__(self, config):
        """
        初始化PPOScheduler。

        参数:
            config (dict): 配置字典，包含网络参数、超参数等。
        """
        self.config = config
        self.gamma = config.get('gamma', 0.99)  # 折扣因子
        self.lamda = config.get('lamda', 0.95)  # GAE参数
        self.clip_eps = config.get('clip_eps', 0.2)  # PPO剪切参数
        self.lr = config.get('learning_rate', 1e-4)  # 学习率
        self.batch_size = config.get('batch_size', 64)  # 每个batch的大小
        self.epochs = config.get('epochs', 10)  # PPO更新的epochs
        self.memory = deque(maxlen=config.get('memory_size', 10000))  # 经验池

        # 初始化GNN模型和编码器
        task_emb_dim = config.get('task_emb_dim', 64)
        cluster_emb_dim = config.get('cluster_emb_dim', 128)
        num_heads = config.get('num_heads', 4)
        num_layers = config.get('num_layers', 2)
        dropout = config.get('dropout', 0.6)

        # 初始化GNN
        self.task_gnn = TaskGNN(input_dim=config['task_input_dim'],
                                hidden_dim=config['task_hidden_dim'],
                                output_dim=task_emb_dim,
                                num_heads=num_heads,
                                num_layers=num_layers,
                                dropout=dropout)

        self.task_agg_gnn = TaskAggregationGNN(input_dim=task_emb_dim,
                                               hidden_dim=config['agg_hidden_dim'],
                                               output_dim=task_emb_dim,
                                               num_heads=num_heads,
                                               num_layers=num_layers,
                                               dropout=dropout)

        self.cluster_encoder = ClusterEncoder(input_dim=config['cluster_input_dim'],
                                             hidden_dim=config['cluster_hidden_dim'],
                                             output_dim=cluster_emb_dim)

        # 初始化策略网络和价值网络
        num_ops = config['num_ops']
        num_tasks = config['num_tasks']
        num_nodes = config['num_nodes']
        state_dim = task_emb_dim + cluster_emb_dim

        self.policy_net = PolicyNetwork(input_dim=state_dim,
                                        hidden_dim=config['policy_hidden_dim'],
                                        num_ops=num_ops,
                                        num_tasks=num_tasks,
                                        num_nodes=num_nodes)

        self.value_net = ValueNetwork(input_dim=state_dim,
                                      hidden_dim=config['value_hidden_dim'])

        # 初始化优化器
        self.optimizer_policy = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.optimizer_value = optim.Adam(self.value_net.parameters(), lr=self.lr)

        # 定义损失函数
        self.value_loss_fn = nn.MSELoss()

    def run(self, cluster, tasks, monitor):
        """
        主循环，运行调度器直到所有任务完成。

        参数:
            cluster (Cluster): 集群对象，管理资源和任务分配。
            tasks (list): Task实例的列表。
            monitor (Monitor): 监控对象，用于更新和记录集群状态。
        """
        logging.info("PPO Scheduler started.")
        self.current_time = tasks[0].create_time if tasks else 0
        self.completed_tasks = []

        while len(self.completed_tasks) < len(tasks):
            # 1. 任务释放
            self.release_tasks(cluster)

            # 2. 新任务到达并加入等待队列
            self.add_new_tasks(cluster, tasks)

            # 3. 获取当前状态
            state = self.get_state(cluster, tasks)

            # 4. 选择并执行动作
            action = self.select_action(state)
            reward, done = self.execute_action(action, cluster, tasks)

            # 5. 获取下一个状态
            next_state = self.get_state(cluster, tasks)

            # 6. 存储经验
            self.memory.append(Experience(state, action, reward, next_state, done))

            # 7. 更新模型
            self.update_model()

            # 8. 更新集群的状况信息
            self.update_cluster(cluster, tasks, monitor)

            # 9. 时间递增
            self.current_time += self.config.get('time_step', 1)

        logging.info("PPO Scheduler completed all tasks.")

    def release_tasks(self, cluster):
        """
        释放已完成的任务。

        参数:
            cluster (Cluster): 集群对象。
        """
        completed = cluster.release_completed_tasks(self.current_time)
        self.completed_tasks.extend(completed)
        if completed:
            logging.info(f"Released {len(completed)} tasks at time {self.current_time}.")

    def add_new_tasks(self, cluster, tasks):
        """
        添加新到达的任务到等待队列。

        参数:
            cluster (Cluster): 集群对象。
            tasks (list): Task实例的列表。
        """
        for task in tasks:
            if task.create_time <= self.current_time and task.status == "UNARRIVAL":
                cluster.add_to_waiting_queue(task)
                task.status = "WAITING"
                logging.info(f"Task {task.task_id} arrived and added to waiting queue.")

    def get_state(self, cluster, tasks):
        """
        构建当前状态的嵌入向量。

        参数:
            cluster (Cluster): 集群对象。
            tasks (list): Task实例的列表。

        返回:
            Tensor: 拼接后的状态向量，形状 [task_emb_dim + cluster_emb_dim]。
        """
        # 1. 获取所有正在运行或等待的任务
        active_tasks = cluster.get_active_tasks()

        # 2. 为每个任务构建阶段图并提取嵌入
        task_embeddings = []
        for task in active_tasks:
            # 计算任务已运行的阶段
            num_stages = int(torch.ceil(torch.tensor(task.duration_time / task.checkpoint_time)).item())
            # 假设task.current_stage记录当前运行的阶段
            current_stage = task.current_stage  # 0-based index
            # 构建阶段节点特征
            stage_features = []
            positions = []
            for stage in range(num_stages):
                stage_duration = min(task.checkpoint_time,
                                     task.duration_time - stage * task.checkpoint_time)
                stage_feature = torch.tensor([
                    task.cards,  # GPU需求
                    task.node_num,  # 节点需求
                    stage_duration / 1000.0,  # 归一化的执行时间
                    task.checkpoint_time / 100.0  # 归一化的checkpoint周期
                ], dtype=torch.float32)
                stage_features.append(stage_feature)
                positions.append(stage)  # 位置编码

            if stage_features:
                stage_nodes = torch.stack(stage_features)  # [num_stages, feature_dim]
                positions_tensor = torch.tensor(positions, dtype=torch.long)
            else:
                stage_nodes = torch.empty((0, 4), dtype=torch.float32)
                positions_tensor = torch.empty((0,), dtype=torch.long)

            # 构建边索引，顺序依赖
            edge_index = []
            for i in range(num_stages - 1):
                edge_index.append([i, i + 1])
                edge_index.append([i + 1, i])
            if edge_index:
                edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # [2, num_edges]
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)

            # 任务阶段图嵌入
            task_embedding = self.task_gnn(stage_nodes, edge_index, positions_tensor)  # [num_stages, task_emb_dim]
            # 选择当前运行的阶段和其子阶段
            if num_stages > current_stage:
                active_embedding = torch.mean(task_embedding[current_stage:], dim=0)  # [task_emb_dim]
            else:
                active_embedding = torch.mean(task_embedding, dim=0)  # [task_emb_dim]
            task_embeddings.append(active_embedding)

        # 3. 聚合所有任务的嵌入
        if task_embeddings:
            task_embeddings_tensor = torch.stack(task_embeddings)  # [num_tasks, task_emb_dim]
            # 构建任务聚合图（全连接）
            agg_edge_index = []
            num_tasks = task_embeddings_tensor.size(0)
            for i in range(num_tasks):
                for j in range(i + 1, num_tasks):
                    agg_edge_index.append([i, j])
                    agg_edge_index.append([j, i])
            if agg_edge_index:
                agg_edge_index = torch.tensor(agg_edge_index, dtype=torch.long).t().contiguous()  # [2, num_edges]
            else:
                agg_edge_index = torch.empty((2, 0), dtype=torch.long)
            global_task_embedding = self.task_agg_gnn(task_embeddings_tensor, agg_edge_index)  # [num_tasks, task_emb_dim]
            global_task_embedding = torch.mean(global_task_embedding, dim=0)  # [task_emb_dim]
        else:
            global_task_embedding = torch.zeros(self.task_agg_gnn.convs[-1].out_channels, dtype=torch.float32)

        # 4. 编码集群状态
        cluster_metrics = torch.tensor([
            cluster.utilization,          # 集群利用率
            cluster.fragmentation_rate,  # 碎片率
            cluster.throughput,           # 吞吐量
            cluster.avg_waiting_time      # 平均等待时间
        ], dtype=torch.float32)
        cluster_embedding = self.cluster_encoder(cluster_metrics)  # [cluster_emb_dim]

        # 5. 拼接任务嵌入和集群嵌入
        combined_state = torch.cat([global_task_embedding, cluster_embedding], dim=0)  # [task_emb_dim + cluster_emb_dim]

        return combined_state  # [task_emb_dim + cluster_emb_dim]

    def select_action(self, state):
        """
        根据当前状态选择动作。

        参数:
            state (Tensor): 当前状态向量，形状 [task_emb_dim + cluster_emb_dim]。

        返回:
            tuple: (op, task_id, src_node, des_node)
        """
        self.policy_net.eval()
        with torch.no_grad():
            # 添加batch维度
            state = state.unsqueeze(0)  # [1, task_emb_dim + cluster_emb_dim]
            # 通过策略网络获取概率分布
            op_probs, task_id_probs, src_node_id_probs, des_node_id_probs = self.policy_net(state)

        # 采样操作类型
        op_dist = torch.distributions.Categorical(op_probs)
        op = op_dist.sample().item()

        # 根据操作类型采样其他动作分量
        if op == 0:
            # 无操作，其他动作分量无意义
            task_id = -1
            src_node = -1
            des_node = -1
        elif op == 1:
            # 放置操作，只需要 des_node_id
            task_id = self.sample_task_id(task_id_probs)
            src_node = -1
            des_node = torch.distributions.Categorical(des_node_id_probs).sample().item()
        elif op == 2:
            # 迁移操作，需要 task_id, src_node_id, des_node_id
            task_id = self.sample_task_id(task_id_probs)
            src_node = torch.distributions.Categorical(src_node_id_probs).sample().item()
            des_node = torch.distributions.Categorical(des_node_id_probs).sample().item()
        elif op == 3:
            # 停止操作，需要 task_id 和 src_node_id
            task_id = self.sample_task_id(task_id_probs)
            src_node = torch.distributions.Categorical(src_node_id_probs).sample().item()
            des_node = -1
        else:
            # 默认无操作
            task_id = -1
            src_node = -1
            des_node = -1

        return op, task_id, src_node, des_node

    def sample_task_id(self, task_id_probs):
        """
        根据任务ID的概率分布采样一个任务ID。

        参数:
            task_id_probs (Tensor): 任务ID的概率分布，形状 [1, num_tasks]。

        返回:
            int: 采样得到的任务ID。
        """
        task_dist = torch.distributions.Categorical(task_id_probs)
        task_id = task_dist.sample().item()
        return task_id

    def execute_action(self, action, cluster, tasks):
        """
        执行动作并计算奖励。

        参数:
            action (tuple): (op, task_id, src_node, des_node)
            cluster (Cluster): 集群对象。
            tasks (list): Task实例的列表。

        返回:
            tuple: (reward, done)
        """
        op, task_id, src_node, des_node = action
        reward = 0
        done = False

        if op == 0:
            # 无操作
            logging.debug("Action: No operation.")
            pass
        elif op == 1:
            # 放置任务
            task = cluster.get_task_by_id(task_id)
            if task and cluster.can_allocate(task, des_node):
                cluster.allocate_task(task, des_node)
                cluster.remove_from_waiting_queue(task)
                reward += self.compute_reward(cluster, task)
                logging.info(f"Action: Allocated Task {task_id} to Node {des_node}.")
            else:
                # 无法分配，给予负奖励
                reward -= 1
                logging.warning(f"Action Failed: Cannot allocate Task {task_id} to Node {des_node}.")
        elif op == 2:
            # 迁移任务
            task = cluster.get_task_by_id(task_id)
            if task and cluster.can_migrate(task, src_node, des_node):
                cluster.migrate_task(task, src_node, des_node)
                reward += self.compute_reward(cluster, task)
                logging.info(f"Action: Migrated Task {task_id} from Node {src_node} to Node {des_node}.")
            else:
                # 无法迁移，给予负奖励
                reward -= 1
                logging.warning(f"Action Failed: Cannot migrate Task {task_id} from Node {src_node} to Node {des_node}.")
        elif op == 3:
            # 停止任务
            task = cluster.get_task_by_id(task_id)
            if task and cluster.can_stop(task, src_node):
                cluster.stop_task(task, src_node)
                cluster.add_to_waiting_queue(task)
                reward -= self.compute_penalty()
                logging.info(f"Action: Stopped Task {task_id} on Node {src_node}.")
            else:
                # 无法停止，给予负奖励
                reward -= 1
                logging.warning(f"Action Failed: Cannot stop Task {task_id} on Node {src_node}.")
        else:
            # 无效操作
            logging.warning(f"Invalid Action Type: {op}")
            reward -= 1

        # 计算奖励，考虑集群利用率和任务完成时间
        utilization = cluster.get_utilization()
        fragmentation_rate = cluster.get_fragmentation_rate()
        throughput = cluster.get_throughput()
        avg_waiting_time = cluster.get_avg_waiting_time()
        reward += self.reward_function(utilization, fragmentation_rate, throughput, avg_waiting_time)

        # 检查是否完成所有任务
        if len(self.completed_tasks) == len(tasks):
            done = True

        return reward, done

    def reward_function(self, utilization, fragmentation_rate, throughput, avg_waiting_time):
        """
        奖励函数设计，基于集群的多个指标。

        参数:
            utilization (float): 集群利用率。
            fragmentation_rate (float): 集群碎片率。
            throughput (float): 集群吞吐量。
            avg_waiting_time (float): 平均等待时间。

        返回:
            float: 奖励值。
        """
        alpha = self.config.get('alpha', 1.0)
        beta = self.config.get('beta', 1.0)
        gamma = self.config.get('gamma', 1.0)
        delta = self.config.get('delta', 1.0)

        utilization_reward = alpha * utilization
        fragmentation_penalty = -beta * fragmentation_rate
        throughput_reward = gamma * throughput
        waiting_time_penalty = -delta * avg_waiting_time

        total_reward = utilization_reward + fragmentation_penalty + throughput_reward + waiting_time_penalty
        return total_reward

    def compute_reward(self, cluster, task):
        """
        计算任务成功分配后的奖励。

        参数:
            cluster (Cluster): 集群对象。
            task (Task): 任务对象。

        返回:
            float: 奖励值。
        """
        return 1.0  # 示例：任务成功分配后给予正奖励

    def compute_penalty(self):
        """
        计算停止任务的惩罚。

        返回:
            float: 惩罚值。
        """
        return 1.0  # 示例：停止任务给予负奖励

    def update_model(self):
        """
        使用经验池中的经验来更新策略网络和价值网络。
        """
        if len(self.memory) < self.batch_size:
            return  # 如果经验池中的经验不足，暂不更新

        # 从经验池中采样一个batch
        experiences = [self.memory.popleft() for _ in range(self.batch_size)]
        batch = Experience(*zip(*experiences))

        # 转换为张量
        states = torch.stack(batch.state)  # [batch_size, state_dim]
        actions = torch.tensor(batch.action, dtype=torch.long)  # [batch_size, 4]
        rewards = torch.tensor(batch.reward, dtype=torch.float32)  # [batch_size]
        next_states = torch.stack(batch.next_state)  # [batch_size, state_dim]
        dones = torch.tensor(batch.done, dtype=torch.float32)  # [batch_size]

        # 计算价值和优势
        values = self.value_net(states).squeeze()  # [batch_size]
        next_values = self.value_net(next_states).squeeze()  # [batch_size]
        targets = rewards + self.gamma * next_values * (1 - dones)
        advantages = targets - values.detach()

        # 更新价值网络
        value_loss = self.value_loss_fn(values, targets)
        self.optimizer_value.zero_grad()
        value_loss.backward()
        self.optimizer_value.step()

        # 更新策略网络
        self.policy_net.train()
        op_probs, task_id_probs, src_node_id_probs, des_node_id_probs = self.policy_net(states)

        # 定义分布
        op_dist = torch.distributions.Categorical(op_probs)
        task_dist = torch.distributions.Categorical(task_id_probs)
        src_dist = torch.distributions.Categorical(src_node_id_probs)
        des_dist = torch.distributions.Categorical(des_node_id_probs)

        # 计算log概率
        log_prob_op = op_dist.log_prob(actions[:, 0])
        log_prob_task = task_dist.log_prob(actions[:, 1])
        log_prob_src = src_dist.log_prob(actions[:, 2])
        log_prob_des = des_dist.log_prob(actions[:, 3])

        # 总的log概率
        log_probs = log_prob_op + log_prob_task + log_prob_src + log_prob_des

        # 计算策略损失
        ratios = torch.exp(log_probs)  # [batch_size]
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # 反向传播并优化策略网络
        self.optimizer_policy.zero_grad()
        policy_loss.backward()
        self.optimizer_policy.step()

    def update_cluster(self, cluster, tasks, monitor):
        """
        更新集群的状况信息。

        参数:
            cluster (Cluster): 集群对象。
            tasks (list): Task实例的列表。
            monitor (Monitor): 监控对象。
        """
        # 调用监控对象更新集群信息
        cluster_metrics = monitor.monitor(cluster, tasks, self.current_time)
        # 可能需要记录或使用这些信息
        logging.info(f"Cluster Metrics at time {self.current_time}: {cluster_metrics}")

    def save_model(self, path):
        """
        保存策略网络和价值网络的模型参数。

        参数:
            path (str): 模型保存路径。
        """
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'value_net_state_dict': self.value_net.state_dict(),
        }, path)
        logging.info(f"Model saved to {path}.")

    def load_model(self, path):
        """
        加载策略网络和价值网络的模型参数。

        参数:
            path (str): 模型加载路径。
        """
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_net_state_dict'])
        logging.info(f"Model loaded from {path}.")
