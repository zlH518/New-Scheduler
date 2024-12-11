import torch
import torch.nn as nn
import torch.nn.functional as F




import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from collections import deque, namedtuple

from ..baseSchedulers import Scheduler

# 定义经验元组
Experience = namedtuple('Experience', ['state', 'op', 'task_id', 'src_node', 'des_node', 'reward', 'next_state', 'done'])

class PPO(Scheduler):
    def __init__(self, schedulerConfig, num_tasks, num_nodes, num_ops, state_size, action_size):
        super(PPO, self).__init__(schedulerConfig)
        self.name = self.schedulerConfig['name']
        self.timeStep = self.schedulerConfig['time_step']
        self.completed_tasks = []

        # 超参数设置
        self.gamma = 0.99  # 折扣因子
        self.lamda = 0.95  # GAE参数
        self.clip_eps = 0.2  # PPO剪切参数
        self.lr = 1e-4  # 学习率
        self.batch_size = 64  # 每个batch大小
        self.epochs = 10  # PPO更新的epochs
        self.memory = deque(maxlen=10000)  # 经验池

        # 初始化策略网络和价值网络
        self.policy_net = PolicyNetwork(input_size=state_size, num_tasks=num_tasks, num_nodes=num_nodes, num_ops=num_ops)
        self.value_net = ValueNetwork(input_size=state_size)
        self.optimizer_policy = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.optimizer_value = optim.Adam(self.value_net.parameters(), lr=self.lr)

        # 损失函数
        self.value_loss_fn = nn.MSELoss()

    def __init__run(self, cluster, tasks, monitor):
        cluster.tasks = tasks
        self.monitor = monitor
        print(f"-----------{self.name} begin!!-------------")
        self.startTime = tasks[0].create_time
        self.currentTime = self.startTime
        print(f'start_time:{self.startTime}')
        logging.info(f'start_time:{self.startTime}')
        self.currentTaskIndex = 0
        self.lastTaskIndex = len(tasks)
        self.info = {
            'timestamp': self.currentTime,
            'free_rate': 1.0,
            'throughput': 0.0,
            'fragment_rate': 0.0,
            'avg_waiting_time': 0.0,
            'avg_completion_time': 0.0,
            'avg_migration_times': 0.0,
            'unused_node_num': 500,
            'num task in wl': 0
        }

    def __release_tasks(self, cluster):
        self.completed_tasks += cluster.release_task(self.currentTime)

    def __new_tasks_arrival(self, tasks):
        if self.currentTaskIndex < self.lastTaskIndex:
            while self.currentTaskIndex < self.lastTaskIndex and tasks[self.currentTaskIndex].create_time <= self.currentTime:
                self.waitingList.add_task(tasks[self.currentTaskIndex])
                self.currentTaskIndex += 1

    def __select_and_execute_action(self, cluster, tasks):
        state = self.get_state(cluster, tasks)
        op, task_id, src_node, des_node = self.select_action(state)
        reward, done = self.execute_action(op, task_id, src_node, des_node, cluster, tasks)
        next_state = self.get_state(cluster, tasks)
        self.memory.append(Experience(state, op, task_id, src_node, des_node, reward, next_state, done))
        self.update_model()  # 更新模型

    def __update_info(self, cluster, tasks):
        self.info = self.monitor.monitor(cluster, tasks, self.currentTime, self.timeStep)

    def __time_add(self):
        self.currentTime += self.timeStep

    def run(self, cluster, tasks, monitor):
        # 初始化
        self.__init__run(cluster, tasks, monitor)

        while len(self.completed_tasks) != len(tasks):
            # 1. 任务释放
            self.__release_tasks(cluster)

            # 2. 新任务到来加入等待列表
            self.__new_tasks_arrival(tasks)

            # 3. 选择并执行动作
            self.__select_and_execute_action(cluster, tasks)

            # 4. 更新集群的状况信息
            self.__update_info(cluster, tasks)

            # 5. 时间递增
            self.__time_add()

    def get_state(self, cluster, tasks):
        # 生成状态向量（集群和任务特征的拼接）
        cluster_state = self.get_cluster_state(cluster)
        task_state = self.get_task_state(tasks)
        state = torch.cat([cluster_state, task_state.flatten()]).float()
        return state

    def get_cluster_state(self, cluster):
        # 示例：假设每个节点的GPU使用情况被嵌入为一个向量
        # 这里需要根据具体实现进行调整
        node_states = []
        for node in cluster.nodes:
            # 假设每个节点有8个GPU，0表示空闲，1表示使用中
            gpu_usage = node.get_gpu_usage()  # 返回一个长度为8的列表或数组
            node_states.extend(gpu_usage)
        cluster_embedding = torch.tensor(node_states, dtype=torch.float32)
        return cluster_embedding

    def get_task_state(self, tasks):
        task_embeddings = []
        # 只考虑等待队列中的任务
        waiting_tasks = self.waitingList.get_waiting_tasks()
        for task in waiting_tasks:
            task_embeddings.append(self.get_task_embedding(task))
        
        # 填充任务嵌入，确保长度固定
        max_num_tasks = 50  # 假设最大任务数为50
        task_embeddings = torch.stack(task_embeddings) if task_embeddings else torch.zeros(1, 32)
        padded_embeddings = self.pad_task_embeddings(task_embeddings, max_num_tasks=max_num_tasks)
        return padded_embeddings

    def get_task_embedding(self, task):
        # 每个任务的32维嵌入向量，使用任务的特征来生成
        # 这里可以使用任务的GPU需求、节点需求、执行时间、checkpoint周期等特征
        embedding = torch.zeros(32)
        embedding[0] = task.gpu  # GPU需求
        embedding[1] = task.node_requirement  # 节点需求
        embedding[2] = task.exec_time / 1000.0  # 归一化的执行时间
        embedding[3] = task.checkpoint_period / 100.0  # 归一化的checkpoint周期
        # 其余维度可以用来表示其他特征或通过嵌入层学习
        return embedding

    def pad_task_embeddings(self, task_embeddings, max_num_tasks):
        # 填充任务嵌入
        if task_embeddings.size(0) >= max_num_tasks:
            return task_embeddings[:max_num_tasks]
        else:
            padding = torch.zeros(max_num_tasks - task_embeddings.size(0), task_embeddings.size(1))
            padded_embeddings = torch.cat([task_embeddings, padding], dim=0)
            return padded_embeddings

    def select_action(self, state):
        self.policy_net.eval()
        with torch.no_grad():
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
        # 采样有效的任务ID
        # 假设等待队列中有任务可供选择
        waiting_tasks = self.waitingList.get_waiting_tasks()
        if not waiting_tasks:
            return -1
        task_dist = torch.distributions.Categorical(task_id_probs[:len(waiting_tasks)])
        task_index = task_dist.sample().item()
        return waiting_tasks[task_index].id

    def execute_action(self, op, task_id, src_node, des_node, cluster, tasks):
        reward = 0
        done = False

        if op == 0:
            # 无操作
            pass
        elif op == 1:
            # 放置任务
            task = self.waitingList.get_task_by_id(task_id)
            if task and cluster.can_allocate(task, des_node):
                cluster.allocate_task(task, des_node)
                self.waitingList.remove_task(task)
                reward += self.compute_reward(cluster, task)
            else:
                # 无法分配，给予负奖励
                reward -= 1
        elif op == 2:
            # 迁移任务
            task = cluster.get_task_by_id(task_id)
            if task and cluster.can_migrate(task, src_node, des_node):
                cluster.migrate_task(task, src_node, des_node)
                reward += self.compute_reward(cluster, task)
            else:
                # 无法迁移，给予负奖励
                reward -= 1
        elif op == 3:
            # 停止任务
            task = cluster.get_task_by_id(task_id)
            if task and cluster.can_stop(task, src_node):
                cluster.stop_task(task, src_node)
                self.waitingList.add_task(task)
                reward -= self.compute_penalty()
            else:
                # 无法停止，给予负奖励
                reward -= 1

        # 计算奖励，考虑集群利用率和任务完成时间
        utilization = cluster.get_utilization()
        avg_waiting_time = self.waitingList.get_avg_waiting_time()
        reward += self.reward_function(utilization, avg_waiting_time)

        # 检查是否完成
        if len(self.completed_tasks) == len(tasks):
            done = True

        return reward, done

    def reward_function(self, cluster_utilization, avg_waiting_time):
        # 奖励函数设计
        # 提高集群利用率并减少平均等待时间
        alpha = 1.0
        beta = 1.0
        utilization_reward = alpha * cluster_utilization
        waiting_time_penalty = -beta * avg_waiting_time
        return utilization_reward + waiting_time_penalty

    def compute_reward(self, cluster, task):
        # 根据具体需求设计
        # 示例：任务成功分配后给予正奖励
        return 1.0

    def compute_penalty(self):
        # 根据具体需求设计
        # 示例：停止任务给予负奖励
        return 1.0

    def update_model(self):
        if len(self.memory) < self.batch_size:
            return

        # 转换经验为批量数据
        experiences = list(self.memory)[-self.batch_size:]
        states = torch.stack([e.state for e in experiences])
        ops = torch.tensor([e.op for e in experiences], dtype=torch.long)
        task_ids = torch.tensor([e.task_id for e in experiences], dtype=torch.long)
        src_nodes = torch.tensor([e.src_node for e in experiences], dtype=torch.long)
        des_nodes = torch.tensor([e.des_node for e in experiences], dtype=torch.long)
        rewards = torch.tensor([e.reward for e in experiences], dtype=torch.float32)
        next_states = torch.stack([e.next_state for e in experiences])
        dones = torch.tensor([e.done for e in experiences], dtype=torch.float32)

        # 计算价值和优势
        values = self.value_net(states).squeeze()
        next_values = self.value_net(next_states).squeeze()
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
        
        # 计算操作类型的log概率
        op_dist = torch.distributions.Categorical(op_probs)
        log_probs_op = op_dist.log_prob(ops)
        
        # 计算任务ID的log概率
        log_probs_task = torch.log(task_id_probs.gather(1, task_ids.unsqueeze(1)).squeeze(1) + 1e-10)
        
        # 计算源节点ID的log概率
        # 对于不需要源节点的操作，log_probs_src为0
        mask_src = (ops == 2) | (ops == 3)
        log_probs_src = torch.zeros_like(ops, dtype=torch.float32)
        valid_src = mask_src.nonzero(as_tuple=False).squeeze()
        if valid_src.numel() > 0:
            src_dist = torch.distributions.Categorical(src_node_id_probs[valid_src])
            log_probs_src[valid_src] = src_dist.log_prob(src_nodes[valid_src])
        
        # 计算目标节点ID的log概率
        # 对于不需要目标节点的操作，log_probs_des为0
        mask_des = (ops == 1) | (ops == 2)
        log_probs_des = torch.zeros_like(ops, dtype=torch.float32)
        valid_des = mask_des.nonzero(as_tuple=False).squeeze()
        if valid_des.numel() > 0:
            des_dist = torch.distributions.Categorical(des_node_id_probs[valid_des])
            log_probs_des[valid_des] = des_dist.log_prob(des_nodes[valid_des])
        
        # 总的log概率
        log_probs = log_probs_op + log_probs_task + log_probs_src + log_probs_des

        # 计算策略损失
        ratios = torch.exp(log_probs)  # 假设旧策略的概率为1（需要保存旧策略的概率）
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # 反向传播并优化策略网络
        self.optimizer_policy.zero_grad()
        policy_loss.backward()
        self.optimizer_policy.step()

        # 清空经验池
        for _ in range(self.batch_size):
            self.memory.popleft()





# 决策网络
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, num_tasks, num_nodes, num_ops):
        super(PolicyNetwork, self).__init__()
        
        # 共享的网络层
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        
        # 输出层
        self.fc_op = nn.Linear(128, num_ops)  # 操作类型的输出
        self.fc_task_id = nn.Linear(128, num_tasks)  # 任务ID的输出
        self.fc_src_node_id = nn.Linear(128, num_nodes)  # 源节点ID的输出
        self.fc_des_node_id = nn.Linear(128, num_nodes)  # 目标节点ID的输出
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # 各部分输出
        op_logits = self.fc_op(x)
        task_id_logits = self.fc_task_id(x)
        src_node_id_logits = self.fc_src_node_id(x)
        des_node_id_logits = self.fc_des_node_id(x)
        
        # Softmax转化为概率分布
        op_probs = F.softmax(op_logits, dim=-1)
        task_id_probs = F.softmax(task_id_logits, dim=-1)
        src_node_id_probs = F.softmax(src_node_id_logits, dim=-1)
        des_node_id_probs = F.softmax(des_node_id_logits, dim=-1)
        
        return op_probs, task_id_probs, src_node_id_probs, des_node_id_probs

# 价值网络
class ValueNetwork(nn.Module):
    def __init__(self, input_size):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value
