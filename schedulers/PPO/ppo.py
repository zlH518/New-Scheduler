import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from collections import deque
from ..baseSchedulers import Scheduler

import torch
import torch.nn as nn
import torch.nn.functional as F

#决策网络
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, num_tasks, num_nodes, num_ops):
        super(PolicyNetwork, self).__init__()
        
        # 网络层
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        
        # 输出层
        self.fc_task_id = nn.Linear(64, num_tasks)  # 任务ID的输出
        self.fc_src_node_id = nn.Linear(64, num_nodes)  # 源节点ID的输出
        self.fc_des_node_id = nn.Linear(64, num_nodes)  # 目标节点ID的输出
        self.fc_op = nn.Linear(64, num_ops)  # 操作类型的输出（放置、停止、迁移、无操作）
        
    def forward(self, x):
        # 输入x通过两层全连接网络
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # 各部分输出
        task_id_logits = self.fc_task_id(x)
        src_node_id_logits = self.fc_src_node_id(x)
        des_node_id_logits = self.fc_des_node_id(x)
        op_logits = self.fc_op(x)
        
        # Softmax转化为概率分布
        task_id_probs = F.softmax(task_id_logits, dim=-1)
        src_node_id_probs = F.softmax(src_node_id_logits, dim=-1)
        des_node_id_probs = F.softmax(des_node_id_logits, dim=-1)
        op_probs = F.softmax(op_logits, dim=-1)
        
        return task_id_probs, src_node_id_probs, des_node_id_probs, op_probs


# 神经网络：价值网络
class ValueNetwork(nn.Module):
    def __init__(self, input_size):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class PPO(Scheduler):
    def __init__(self, schedulerConfig):
        super(PPO, self).__init__(schedulerConfig)
        self.name = self.schedulerConfig['name']
        self.timeStep = self.schedulerConfig['time_step']
        self.completed_tasks = []

        # 超参数设置
        self.gamma = 0.99  # 折扣因子
        self.lamda = 0.95  # GAE（Generalized Advantage Estimation）参数
        self.clip_eps = 0.2  # PPO剪切参数
        self.lr = 1e-3  # 学习率
        self.batch_size = 64  # 每个batch大小
        self.memory = deque(maxlen=1000)  # 用于存储经验池

        # 初始化策略网络和价值网络
        self.policy_net = PolicyNetwork(input_size=100, output_size=5)  # 假设状态大小为100，动作空间大小为5
        self.value_net = ValueNetwork(input_size=100)
        self.optimizer_policy = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.optimizer_value = optim.Adam(self.value_net.parameters(), lr=self.lr)

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
            while tasks[self.currentTaskIndex].create_time <= self.currentTime:
                self.waintingList.add_task(tasks[self.currentTaskIndex])
                self.currentTaskIndex += 1
                if self.currentTaskIndex == self.lastTaskIndex:
                    break

    def __select_and_execute_action(self, cluster, tasks):
        state = self.get_state(cluster, tasks)
        action = self.select_action(state)
        reward, done = self.execute_action(action, cluster, tasks)
        self.memory.append((state, action, reward, done))  # 将经验加入经验池
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
        state = torch.cat([cluster_state, task_state.flatten()])
        return state

    def get_cluster_state(self, cluster):
        # 128维的集群状态向量，使用embedding或者其他方式生成
        cluster_embedding = torch.zeros(128)  # 示例：此处应为集群状态特征的嵌入
        return cluster_embedding

    def get_task_state(self, tasks):
        task_embeddings = []
        for task in tasks:
            task_embeddings.append(self.get_task_embedding(task))
        
        # 填充任务嵌入，确保长度固定
        task_embeddings = torch.stack(task_embeddings)
        padded_embeddings = self.pad_task_embeddings(task_embeddings, max_num_tasks=10)  # 假设最大任务数为10
        return padded_embeddings

    def get_task_embedding(self, task):
        # 每个任务的32维嵌入向量，使用任务的特征来生成
        task_embedding = torch.zeros(32)  # 示例：此处应为任务特征的嵌入
        return task_embedding

    def pad_task_embeddings(self, task_embeddings, max_num_tasks):
        # 填充任务嵌入
        padded_embeddings = torch.zeros(max_num_tasks, 32)
        padded_embeddings[:len(task_embeddings)] = task_embeddings
        return padded_embeddings

    def select_action(self, state):
        # 使用策略网络选择动作
        action_probs = self.policy_net(state)
        action = torch.multinomial(action_probs, 1).item()  # 从概率分布中抽样
        return action

    def execute_action(self, action, cluster, tasks):
        # 根据选定的动作执行，并返回奖励和任务是否完成
        reward = 0
        done = False
        # 执行动作，这里可以根据任务调度的情况进行实际的任务分配、迁移等
        # 假设每次执行后任务完成度增加，返回奖励
        return reward, done

    def reward_function(self, cluster_utilization, avg_waiting_time):
        utilization_reward = cluster_utilization  # 直接奖励集群利用率
        waiting_time_penalty = -avg_waiting_time  # 通过惩罚等待时间来鼓励减少等待
        return utilization_reward + waiting_time_penalty

    def update_model(self):
        # 从经验池中采样数据并进行在线学习
        if len(self.memory) < self.batch_size:
            return

        # 随机采样一个batch
        batch = np.random.choice(self.memory, self.batch_size, replace=False)
        states, actions, rewards, dones = zip(*batch)

        # 将状态转换为张量
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # 计算目标值和优势
        values = self.value_net(states)
        next_values = self.value_net(states)  # 可以考虑使用下一个状态的值来估算
        advantages = rewards + self.gamma * next_values - values

        # 更新价值网络
        value_loss = advantages.pow(2).mean()
        self.optimizer_value.zero_grad()
        value_loss.backward()
        self.optimizer_value.step()

        # 更新策略网络
        log_probs = torch.log(self.policy_net(states)[actions])
        policy_loss = -(log_probs * advantages).mean()
        self.optimizer_policy.zero_grad()
        policy_loss.backward()
        self.optimizer_policy.step()
