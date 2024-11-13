import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

class TaskSchedulerRL:
    def __init__(self, state_dim, action_dim, hidden_dim=128, gamma=0.99, lr=1e-4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1

        # 定义神经网络
        self.model = self.build_model(state_dim, action_dim, hidden_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.memory = deque(maxlen=5000)

    def build_model(self, input_dim, output_dim, hidden_dim):
        """构建神经网络模型"""
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def predict(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            return self.model(state).squeeze(0).numpy()

    def train(self, batch_size=64):
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # 计算 Q 值
        q_values = self.model(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # 计算目标 Q 值
        with torch.no_grad():
            next_q_values = self.model(next_states).max(1)[0]
        target = rewards + self.gamma * next_q_values * (1 - dones)

        # 更新网络
        loss = nn.MSELoss()(q_value, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.action_dim)
        q_values = self.predict(state)
        return np.argmax(q_values)

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# 调度函数
def schedule_task(cluster_state, task_list):
    """
    输入:
    - cluster_state: 包含集群的状态信息，如空闲率、碎片率、吞吐量
    - task_list: 包含任务列表，每个任务的资源需求

    输出:
    - 调度任务的索引
    """
    # 初始化调度器
    state_dim = len(cluster_state) + len(task_list[0])
    action_dim = len(task_list)
    scheduler = TaskSchedulerRL(state_dim, action_dim)

    # 构建状态向量
    state = np.concatenate([cluster_state] + [task_list.flatten()])
    
    # 使用强化学习模型选择任务
    action = scheduler.choose_action(state)
    selected_task = task_list[action]

    # 更新调度策略
    reward = calculate_reward(cluster_state, selected_task)
    # next_state = update_cluster_state(cluster_state, selected_task)
    # scheduler.store_transition(state, action, reward, next_state, done=False)
    # scheduler.train()

    return selected_task


def calculate_reward(cluster_state, task):
    idle_rate, fragment_rate, throughput = cluster_state
    cards_needed, nodes_needed = task['cards'], task['node_num']
    
    # 惩罚碎片率高和空闲率高
    reward = -fragment_rate - idle_rate

    # 奖励调度高吞吐量的任务
    reward += throughput * 0.5

    # 惩罚过多等待时间
    if task['waiting_time'] > 100:
        reward -= 0.5

    return reward
