import gym
from gym import spaces
import numpy as np

class AITaskSchedulingEnv(gym.Env):
    def __init__(self, num_nodes):
        super(AITaskSchedulingEnv, self).__init__()
        self.num_nodes = num_nodes
        self.gpus_per_node = 8
        # 定义动作空间
        # op: 0-3, task_id: 0-最大任务数, src_node_id: 0-num_nodes-1, des_node_id: 0-num_nodes-1
        self.action_space = spaces.MultiDiscrete([4, 100, num_nodes, num_nodes])
        # 定义状态空间（示例）
        # 包含每个节点的GPU使用情况和等待队列中任务的特征
        self.observation_space = spaces.Box(low=0, high=1, shape=(num_nodes * 8 + 100 * 4,), dtype=np.float32)
        # 初始化状态
        self.reset()
    
    def reset(self):
        self.node_gpus = np.zeros((self.num_nodes, self.gpus_per_node), dtype=np.float32)
        self.waiting_tasks = []  # 列表存储等待任务
        self.running_tasks = {}  # 任务ID映射到节点和剩余时间
        self.current_step = 0
        return self._get_state()
    
    def step(self, action):
        op, task_id, src_node, des_node = action
        reward = 0
        done = False
        info = {}
        
        # 执行动作
        if op == 1:
            # 分配任务到des_node
            # 检查资源是否足够
            task = self.waiting_tasks.pop(task_id)
            if self._can_allocate(task, des_node):
                self._allocate(task, des_node)
                reward += 1  # 简单奖励
            else:
                self.waiting_tasks.append(task_id)  # 回退
        elif op == 2:
            # 迁移任务
            if task_id in self.running_tasks:
                if self._can_allocate(task_id, des_node):
                    self._migrate(task_id, src_node, des_node)
                    reward += 1
        elif op == 3:
            # 停止任务
            if task_id in self.running_tasks:
                self._stop(task_id, src_node)
                self.waiting_tasks.append(task_id)
                reward -= 1  # 惩罚
        # op == 0: 不做操作
        
        # 更新任务执行
        self._update_tasks()
        
        # 计算奖励
        reward += self._compute_reward()
        
        self.current_step += 1
        return self._get_state(), reward, done, info
    
    def _get_state(self):
        # 简化状态表示，实际需要根据具体情况设计
        node_state = self.node_gpus.flatten()
        task_state = np.zeros(100 * 4)  # 假设最多100个等待任务，每个任务4个特征
        for i, task in enumerate(self.waiting_tasks[:100]):
            task_state[i*4:(i+1)*4] = task  # 假设任务已编码为4维特征
        state = np.concatenate([node_state, task_state])
        return state
    
    def _can_allocate(self, task, node):
        # 检查目标节点是否有足够的GPU
        required_gpus = task['gpu']
        available_gpus = np.sum(self.node_gpus[node] == 0)
        return available_gpus >= required_gpus
    
    def _allocate(self, task, node):
        # 分配任务到节点，更新GPU使用情况
        gpus = task['gpu']
        indices = np.where(self.node_gpus[node] == 0)[0][:gpus]
        self.node_gpus[node][indices] = 1  # 标记为使用中
        self.running_tasks[task['id']] = {'node': node, 'remaining_time': task['exec_time']}
    
    def _migrate(self, task_id, src_node, des_node):
        # 迁移任务，从src_node到des_node
        task = self.running_tasks.pop(task_id)
        self.node_gpus[src_node] -= 1  # 释放GPU
        self._allocate(task, des_node)
    
    def _stop(self, task_id, src_node):
        # 停止任务，释放GPU
        self.running_tasks.pop(task_id)
        self.node_gpus[src_node] -= 1
    
    def _update_tasks(self):
        # 更新所有运行中任务的剩余时间
        completed_tasks = []
        for task_id, info in self.running_tasks.items():
            info['remaining_time'] -= 1
            if info['remaining_time'] <= 0:
                completed_tasks.append(task_id)
        for task_id in completed_tasks:
            node = self.running_tasks[task_id]['node']
            self.node_gpus[node] -= 1  # 释放GPU
            del self.running_tasks[task_id]
    
    def _compute_reward(self):
        # 简单奖励，基于集群利用率
        utilization = np.sum(self.node_gpus) / (self.num_nodes * self.gpus_per_node)
        return utilization
    
    def render(self, mode='human'):
        pass  # 可视化实现

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

# 创建环境
env = AITaskSchedulingEnv(num_nodes=10)
check_env(env)

# 创建PPO代理
model = PPO('MlpPolicy', env, verbose=1)

# 训练代理
model.learn(total_timesteps=100000)

# 保存模型
model.save("ppo_ai_task_scheduling")

# 加载模型
model = PPO.load("ppo_ai_task_scheduling")

# 评估
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    if done:
        obs = env.reset()
