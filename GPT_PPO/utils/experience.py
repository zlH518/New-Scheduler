# utils/experience.py

from collections import namedtuple

# 定义经验元组，包含状态、动作、奖励、下一个状态和done标志
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])
