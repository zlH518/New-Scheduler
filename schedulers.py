# schedulers.py
import logging
# from rl_agent import DQNAgent
import random

class Scheduler:
    def __init__(self,config):
        self.name = config['name']

    def run(self, culster, tasks, monitor):
        pass
    

class WaitingList:
    def __init__(self, wlConfig):
        self.wl = []
        self.config = wlConfig
        self.priority = wlConfig["wl_priority"]
        self.task_num = 0
        self.avg_cards = 0
        if self.priority == "highest response ratio next":
            self.alpha = wlConfig["wl_alpha"]

    def add_task(self, task):
        task.status = 'WAIT'
        self.wl.append(task)
        self.avg_cards = (self.task_num * self.avg_cards + task.cards) / (self.task_num + 1)
        self.task_num += 1

    def pop_task(self, current_time, info):
        if len(self.wl) == 0:
            return None
        if len(self.wl) == 1:
            return self.wl[0]
        if self.priority == "cards_big_first":
            tasks = sorted(self.wl, key=lambda task: task.cards * task.node_num, reverse=True)
            return tasks[0] 
        elif self.priority == "cards_small_first":
            tasks = sorted(self.wl, key=lambda task: task.cards * task.node_num, reverse=False)
            return tasks[0]
        elif self.priority == "first come first sever":
            tasks = sorted(self.wl, key=lambda task: task.create_time, reverse=False)
            return tasks[0]
        elif self.priority == "last come last sever":
            tasks = sorted(self.wl, key=lambda task: task.create_time, reverse=True)
            return tasks[0]
        elif self.priority == "short_time_first":
            tasks = sorted(self.wl, key=lambda task: task.duration_time, reverse=False)
            return tasks[0]
        elif self.priority == "long_time_first":
            tasks = sorted(self.wl, key=lambda task: task.duration_time, reverse=True)
            return tasks[0]
        elif self.priority == "highest response ratio next":
            for task in self.wl:
                waiting_time = current_time - task.create_time
                response_ratio = float((waiting_time + task.duration_time) / float(task.duration_time))
                task.response_ratio = response_ratio
            tasks = sorted(self.wl, key=lambda task: task.response_ratio, reverse=True)
            return tasks[0]
        elif self.priority == "dynamic":    
            """
            1.如果集群空闲率超过阈值,那么就大作业优先，否则小作业优先
            2.如果wl中等待任务太多,超过阈值，那么就说明大作业卡住了，转化为小作业优先
            """
            if len(self.wl) > self.config["wl_tasks_num"]:                  #等待任务太多，紧急转化为小作业优先
                tasks = sorted(self.wl, key=lambda task: task.cards * task.node_num, reverse=False)
                return tasks[0]
            elif info["free_rate"] > self.config["wl_free_max_rate"]:         #空闲率太高,则大作业优先
                tasks = sorted(self.wl, key=lambda task: task.cards * task.node_num, reverse=True)
                return tasks[0] 
            elif info["free_rate"] < self.config["wl_free_min_rate"]:         #空闲率太低,则小作业优先
                tasks = sorted(self.wl, key=lambda task: task.cards * task.node_num, reverse=False)
                return tasks[0]
            else:
                tasks = sorted(self.wl, key=lambda task: task.duration_time, reverse=False)       #默认情况，短任务优先
                return tasks[0]
        elif self.priority == "lottery":
            tickets = []
            total_tickets = 0
            for task in self.wl:
                waiting_time = current_time - task.create_time
                task_duration = max(task.duration_time, 1)
                task.tickets = int((waiting_time / task_duration ) * 100)
                tickets.append(task.tickets)
                total_tickets += task.tickets
            draw = random.randint(1, total_tickets)
            cumulative_tickets = 0
            for task, ticket_count in zip(self.wl, tickets):
                cumulative_tickets += ticket_count
                if draw <= cumulative_tickets:
                    return task
        else:
            raise ValueError(f'invalid priority setting: {self.priority} in waiting list')

    def delete_task(self, task_id):
        self.wl = list(filter(lambda task: task.task_id != task_id, self.wl))

    def __len__(self):
        return len(self.wl)

"""
1.任务释放
2.新任务到来加入wl
3.从wl中取出任务,找合适的节点放置
4.如果成功放置,则在wl中删除任务,否则不做处理
5.记录各种数据
6.时间继续增加
"""
class FCFS(Scheduler):
    def __init__(self, config):
        Scheduler.__init__(self,config)
        self.completed_tasks = []
        self.wlConfig = {}
        self.wlConfig['wl_priority'] = config["wl_priority"]
        if self.wlConfig['wl_priority'] == "highest response ratio next":
            self.wlConfig['wl_alpha'] = config['wl_alpha']
        elif self.wlConfig['wl_priority'] == "dynamic":
            self.wlConfig['wl_free_max_rate'] = config["wl_free_max_rate"]
            self.wlConfig['wl_free_min_rate'] = config["wl_free_min_rate"]
            self.wlConfig['wl_tasks_num'] = config["wl_tasks_num"]
        else:
            self.wlConfig['wl_alpha'] = None
        self.time_step = config['time_step']
    
    def run(self, cluster, tasks, monitor):
        #模拟tasks在集群中的过程
        cluster.tasks = tasks
        print(f"-----------{self.name} begin!!-------------")
        start_time = tasks[0].create_time
        current_time = start_time
        print(f'start_time:{start_time}')
        logging.info(f'start_time:{start_time}')
        index = 0
        wl = WaitingList(self.wlConfig)
        info ={
            'timestamp': current_time,
            'free_rate': 1.0,
            'throughput': 0.0,
            'fragment_rate': 0.0,
            'avg_waiting_time': 0.0,
            'avg_completion_time': 0.0,
            'avg_migration_times': 0.0,
            'unused_node_num': 500,
            'num task in wl': 0
        }
        while len(self.completed_tasks) != len(tasks):
            #1.任务释放
            self.completed_tasks += cluster.release_task(current_time)
            
            #2.新任务到来加入wl
            if index < len(tasks):
                while tasks[index].create_time <= current_time:
                    # print(f"{tasks[index].task_id} task is in wl")
                    wl.add_task(tasks[index])
                    index += 1
                    if index == len(tasks):
                        break
            
            #3.从wl中取出任务，然后放置到集群中
            if len(wl) > 0:
                status = True
                while status and len(wl) > 0:
                    task = wl.pop_task(current_time, info)
                    if task is not None:
                        status = cluster.add_task(current_time, task)
                    
                        #4.如果放置成功，则从wl中删除，否则继续等待
                        if status:  
                            # print(f"task_{task.task_id} is in node")
                            wl.delete_task(task.task_id)
            
            #5.监控各种信息
            info = monitor.monitor(cluster, tasks, current_time)
            # wl_first_task = wl.pop_task(current_time)
            # if wl_first_task is not None:
            #     print(info + f', wl_first_task.cards:{wl_first_task.cards}, wl_first_task.node_num:{wl_first_task.node_num}')
            # else:
            #     print(info + ', wl_first_task.cards:0, wl_first_task.node_num:0')
            #6.时间继续增加
            current_time += self.time_step
            


class MIG(Scheduler):
    def __init__(self, config):
        Scheduler.__init__(self,config)
    
    def run(self, cluster, tasks, monitor):
        #模拟tasks在集群中的过程
        print(f"-----------{self.name} begin!!-------------")
        start_time = tasks[0].create_time
        current_time = start_time
        print(f'start_time:{start_time}')
        logging.info(f'start_time:{start_time}')
        index = 0
        wl = WaitingList(self.waitingListPriority)
        while len(self.completed_tasks) != len(tasks):
            #1.任务释放
            self.completed_tasks += cluster.release_task(current_time)
            
            #2.新任务到来加入wl
            if index < len(tasks):
                while tasks[index].create_time <= current_time:
                    # print(f"{tasks[index].task_id} task is in wl")
                    wl.add_task(tasks[index])
                    index += 1
                    if index == len(tasks):
                        break
            
            #3.从wl中取出任务，然后放置到集群中
            if len(wl) > 0:
                status = True
                while status and len(wl) > 0:
                    task = wl.pop_task(current_time)
                    if task is not None:
                        status = cluster.add_task(current_time, task)
                    
                        #4.如果放置成功，则从wl中删除，否则继续等待
                        if status:  
                            # print(f"task_{task.task_id} is in node")
                            wl.delete_task(task.task_id)
            
            #5.如果waiting list中有任务，并且有碎片可以用，那么就启动迁移
            if len(wl) > 0:
                task = wl.pop_task(current_time)        #迁移也需要有一定的策略考虑，仅考虑长任务迁移，防止小任务挤占大任务


            #6.监控各种信息
            monitor.monitor(cluster, tasks, current_time)

            #7.时间继续增加
            current_time += self.time_step



# class RLScheduler(Scheduler):
#     def __init__(self, config):
#         super().__init__(config)
#         self.completed_tasks = []
#         self.rl_agent = DQNAgent(44, 10)
#         self.waitingListPriority = config['wl_priority']
#         self.waitingListAlpha = config.get('wl_alpha', None)
#         self.time_step = config['time_step']


    
#     def run(self, cluster, tasks, monitor):
#         """
#         使用强化学习进行任务调度
#         """
#         cluster.tasks = tasks
#         print(f"-----------{self.name} begin!!-------------")
#         start_time = tasks[0].create_time
#         current_time = start_time
#         index = 0
#         wl = WaitingList(self.waitingListPriority, self.waitingListAlpha)
        
#         while len(self.completed_tasks) != len(tasks):
#             # 1. 任务释放
#             self.completed_tasks += cluster.release_task(current_time)
            
#             # 2. 新任务到来加入 waiting list
#             if index < len(tasks):
#                 while tasks[index].create_time <= current_time:
#                     wl.add_task(tasks[index])
#                     index += 1
#                     if index == len(tasks):
#                         break
            
#             # 3. 从 waiting list 中选择任务进行调度
#             if len(wl) > 0:
#                 status = True
#                 while status and len(wl) > 0:
#                     # 获取当前集群状态作为强化学习模型的输入
#                     state = self.get_state(cluster, wl)
                    
#                     # 使用强化学习模型选择任务
#                     action_index = self.rl_agent.choose_action(state)
#                     task = wl.get_task_by_index(action_index)
                    
#                     if task is not None:
#                         # 尝试将任务放置到集群
#                         status = cluster.add_task(current_time, task)
                        
#                         # 4. 如果任务放置成功，更新 waiting list
#                         if status:
#                             wl.delete_task(task.task_id)
#                             reward = self.calculate_reward(cluster, task)
#                         else:
#                             reward = -1  # 放置失败给予负奖励

            
#             # 5. 监控集群状态
#             new_state = monitor.monitor(cluster, tasks, current_time)
#             self.rl_agent.store_transition(state, action_index, reward, new_state, done=False)
#             self.rl_agent.train()
            
#             # 6. 时间推进
#             current_time += self.time_step
    
#     def get_state(self, cluster, wl):
#         """
#         获取当前状态向量，包括集群状况和等待列表中的任务信息
#         """
#         # 提取集群状态
#         idle_rate = cluster.get_idle_rate()
#         fragment_rate = cluster.get_fragment_rate()
#         throughput = cluster.get_throughput()
        
#         # 提取 waiting list 中任务的特征
#         wl_features = []
#         for task in wl.tasks:
#             wl_features.append([task.cards, task.node_num, task.create_time])
        
#         # 将集群状态和任务特征拼接为状态向量
#         state = np.concatenate([[idle_rate, fragment_rate, throughput]] + wl_features)
#         return state

#     def calculate_reward(self, cluster, task):
#         """
#         根据集群状态和任务信息计算奖励
#         """
#         idle_rate = cluster.get_idle_rate()
#         fragment_rate = cluster.get_fragment_rate()
#         waiting_time = task.start_time - task.create_time if task.start_time else 0
        
#         # 奖励函数的定义可以根据你的优化目标调整
#         reward = -fragment_rate - waiting_time / 1000  # 惩罚碎片率和长等待时间
#         return reward
