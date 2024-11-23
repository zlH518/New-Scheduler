# schedulers.py
import logging
# from rl_agent import DQNAgent
from waitingList import WaitingList
from migrateSolver import MigrationSolver

class Scheduler:
    def __init__(self,config):
        self.name = config['name']

    def run(self, culster, tasks, monitor):
        pass
    

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
        migrator = MigrationSolver()
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
            info, is_migrate, is_stop_big = monitor.monitor(cluster, tasks, wl, current_time)
            if is_migrate:
                self.migrate(cluster, tasks, wl, current_time)
            if is_stop_big:
                self.stop_big(cluster, tasks, wl, current_time)
                
            # wl_first_task = wl.pop_task(current_time)
            # if wl_first_task is not None:
            #     print(info + f', wl_first_task.cards:{wl_first_task.cards}, wl_first_task.node_num:{wl_first_task.node_num}')
            # else:
            #     print(info + ', wl_first_task.cards:0, wl_first_task.node_num:0')
            #6.时间继续增加
            current_time += self.time_step
    


    def stop_big(self, cluster, tasks, wl, current_time):
        pass
    #! TODO
