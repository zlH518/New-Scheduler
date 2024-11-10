# schedulers.py
import logging

class Scheduler:
    def __init__(self,config):
        self.name = config['name']

    def run(self, culster, tasks, monitor):
        pass
    

class WaitingList:
    def __init__(self, priority, alpha = None):
        self.wl = []
        self.priority = priority
        if self.priority == "highest response ratio next":
            self.alpha = alpha

    def add_task(self, task):
        task.status = 'WAIT'
        self.wl.append(task)

    def pop_task(self, current_time):
        if len(self.wl) == 0:
            return None
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
        elif self.priority == "highest response ratio next":
            for task in self.wl:
                waiting_time = current_time - task.create_time
                response_ratio = float((waiting_time + self.alpha * task.cards * task.node_num) / float(self.alpha * task.node_num * task.cards))
                task.response_ratio = response_ratio
            tasks = sorted(self.wl, key=lambda task: task.response_ratio, reverse=True)
            return tasks[0]
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
        self.waitingListPriority = config['wl_priority']
        if self.waitingListPriority == "highest response ratio next":
            self.waitingListAlpha = config['wl_alpha']
        else:
            self.waitingListAlpha = None
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
        wl = WaitingList(self.waitingListPriority, self.waitingListAlpha)
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
