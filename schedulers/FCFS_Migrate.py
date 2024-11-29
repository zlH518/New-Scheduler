import logging

from .baseSchedulers import Scheduler



class FCFS_Migrate(Scheduler):
    def __init__(self, schedulerConfig):
        Scheduler.__init__(self,schedulerConfig)
        self.name = self.schedulerConfig['name']
        self.timeStep = self.schedulerConfig['time_step']
        self.fragment_alpha_rate = self.schedulerConfig['fragment_alpha_rate']

        self.completed_tasks = []

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
        self.info ={
            'timestamp': self.currentTime,
            'free_rate': 1.0,
            'throughput': 0.0,
            'fragment_rate': 0.0,
            'avg_waiting_time': 0.0,
            'avg_completion_time': 0.0,
            'avg_migration_times': 0.0,
            'unused_node_num': 500,
            'num task in wl': 0,
            'arrival_task_num': 0
        }

    def __release_tasks(self, cluster):
        self.completed_tasks += cluster.release_task(self.currentTime)

    
    def __new_tasks_arrival(self, tasks):
        if self.currentTaskIndex < self.lastTaskIndex:
            self.arrival_task_num = 0
            while tasks[self.currentTaskIndex].create_time <= self.currentTime:
                # print(f"{tasks[index].task_id} task is in wl")
                self.arrival_task_num = self.arrival_task_num + 1
                self.waintingList.add_task(tasks[self.currentTaskIndex])
                self.currentTaskIndex += 1
                if self.currentTaskIndex == self.lastTaskIndex:
                    break

    def __put_task_from_wl_2_cluster(self, cluster):
        task_from_wl ={}
        if len(self.waintingList) > 0:
            status = True
            while status and len(self.waintingList) > 0:
                task = self.waintingList.pop_task(self.currentTime, self.info)      #wl根据集群的状况和当前时刻选择弹出一个任务
                if task is not None:
                    status = cluster.add_task(self.currentTime, task)
                
                    #如果放置成功，则从wl中删除，否则继续等待
                    if status:  
                        # print(f"task_{task.task_id} is in node")
                        self.waintingList.delete_task(task.task_id)
                task_from_wl = task
        return task_from_wl

    def __update_info(self,cluster, tasks):
        self.info = self.monitor.monitor(cluster, tasks, self.currentTime, self.timeStep, self.arrival_task_num)

    def __migrate(self, cluster, tasks):
        if self.info['free_rate'] == 0.0:
            return False
        if self.info['fragment_rate'] > self.fragment_alpha_rate:
            allow_migrate_nodes = list(filter(lambda node: node.cards !=0 and node.cards != 8, cluster.nodes))
            allow_migrate_tasks = []
            for node in allow_migrate_nodes:
                for task in node.tasks:
                    if task.migration_times < self.schedulerConfig['migrate_times']:
                        allow_migrate_tasks.append(task)
            if len(allow_migrate_tasks) or len(allow_migrate_nodes) <= 1:
                return False
            print(len(allow_migrate_tasks), len(allow_migrate_nodes))
            ans = self.migrateSolver.solver(allow_migrate_nodes, allow_migrate_tasks, self.currentTime)     #! 获取迁移求解器的答案，答案为source_node和target_node的编号和任务的id
        else:
            return False
    def __adjust(self, cluster):
        #TODO:根据当前的集群状况来决定是否调整，调整目前仅包括停止大的任务
        pass

    def __time_add(self):
        self.currentTime += self.timeStep

    def run(self, cluster, tasks, monitor):
        #初始化
        self.__init__run(cluster, tasks, monitor)

        while len(self.completed_tasks) != len(tasks):
            #1.任务释放
            self.__release_tasks(cluster)

            #2.新任务到来加入wl
            self.__new_tasks_arrival(tasks)
            
            #3.从wl中取出任务，然后放置到集群中
            task_from_wl = self.__put_task_from_wl_2_cluster(cluster)

            #4.更新集群的状况信息
            self.__update_info(cluster, tasks)

            #5.根据当前集群的各种状况，来决定是否启动迁移
            self.__migrate(cluster, tasks)

            #6.根据当前集群的各种状况，来决定是否启动调整
            self.__adjust(cluster)

            #打印信息
            print(f"time: {self.currentTime}, effiency:{(1.0-self.info['free_rate'])*100}%, task num in wl:{self.info['num task in wl']}, unused_nodes:{self.info['unused_node_num']}, fragment_rate:{self.info['fragment_rate']},tasks from wl: {task_from_wl}")

            #7.时间递增    
            self.__time_add()