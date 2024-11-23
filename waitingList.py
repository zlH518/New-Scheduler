import random



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