import logging

class Node:
    Node_id = 0
    def __init__(self, nodeConfig):
        self.Id = Node.Node_id
        Node.Node_id += 1
        self.cards = nodeConfig['cards_per_node']  
        self.max_cards = nodeConfig['cards_per_node']
        self.tasks = []
        self.release_point = float('-inf')
        self.tasks_id = []
    
    def add_task(self, task, current_time):
        self.tasks_id.append(task.task_id)
        if task.node_num <= 1:
            task.node_id.append(self.Id)
            task.real_start_time = current_time
            task.status = "RUNNING"
            task.real_queue_time = current_time - task.create_time
            if len(self.tasks) == 0:
                self.release_point = current_time + task.duration_time
            else:
                self.release_point = max(self.release_point, current_time + task.duration_time)
            self.tasks.append(task)
            self.cards -= task.cards
        else:
            task.node_id.append(self.Id)
            task.real_start_time = current_time
            task.status = "RUNNING"
            task.real_queue_time = current_time - task.create_time
            if len(self.tasks) == 0:
                self.release_point = current_time + task.duration_time
            else:
                self.release_point = max(self.release_point, current_time + task.duration_time)
            self.tasks.append(task)
            self.cards -= self.max_cards


    def delete_task(self, task_id):
        self.tasks = list(filter(lambda task: task.task_id != task_id, self.tasks))

    def release_task(self, task_id, current_time):
        num = 0
        for task in self.tasks:             
            if task.task_id == task_id:
                if task.node_num > 1:
                    self.cards = self.max_cards
                else:
                    self.cards += task.cards
                task.real_end_time = current_time
                task.status = "DONE"
                logging.info(f"task_{task.task_id} released")
                self.delete_task(task_id)
                num += 1
                assert num == 1
                break
        if len(self.tasks) == 0:
            self.release_point = float('-inf')
        else:
            self.release_point = max(task.real_start_time + task.duration_time for task in self.tasks)


class Cluster:
    def __init__(self, clusterConfig):
        self.nodes = []
        self.cards_per_node = clusterConfig['cards_per_node']
        self.node_num = clusterConfig['node_num']
        self.priority = clusterConfig['priority']
        self.tasks = None
        for _ in range(clusterConfig['node_num']):
            self.nodes.append(Node(clusterConfig))
    
    def release_task(self, current_time):
        completed_tasks = []
        running_tasks = list(filter(lambda task: task.status == "RUNNING", self.tasks.tasks))
        for task in running_tasks:
            if current_time - task.real_start_time >= task.duration_time:
                for node_id in task.node_id:
                    self.nodes[node_id].release_task(task.task_id, current_time)
                completed_tasks.append(task)
        return completed_tasks


    #! 任务释放没有释放干净，需要debug代码
    def find_node(self, task, current_time):
        #针对multi
        if task.node_num != 1:
            nodes = list(filter(lambda node: node.cards == self.cards_per_node, self.nodes))
            if len(nodes) < task.node_num:
                return None
            else:
                return nodes[:task.node_num]

        if self.priority == "best fit":
            nodes = list(filter(lambda node: node.cards-task.cards >= 0, self.nodes))
            nodes = sorted(nodes, key=lambda node: node.cards, reverse=False)
            if len(nodes) == 0:
                return None
            return [nodes[0]]
        elif self.priority == "big first":
            nodes = list(filter(lambda node: node.cards - task.cards >= 0, self.nodes))
            nodes = sorted(nodes, key=lambda node: node.cards, reverse=True)
            if len(nodes) == 0:
                return None
            return [nodes[0]]
        elif self.priority == "best similar":   #按照节点中的任务结束时间点，选择最接近的那个
            nodes = list(filter(lambda node: node.cards - task.cards>= 0, self.nodes))
            nodes = sorted(nodes, key=lambda node: node.release_point, reverse=False)
            if len(nodes) == 0:
                return None
            return [nodes[0]]


    def add_task(self, current_time, task):        
        nodes = self.find_node(task, current_time)   
        if nodes is None:
            return False
        for node in nodes:
            node.add_task(task, current_time)
        return True
    
    def migrate_task(self, node_src_Id, node_des_Id, current_time):
        node_src, node_des = self.nodes[node_src_Id], self.nodes[node_des_Id]
        migrate_tasks = node_src.tasks
        node_src.cards = node_src.max_cards
        node_src.tasks = []
        node_src.release_point = float('-inf')
        for task in migrate_tasks:
            task.node_id.clear()
            task.node_id.append(node_des_Id)
            node_des.tasks.append(task)
            node_des.cards -= task.cards
            node_des.release_point = max(node_des.release_point, current_time + task.duration_time)

