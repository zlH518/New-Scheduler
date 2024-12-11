import os
import pandas as pd
import logging
import json
import numpy as np
import sys

class Monitor:
    def __init__(self, monitorConfig):
        self.name = monitorConfig['name']
        self.save_path = monitorConfig['save_path']
        self.frag_alpha = monitorConfig['frag_alpha']
        self.free_rate_alpha = monitorConfig['free_rate_alpha']

    def monitor(self, cluster, tasks, current_time, timestep, arrival_task_num):   
        monitoring_data = self.__get_state(cluster, tasks, current_time, timestep)
        monitoring_data['arrival_task_num'] = arrival_task_num
        # print(f"current_time:{current_time}, free_rate:{monitoring_data['free_rate']}, task num in wl:{monitoring_data['num task in wl']}, unused_nodes:{monitoring_data['unused_node_num']}, fragment_rate:{monitoring_data['fragment_rate']}")

        if os.path.exists(self.save_path):
            try:
                with open(self.save_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if not isinstance(data, list):
                        data = []
            except json.JSONDecodeError:
                logging.warning(f"JSON 文件格式错误，重新创建文件: {self.save_path}")
                data = []
        else:
            data = []

        data.append(monitoring_data)

        with open(self.save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        logging.info(f"Monitoring data saved to {self.save_path}")
        return monitoring_data
        
    def __get_state(self, cluster, tasks, current_time, timestep):
        """
        1:集群中的信息：
            集群空闲率:空闲的卡除以总的卡数
            集群的吞吐量:应该按照天数来统计,统计这一天中completed的任务数量
            集群的碎片率:碎片卡数量(只包含节点中没有用上的卡,完全空闲的节点不考虑)

        2:任务的信息
            任务的平均等待时间:应该统计一天中到达时间在今天的任务和还在运行的任务的平均等待时间
            任务的平均完成时间:只包括已经完成的任务，在过去一天中
            每个任务的迁移次数:
        """
        total_cards = 0
        free_cards = 0
        pieces_cards = 0
        # 1. 集群信息
        total_cards = cluster.node_num * cluster.cards_per_node
        for node in cluster.nodes:
            if node.cards != cluster.cards_per_node:
                pieces_cards += node.cards
            free_cards += node.cards
        
        
        completed_tasks = list(filter(lambda task: task.status == "DONE", tasks.tasks))
        wait_tasks = list(filter(lambda task: task.status == "WAIT", tasks.tasks))
        running_tasks = list(filter(lambda task: task.status == "RUNNING", tasks.tasks))
        unarrival_tasks = list(filter(lambda task: task.status == "UNARRIVAL", tasks.tasks))
        
        
        used_nodes = list(filter(lambda node: node.cards != 8, cluster.nodes))

        #1.1 集群空闲率
        free_rate = float(free_cards / float(total_cards))

        #1.2 集群吞吐量
        throughput = float(len(completed_tasks) / float(current_time) * 24) if current_time != 0 else 0.0

        #1.3 集群碎片率
        fragment_rate = float(pieces_cards / float(len(used_nodes)*8)) if len(used_nodes) != 0 else 0.0



        #2.1 任务的平均等待时间
        waiting_time = []
        completed_time = []
        predict_waiting_time = []
        duration_time = []
        arrival_tasks = list(filter(lambda task: task.status != "UNARRIVAL", tasks.tasks))
        for task in arrival_tasks:
            if task.status == "WAIT":
                waiting_time.append(current_time - task.create_time)
            elif task.status == "RUNNING":
                waiting_time.append(task.real_start_time - task.create_time)
                duration_time.append(current_time - task.real_start_time)
            elif task.status == "DONE":
                waiting_time.append(task.real_start_time - task.create_time)
                completed_time.append(task.real_end_time - task.create_time)
                duration_time.append(task.real_end_time - task.real_start_time)
                predict_waiting_time.append(task.real_end_time - task.real_start_time)
                if task.real_end_time - task.create_time < task.real_end_time - task.real_start_time:
                    input()
                assert task.real_end_time - task.create_time >= task.real_end_time - task.real_start_time
                assert len(completed_time) == len(predict_waiting_time)

        avg_waiting_time = np.mean(waiting_time)
        avg_completed_time = np.mean(completed_time)
        avg_duration_time = np.mean(duration_time)
        avg_predict_waiting_time = np.mean(predict_waiting_time)

        monitoring_data = {
            'timestamp': current_time,
            'free_rate': free_rate,
            'throughput': throughput,
            'fragment_rate': fragment_rate,
            'avg_waiting_time': avg_waiting_time,
            'avg_completed_time': avg_completed_time,
            'predict_waiting_time': avg_predict_waiting_time,
            'avg_duration_time': avg_duration_time,
            'unused_node_num': cluster.node_num - len(used_nodes),
            'num_task_in_wl': len(wait_tasks),
            'num_task_running': len(running_tasks),
            'num_task_completed': len(completed_tasks),
        }
        return monitoring_data