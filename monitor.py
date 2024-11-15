import os
import pandas as pd
import logging

class Monitor:
    def __init__(self, monitorConfig):
        self.name = monitorConfig['name']
        self.save_path = monitorConfig['save_path']
        self.frag_alpha = monitorConfig['frag_alpha']
        self.free_rate_alpha = monitorConfig['free_rate_alpha']

    def monitor(self, cluster, tasks, current_time):
    
        # print(current_time)
        monitoring_data = self.get_state(cluster, tasks, current_time)
        print(f'current_time:{current_time}, free_rate:{monitoring_data['free_rate']}, task num in wl:{monitoring_data['num task in wl']}, unused_nodes:{monitoring_data['unused_node_num']}, fragment_rate:{monitoring_data['fragment_rate']}')

        if os.path.exists(self.save_path):
            df_existing = pd.read_csv(self.save_path)
            df_new = pd.DataFrame([monitoring_data])
            df = pd.concat([df_existing, df_new], ignore_index=True)
            df.to_csv(self.save_path, index=False)
        else:
            df = pd.DataFrame([monitoring_data])
            df.to_csv(self.save_path, index=False)

        logging.info(f"Monitoring data saved to {self.save_path}")

        #! 判断是否需要进行碎片整理
        # is_migrate = self.check_fragment(cluster, tasks, current_time)
        
        #! 判断是否需要把大的任务先停
        # if state:
        #     self.defrag(cluster, tasks, current_time, state)
        # is_stop_big = self.check_stopBig()
        
        # return monitoring_data, is_migrate, is_stop_big
        return monitoring_data


    def check_fragment(self, cluster, tasks, current_time):
        # ! 根据集群状况，判断是否需要进行碎片整理， 碎片太多，占据了空闲率的一半, 并且空闲率超过阈值了，那就需要整理了
        state = self.get_state(cluster, tasks, current_time)
        if state['free_rate'] >= self.free_rate_alpha and float(state['fragment_rate']/state['free_rate']) >= self.frag_alpha:       #! 需要整理了
            return True
        return False
        
    
    def defrag(self, cluster, tasks, current_time, state):
        #! 整理的思路，首先筛选未填满的节点，按照碎片升序，任务数量升序的方法排序
        #! 然后找到两个最配对的节点，由迁移代价较少的那一个节点将任务迁移到另外一个节点中
        used_nodes = [node for node in cluster.nodes if node.cards != cluster.cards_per_node and node.cards != 0]
        if not used_nodes:
            return

        # 按碎片数量和任务数量排序
        used_nodes.sort(key=lambda n: (n.cards, len(n.tasks)))

        for i in range(len(used_nodes) - 1):
            node_a = used_nodes[i]
            node_b = used_nodes[len(used_nodes)-1 - i]

            # 选择迁移代价较小的节点进行任务迁移
            if node_a.cards + node_b.cards <= cluster.cards_per_node:
                if len(node_a.tasks) < len(node_b.tasks):
                    node_src_Id = node_a.Id
                    node_des_Id = node_b.Id 
                else:
                    node_src_Id = node_b.Id
                    node_des_Id = node_a.Id 
                cluster.migrate_task(node_src_Id, node_des_Id, current_time)
        return None

    def get_state(self, cluster, tasks, current_time):
        """
        1:集群中的信息：
            集群空闲率:空闲的卡除以总的卡数
            集群的吞吐量:只统计所有任务完成之后,任务的完成量除以时间
            集群的碎片率:碎片卡数量(只包含节点中没有用上的卡,完全空闲的节点不考虑)

        2:任务的信息
            任务的平均等待时间:等待时间包括正在等待和已经完成的任务
            任务的平均完成时间:只包括已经完成的任务
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
        
        # 2. 任务信息
        completed_tasks = list(filter(lambda task: task.status == "DONE", tasks.tasks))
        wait_tasks = list(filter(lambda task: task.status == "WAIT", tasks.tasks))
        running_tasks = list(filter(lambda task: task.status == "RUNNING", tasks.tasks))
        unarrival_tasks = list(filter(lambda task: task.status == "UNARRIVAL", tasks.tasks))
        used_nodes = list(filter(lambda node: node.cards != 8, cluster.nodes))


        #1.1 集群空闲率
        free_rate = float(free_cards / float(total_cards))

        #1.2 集群吞吐量每天
        throughput = float(len(completed_tasks) / float(current_time) * 60 * 60 * 24) if current_time != 0 else 0.0

        #1.3 集群碎片率
        fragment_rate = float(pieces_cards / float(len(used_nodes)*8)) if len(used_nodes) != 0 else 0.0

        #2.1 任务的平均等待时间
        avg_waiting_time = float(sum(current_time - task.create_time for task in wait_tasks) + \
                                 sum(task.real_queue_time for task in completed_tasks) + \
                                 sum(task.real_queue_time for task in running_tasks)) \
                            / float((len(wait_tasks) + len(completed_tasks) + len(running_tasks)))

        # 2.2 任务的平均完成时间
        avg_completion_time = float(sum(task.real_end_time - task.create_time for task in completed_tasks) \
                            / float(len(completed_tasks))) if len(completed_tasks) != 0 else 0.0

        # 2.3 任务的平均迁移次数
        avg_migration_times = float(sum(task.migration_times for task in completed_tasks) \
                            / float(len(completed_tasks))) if len(completed_tasks) != 0 else 0.0

        #将数据存储为字典
        monitoring_data = {
            'timestamp': current_time,
            'free_rate': free_rate,
            'throughput': throughput,
            'fragment_rate': fragment_rate,
            'avg_waiting_time': avg_waiting_time,
            'avg_completion_time': avg_completion_time,
            'avg_migration_times': avg_migration_times,
            'unused_node_num': cluster.node_num-len(used_nodes),
            'num task in wl': len(wait_tasks)
        }
        # print(monitoring_data)
        return monitoring_data