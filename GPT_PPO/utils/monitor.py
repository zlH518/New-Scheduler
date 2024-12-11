# utils/monitor.py

import logging

class Monitor:
    """
    Monitor类用于监控集群的状况信息，更新并记录相关指标。
    """

    def __init__(self):
        """
        初始化Monitor实例。
        """
        pass

    def monitor(self, cluster, tasks, current_time):
        """
        更新并返回集群的状况信息。

        参数:
            cluster (Cluster): 集群对象。
            tasks (list): Task实例的列表。
            current_time (int): 当前时间。

        返回:
            dict: 集群指标的字典。
        """
        # 更新集群的各种指标
        utilization = cluster.get_utilization()
        fragmentation_rate = cluster.get_fragmentation_rate()
        throughput = cluster.get_throughput()
        avg_waiting_time = cluster.get_avg_waiting_time()

        cluster_metrics = {
            'utilization': utilization,
            'fragmentation_rate': fragmentation_rate,
            'throughput': throughput,
            'avg_waiting_time': avg_waiting_time
        }

        # 记录或打印集群指标
        logging.info(f"Monitor at time {current_time}: {cluster_metrics}")

        return cluster_metrics
