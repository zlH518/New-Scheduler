import os
import yaml
from datetime import datetime
import logging
import importlib


from cluster import Cluster
from task import Tasks
from monitor import Monitor
import schedulers


class Executor:
    def __init__(self, config):
        self.config = config
        self.schedulers = {}
        self.cluster = None
        self.tasks = {}
        self.__init_schedulers(config['schedulers'])
        self.__init_cluster(config['cluster'])
        self.__init_tasks(config['tasks'])
        self.__init_monitor(config['monitor'])

    def __init_monitor(self, monitorConfig):
        self.monitor = monitorConfig

    def __init_cluster(self, clusterConfig):
        self.cluster = clusterConfig

    def __init_schedulers(self, schedulersConfig):
        for config in schedulersConfig:
            strategy_name = config['name']
            self.schedulers[strategy_name] = config

    def __init_tasks(self, tasksConfig):
        for config in tasksConfig:
            self.tasks[config['name']] = config


    def executor(self):
        for scheduler_name, schedulerConfig in self.schedulers.items():
            schedulers = importlib.import_module(f'schedulers.{scheduler_name}')
            scheduler_class = getattr(schedulers, scheduler_name, None)
            if scheduler_class is None:
                raise ValueError(f"Scheduler class '{scheduler_name}' not found in 'schedulers' module")
            scheduler = scheduler_class(schedulerConfig)
            for task_name, config in self.tasks.items():
                cluster = Cluster(self.cluster)  
                tasks = Tasks(config)
                monitorConfig = self.monitor
                monitorConfig["save_path"] = os.path.join(monitorConfig['base_path'],scheduler_name + f"_{self.config['timestamp']}.json")
                monitor = Monitor(self.monitor)
                scheduler.run(cluster, tasks, monitor)
                self.__save_config()

    def __save_config(self):
        """
        Save the current config to the specified directory as a YAML file with a unique name.
        The filename will be based on the current timestamp.
        """
        if not os.path.exists(self.config['config']['save_path']):
            os.makedirs(self.config['config']['save_path'])

        #// !这里改成self.timestamp，保持配置文件和监控的信息文件名字一致
        save_path = os.path.join(self.config['config']['save_path'], f"config_{self.config['timestamp']}.yaml")

        with open(save_path, 'w') as yaml_file:
            yaml.dump(self.config, yaml_file, default_flow_style=False, allow_unicode=True)

        logging.info(f"Config saved to {save_path}")
        print(f"Config saved to {save_path}")