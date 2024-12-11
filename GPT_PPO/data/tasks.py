# data/tasks.py

import pandas as pd
import logging
import json
import os
import glob
from .task import Task

class Tasks:
    """
    Tasks类用于处理和管理多个Task实例，从配置文件读取任务数据，进行预处理和分阶段。
    """

    def __init__(self, taskConfig):
        """
        初始化Tasks实例。

        参数:
            taskConfig (dict): 任务配置，包括文件路径、类型等。
        """
        self.config = taskConfig
        self.name = taskConfig["name"]
        self.type = taskConfig["type"]
        self.tasks = None

        if self.name == 'snt9b':
            if len(os.listdir(self.config['save_csv_path'])) == 0:
                self.transferSnt9b()
                self.merge()
            self.filter()
            self.tasks = self.read_and_create_Tasks()
        elif self.name == "mydata":  # 改成elif
            self.tasks = self.read_and_create_Tasks()

    def transferSnt9b(self):
        """
        将SNT9B格式的JSON文件转换为CSV格式，并保存到指定路径。
        """
        csv_paths = []
        json_file_paths = glob.glob(self.config['path'])
        for index, path in enumerate(json_file_paths):
            with open(path, 'r') as f:
                try:
                    json_data = json.load(f)
                    dft = pd.DataFrame(json_data)
                    dft.reset_index(drop=True, inplace=True)
                    output_file_path = os.path.join(self.config['save_csv_path'], f'file_{index}.csv')
                    dft.to_csv(output_file_path, index=False)
                    csv_paths.append(output_file_path)
                    logging.info(f'{path} is read and converted to CSV successfully.')
                except Exception as e:
                    logging.error(f'Error processing {path}: {e}')
        self.csv_paths = csv_paths
        logging.info('All JSON files have been converted to CSV.')

    def merge(self):
        """
        将多个CSV文件合并为一个大的CSV文件。
        """
        dfs = []
        for path in self.csv_paths:
            try:
                df = pd.read_csv(path, low_memory=False)
                dfs.append(df)
            except Exception as e:
                logging.error(f'Error reading {path}: {e}')

        if dfs:
            merged_df = pd.concat(dfs, ignore_index=True)
            merged_df.to_csv(self.config['merge_data_path'], index=False)
            logging.info(f'Merged CSV saved to {self.config["merge_data_path"]}.')
        else:
            logging.warning('No CSV files to merge.')

    def filter(self):
        """
        根据任务类型（单节点或多节点）过滤和处理任务数据。
        """
        if self.config['type'] == 'single':
            # 处理单节点任务
            data = pd.read_csv(self.config['merge_data_path'],
                               usecols=['metadata.create_time', 'status.duration', 'status.start_time',
                                        'spec.resource.flavor_id', 'spec.resource.node_count', 'status.phase'],
                               low_memory=False)
            data = data[(data['spec.resource.node_count'] == 1)]
            data = data[(data['status.phase'] == 'Completed')]
            data = data[['metadata.create_time', 'status.duration', 'status.start_time',
                        'spec.resource.flavor_id', 'spec.resource.node_count', 'status.phase']]
            data['spec.resource.flavor_id'] = data['spec.resource.flavor_id'].replace({
                'modelarts.pool.visual.8xlarge': 8,
                'modelarts.pool.visual.4xlarge': 4,
                'modelarts.pool.visual.2xlarge': 2,
                'modelarts.pool.visual.xlarge': 1
            })
            data.rename(columns={
                'metadata.create_time': 'create_time',
                'status.duration': 'duration_time',
                'status.start_time': 'start_time',
                'spec.resource.flavor_id': 'cards',
                'spec.resource.node_count': 'node_num'
            }, inplace=True)
            min_create_time = data['create_time'].min()

            # 归一化时间
            data['create_time'] = ((data['create_time'] - min_create_time) / 1000).astype(int)
            data['start_time'] = ((data['start_time'] - min_create_time) / 1000).astype(int)
            data['duration_time'] = (data['duration_time'] / 1000).astype(int)
            data['gpu_time'] = 0
            data['migration_cost'] = 0
            data['pre_queue_time'] = 0
            data['checkpoint_time'] = self.config.get('checkpoint_time', 2)  # 默认2小时
            data['scheduling_time'] = data['start_time'] - data['create_time']
            data = data[(data['create_time'] >= 0) & (data['start_time'] >= 0) & (data['duration_time'] > 0)]
            data.to_csv(self.config['useful_data_path'], index=False)
            logging.info(f"Useful data for 'single' type tasks saved in {self.config['useful_data_path']}.")
        
        elif self.config['type'] == 'multi':
            # 处理多节点任务
            data = pd.read_csv(self.config['merge_data_path'],
                               usecols=['metadata.create_time', 'status.duration', 'status.start_time',
                                        'spec.resource.flavor_id', 'spec.resource.node_count', 'status.phase'],
                               low_memory=False)
            data = data[(data['spec.resource.node_count'] <= 500)]
            data = data[(data['status.phase'] == 'Completed')]
            data = data[['metadata.create_time', 'status.duration', 'status.start_time',
                        'spec.resource.flavor_id', 'spec.resource.node_count', 'status.phase']]
            data['spec.resource.flavor_id'] = data['spec.resource.flavor_id'].replace({
                'modelarts.pool.visual.8xlarge': 8,
                'modelarts.pool.visual.4xlarge': 4,
                'modelarts.pool.visual.2xlarge': 2,
                'modelarts.pool.visual.xlarge': 1
            })
            data.rename(columns={
                'metadata.create_time': 'create_time',
                'status.duration': 'duration_time',
                'status.start_time': 'start_time',
                'spec.resource.flavor_id': 'cards',
                'spec.resource.node_count': 'node_num'
            }, inplace=True)
            min_create_time = data['create_time'].min()

            # 归一化时间
            data['create_time'] = ((data['create_time'] - min_create_time) / 1000).astype(int)
            data['start_time'] = ((data['start_time'] - min_create_time) / 1000).astype(int)
            data['duration_time'] = (data['duration_time'] / 1000).astype(int)
            data['gpu_time'] = 0
            data['migration_cost'] = 0
            data['pre_queue_time'] = 0
            data['checkpoint_time'] = self.config.get('checkpoint_time', 2)  # 默认2小时
            data['scheduling_time'] = data['start_time'] - data['create_time']
            data = data[(data['create_time'] >= 0) & (data['start_time'] >= 0) & (data['duration_time'] > 0)]
            data.to_csv(self.config['useful_data_path'], index=False)
            logging.info(f"Useful data for 'multi' type tasks saved in {self.config['useful_data_path']}.")

    def read_and_create_Tasks(self):
        """
        从预处理后的CSV文件中读取数据并创建Task实例列表。
        
        返回:
            list: Task实例的列表，按创建时间排序。
        """
        tasks = []
        try:
            data = pd.read_csv(self.config['useful_data_path'],
                               usecols=['create_time', 'start_time', 'cards',
                                        'gpu_time', 'duration_time', 'migration_cost',
                                        'pre_queue_time', 'node_num'])
            logging.info(f"Successfully read tasks from {self.config['useful_data_path']}.")
            for row in data.itertuples(index=False):
                tasks.append(Task(
                    create_time=row.create_time,
                    start_time=row.start_time,
                    cards=row.cards,
                    duration_time=row.duration_time,
                    gpu_time=row.gpu_time,
                    migration_cost=row.migration_cost,
                    pre_queue_time=row.pre_queue_time,
                    node_num=row.node_num
                ))
        except Exception as e:
            logging.error(f"Error reading {self.config['useful_data_path']}: {e}")
        logging.info(f"The number of tasks created: {len(tasks)}")
        tasks = sorted(tasks, key=lambda task: task.create_time)
        return tasks

    def __len__(self):
        """
        返回任务列表的长度。
        """
        return len(self.tasks)

    def __getitem__(self, index):
        """
        通过索引访问任务列表。
        """
        return self.tasks[index]
