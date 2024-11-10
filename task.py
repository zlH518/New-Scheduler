import pandas as pd
import logging
import json
import os
import glob

class Task:
    TaskId = 0

    def __init__(self, create_time, start_time, cards, duration_time,
                 gpu_time, migration_cost, pre_queue_time, node_num):
        self.task_id = Task.TaskId
        self.create_time = create_time
        self.start_time = start_time
        self.cards = cards
        self.gpu_time = gpu_time
        self.duration_time = duration_time
        self.migration_cost = migration_cost
        self.pre_queue_time = pre_queue_time
        self.node_num = node_num
        self.migration_times = 0
        self.migration_time = 0
        self.checkpoint_time = 0
        self.node_id = []
        self.status = "UNARRIVAL"

        # real_parameter
        self.real_start_time = None
        self.real_end_time = None
        Task.TaskId += 1

    def __repr__(self):
        return (f"Task(task_id={self.task_id}, create_time={self.create_time}, "
                f"start_time={self.start_time}, cards={self.cards}, "
                f"duration_time={self.duration_time}, gpu_time={self.gpu_time}\n")








class Tasks:
    def __init__(self, taskConfig):
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
        elif self.name == "mydata":               #// ! 改成elif
            self.tasks = self.read_and_create_Tasks()
            

    def transferSnt9b(self):
        csv_paths = []
        json_file_paths = glob.glob(self.config['path'])
        for index, path in enumerate(json_file_paths):
            with open(path, 'r') as f:
                json_data = json.load(f)
                dft = pd.DataFrame(json_data)
                dft.reset_index(drop=True, inplace=True)
                output_file_path = os.path.join(self.config['save_csv_path'], f'file_{index}.csv')
                dft.to_csv(output_file_path, index=False)
                csv_paths.append(output_file_path)      #// ! TODO：添加一下logging
                logging.info(f'{path} is read done')
        self.csv_paths = csv_paths
        logging.info('file transfer is completed')

    def merge(self):
        dfs = []
        for path in self.csv_paths:
            df = pd.read_csv(path, low_memory=False)      #// ! set low_memory=False
            dfs.append(df)

        if dfs:
            merged_df = pd.concat(dfs, ignore_index=True)
            merged_df.to_csv(self.config['merge_data_path'], index=False)

    def filter(self):
        if self.config['type'] == 'single':
            data = pd.read_csv(self.config['merge_data_path'],usecols=['metadata.create_time', 'status.duration', 'status.start_time', 'spec.resource.flavor_id', 'spec.resource.node_count'],low_memory=False)
            data = data[(data['spec.resource.node_count'] == 1)]
            data = data[['metadata.create_time', 'status.duration', 'status.start_time', 'spec.resource.flavor_id', 'spec.resource.node_count']]
            data['spec.resource.flavor_id'] = data['spec.resource.flavor_id'].replace({
                'modelarts.pool.visual.8xlarge': 8,
                'modelarts.pool.visual.4xlarge': 4,
                'modelarts.pool.visual.2xlarge': 2,
                'modelarts.pool.visual.xlarge': 1
            })
            data.rename(columns={'metadata.create_time': 'create_time'}, inplace=True)
            data.rename(columns={'status.duration': 'duration_time'}, inplace=True)
            data.rename(columns={'status.start_time': 'start_time'}, inplace=True)
            data.rename(columns={'spec.resource.flavor_id': 'cards'}, inplace=True)
            data.rename(columns={'spec.resource.node_count': 'node_num'}, inplace=True)
            min_create_time = data['create_time'].min()

            data['create_time'] = ((data['create_time'] - min_create_time)/1000).astype(int)
            data['start_time'] = ((data['start_time'] - min_create_time)/1000).astype(int)
            data['duration_time'] = (data['duration_time']/1000).astype(int)
            data['gpu_time'] = 0                    
            data['migration_cost'] = 0
            data['pre_queue_time'] = 0
            data['checkpoint_time'] = 0 
            data['scheduling_time'] = data['start_time'] - data['create_time']
            data = data[(data['create_time'] >= 0) & (data['start_time'] >= 0) & (data['duration_time'] > 0)]
            data.to_csv(self.config['useful_data_path'], index=False)
            logging.info(f'useful data is saved in {self.config['useful_data_path']}')
      
        elif self.config['type'] == 'multi':
            data = pd.read_csv(self.config['merge_data_path'],usecols=['metadata.create_time', 'status.duration', 'status.start_time', 'spec.resource.flavor_id', 'spec.resource.node_count'], low_memory=False)
            data = data[(data['spec.resource.node_count'] <= 500)]
            data = data[['metadata.create_time', 'status.duration', 'status.start_time', 'spec.resource.flavor_id', 'spec.resource.node_count']]
            data['spec.resource.flavor_id'] = data['spec.resource.flavor_id'].replace({
                'modelarts.pool.visual.8xlarge': 8,
                'modelarts.pool.visual.4xlarge': 4,
                'modelarts.pool.visual.2xlarge': 2,
                'modelarts.pool.visual.xlarge': 1
            })
            data.rename(columns={'metadata.create_time': 'create_time'}, inplace=True)
            data.rename(columns={'status.duration': 'duration_time'}, inplace=True)
            data.rename(columns={'status.start_time': 'start_time'}, inplace=True)
            data.rename(columns={'spec.resource.flavor_id': 'cards'}, inplace=True)
            data.rename(columns={'spec.resource.node_count': 'node_num'}, inplace=True)
            min_create_time = data['create_time'].min()

            data['create_time'] = ((data['create_time'] - min_create_time)/1000).astype(int)
            data['start_time'] = ((data['start_time'] - min_create_time)/1000).astype(int)
            data['duration_time'] = (data['duration_time']/1000).astype(int)
            data['gpu_time'] = 0                    
            data['migration_cost'] = 0
            data['pre_queue_time'] = 0
            data['checkpoint_time'] = 0 
            data['scheduling_time'] = data['start_time'] - data['create_time']
            data = data[(data['create_time'] >= 0) & (data['start_time'] >= 0) & (data['duration_time'] > 0)]
            data.to_csv(self.config['useful_data_path'], index=False)
            logging.info(f'useful data is saved in {self.config['useful_data_path']}')


    def read_and_create_Tasks(self):
        tasks = []
        try:
            data = pd.read_csv(self.config['useful_data_path'], usecols=['create_time', 'start_time', 'cards', 'gpu_time', 'duration_time', 'migration_cost', 'pre_queue_time', 'node_num'])
            logging.info(msg=str(f"file:{self.config['useful_data_path']} read success!"))
            for row in data.itertuples():
                tasks.append(Task(create_time=row.create_time, start_time=row.start_time,
                    cards=row.cards, duration_time=row.duration_time,
                    gpu_time=row.gpu_time, migration_cost=row.migration_cost,
                    pre_queue_time=row.pre_queue_time, node_num = row.node_num))
        except Exception as e:
            print(f"file:{self.config['useful_data_path']}: {e}")
        logging.info(msg=f"the number of task is {len(tasks)}")
        tasks = sorted(tasks, key=lambda task: task.create_time, reverse=False)
        return tasks
    
    def __len__(self):
        return len(self.tasks)
    
    def __getitem__(self, index):
        return self.tasks[index]