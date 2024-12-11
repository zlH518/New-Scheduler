import glob
import pandas as pd
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
from datetime import datetime


SOURCE_DATA_PATH = 'data/snt9b/sourceDataSnt9b/*.json'
SAVE_EXCEL_PATH = 'analysis/excel'
MERGE_DATA_PATH = 'analysis/excel/merge_data.xlsx'
FILTER_TYPE ='multi'
IS_COMPLETED = True
CDF_PATH = 'analysis/CDF'
FIGURE_PATH = 'analysis/Figure'








def plot_hourly_task_arrivals(folder_path):
    """
    读取指定文件夹中的所有以'file_'开头的Excel文件，筛选符合条件的任务，
    并在同一图中绘制每个文件每小时任务到来数量的折线图。
    
    参数:
    - folder_path: str, 文件夹路径
    """
    # 检查文件夹是否存在
    if not os.path.isdir(folder_path):
        print(f"文件夹未找到: {folder_path}")
        return

    # 使用glob筛选以'file_'开头的Excel文件（支持.xlsx和.xls）
    pattern_xlsx = os.path.join(folder_path, 'merge_data.xlsx')
    excel_files = glob.glob(pattern_xlsx)

    if not excel_files:
        print(f"在文件夹中未找到以'file_'开头的Excel文件。")
        return

    print(f"找到 {len(excel_files)} 个Excel文件。")

    # 设置绘图风格
    plt.figure(figsize=(14, 7))
    max_task_count = 0
    
    # 为每个文件绘制折线
    for file in excel_files:
        try:
            print(f"读取文件: {file}")
            df = pd.read_excel(file,usecols=['spec.resource.flavor_id','spec.resource.node_count','metadata.create_time'])

            # 检查必要的列是否存在
            required_columns = [
                'spec.resource.flavor_id',
                'spec.resource.node_count',
                'metadata.create_time'
            ]
            if not all(col in df.columns for col in required_columns):
                print(f"文件 {file} 中缺少必要的列。跳过此文件。")
                continue

            # 筛选符合条件的任务
            condition1 = (
                (df['spec.resource.flavor_id'] == "modelarts.pool.visual.xlarge") &
                (df['spec.resource.node_count'] == 1)
            )
            condition2 = (
                (df['spec.resource.flavor_id'] != "modelarts.pool.visual.xlarge") &
                (df['spec.resource.node_count'] >= 1)
            )
            filtered_df = df[condition1 | condition2]
            print(f"文件 {file} 筛选后的数据量: {len(filtered_df)} 条")

            if filtered_df.empty:
                print(f"文件 {file} 中没有符合条件的任务。跳过此文件。")
                continue

            # 将metadata.create_time从毫秒转换为datetime
            try:
                # 确保create_time是数值类型
                filtered_df['metadata.create_time'] = pd.to_numeric(filtered_df['metadata.create_time'], errors='coerce')
                # 删除无法转换的行
                filtered_df = filtered_df.dropna(subset=['metadata.create_time'])
                # 转换为秒并转换为datetime
                filtered_df['create_datetime'] = pd.to_datetime(filtered_df['metadata.create_time'] / 1000, unit='s')
            except Exception as e:
                print(f"文件 {file} 转换时间戳时出错: {e}")
                continue

            # 设置create_datetime为索引
            filtered_df.set_index('create_datetime', inplace=True)

            # 按小时统计任务数量
            task_counts = filtered_df.resample('H').size()
            max_task_count = max(max_task_count, task_counts.max())

            # 为了在同一图中对齐所有文件的时间范围，记录所有时间索引
            if 'all_times' not in locals():
                all_times = task_counts.index
            else:
                all_times = all_times.union(task_counts.index)

            # 绘制折线
            label = os.path.basename(file)
            plt.bar(task_counts.index, task_counts.values, alpha=0.2)

        except Exception as e:
            print(f"读取或处理文件 {file} 时出错: {e}")
            continue

    # 设置图表格式
    plt.xlabel('time(h)')
    plt.ylabel('number of tasks')
    plt.title('task arrivals per hour')
    # plt.ylim(0, max_task_count + 1)  # 为y轴设置合适的范围
    plt.grid(True)
    plt.gca().spines['top'].set_visible(False)  
    plt.gca().spines['right'].set_visible(False)  
    # plt.gca().spines['left'].set_position('zero')  
    # plt.gca().spines['bottom'].set_position('zero') 
    # plt.tight_layout()
    # plt.xticks(rotation=45)
    plt.savefig(f"{os.path.join(FIGURE_PATH,'tasks_arrivals.png')}", dpi=300)



def plot_daily_hourly_task_arrivals(folder_path):
    """
    读取指定文件夹中的所有以'file_'开头的Excel文件，筛选符合条件的任务，
    并按每天24小时绘制任务到达数量的柱状图。
    
    参数:
    - folder_path: str, 文件夹路径
    """
    # 检查文件夹是否存在
    if not os.path.isdir(folder_path):
        print(f"文件夹未找到: {folder_path}")
        return

    # 使用glob筛选以'file_'开头的Excel文件（支持.xlsx和.xls）
    pattern_xlsx = os.path.join(folder_path, 'merge_data.xlsx')
    excel_files = glob.glob(pattern_xlsx)

    if not excel_files:
        print(f"在文件夹中未找到以'file_'开头的Excel文件。")
        return

    print(f"找到 {len(excel_files)} 个Excel文件。")

    # 设置绘图风格
    plt.figure(figsize=(14, 7))

    # 用于跟踪每小时的任务数量
    hourly_task_counts = [0] * 24  # 用一个24小时长度的列表存储每小时的任务数量

    # 为每个文件统计任务的到达时间
    for file in excel_files:
        try:
            print(f"读取文件: {file}")
            df = pd.read_excel(file,usecols=['spec.resource.flavor_id','spec.resource.node_count','metadata.create_time'])


            # 检查必要的列是否存在
            required_columns = [
                'spec.resource.flavor_id',
                'spec.resource.node_count',
                'metadata.create_time'
            ]
            if not all(col in df.columns for col in required_columns):
                print(f"文件 {file} 中缺少必要的列。跳过此文件。")
                continue

            # 筛选符合条件的任务
            condition1 = (
                (df['spec.resource.flavor_id'] == "modelarts.pool.visual.xlarge") &
                (df['spec.resource.node_count'] == 1)
            )
            condition2 = (
                (df['spec.resource.flavor_id'] != "modelarts.pool.visual.xlarge") &
                (df['spec.resource.node_count'] >= 1)
            )
            filtered_df = df[condition1 | condition2]
            print(f"文件 {file} 筛选后的数据量: {len(filtered_df)} 条")

            if filtered_df.empty:
                print(f"文件 {file} 中没有符合条件的任务。跳过此文件。")
                continue

            # 将metadata.create_time从毫秒转换为datetime
            try:
                # 确保create_time是数值类型
                filtered_df['metadata.create_time'] = pd.to_numeric(filtered_df['metadata.create_time'], errors='coerce')
                # 删除无法转换的行
                filtered_df = filtered_df.dropna(subset=['metadata.create_time'])
                # 转换为秒并转换为datetime
                filtered_df['create_datetime'] = pd.to_datetime(filtered_df['metadata.create_time'] / 1000, unit='s')
            except Exception as e:
                print(f"文件 {file} 转换时间戳时出错: {e}")
                continue

            # 提取小时信息
            filtered_df['hour'] = filtered_df['create_datetime'].dt.hour

            # 按小时统计任务数量
            hourly_counts = filtered_df['hour'].value_counts().sort_index()

            # 将每个文件的任务数量添加到对应小时的统计数据中
            for hour, count in hourly_counts.items():
                hourly_task_counts[hour] += count

        except Exception as e:
            print(f"读取或处理文件 {file} 时出错: {e}")
            continue

    # 绘制柱状图
    hours = list(range(24))
    plt.bar(hours, hourly_task_counts, color='skyblue', alpha=0.7)

    # 设置图表格式
    plt.xlabel('hour')  # X 轴标签
    plt.ylabel('number of task arrivals')  # Y 轴标签
    plt.title('average hourly task arrivals')
    plt.xticks(hours)  # 设置x轴刻度为0-23小时
    plt.gca().spines['top'].set_visible(False)  
    plt.gca().spines['right'].set_visible(False)  
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(f"{os.path.join(FIGURE_PATH,'tasks_arrivals_per_hours.png')}", dpi=300)



def transferSnt9b():
    excel_paths = []
    json_file_paths = glob.glob(SOURCE_DATA_PATH)
    for index, path in enumerate(json_file_paths):
        with open(path, 'r') as f:
            json_data = json.load(f)
            dft = pd.DataFrame(json_data)
            dft.reset_index(drop=True, inplace=True)
            output_file_path = os.path.join(SAVE_EXCEL_PATH, f'file_{index}.xlsx')
            dft.to_excel(output_file_path, index=False)
            excel_paths.append(output_file_path)      
            print(f'{path} is read done')
    print('file transfer is completed')

def merge():
    dfs = []
    excel_paths = glob.glob(os.path.join(SAVE_EXCEL_PATH, '*.xlsx'))
    for path in excel_paths:
        df = pd.read_excel(path)      #// ! set low_memory=False
        dfs.append(df)

    if dfs:
        merged_df = pd.concat(dfs, ignore_index=True)
        merged_df.to_excel(MERGE_DATA_PATH, index=False)

def filter(path, is_completed):
    USEFUL_DATA_PATH = path
    if type == 'single':
        data = pd.read_excel(MERGE_DATA_PATH,usecols=['metadata.create_time', 'status.duration', 'status.start_time', 'spec.resource.flavor_id', 'spec.resource.node_count', 'status.phase'])
        data = data[(data['spec.resource.node_count'] == 1)]
        if is_completed:
            data = data[(data['status.phase'] == 'Completed')]
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
        data.to_excel(USEFUL_DATA_PATH, index=False)
        print(f"useful data is saved in {USEFUL_DATA_PATH}")
    
    elif type == 'multi':
        data = pd.read_excel(MERGE_DATA_PATH,usecols=['metadata.create_time', 'status.duration', 'status.start_time', 'spec.resource.flavor_id', 'spec.resource.node_count', 'status.phase'])
        data = data[(data['spec.resource.node_count'] <= 500)]
        if is_completed:
            data = data[(data['status.phase'] == 'Completed')]
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
        data.to_excel(USEFUL_DATA_PATH, index=False)
        print(f"useful data is saved in {USEFUL_DATA_PATH}")




def merge_dataprepare():
    # transferSnt9b()
    # merge()
    filter(FILTER_TYPE, IS_COMPLETED)


def analysis(type, is_completed, target):
    USEFUL_DATA_PATH = os.path.join(SAVE_EXCEL_PATH, type +'_' + str(is_completed) + '.xlsx')
    print(f'---------{type}-{is_completed}---------')
    data = pd.read_excel(USEFUL_DATA_PATH)
    print(f'Task type: {type}')
    print(f'Is completed: {is_completed}')
    print(data.describe())
    if target == "duration_time":
        plot_CDF_duration(USEFUL_DATA_PATH, target)
    elif target == "startA_time" or target == "create_time":
        plot_CDF(USEFUL_DATA_PATH, target)
    elif target == "cards":
        plot_CDF_cards(USEFUL_DATA_PATH, target)
    elif target == "gpu_time":
        plot_CDF_gpu_time(USEFUL_DATA_PATH, target)
    print('----------------------------------------')


def plot_CDF_gpu_time(data_path, target: str):
    data = np.array(pd.read_excel(data_path, usecols=['cards', 'duration_time', 'node_num']))
    
    y = {}
    for d in data:
        d = float(d[0] * d[1] * d[2])
        if d not in y:
            y[d] = 1
        else:
            y[d] += 1
    
    x = sorted(y.keys())
    z = [float(y[k]) / len(data) * 100 for k in x] 

    z_cdf = np.cumsum(z) 
    
    plt.plot(x, z_cdf, label=target)
    plt.xlabel(target, labelpad=10)  
    plt.ylabel('CDF(%)', labelpad=10)  
    plt.title(f'CDF of {target}')
    plt.xscale('log')
    
    plt.gca().spines['top'].set_visible(False)  
    plt.gca().spines['right'].set_visible(False)  
    plt.grid(True, which='both', axis='both', linewidth=0.7, alpha=0.5)  
    plt.gca().set_axisbelow(True)

    plt.legend()
    plt.savefig(f'{os.path.join(CDF_PATH, target)}_cdf_log.png', dpi=300)


def plot_CDF(data_path, target: str):
    data = np.array(pd.read_excel(data_path, usecols=[target]))
    
    y = {}
    for d in data:
        d = float(d)
        if d not in y:
            y[d] = 1
        else:
            y[d] += 1
    
    x = sorted(y.keys())
    z = [float(y[k]) / len(data) * 100 for k in x] 

    z_cdf = np.cumsum(z) 
    
    plt.plot(x, z_cdf, label=target)
    plt.xlabel(target)  
    plt.ylabel('CDF(%)') 
    plt.title('CDF of' + ' ' + target) 
    my_x_ticks = np.arange(0*1e6, 10*1e6, 1e6)
    plt.xticks(my_x_ticks)
    my_y_ticks = np.arange(0, 110, 25)
    plt.yticks(my_y_ticks)
    plt.gca().spines['top'].set_visible(False)  
    plt.gca().spines['right'].set_visible(False)  

    plt.gca().spines['left'].set_position('zero')  
    plt.gca().spines['bottom'].set_position('zero') 

    plt.grid(True, which='both', axis='both', linewidth=0.7, alpha=0.5)  

    plt.gca().set_axisbelow(True)
    
    plt.gca().set_xlim([min(x), max(x)])
    plt.gca().set_ylim([0, 100])

    plt.gca().tick_params(axis='x', pad=5)  
    plt.legend()
    plt.savefig(f'{os.path.join(CDF_PATH,target)}_cdf.png',dpi=300)


def plot_CDF_cards(data_path, target: str):
    data = np.array(pd.read_excel(data_path, usecols=[target, 'node_num']))
    
    y = {}
    for d in data:
        d = float(d[0] * d[1])
        if d not in y:
            y[d] = 1
        else:
            y[d] += 1
    
    x = sorted(y.keys())
    z = [float(y[k]) / len(data) * 100 for k in x] 

    z_cdf = np.cumsum(z) 
    
    plt.plot(x, z_cdf, label=target)
    plt.xlabel(target, labelpad=10)  
    plt.ylabel('CDF(%)', labelpad=10)  
    plt.title(f'CDF of {target}')
    plt.xscale('log')
    
    plt.gca().spines['top'].set_visible(False)  
    plt.gca().spines['right'].set_visible(False)  
    plt.grid(True, which='both', axis='both', linewidth=0.7, alpha=0.5)  
    plt.gca().set_axisbelow(True)

    plt.legend()
    plt.savefig(f'{os.path.join(CDF_PATH, target)}_cdf_log.png', dpi=300)


def plot_CDF_duration(data_path, target: str):
    data = np.array(pd.read_excel(data_path, usecols=[target]))
    
    y = {}
    for d in data:
        d = float(d)
        if d not in y:
            y[d] = 1
        else:
            y[d] += 1
    
    x = sorted(y.keys())  
    z = [float(y[k]) / len(data) * 100 for k in x]  
    z_cdf = np.cumsum(z)  
    x = [xi + 1e-5 if xi == 0 else xi for xi in x]  
    
    plt.plot(x, z_cdf, label=target)
    plt.xlabel(target, labelpad=10)  
    plt.ylabel('CDF(%)', labelpad=10)  
    plt.title(f'CDF of {target}')
    plt.xscale('log')
    
    plt.gca().spines['top'].set_visible(False)  
    plt.gca().spines['right'].set_visible(False)  
    plt.grid(True, which='both', axis='both', linewidth=0.7, alpha=0.5)  
    plt.gca().set_axisbelow(True)

    plt.legend()
    plt.savefig(f'{os.path.join(CDF_PATH, target)}_cdf_log.png', dpi=300)
    plt.show()


def main():
    target = 'gpu_time'
    for type in ['multi']:
        for is_completed in [True]:
            analysis(type, is_completed, target)
    
            

    


if __name__ == '__main__':
    # main()
    plot_daily_hourly_task_arrivals('analysis/excel')