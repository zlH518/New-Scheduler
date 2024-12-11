import os
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import List
import logging

FIGURE_PATH = 'analysis/Figure'

def plot_target_metric(folder_path: str, target_metric: str):
    """
    绘制指定文件夹中所有算法的目标指标随时间变化的折线图。

    参数:
        folder_path (str): 包含JSON结果文件的文件夹路径。
        target_metric (str): 要绘制的目标指标名称。
        save_path (str, optional): 图像保存路径。如果为None，则仅显示图像。
    """
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    if not os.path.isdir(folder_path):
        logging.error(f"文件夹路径 '{folder_path}' 不存在或不是一个文件夹。")
        return
    
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    
    if not json_files:
        logging.error(f"在文件夹 '{folder_path}' 中未找到任何JSON文件。")
        return
    
    plt.figure(figsize=(12, 8))
    for json_file in json_files:
        file_path = os.path.join(folder_path, json_file)
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            if not isinstance(data, list):
                logging.warning(f"文件 '{json_file}' 的数据格式不是列表，跳过。")
                continue
            
            timestamps = []
            metrics = []
            for entry in data:
                if 'timestamp' in entry and target_metric in entry:
                    timestamps.append(entry['timestamp'])
                    metrics.append(entry[target_metric])
                else:
                    logging.warning(f"文件 '{json_file}' 中的某些条目缺少 'timestamp' 或 '{target_metric}'，这些条目将被忽略。")
            
            if not timestamps:
                logging.warning(f"文件 '{json_file}' 中没有有效的数据点，跳过。")
                continue
            
            sorted_data = sorted(zip(timestamps, metrics), key=lambda x: x[0])
            sorted_timestamps, sorted_metrics = zip(*sorted_data)
            algorithm_name = os.path.splitext(json_file)[0]  
            plt.plot(sorted_timestamps, sorted_metrics, label=algorithm_name)
        
        except json.JSONDecodeError:
            logging.error(f"文件 '{json_file}' 不是有效的JSON格式，跳过。")
            continue
        except Exception as e:
            logging.error(f"处理文件 '{json_file}' 时发生错误: {e}")
            continue
    plt.xlabel('Timestamp', fontsize=14)
    plt.ylabel(target_metric.replace('_', ' ').title(), fontsize=14)
    plt.title(f'{target_metric.replace("_", " ").title()} Over Time for Different Algorithms', fontsize=16)
    plt.legend(title='Algorithms')
    plt.grid(True, linestyle='--', alpha=0.6)
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.gca().spines['top'].set_visible(False)  
    plt.gca().spines['right'].set_visible(False)  
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_PATH, target_metric) + ".png", dpi=300)



# 示例用法
if __name__ == "__main__":
    folder = "monitor/no_completed/no_migrate_no_step/2000"
    target = "predict_waiting_time"
    plot_target_metric(folder, target)
