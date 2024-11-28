import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging

class Plot:
    def plot(config):
        algorithm_paths = [] 
        for scheduler in config['schedulers']:
            for task in config['tasks']:
                algorithm_paths.append((scheduler['name'], os.path.join(config['monitor']['base_path'], scheduler['name'] + f"_{config['timestamp']}.csv")))

        sns.set(style="whitegrid")
        fig, ax = plt.subplots(figsize=(10, 6))
        for algorithm_name, data_path in algorithm_paths:
            df = pd.read_csv(data_path)
        
        ax.plot(df['timestamp'], df['free_rate'], label=f'Free Rate - {algorithm_name}')
        ax.plot(df['timestamp'], df['throughput'], label=f'Throughput - {algorithm_name}')
        ax.plot(df['timestamp'], df['fragment_rate'], label=f'Fragment Rate - {algorithm_name}')
        ax.plot(df['timestamp'], df['avg_waiting_time'], label=f'Avg Waiting Time - {algorithm_name}')
        
        ax.set_xlabel('Timestamp')
        ax.set_ylabel('Metrics')
        ax.set_title('Comparison of Different Algorithms')
        ax.legend(loc='upper right')

        plt.tight_layout()

        if not os.path.exists(config['plotter']['save_path']):
            os.makedirs(config['plotter']['save_path'])

        save_file_path = os.path.join(config['plotter']['save_path'], 'comparison_of_algorithms.png')  # 保存为PNG图像文件
        plt.savefig(save_file_path)
        logging.info(f"Chart saved to {save_file_path}")
        print(f"Chart saved to {save_file_path}")

