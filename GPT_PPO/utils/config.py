# utils/config.py

import json
import logging

def load_config(config_path):
    """
    从JSON文件加载配置。

    参数:
        config_path (str): 配置文件路径。

    返回:
        dict: 配置字典。
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logging.info(f"Configuration loaded from {config_path}.")
        return config
    except Exception as e:
        logging.error(f"Error loading configuration from {config_path}: {e}")
        raise e
