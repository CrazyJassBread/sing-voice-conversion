#!/usr/bin/env python3
"""
prepare_shared.py

提供用于所有prepare_xx.py脚本的共享并行处理工具。
"""

import os
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import torch

def get_device():
    """返回单个device: 如果有CUDA则使用cuda:0，否则CPU。"""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def run_parallel(audio_list, process_func, num_workers, desc="Processing", initializer=None, initargs=()):
    """
    用Pool对audio_list中的每个音频执行process_func，并显示进度条。
    
    参数：
        audio_list: 待处理的音频描述列表（通常从meta_info.json中读取）。
        process_func: 对单个audio处理的函数，该函数只接受单个audio字典。
        num_workers: 并行进程数。
        desc: 进度条描述文字。
        initializer: Pool初始化函数（可选），用于例如加载模型。
        initargs: initializer的参数（以tuple形式）。
    """
    num_workers = min(num_workers, cpu_count())
    with Pool(processes=num_workers, initializer=initializer, initargs=initargs) as pool:
        list(tqdm(pool.imap_unordered(process_func, audio_list),
                  total=len(audio_list), desc=desc, unit="file"))