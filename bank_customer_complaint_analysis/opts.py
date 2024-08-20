"""使用 argparse 來設定命令列參數，更方便控制變數和對接"""

import argparse
import os
import time


class Opts:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="Training options for Bank narrative classification model"
        )
        self.parser.add_argument(
            "--exp_id", default="default", help="Experiment ID for logging purposes"
        )

        """基本實驗設定"""
        self.parser.add_argument(
            "--file_path",
            default="data/complaints.csv",
            help="Path to the training data CSV file",
        )
        self.parser.add_argument(
            "--max_length",
            type=int,
            default=128,
            help="Number of tokens to truncate or pad the input sequence",
        )

        self.parser.add_argument(
            "--seed", type=int, default=87, help="Random seed for reproducibility"
        )  # 隨機種子

        self.parser.add_argument(
            "--gpu_id",
            type=int,
            default=0,  # 默認使用第一個 GPU
            help="GPU ID to use, -1 for CPU. Use comma for multiple GPUs, e.g., '0,1'",  # 指定 GPU 編號，-1 為 CPU
        )

        self.parser.add_argument(
            "--max_epochs",
            type=int,
            default=4,
            help="Maximum number of epochs to train the model",  # 最大訓練週期
        )

        self.parser.add_argument(
            "--batch_size",
            type=int,
            default=128,
            help="Batch size for training and validation",  # 批次大小
        )

        self.parser.add_argument(
            "--num_workers",
            type=int,
            default=8,
            help="Number of worker threads for data loading",  # 資料加載器的工作數量(cpu)
        )

        self.parser.add_argument(
            "--lr",
            type=float,
            default=1e-3,
            help="Initial learning rate for the optimizer",  # 學習率
        )

        self.parser.add_argument(
            "--eta_min",
            type=float,
            default=1e-6,
            help="Minimum learning rate during training",  # 最小學習率
        )

    def parse(self, args=""):
        if args == "":
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)
        # 設定根目錄
        opt.root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        # 設定設備編號，如果 gpu_id 為 -1 則使用 CPU
        opt.device = f"cuda:{opt.gpu_id}" if opt.gpu_id >= 0 else "cpu"
        # 設定日誌目錄
        time_str = time.strftime("%Y-%m-%d-%H-%M-%S")
        opt.log_dir = os.path.join(opt.root_dir, "runs", opt.exp_id, time_str)

        # 檢查資料集目錄是否存在，不存在則拋出異常
        opt.file_path = os.path.join(opt.root_dir, opt.file_path)
        if not os.path.exists(opt.file_path):
            raise ValueError(f"Data directory {opt.file_path} not found")

        if not os.path.exists(opt.log_dir):  # 如果日誌目錄不存在, 則建立日誌目錄
            os.makedirs(opt.log_dir, exist_ok=True)
            print(f"Created directory {opt.log_dir}")
        return opt
