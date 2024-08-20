import os

import numpy as np
import torch
from torch.distributed import destroy_process_group, init_process_group


def ddp_setup():
    """初始化分散式訓練環境"""
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def ddp_cleanup():
    """清理分散式訓練環境"""
    destroy_process_group()


def seed_everything(seed=42):
    """設置隨機種子"""
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
