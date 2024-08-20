import os
import sys

import torch
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary


class Logger(object):
    def __init__(self, opt):
        """紀錄訓練過程的資訊"""
        self.log_dir = opt.log_dir
        file_name = os.path.join(self.log_dir, "opt.txt")  # opt.txt: 紀錄訓練過程的參數

        # 解析 opt 的參數
        args = dict(
            (name, getattr(opt, name)) for name in dir(opt) if not name.startswith("_")
        )
        # 寫入 opt.txt
        with open(file_name, "wt") as opt_file:
            opt_file.write("==> torch version: {}\n".format(torch.__version__))
            opt_file.write(
                "==> cudnn version: {}\n".format(torch.backends.cudnn.version())
            )
            opt_file.write("==> Cmd:\n")
            opt_file.write(str(sys.argv))
            opt_file.write("\n==> Opt:\n")
            for k, v in sorted(args.items()):
                opt_file.write("  %s: %s\n" % (str(k), str(v)))

        """初始化 tensorboard"""
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.writer.add_hparams(args, {})  # 將參數寫入 tensorboard

    def add_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def close(self):
        self.writer.close()

    def write_model_summary(self, model):
        summary_str = summary(model)
        with open(os.path.join(self.log_dir, "model_summary.txt"), "w") as f:
            f.write(str(summary_str))
