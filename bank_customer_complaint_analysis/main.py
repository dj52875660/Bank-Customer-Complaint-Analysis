import argparse

from dataset import TextDataset
from logger import Logger
from models import *
from opts import Opts
from torch.utils.data import DataLoader
from trainer import Trainer
from utils import seed_everything


def main(opt: argparse.Namespace):
    seed_everything(opt.seed)  # 設定隨機種子
    logger = Logger(opt)  # 設定 logger
    train_dataset = TextDataset(
        root=opt.file_path,
        seed=opt.seed,
        mode='train',
        max_length=opt.max_length
    )
    val_dataset = TextDataset(
        root=opt.file_path,
        seed=opt.seed,
        mode='val',
        max_length=opt.max_length
    )
    test_dataset = TextDataset(
        root=opt.file_path,
        seed=opt.seed,
        mode='test',
        max_length=opt.max_length
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        shuffle=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        shuffle=False
    )
    model = TextClassificationModel(
        num_labels=train_dataset.num_classes
    )
    logger.write_model_summary(model)
    trainer = Trainer(
        opt=opt,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        logger=logger,
    )
    trainer.run()
    logger.close()


if __name__ == "__main__":
    opt = Opts().parse(
        [
            "--max_epochs",
            "4",
            "--max_length",
            "128",
            "--gpu_id",
            "0",
            "--lr",
            "1e-7",
        ]
    )
    main(opt)    