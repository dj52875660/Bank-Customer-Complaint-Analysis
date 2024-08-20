import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from logger import Logger
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction="sum"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = nn.functional.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-BCE_loss)
        F_loss = (1 - pt) ** self.gamma * BCE_loss

        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets.data.view(-1))
            F_loss = alpha_t * F_loss

        if self.reduction == "mean":
            return F_loss.mean()
        elif self.reduction == "sum":
            return F_loss.sum()
        else:
            return F_loss


class Trainer:
    def __init__(
        self,
        opt: argparse.Namespace,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        logger: Logger,
        num_classes: int = 5,
    ):
        self.epochs_run = 0
        # 解析 opt 的必要參數
        self.lr = opt.lr
        self.device = opt.device
        self.eta_min = opt.eta_min
        self.max_epochs = opt.max_epochs
        self.log_dir = opt.log_dir
        self.step = 0
        self.num_classes = num_classes

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.criterion = nn.CrossEntropyLoss()
        # self.criterion = FocalLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        self.lr_scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=2e-5,
            epochs=self.max_epochs,
            steps_per_epoch=len(self.train_loader),
            pct_start=0.15,
            anneal_strategy="cos",
        )

        self.accuracy_metric = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        ).to(self.device)
        self.precision_metric = torchmetrics.Precision(
            task="multiclass", num_classes=num_classes
        ).to(self.device)
        self.recall_metric = torchmetrics.Recall(
            task="multiclass", num_classes=num_classes
        ).to(self.device)
        self.f1_metric = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes
        ).to(self.device)

        self.writer = logger

    def log_metrics(self, prefix: str, epoch: int):
        self.writer.add_scalar(
            f"{prefix}/Accuracy", self.accuracy_metric.compute().item(), epoch
        )
        self.writer.add_scalar(
            f"{prefix}/Precision", self.precision_metric.compute().item(), epoch
        )
        self.writer.add_scalar(
            f"{prefix}/Recall", self.recall_metric.compute().item(), epoch
        )
        self.writer.add_scalar(f"{prefix}/F1", self.f1_metric.compute().item(), epoch)

        self.accuracy_metric.reset()
        self.precision_metric.reset()
        self.recall_metric.reset()
        self.f1_metric.reset()

    def train_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"[{self.device}] Train Epoch {epoch:2d}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(input_ids, attention_mask)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

            self.accuracy_metric(outputs, labels)
            self.precision_metric(outputs, labels)
            self.recall_metric(outputs, labels)
            self.f1_metric(outputs, labels)

            self.writer.add_scalar(
                "Train/Loss", total_loss / (self.step + 1), self.step
            )
            self.writer.add_scalar(
                "Learning Rate", self.optimizer.param_groups[0]["lr"], self.step
            )
            self.log_metrics("Train", self.step)

            self.step += 1
            self.lr_scheduler.step()

    def val_epoch(self, epoch: int):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"[{self.device}] Val Epoch {epoch:2d}")
            for batch in pbar:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                self.accuracy_metric(outputs, labels)
                self.precision_metric(outputs, labels)
                self.recall_metric(outputs, labels)
                self.f1_metric(outputs, labels)

        self.writer.add_scalar("Val/Loss", total_loss, epoch)
        self.log_metrics("Val", epoch)

    def test_epoch(self, epoch: int):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc=f"[{self.device}] Test Epoch {epoch:2d}")
            for batch in pbar:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                self.accuracy_metric(outputs, labels)
                self.precision_metric(outputs, labels)
                self.recall_metric(outputs, labels)
                self.f1_metric(outputs, labels)

        self.writer.add_scalar("Test/Loss", total_loss, epoch)
        self.log_metrics("Test", epoch)

    def run(self):
        """主訓練循環"""
        for epoch in range(self.max_epochs):
            self.train_epoch(epoch)  # 訓練一個epoch
            # self.lr_scheduler.step()  # 更新學習率調整器
            self.val_epoch(epoch)
            self.test_epoch(epoch)
