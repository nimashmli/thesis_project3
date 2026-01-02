import torch
from torch import nn, optim
from tqdm import tqdm
import os
import json
import numpy as np
from pathlib import Path

class Trainer:
    def __init__(self, model, train_loader, test_loader, device,label_method,
                 optimizer_cls=optim.Adam, lr=1e-3, epochs=10, loss_fn=None,
                 checkpoint_path="checkpoint.pth", log_path="training_log.json",
                 verbose=False, save_each_epoch=True, config_path=None):
        """
        کلاس آموزش با قابلیت ادامه از چک‌پوینت.
        
        Parameters:
        -----------
        config_path : str or Path, optional
            مسیر فایل config برای به‌روزرسانی وضعیت آزمایش
        """
        self.label_method = label_method
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.epochs = epochs
        self.optimizer = optimizer_cls(self.model.parameters(), lr=lr)
        self.loss_fn = loss_fn if loss_fn else  nn.CrossEntropyLoss()
        self.checkpoint_path = checkpoint_path
        self.log_path = log_path
        self.config_path = config_path
        self.start_epoch = 1
        self.verbose = verbose
        self.save_each_epoch = save_each_epoch
        self.history = {
                "epoch": [],
                "train_loss":[],
                "val_loss": [],
                "train_acc": [] , 
                "val_acc": []
            }

        # اگر چک‌پوینت قبلی هست، لود کن
        self._load_checkpoint()

    def _save_checkpoint(self, epoch):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history
        }
        torch.save(checkpoint, self.checkpoint_path)
        with open(self.log_path, "w") as f:
            json.dump(self.history, f, indent=4)
        
        # به‌روزرسانی config اگر موجود باشد
        if self.config_path and os.path.exists(self.config_path):
            try:
                from experiment_manager import ExperimentManager
                manager = ExperimentManager()
                manager.update_experiment_config(
                    self.config_path,
                    current_epoch=epoch,
                    completed_epochs=epoch,
                    status="in_progress"
                )
            except Exception as e:
                # اگر خطا رخ داد، ادامه بده
                pass

    def _load_checkpoint(self):
        if os.path.exists(self.checkpoint_path):
            print(f"[INFO] Loading checkpoint from '{self.checkpoint_path}'...")
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.history = checkpoint.get("history", [])
            self.start_epoch = checkpoint["epoch"] + 1
            #print(f"[INFO] Resuming from epoch {self.start_epoch}")
        else:
            pass
            #print("[INFO] No checkpoint found. Starting from scratch.")

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0
        loop = tqdm(self.train_loader, desc="Training", leave=False)
        for x, y in loop:
            x, y = x.to(self.device), y.to(self.device).long()

            self.optimizer.zero_grad()
            y_pred = self.model(x)
            loss = self.loss_fn(y_pred, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        return total_loss / len(self.train_loader)

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device).long()
                y_pred = self.model(x)
                loss = self.loss_fn(y_pred, y)
                total_loss += loss.item()*x.size(0)

                predicted = torch.argmax(y_pred , dim=1)
                correct += (predicted == y.int()).sum().item()
                total += y.size(0)

        accuracy_test = 100 * correct / total
        loss_test = total_loss / total
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device).long()
                y_pred = self.model(x)
                loss = self.loss_fn(y_pred, y)
                total_loss += loss.item()*x.size(0)

                predicted = torch.argmax(y_pred , dim=1)
                correct += (predicted == y.int()).sum().item()
                total += y.size(0)

        accuracy_train = 100 * correct / total
        loss_train = total_loss / total
        return loss_train , loss_test,accuracy_train ,  accuracy_test

    def fit(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
            train_loss = self.train_one_epoch()
            loss_train , loss_test,accuracy_train ,  accuracy_test = self.evaluate()

            # ذخیره در history
            self.history['epoch'].append(epoch)
            self.history['train_loss'].append(loss_train)
            self.history['val_loss'].append(loss_test)
            self.history['train_acc'].append(accuracy_train)
            self.history['val_acc'].append(accuracy_test)

            if self.verbose:
                print(f"Epoch [{epoch}/{self.epochs}] Train Loss: {loss_train:.4f} | Val Loss: {loss_test:.4f} | Train Acc: {accuracy_train:.2f}% | Val Acc: {accuracy_test:.2f}%")

            # ذخیره چک‌پوینت
            if self.save_each_epoch:
                self._save_checkpoint(epoch)

        # به‌روزرسانی نهایی config
        if self.config_path and os.path.exists(self.config_path):
            try:
                from experiment_manager import ExperimentManager
                manager = ExperimentManager()
                manager.update_experiment_config(
                    self.config_path,
                    current_epoch=self.epochs,
                    completed_epochs=self.epochs,
                    status="completed"
                )
            except Exception as e:
                pass

        print(
            f"Train Loss: {np.mean(self.history['train_loss'][-5:]):.4f} | "
            f"Val Loss: {np.mean(self.history['val_loss'][-5:]):.4f} | "
            f"Train Acc: {np.mean(self.history['train_acc'][-5:]):.2f}% | "
            f"Val Acc: {np.mean(self.history['val_acc'][-5:]):.2f}%"
        )

        return self.history


