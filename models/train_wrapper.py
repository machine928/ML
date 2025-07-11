import logging
import os.path
import copy
import torch
import torch.nn as nn
import random
import numpy as np

from eval import plot

class Trainer:
    def __init__(self, train_cfg, data_loader, val_loader, model, criterion, device="cuda", seed=42):
        self.cfg = train_cfg
        self.device = device
        self.data_loader = data_loader
        self.val_loader = val_loader
        self.model = model.to(self.device)
        self.criterion = criterion

        self.set_seed(seed)

    def train(self, optimizer, val_set):
        save_path_root = self.cfg["output_dir"]
        os.makedirs(save_path_root, exist_ok=True)

        best_loss = float("inf")
        best_model = None
        for epoch in range(self.cfg["epoch"]):
            epoch_loss = 0
            for batch_x, batch_y in self.data_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                pred = self.model(batch_x)
                loss = self.criterion(pred, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            logging.info(f"Epoch {epoch+1} | Loss: {epoch_loss / len(self.data_loader):.4f}")

            if epoch % self.cfg["val_every_ep"] == 0 and epoch != 0:
                mse, mae = self.validate()
                if mse + mae < best_loss:
                    self.save_weight(epoch, self.model, os.path.join(save_path_root, f'epoch_{epoch}.pth'))
                    best_loss = mse + mae
                    best_model = copy.deepcopy(self.model)
        self.save_weight(-1, best_model, os.path.join(save_path_root, f'best.pth'))

    def validate(self):
        self.model.eval()

        val_loss_mse = 0.0
        val_loss_mae = 0.0
        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x)

                # MSE Loss
                mse_loss = nn.MSELoss()(pred, y)
                val_loss_mse += mse_loss.item()

                # MAE Loss
                mae_loss = nn.L1Loss()(pred, y)
                val_loss_mae += mae_loss.item()
        # 计算平均验证损失
        avg_val_loss_mse = val_loss_mse / len(self.val_loader)
        avg_val_loss_mae = val_loss_mae / len(self.val_loader)

        logging.info(f"Validation MSE Loss: {avg_val_loss_mse:.4f}")
        logging.info(f"Validation MAE Loss: {avg_val_loss_mae:.4f}")

        self.model.train()

        return avg_val_loss_mse, avg_val_loss_mae

    @staticmethod
    def set_seed(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    @staticmethod
    def save_weight(epoch, model, save_path):
        torch.save({
            'epoch': epoch,
            "model_state_dict": model.state_dict()
        }, save_path)

        logging.info(f"New best epoch, weight has been saved to {save_path}")
