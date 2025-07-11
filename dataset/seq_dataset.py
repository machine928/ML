import logging
import random
import pandas as pd
import numpy as np
import torch
import math

from scipy.ndimage import gaussian_filter1d
from omegaconf import DictConfig
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from tsaug import AddNoise, Drift, TimeWarp


class ElectricityDataset:
    """
    电力数据集：用于读取预处理后的电力数据
    """
    def __init__(self, data_cfg: DictConfig, seq_type="short_seq", dataset="train_set", augment=False, aug_prob=0.5, extend=False, denoise=False):
        self.cfg = data_cfg
        self.seq_type = seq_type
        self.dataset = dataset
        self.window_size = self.cfg["data_window"][self.seq_type]["data_size"]
        self.window_step = self.cfg["data_window"][self.seq_type]["step"]
        self.pred_size = self.cfg["data_window"][self.seq_type]["pred_size"]
        self.df = pd.read_csv(self.cfg["datasets"]["aggregated"][self.dataset]["directory"])
        self.scaler = {"x_scaler": MinMaxScaler(), "y_scaler": MinMaxScaler()}
        self.x_shape = None
        self.y_shape = None

        self.extend = extend
        self.augment = augment
        self.aug_prob = aug_prob
        self.augmenter = self.build_augmenter()
        self.denoise = denoise

        self.set_seed()

    def dynamic_window(self, magnification=1.5, sub_window_size=30, sub_pred_size=30, start=0):
        assert self.window_size / sub_window_size != sub_pred_size / sub_pred_size, "sub_window does not meet requirements"

        data = self.df
        x, y = [], []
        sub_x, sub_y = [], []
        for _ in range(len(data)):
            in_end = start + sub_window_size
            out_end = in_end + sub_pred_size

            if out_end < len(data):
                sub_x.append(data.iloc[start:in_end, 1:])
                sub_y.append(data.iloc[in_end:out_end, 1])
            start += sub_window_size

        for _ in range(int(len(data) * (magnification - 1))):
            x_samp, y_samp = pd.DataFrame(), pd.DataFrame()
            # 随机在子窗口内选择一部分样本进行拼接
            idx = [random.randint(0, len(sub_x) - 1) for _ in range(self.window_size // sub_window_size)]
            for i in idx:
                x_samp = pd.concat([x_samp, sub_x[i]], axis=0, ignore_index=True)
                y_samp = pd.concat([y_samp, sub_y[i]], axis=0, ignore_index=True)
            x.append(x_samp)
            y.append(y_samp.iloc[:, 0])

        return x, y

    def sliding_window(self, start=0):
        data = self.df
        x, y = [], []

        for _ in range(len(data)):
            in_end = start + self.window_size
            out_end = in_end + self.pred_size

            if out_end < len(data):
                x.append(data.iloc[start:in_end, 1:])
                y.append(data.iloc[in_end:out_end, 1])
            start += self.window_step

        if self.extend and self.dataset == "train_set":
            scale = math.gcd(self.window_size, self.pred_size) if self.window_size != self.pred_size else 3
            add_x, add_y = self.dynamic_window(sub_window_size=self.window_size // scale, sub_pred_size=self.pred_size // scale)
            x.extend(add_x)
            y.extend(add_y)

        x, y = np.array(x), np.array(y)

        if self.denoise and self.dataset == "train_set":
            x = self.gaussian_denoise(x)

        self.x_shape = x.shape
        self.y_shape = y.shape

        logging.info(f"{self.dataset} shape: x: {x.shape}, y: {y.shape}")

        return x, y

    def get_loader(self, batch_size):
        x, y = self.sliding_window()
        x, y = self.normalizing_data(x, y)
        if self.augment and self.dataset == "train_set":
            x = self.data_enhance(x)
        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        dataset = TensorDataset(x_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        return loader

    def normalizing_data(self, x, y):
        x = np.nan_to_num(x, nan=0.0) if np.isnan(x).any() else x
        y = np.nan_to_num(y, nan=0.0) if np.isnan(y).any() else y

        # 对特征列做归一化
        x_scaled = self.scaler["x_scaler"].fit_transform(x.reshape(-1, x.shape[-1])).reshape(x.shape)
        y_scaled = self.scaler["y_scaler"].fit_transform(y)

        return x_scaled, y_scaled

    def get_data_shape(self):
        return self.x_shape, self.y_shape

    @staticmethod
    def build_augmenter():
        return (
                AddNoise(scale=0.02)
                + Drift(max_drift=0.05)
                + TimeWarp(n_speed_change=3,
                           max_speed_ratio=2.0)
        )

    @staticmethod
    def set_seed(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    @staticmethod
    def gaussian_denoise(x, sigma=1.0):
        return gaussian_filter1d(x, sigma=sigma, axis=0)

    def data_enhance(self, x):
        if self.augment and np.random.rand() < self.aug_prob:
            x = self.augmenter.augment(x.astype(np.float32, copy=True))
        return x
