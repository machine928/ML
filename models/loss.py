import torch
import torch.nn as nn
import torch.nn.functional as F

class AvgLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss(reduction=reduction)
        self.mae = nn.L1Loss(reduction=reduction)

    def forward(self, pred, target):
        loss_mse = self.mse(pred, target)
        loss_mae = self.mae(pred, target)
        return self.alpha * loss_mse + self.beta * loss_mae


class HuberTimeDecayLoss(nn.Module):
    def __init__(self, delta=1.0, gamma=0.97):
        super().__init__()
        self.delta = delta
        self.gamma = gamma

    def forward(self, pred, target):
        err = pred - target                       # (B, T)
        abs_err = torch.abs(err)

        # Huber
        quadratic = torch.minimum(abs_err, torch.tensor(self.delta))
        linear = abs_err - quadratic
        huber = 0.5 * quadratic**2 + self.delta * linear

        # 时间衰减权重 w_t = γ^t
        T = err.size(1)
        weights = self.gamma ** torch.arange(T, device=pred.device)
        loss = (huber * weights).mean()
        return loss


class TrendLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        # pred, target: (B, T)
        pred_diff = pred[:, 1:] - pred[:, :-1]
        target_diff = target[:, 1:] - target[:, :-1]

        # 趋势方向一致性（用 cosine 或 sign）
        trend_loss = 1 - nn.functional.cosine_similarity(pred_diff, target_diff, dim=1).mean()

        # 可选组合 MSE
        mse_loss = self.mse(pred, target)

        return mse_loss + trend_loss  # 可加权组合


class PeakWeightedMSELoss(nn.Module):
    r"""对峰值/陡坡点加权的 MSE。
    Args:
        alpha (float): 峰段权重系数 (>=0)。
        threshold (float): 峰判定阈值，取 |Δy| > threshold * std(Δy) 视为峰。
    """
    def __init__(self, alpha: float = 3.0, threshold: float = 0.15):
        super().__init__()
        self.alpha = alpha
        self.threshold = threshold

    @torch.no_grad()
    def _peak_mask(self, y: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(y[..., 1:] - y[..., :-1])
        # 在首位补 0 保持长度一致
        mask = torch.cat([torch.zeros_like(diff[..., :1]), diff], dim=-1)
        return (mask > self.threshold * diff.std(dim=-1, keepdim=True)).float()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weight = 1.0 + self.alpha * self._peak_mask(target)
        return torch.mean(weight * (pred - target).pow(2))

class FourierLoss(nn.Module):
    """
    将预测和真实值变换到频域，比较其频谱的 MAE
    强调周期性模式的一致性
    """
    def __init__(self):
        super(FourierLoss, self).__init__()

    def forward(self, y_pred, y_true):
        # 计算频谱
        fft_pred = torch.fft.fft(y_pred, dim=1)
        fft_true = torch.fft.fft(y_true, dim=1)
        loss = torch.mean(torch.abs(torch.abs(fft_pred) - torch.abs(fft_true)))
        return loss


class PeriodicConsistencyLoss(nn.Module):
    """
    周期一致性损失
    利用电力数据的日/周周期性特征
    """

    def __init__(self, base_loss_fn, period=24, consistency_lambda=0.2):
        super().__init__()
        self.base_loss_fn = base_loss_fn
        self.period = period
        self.consistency_lambda = consistency_lambda

    def forward(self, y_pred, y_true):
        base_loss = self.base_loss_fn(y_pred, y_true)

        # 确保序列足够长以提取周期
        if y_pred.size(1) > self.period:
            current_period = y_pred[:, -self.period:]
            prev_period = y_pred[:, -2 * self.period:-self.period]

            # 计算相邻周期相似性
            period_loss = F.mse_loss(current_period, prev_period)
            return base_loss + self.consistency_lambda * period_loss

        return base_loss