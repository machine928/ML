import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from models.lstm import LSTM


class FeatureAttention(nn.Module):

    def __init__(self, n_features, d_k, n_head=1, dropout=0.1):
        super().__init__()

        self.n_features = n_features
        self.n_head = n_head
        self.d_k = d_k or n_features // n_head

        assert n_features % n_head == 0, "n_features 必须能被 n_head 整除"

        self.q_proj = nn.Linear(n_features, n_head * self.d_k, bias=False)
        self.k_proj = nn.Linear(n_features, n_head * self.d_k, bias=False)
        self.v_proj = nn.Linear(n_features, n_head * self.d_k, bias=False)

        self.o_proj = nn.Linear(n_head * self.d_k, n_features, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_attn=False):
        """
        x : (B, T, F)
        return_attn : 若 True，则同时返回注意力权重 (B, T, n_head, F, F)
        """
        B, T, F_dim = x.size()

        # 生成 Q, K, V
        # (B, T, n_head, d_k)
        Q = self.q_proj(x).view(B, T, self.n_head, self.d_k)
        K = self.k_proj(x).view(B, T, self.n_head, self.d_k)
        V = self.v_proj(x).view(B, T, self.n_head, self.d_k)

        # 计算注意力分数
        Q_flat = Q.reshape(B * T, self.n_head, self.d_k)
        K_flat = K.reshape(B * T, self.n_head, self.d_k)
        V_flat = V.reshape(B * T, self.n_head, self.d_k)

        # scores_flat : (B*T, n_head, 1, d_k) ⋅ (B*T, n_head, d_k, 1) → (B*T, n_head, 1, 1)
        scores_flat = torch.einsum('bhd,bjd->bhj', Q_flat, K_flat) / math.sqrt(self.d_k)
        attn_flat = F.softmax(scores_flat, dim=-1)          # (B*T, n_head, d_k)
        attn_flat = self.dropout(attn_flat)

        context_flat = torch.einsum('bhf,bhf->bhf', attn_flat, V_flat)

        # 还原尺寸
        context = context_flat.view(B, T, self.n_head * self.d_k)  # (B, T, F)

        out = self.o_proj(context)

        if return_attn:
            attn = attn_flat.view(B, T, self.n_head, self.d_k)
            return out, attn
        return out


class DynamicFeatureSelector(nn.Module):
    def __init__(self, n_features, d_hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_features, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, n_features),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, T, F)
        attn = self.mlp(x)         # (B, T, F)
        x = x * attn               # 加权特征
        return x


class AdaLstm(nn.Module):
    def __init__(self, input_size, n_features, hidden_size, output_size, transformer,
                 n_head=1, d_k=64, num_layers=4, l_dropout=0.2):
        super().__init__()

        self.transformer = transformer
        self.feat_selector = DynamicFeatureSelector(n_features=n_features, d_hidden=d_k)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=l_dropout, bidirectional=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

        self.lstm_fc = nn.Linear(hidden_size * 2, output_size)
        self.out_fc = nn.Linear(output_size * 2, output_size)

    def forward(self, x):
        # 自适应特征选择
        # x = self.feat_selector(x)
        # out, _ = self.lstm(x)
        # out = self.linear(out[:, -1, :])

        # 自适应特征选择 + 集成模型
        x = self.feat_selector(x)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.lstm_fc(lstm_out[:, -1, :])
        transformer_out = self.transformer(x)
        combined = torch.cat((lstm_out, transformer_out), dim=-1)
        out = self.out_fc(combined)

        return out

