import math, torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = - torch.exp(torch.arange(0, d_model, 2) * (math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]


class Transformer(nn.Module):
    def __init__(self,
                 n_features,
                 d_model=128,
                 nhead=8,
                 num_encoder_layers=3,
                 num_decoder_layers=3,
                 dim_feedforward=256,
                 dropout=0.1,
                 pred_len=90):
        super().__init__()
        self.d_model = d_model
        # 特征映射
        self.enc_in = nn.Linear(n_features, d_model)
        self.dec_in = nn.Linear(1, d_model)
        # 位置编码
        self.pos_enc = PositionalEncoding(d_model)
        # Transformer
        self.transformer = nn.Transformer(d_model=d_model,
                                          nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout,
                                          batch_first=True)
        # 输出头
        self.proj = nn.Linear(d_model, 1)
        self.pred_len = pred_len

    def forward(self, src):
        B = src.size(0)

        # --- Encoder ---
        enc = self.enc_in(src)                 # (B, 90, d_model)
        enc = self.pos_enc(enc)

        # --- Decoder ---
        ys = src[:, -1:, :1].detach() * 0.0  # (B, 1, 1)   dummy start
        preds = []
        for _ in range(self.pred_len):
            dec_in = self.dec_in(ys)          # (B, t, d_model)
            dec_in = self.pos_enc(dec_in)

            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                dec_in.size(1)).to(src.device)

            out = self.transformer(enc, dec_in, tgt_mask=tgt_mask)
            next_step = self.proj(out[:, -1])  # (B, 1)
            preds.append(next_step)

            ys = torch.cat([ys, next_step.unsqueeze(1)], dim=1)

        preds = torch.cat(preds, dim=1)        # (B, 90)
        return preds
