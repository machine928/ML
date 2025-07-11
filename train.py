import hydra
import torch
import torch.nn as nn

from dataset.seq_dataset import ElectricityDataset
from models.lstm import LSTM
from models.transformer import Transformer
from models.model import AdaLstm
from models.train_wrapper import Trainer


@hydra.main(version_base="1.3", config_path="config", config_name="train_config.yaml")
def train(cfg):
    print(cfg)
    device = cfg["device"]
    train_type = cfg["train_type"]
    train_cfg = cfg["training_settings"][train_type]
    model_cfg = cfg["model"][cfg["model_name"]]

    # 准备数据集
    dataset = ElectricityDataset(cfg["data"], seq_type=cfg["train_type"], augment=cfg["augment"], aug_prob=cfg["aug_prob"])
    data_loader = dataset.get_loader(train_cfg["batch_size"])
    val_set = ElectricityDataset(cfg["data"], seq_type=cfg["train_type"], dataset="test_set")
    val_loader = val_set.get_loader(batch_size=1)
    x_shape, y_shape = dataset.get_data_shape()

    # 加载模型
    if cfg["model_name"] == "lstm":
        model = LSTM(input_size=x_shape[2],
                     hidden_size=model_cfg["hidden_size"],
                     output_size=y_shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"])
        criterion = nn.MSELoss()
    elif cfg["model_name"] == "transformer":
        model = Transformer(n_features=x_shape[2],
                            d_model=model_cfg["d_model"],
                            nhead=model_cfg["n_head"],
                            num_encoder_layers=model_cfg["num_enc_layers"],
                            num_decoder_layers=model_cfg["num_dec_layers"],
                            dim_feedforward=model_cfg["dim_feedforward"],
                            dropout=model_cfg["dropout"],
                            pred_len=y_shape[1])
        optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"])
        criterion = nn.MSELoss()
    elif cfg["model_name"] == "ada_lstm":
        tf_cfg = cfg["model"]["transformer"]
        tf = Transformer(n_features=x_shape[2],
                         d_model=tf_cfg["d_model"],
                         nhead=tf_cfg["n_head"],
                         num_encoder_layers=tf_cfg["num_enc_layers"],
                         num_decoder_layers=tf_cfg["num_dec_layers"],
                         dim_feedforward=tf_cfg["dim_feedforward"],
                         dropout=tf_cfg["dropout"],
                         pred_len=y_shape[1])
        model = AdaLstm(input_size=x_shape[2],
                        n_features=model_cfg["n_features"],
                        hidden_size=model_cfg["hidden_size"],
                        output_size=y_shape[1],
                        transformer=tf,
                        n_head=model_cfg["n_head"],
                        d_k=model_cfg["d_k"],
                        num_layers=model_cfg["num_layers"],
                        l_dropout=model_cfg["dropout"])
        optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"], weight_decay=1e-5)
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"There is no model named {cfg['model_name']}.")

    # 训练
    print(model)
    trainer = Trainer(train_cfg=train_cfg,
                      data_loader=data_loader,
                      val_loader=val_loader,
                      model=model,
                      criterion=criterion,
                      device=device)
    trainer.train(optimizer, val_set)


if __name__ == '__main__':
    train()
