import os.path
import hydra
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import statistics

from dataset.seq_dataset import ElectricityDataset
from models.lstm import LSTM, load_model
from models.transformer import Transformer
from models.model import AdaLstm


def plot(y_gt, y_pred, index, save_path):
    plt.figure(figsize=(10, 6))

    plt.plot(y_gt, label="Real Data", color="blue", linestyle="-", linewidth=2)
    plt.plot(y_pred, label="Predicted Data", color="red", linestyle="--", linewidth=2)

    plt.title(f"Real vs Predicted Data - Sample {index + 1}")
    plt.xlabel("Time Steps")
    plt.ylabel("Values")
    plt.legend()

    plt.savefig(save_path)
    print(f"The plot of sample {index} has been saved to {save_path}")

    plt.close()


@hydra.main(version_base="1.3", config_path="config", config_name="eval_config_ada_lstm.yaml")
def evaluate(cfg):
    model_cfg = cfg["model"][cfg["model_name"]]
    # 加载测试集
    val_set = ElectricityDataset(cfg["data"], cfg["eval_type"], dataset="test_set")
    val_loader = val_set.get_loader(1)
    x_shape, y_shape = val_set.get_data_shape()

    # 加载模型
    if cfg["model_name"] == "lstm":
        model = LSTM(input_size=x_shape[2],
                     hidden_size=model_cfg["hidden_size"],
                     output_size=y_shape[1])
    elif cfg["model_name"] == "transformer":
        model = Transformer(n_features=x_shape[2],
                            d_model=model_cfg["d_model"],
                            nhead=model_cfg["n_head"],
                            num_encoder_layers=model_cfg["num_enc_layers"],
                            num_decoder_layers=model_cfg["num_dec_layers"],
                            dim_feedforward=model_cfg["dim_feedforward"],
                            dropout=model_cfg["dropout"],
                            pred_len=y_shape[1])
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
    else:
        raise ValueError(f"There is no model named {cfg['model_name']}.")

    model = load_model(model, cfg["weight_path"]).to(cfg["device"])
    print(model)
    model.eval()
    output_dir = cfg['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    mse_list, mae_list = [], []
    gt_list, pred_list = [], []
    sum_mse, sum_mae = 0.0, 0.0
    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            x, y = x.to(cfg["device"]), y.to(cfg["device"])
            pred = model(x)

            save_path = os.path.join(output_dir, f"sample_{i}.png")
            y_ori = torch.from_numpy(val_set.scaler["y_scaler"].inverse_transform(y.cpu().detach().numpy()))
            y_pred_ori = torch.from_numpy(val_set.scaler["y_scaler"].inverse_transform(pred.cpu().detach().numpy()))

            if not cfg["eval_type"] == "one_seq":
                plot(y_ori[0], y_pred_ori[0], i, save_path)

                # MSE Loss
                mse_loss = nn.MSELoss()(pred, y).item()
                mse_list.append(mse_loss)
                sum_mse += mse_loss

                # MAE Loss
                mae_loss = nn.L1Loss()(pred, y).item()
                mae_list.append(mae_loss)
                sum_mae += mae_loss
            else:
                gt_list.append(y_ori[0][0].item())
                pred_list.append(y_pred_ori[0][0].item())

    avg_mse = sum_mse / len(val_loader)
    std_mse = statistics.stdev(mse_list)
    print(f"Average MSE of {len(val_loader)} samples:", avg_mse)
    print(f"The standard deviation of MSE", std_mse)

    avg_mae = sum_mae / len(val_loader)
    std_mae = statistics.stdev(mae_list)
    print(f"Average MAE of {len(val_loader)} samples:", avg_mae)
    print(f"The standard deviation of MAE", std_mae)


if __name__ == '__main__':
    evaluate()
