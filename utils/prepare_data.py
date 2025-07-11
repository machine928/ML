import pandas as pd
import hydra
import logging


@hydra.main(version_base="1.3", config_path="../config", config_name="train_config.yaml")
def aggregate_data(cfg):
    data_cfg = cfg["data"]
    # 读取CSV
    df = pd.read_csv(data_cfg["datasets"]["raw"]["test_set"]["directory"], parse_dates=['DateTime'])

    try:
        # 创建新的列 sub_metering_remainder
        df['Sub_metering_remainder'] = (df['Global_active_power'] * 1000 / 60) - (
            df['Sub_metering_1'] + df['Sub_metering_2'] + df['Sub_metering_3'])
    except TypeError as e:
        # 强制转换为 float 类型
        cols_to_convert = [
            'Global_active_power',
            'Global_reactive_power',
            'Sub_metering_1',
            'Sub_metering_2',
            'Sub_metering_3',
            'Voltage',
            'Global_intensity',
            'RR',
            'NBJRR1',
            'NBJRR5',
            'NBJRR10',
            'NBJBROU'
        ]

        for col in cols_to_convert:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['Sub_metering_remainder'] = (df['Global_active_power'] * 1000 / 60) - (
                df['Sub_metering_1'] + df['Sub_metering_2'] + df['Sub_metering_3'])

    # 将 sub_metering_remainder 插入在 Sub_metering_3 后面
    cols = list(df.columns)
    insert_pos = cols.index('Sub_metering_3') + 1
    cols = cols[:insert_pos] + ['Sub_metering_remainder'] + cols[insert_pos:-1]
    df = df[cols + ['Sub_metering_remainder']] if 'Sub_metering_remainder' not in cols else df[cols]

    # 提取日期列
    df['Date'] = df['DateTime'].dt.date

    # 指定聚合方式
    agg_dict = {
        'Global_active_power': 'sum',
        'Global_reactive_power': 'sum',
        'Sub_metering_1': 'sum',
        'Sub_metering_2': 'sum',
        'Sub_metering_3': 'sum',
        'Sub_metering_remainder': 'sum',
        'Voltage': 'mean',
        'Global_intensity': 'mean',
        'RR': 'first',
        'NBJRR1': 'first',
        'NBJRR5': 'first',
        'NBJRR10': 'first',
        'NBJBROU': 'first'
    }

    # 按天聚合
    df_daily = df.groupby('Date').agg(agg_dict).reset_index()

    # 保存为新的CSV文件
    output_path = data_cfg["datasets"]["aggregated"]["test_set"]["directory"]
    df_daily.to_csv(output_path, index=False)


if __name__ == '__main__':
    aggregate_data()
