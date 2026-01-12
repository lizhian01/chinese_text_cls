# src/config.py
from dataclasses import dataclass

@dataclass
class Config:
    # data
    data_dir: str = r"data\thucnews_10c_200k"
    train_file: str = "train.csv"
    val_file: str = "val.csv"   # 你的验证集文件名是 val.csv
    test_file: str = "test.csv"

    # artifacts
    out_dir: str = r"outputs\transformer"
    vocab_path: str = r"outputs\transformer\vocab.json"
    label_path: str = r"outputs\transformer\label2id.json"

    # training
    seed: int = 42
    epochs: int = 3
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.01
    max_len: int = 256
    num_workers: int = 0

    # model
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 4
    dim_ff: int = 512
    dropout: float = 0.1

    # vocab
    vocab_size: int = 50000
    min_freq: int = 2

    # artifacts (add)
    out_dir_cnn: str = r"outputs\cnn"
    out_dir_lstm: str = r"outputs\lstm"

CFG = Config()
