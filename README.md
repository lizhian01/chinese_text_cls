# 中文文本分类（THUCNews 10 类）

本仓库提供一个中文新闻文本分类的完整流程，覆盖数据准备、字符级 CNN/LSTM/Transformer 训练、评估与结果汇总。

## 目录结构

```
.
├── outputs/                 # 训练/评估产物与汇总
│   ├── cnn/
│   ├── lstm/
│   ├── transformer/
│   └── summary/
├── src/
│   ├── build_index.py        # 从 THUCNews 原始 txt 生成 thucnews_all.csv
│   ├── make_10c_200k.py       # 采样 10 类 200k 子集并划分 train/val/test
│   ├── data_char.py           # 字符级数据处理
│   ├── models_cnn.py          # 字符级 CNN
│   ├── models_lstm.py         # 字符级 BiLSTM
│   ├── models_transformer.py  # 字符级 Transformer
│   ├── train_cnn_char.py      # 训练 CNN（字符级）
│   ├── train_lstm_char.py     # 训练 LSTM（字符级）
│   ├── train_transformer_char.py # 训练 Transformer（字符级）
│   ├── summarize_results.py   # 汇总测试集指标与混淆矩阵
│   ├── train_baseline.py      # 玩具基线（需自备 toy.csv）
│   └── train_cnn.py           # 玩具 CNN（需自备 toy.csv）
└── requirements.txt
```

## 环境准备

> 建议使用 Python 3.9+，并在虚拟环境中安装依赖。

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

注意：当前 `requirements.txt` 以 UTF-16 编码保存，如果 pip 报编码错误，可先转换再安装：

```bash
python - <<'PY'
from pathlib import Path
raw = Path('requirements.txt').read_text(encoding='utf-16')
Path('requirements-utf8.txt').write_text(raw, encoding='utf-8')
PY
pip install -r requirements-utf8.txt
```

## 数据准备

### 1) 生成全量 CSV（可选）

若你已有 THUCNews 原始 txt 数据集，可先生成 `data/thucnews_all.csv`：

1. 修改 `src/build_index.py` 中的 `SRC_ROOT` 为本地解压目录。
2. 执行：

```bash
python -m src.build_index
```

### 2) 构建 10 类 200k 子集并划分数据集

```bash
python -m src.make_10c_200k
```

产出：

```
data/thucnews_10c_200k/
├── train.csv
├── val.csv
└── test.csv
```

### 3) 配置路径

`src/config.py` 中默认使用 Windows 路径写法（如 `data\thucnews_10c_200k`）。
若在 Linux/macOS 环境，请按需修改为 Unix 风格路径（如 `data/thucnews_10c_200k`）。

## 训练与评估

本项目使用**字符级**建模（CNN/LSTM/Transformer），训练后会自动在测试集上生成报告与混淆矩阵。

```bash
# Transformer
python -m src.train_transformer_char

# CNN
python -m src.train_cnn_char

# LSTM
python -m src.train_lstm_char
```

训练完成后可生成汇总报告：

```bash
python -m src.summarize_results
```

输出路径：

```
outputs/
├── cnn/
│   ├── best.pt
│   ├── test_report.txt
│   └── confusion_matrix.png
├── lstm/
├── transformer/
└── summary/
    ├── metrics.csv
    ├── metrics.md
    ├── confusion_matrix_cnn.png
    ├── confusion_matrix_lstm.png
    └── confusion_matrix_transformer.png
```

## 结果（Test Set）

以下结果来自 `outputs/summary/metrics.md`，供复现实验对比：

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| transformer | 0.9378 | 0.9240 | 0.9379 |
| cnn | 0.9426 | 0.9313 | 0.9424 |
| lstm | 0.9353 | 0.9204 | 0.9352 |

### 混淆矩阵

![transformer confusion matrix](outputs/summary/confusion_matrix_transformer.png)
![cnn confusion matrix](outputs/summary/confusion_matrix_cnn.png)
![lstm confusion matrix](outputs/summary/confusion_matrix_lstm.png)

## 常见问题

- **没有数据集怎么办？**
  - 本项目默认不包含 THUCNews 原始数据，需要自行下载。
  - 若仅想快速跑通流程，可准备一个简单的 `data/toy.csv`，字段为 `text,label`，并运行 `python -m src.train_baseline` 或 `python -m src.train_cnn`。

- **GPU 不可用？**
  - 脚本会自动回退到 CPU，但训练会显著变慢。

## 复现实验建议

1. 固定随机种子（默认已在 `src/config.py` 中设置）。
2. 保持数据集版本一致（10 类、200k 样本、8/1/1 划分）。
3. 使用相同的最大长度、词表大小与训练轮数（见 `src/config.py`）。
