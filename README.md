# 基于 Transformer 的中文文本分类与可解释性分析
（CNN / BiLSTM / Transformer 对比）

## 1. 项目简介

本项目以中文新闻文本为研究对象，基于 THUCNews 数据集构建 10 类文本分类任务，分别实现并对比了 CNN、BiLSTM 与 Transformer 三种深度学习模型在中文文本分类场景下的性能表现。

在保证训练数据与评估流程一致的前提下，系统分析了不同模型在整体准确率、类别稳定性以及语义判别能力方面的差异，并进一步通过 Attention 可视化手段，对 Transformer 模型的预测依据进行了可解释性分析。

## 2. 实验环境

- 操作系统：Windows 11
- CPU：Intel / AMD 通用平台
- GPU：NVIDIA GeForce GTX 1650 Ti
- 显存：4GB
- 内存：16GB

### 软件环境

- Python 3.12
- PyTorch（CUDA 版本）
- numpy / pandas / scikit-learn
- matplotlib

推荐使用 `venv` 虚拟环境运行。

### 环境搭建

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

## 3. 项目结构说明

```text
chinese_text_cls/
├── data/
│   └── thucnews_10c_200k/        # 处理后的数据
├── outputs/
│   ├── transformer/
│   │   ├── best.pt               # 最优模型权重
│   │   ├── test_report.txt
│   │   ├── confusion_matrix.png
│   │   ├── attention_cls_bar.png
│   │   └── attention_cls_heat.png
│   ├── cnn/
│   └── lstm/
├── src/
│   ├── make_10c_200k.py          # 数据构建
│   ├── train_transformer_char.py
│   ├── train_cnn_char.py
│   ├── train_lstm_char.py
│   ├── visualize_attention.py
│   └── utils.py
├── .venv/                        # 虚拟环境
├── requirements.txt
└── README.md
```

## 4. 数据集说明

原始数据：THUCNews 中文新闻数据集。

选取类别（10 类）：

- 体育、娱乐、家居、教育、时政
- 游戏、社会、科技、股票、财经

数据经清洗、去噪与类别筛选后，构建规模为：

| 数据集 | 样本数 |
|---|---:|
| Train | 160,000 |
| Validation | 20,000 |
| Test | 20,000 |

### 数据构建

1. 生成全量 CSV（可选）：
   - 修改 `src/build_index.py` 中的 `SRC_ROOT` 为本地解压目录。
   - 运行：

```bash
python -m src.build_index
```

2. 生成 10 类 200k 子集并划分数据集：

```bash
python -m src.make_10c_200k
```

3. 路径配置：
   - `src/config.py` 中默认使用 Windows 路径写法（如 `data\thucnews_10c_200k`）。
   - Linux/macOS 环境需改为 Unix 风格路径（如 `data/thucnews_10c_200k`）。

## 5. 模型设计

### 5.1 CNN 文本分类模型

- Embedding + 多通道卷积
- MaxPooling + 全连接分类
- 优点：结构简单、训练速度快

### 5.2 BiLSTM 文本分类模型

- 双向 LSTM 编码序列语义
- Mean Pooling 汇总特征
- 优点：擅长捕捉时序语义

### 5.3 Transformer 文本分类模型

- 基于多层 Transformer Encoder
- 自注意力机制捕捉全局语义
- 使用 [CLS] 表示进行分类

## 6. 实验结果与对比分析

### 6.1 整体性能

在测试集上的实验结果表明，三种模型在整体准确率上均取得了较高性能，其中 Transformer 略优。

- Transformer：≈ 93.7%
- CNN：≈ 93.3%
- BiLSTM：≈ 93.1%

Transformer 在宏平均 F1 值与类别稳定性方面表现更好，尤其在语义相近类别之间具备更强的判别能力。

下表为 `outputs/summary/metrics.md` 中的测试集指标：

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| transformer | 0.9378 | 0.9240 | 0.9379 |
| cnn | 0.9426 | 0.9313 | 0.9424 |
| lstm | 0.9353 | 0.9204 | 0.9352 |

### 6.2 混淆矩阵分析

从混淆矩阵可以看出，模型主要的误分类集中在语义相近类别之间（如“财经”与“股票”、“社会”与“时政”），整体呈现明显的对角线分布，说明模型已较好地学习到类别区分特征。

## 7. Attention 可解释性分析（Transformer）

为提升模型的可解释性，对 Transformer 最后一层 Encoder 的自注意力权重进行了可视化，重点分析 [CLS] token 对输入文本各 token 的注意力分布。

实验结果表明，模型在进行分类决策时，并非对全文均匀关注，而是将注意力集中在少数与类别语义高度相关的关键词上。例如在“体育”新闻中，模型显著关注与比赛、球队或运动事件相关的词汇位置，并最终给出正确预测。

该结果从可解释性角度验证了 Transformer 模型决策的合理性。

## 8. 运行方式（示例）

```bash
# 训练 Transformer
python -m src.train_transformer_char

# 训练 CNN
python -m src.train_cnn_char

# 训练 BiLSTM
python -m src.train_lstm_char

# Attention 可视化
python -m src.visualize_attention
```

训练完成后可生成汇总报告：

```bash
python -m src.summarize_results
```

## 9. 总结

本项目系统性地完成了中文文本分类任务中不同深度学习模型的实现、对比与分析，并通过 Attention 可视化进一步增强了模型的可解释性。实验结果表明，Transformer 在综合性能与语义建模能力方面具备一定优势，适合作为中文文本分类任务的主流模型方案。
