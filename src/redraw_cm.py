# src/redraw_cm.py
import argparse
import os
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd

from src.config import CFG
from src.data_char import TextClsDataset
from src.utils import load_json
from src.models_transformer import TransformerClassifier
from src.models_cnn import TextCNN
from src.models_lstm import BiLSTMClassifier


# ---------- 中文字体设置 ----------
def setup_chinese_font():
    candidates = [
        "Microsoft YaHei",  # Windows
        "SimHei",
        "SimSun",
        "NSimSun",
        "PingFang SC",      # macOS
        "Noto Sans CJK SC", # Linux
        "WenQuanYi Micro Hei"
    ]
    for f in candidates:
        if f in [x.name for x in matplotlib.font_manager.fontManager.ttflist]:
            plt.rcParams["font.sans-serif"] = [f]
            break
    plt.rcParams["axes.unicode_minus"] = False


def plot_confusion_matrix(cm, labels, save_path):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel("True")
    plt.xlabel("Pred")
    plt.savefig(save_path, dpi=200)
    plt.close()


def load_model(model_name, vocab_size, num_labels, device, ckpt_path):
    if model_name == "transformer":
        model = TransformerClassifier(
            vocab_size=vocab_size,
            num_labels=num_labels,
            max_len=CFG.max_len,
            d_model=CFG.d_model,
            nhead=CFG.nhead,
            num_layers=CFG.num_layers,
            dim_ff=CFG.dim_ff,
            dropout=CFG.dropout
        )
    elif model_name == "cnn":
        model = TextCNN(
            vocab_size=vocab_size,
            num_labels=num_labels,
            d_model=CFG.d_model
        )
    elif model_name == "lstm":
        model = BiLSTMClassifier(
            vocab_size=vocab_size,
            num_labels=num_labels,
            d_model=CFG.d_model
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["transformer", "cnn", "lstm"], required=True)
    args = parser.parse_args()

    model_name = args.model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name == "transformer":
        out_dir = CFG.out_dir
    elif model_name == "cnn":
        out_dir = CFG.out_dir_cnn
    else:
        out_dir = CFG.out_dir_lstm

    print(f"Redrawing confusion matrix for [{model_name}]")
    print("Using device:", device)

    vocab = load_json(os.path.join(out_dir, "vocab.json"))
    label2id = load_json(os.path.join(out_dir, "label2id.json"))
    id2label = {v: k for k, v in label2id.items()}

    test_csv = os.path.join(CFG.data_dir, CFG.test_file)
    test_ds = TextClsDataset(
        csv_path=test_csv,
        vocab=vocab,
        label2id=label2id,
        max_len=CFG.max_len
    )

    loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=64,
        shuffle=False
    )

    model = load_model(
        model_name=model_name,
        vocab_size=len(vocab),
        num_labels=len(label2id),
        device=device,
        ckpt_path=os.path.join(out_dir, "best.pt")
    )

    y_true, y_pred = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

    y_true_names = [id2label[i] for i in y_true]
    y_pred_names = [id2label[i] for i in y_pred]
    labels_sorted = [id2label[i] for i in range(len(id2label))]

    cm = confusion_matrix(y_true_names, y_pred_names, labels=labels_sorted)

    setup_chinese_font()
    save_path = os.path.join(out_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, labels_sorted, save_path)

    print(f"✅ Saved new confusion matrix with Chinese labels to:\n{save_path}")


if __name__ == "__main__":
    main()
