# src/train_lstm_char.py
import os
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

from src.config import CFG
from src.utils import set_seed, ensure_dir
from src.data_char import make_loaders
from src.models_lstm import BiLSTMClassifier

def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    losses = []
    ce = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            logits = model(input_ids, attention_mask)
            loss = ce(logits, labels)
            losses.append(loss.item())

            pred = torch.argmax(logits, dim=1)
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(pred.cpu().numpy().tolist())

    acc = accuracy_score(y_true, y_pred)
    return float(np.mean(losses)), float(acc), np.array(y_true), np.array(y_pred)

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

def main():
    set_seed(CFG.seed)
    out_dir = CFG.out_dir_lstm
    ensure_dir(out_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    vocab_path = os.path.join(out_dir, "vocab.json")
    label_path = os.path.join(out_dir, "label2id.json")

    train_loader, val_loader, test_loader, vocab, label2id = make_loaders(
        data_dir=CFG.data_dir,
        train_file=CFG.train_file,
        val_file=CFG.val_file,
        test_file=CFG.test_file,
        vocab_path=vocab_path,
        label_path=label_path,
        vocab_size=CFG.vocab_size,
        min_freq=CFG.min_freq,
        max_len=CFG.max_len,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers
    )

    id2label = {v: k for k, v in label2id.items()}
    num_labels = len(label2id)

    model = BiLSTMClassifier(
        vocab_size=len(vocab),
        num_labels=num_labels,
        d_model=CFG.d_model,
        hidden_size=256,
        num_layers=1,
        dropout=0.2
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    ce = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_path = os.path.join(out_dir, "best.pt")

    for epoch in range(1, CFG.epochs + 1):
        model.train()
        t0 = time.time()
        losses = []

        for step, batch in enumerate(train_loader, start=1):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids, attention_mask)
            loss = ce(logits, labels)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            if step % 200 == 0:
                print(f"epoch {epoch} step {step} / loss {np.mean(losses):.4f}")

        train_loss = float(np.mean(losses))
        val_loss, val_acc, _, _ = evaluate(model, val_loader, device)
        dt = time.time() - t0
        print(f"\nEpoch {epoch} done. time={dt:.1f}s  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model": model.state_dict(), "cfg": CFG.__dict__}, best_path)
            print(f"✅ best saved: {best_path} (val_acc={best_val_acc:.4f})\n")

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, device)
    print(f"\nTEST: loss={test_loss:.4f} acc={test_acc:.4f}")

    y_true_names = [id2label[int(i)] for i in y_true]
    y_pred_names = [id2label[int(i)] for i in y_pred]
    labels_sorted = [id2label[i] for i in range(num_labels)]

    report = classification_report(y_true_names, y_pred_names, labels=labels_sorted, digits=4)
    report_path = os.path.join(out_dir, "test_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Saved report: {report_path}")

    cm = confusion_matrix(y_true_names, y_pred_names, labels=labels_sorted)
    cm_path = os.path.join(out_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, labels_sorted, cm_path)
    print(f"Saved confusion matrix: {cm_path}")

if __name__ == "__main__":
    main()
