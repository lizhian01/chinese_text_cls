import os
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ---------------------------
# 1) 数据集
# ---------------------------
class ToyTextDataset(Dataset):
    def __init__(self, texts, labels, vocab=None, max_len=32):
        self.texts = texts
        self.labels = labels
        self.max_len = max_len

        # 构建词表（空格分词，toy 数据足够）
        if vocab is None:
            vocab = {"[PAD]": 0, "[UNK]": 1}
            for t in texts:
                for w in t.split():
                    if w not in vocab:
                        vocab[w] = len(vocab)
        self.vocab = vocab

    def encode(self, text):
        ids = []
        for w in text.split():
            ids.append(self.vocab.get(w, self.vocab["[UNK]"]))
        # padding/truncation
        ids = ids[: self.max_len]
        if len(ids) < self.max_len:
            ids += [self.vocab["[PAD]"]] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        x = self.encode(self.texts[idx])
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


# ---------------------------
# 2) 最简单的 baseline：Embedding + 平均池化 + 线性分类
#    (先跑通训练闭环，后面替换成 CNN/LSTM/Transformer)
# ---------------------------
class MeanPoolClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, num_classes, pad_id=0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.fc = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        # x: [B, L]
        e = self.emb(x)               # [B, L, D]
        m = e.mean(dim=1)             # [B, D]
        logits = self.fc(m)           # [B, C]
        return logits


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    df = pd.read_csv(os.path.join("data", "toy.csv"))
    texts = df["text"].tolist()
    labels_str = df["label"].tolist()

    le = LabelEncoder()
    labels = le.fit_transform(labels_str)
    num_classes = len(le.classes_)
    print("classes:", list(le.classes_))

    x_train, x_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.3, random_state=42
    )

    train_ds = ToyTextDataset(x_train, y_train, max_len=32)
    val_ds = ToyTextDataset(x_val, y_val, vocab=train_ds.vocab, max_len=32)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4)

    model = MeanPoolClassifier(
        vocab_size=len(train_ds.vocab),
        emb_dim=64,
        num_classes=num_classes,
        pad_id=train_ds.vocab["[PAD]"]
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=2e-3)

    for epoch in range(1, 11):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optim.step()
            total_loss += loss.item()

        # eval
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.numel()

        acc = correct / max(total, 1)
        print(f"epoch {epoch:02d} | loss {total_loss:.4f} | val_acc {acc:.3f}")

    # 预测演示
    demo = "股市 今日 大涨"
    model.eval()
    with torch.no_grad():
        x = train_ds.encode(demo).unsqueeze(0).to(device)
        pred = model(x).argmax(dim=1).item()
    print("demo:", demo, "=>", le.inverse_transform([pred])[0])


if __name__ == "__main__":
    main()
