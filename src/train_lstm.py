import os
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab=None, max_len=32):
        self.texts = texts
        self.labels = labels
        self.max_len = max_len

        if vocab is None:
            vocab = {"[PAD]": 0, "[UNK]": 1}
            for t in texts:
                for w in t.split():
                    if w not in vocab:
                        vocab[w] = len(vocab)
        self.vocab = vocab

    def encode(self, text):
        ids = [self.vocab.get(w, self.vocab["[UNK]"]) for w in text.split()]
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


class LSTMClassifier(nn.Module):
    """
    Embedding -> BiLSTM -> 取最后层的最后时刻(hidden) -> FC
    """
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_classes, num_layers=1, bidir=True, pad_id=0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.bidir = bidir
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidir
        )
        out_dim = hidden_dim * (2 if bidir else 1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(out_dim, num_classes)

    def forward(self, x):
        # x: [B, L]
        e = self.emb(x)  # [B, L, D]
        _, (h_n, _) = self.lstm(e)
        # h_n: [num_layers * num_directions, B, hidden_dim]
        if self.bidir:
            # 取最后一层的正向与反向 hidden 拼接
            h_forward = h_n[-2]   # [B, H]
            h_backward = h_n[-1]  # [B, H]
            h = torch.cat([h_forward, h_backward], dim=1)  # [B, 2H]
        else:
            h = h_n[-1]  # [B, H]

        h = self.dropout(h)
        logits = self.fc(h)
        return logits


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / max(total, 1)


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

    train_ds = TextDataset(x_train, y_train, max_len=32)
    val_ds = TextDataset(x_val, y_val, vocab=train_ds.vocab, max_len=32)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4)

    model = LSTMClassifier(
        vocab_size=len(train_ds.vocab),
        emb_dim=64,
        hidden_dim=64,
        num_classes=num_classes,
        num_layers=1,
        bidir=True,
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

        val_acc = evaluate(model, val_loader, device)
        print(f"epoch {epoch:02d} | loss {total_loss:.4f} | val_acc {val_acc:.3f}")

    demo = "股市 今日 大涨"
    model.eval()
    with torch.no_grad():
        x = train_ds.encode(demo).unsqueeze(0).to(device)
        pred = model(x).argmax(dim=1).item()
    print("demo:", demo, "=>", le.inverse_transform([pred])[0])


if __name__ == "__main__":
    main()
