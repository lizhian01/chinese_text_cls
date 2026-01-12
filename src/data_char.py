# src/data_char.py
import os
import re
import pandas as pd
from collections import Counter
from typing import List, Dict, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

from src.utils import save_json

PAD = "[PAD]"
UNK = "[UNK]"
CLS = "[CLS]"

def normalize_text(s: str) -> str:
    s = str(s)
    s = s.replace("\u3000", " ").replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def build_vocab(texts: List[str], vocab_size: int = 50000, min_freq: int = 2) -> Dict[str, int]:
    counter = Counter()
    for t in texts:
        t = normalize_text(t)
        counter.update(list(t))  # char-level

    vocab = {PAD: 0, UNK: 1, CLS: 2}

    for ch, freq in counter.most_common():
        if freq < min_freq:
            break
        if ch in vocab:
            continue
        vocab[ch] = len(vocab)
        if len(vocab) >= vocab_size:
            break
    return vocab

def encode_text(text: str, vocab: Dict[str, int], max_len: int) -> Tuple[List[int], List[int]]:
    text = normalize_text(text)
    ids = [vocab.get(CLS, 2)]
    for ch in list(text):
        ids.append(vocab.get(ch, vocab.get(UNK, 1)))
        if len(ids) >= max_len:
            break

    attn = [1] * len(ids)
    while len(ids) < max_len:
        ids.append(vocab.get(PAD, 0))
        attn.append(0)
    return ids, attn

class TextClsDataset(Dataset):
    def __init__(self, csv_path: str, vocab: Dict[str, int], label2id: Dict[str, int], max_len: int):
        df = pd.read_csv(csv_path)
        assert "text" in df.columns and "label" in df.columns, f"Need columns text,label in {csv_path}"

        self.texts = df["text"].astype(str).tolist()
        self.labels = df["label"].astype(str).tolist()
        self.vocab = vocab
        self.label2id = label2id
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        x_ids, x_attn = encode_text(self.texts[idx], self.vocab, self.max_len)
        y = self.label2id[self.labels[idx]]
        return {
            "input_ids": torch.tensor(x_ids, dtype=torch.long),
            "attention_mask": torch.tensor(x_attn, dtype=torch.long),
            "label": torch.tensor(y, dtype=torch.long),
        }

def build_label_map(labels: List[str]) -> Dict[str, int]:
    uniq = sorted(list(set(labels)))
    return {lab: i for i, lab in enumerate(uniq)}

def make_loaders(
    data_dir: str,
    train_file: str,
    val_file: str,
    test_file: str,
    vocab_path: str,
    label_path: str,
    vocab_size: int,
    min_freq: int,
    max_len: int,
    batch_size: int,
    num_workers: int
):
    train_csv = os.path.join(data_dir, train_file)
    val_csv = os.path.join(data_dir, val_file)
    test_csv = os.path.join(data_dir, test_file)

    # label map from train
    df_train = pd.read_csv(train_csv)
    label2id = build_label_map(df_train["label"].astype(str).tolist())
    save_json(label2id, label_path)

    # vocab from train texts
    vocab = build_vocab(df_train["text"].astype(str).tolist(), vocab_size=vocab_size, min_freq=min_freq)
    save_json(vocab, vocab_path)

    train_ds = TextClsDataset(train_csv, vocab, label2id, max_len)
    val_ds   = TextClsDataset(val_csv, vocab, label2id, max_len)
    test_ds  = TextClsDataset(test_csv, vocab, label2id, max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, vocab, label2id
