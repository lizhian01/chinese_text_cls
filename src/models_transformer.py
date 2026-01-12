# src/models_transformer.py
import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)  # [L, D]
        position = torch.arange(0, max_len).unsqueeze(1)  # [L,1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even
        pe[:, 1::2] = torch.cos(position * div_term)  # odd
        pe = pe.unsqueeze(0)  # [1,L,D]
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: [B, L, D]
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size: int, num_labels: int,
                 max_len: int, d_model: int, nhead: int, num_layers: int,
                 dim_ff: int, dropout: float):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, num_labels)

    def forward(self, input_ids, attention_mask):
        # input_ids: [B,L], attention_mask: [B,L] (1=valid,0=pad)
        x = self.emb(input_ids)           # [B,L,D]
        x = self.pos(x)

        # src_key_padding_mask: True 表示 PAD 位置
        key_padding_mask = (attention_mask == 0)  # [B,L] bool
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)  # [B,L,D]

        # 取 [CLS] token 位置向量做分类
        cls_vec = x[:, 0, :]              # [B,D]
        cls_vec = self.norm(cls_vec)
        logits = self.fc(cls_vec)         # [B,C]
        return logits
    def forward_with_attn(self, input_ids, attention_mask):
 
        x = self.emb(input_ids)
        x = self.pos(x)

        key_padding_mask = attention_mask == 0

        attn_weights = None
        for i, layer in enumerate(self.encoder.layers):
            x = layer.self_attn(
                x, x, x,
                key_padding_mask=key_padding_mask,
                need_weights=True,
                average_attn_weights=False
            )[0]
            x = layer(x, src_key_padding_mask=key_padding_mask)
            if i == len(self.encoder.layers) - 1:
                # 取最后一层
                attn_weights = layer.self_attn(
                    x, x, x,
                    key_padding_mask=key_padding_mask,
                    need_weights=True,
                    average_attn_weights=False
                )[1]

        pooled = x.mean(dim=1)
        logits = self.fc(pooled)
        return logits, attn_weights
