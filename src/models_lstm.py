# src/models_lstm.py
import torch
import torch.nn as nn

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, num_labels: int, d_model: int = 256,
                 hidden_size: int = 256, num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_labels)

    def forward(self, input_ids, attention_mask):
        # input_ids: [B,L], attention_mask: [B,L] (1 valid, 0 pad)
        x = self.emb(input_ids)  # [B,L,D]
        out, _ = self.lstm(x)    # [B,L,2H]

        # mask mean pooling over valid tokens
        mask = attention_mask.unsqueeze(-1).float()     # [B,L,1]
        out = out * mask
        denom = mask.sum(dim=1).clamp(min=1.0)          # [B,1]
        pooled = out.sum(dim=1) / denom                 # [B,2H]

        pooled = self.dropout(pooled)
        logits = self.fc(pooled)
        return logits
