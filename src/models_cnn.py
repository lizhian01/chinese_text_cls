# src/models_cnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, vocab_size: int, num_labels: int, d_model: int = 256,
                 kernel_sizes=(3,4,5), num_filters: int = 128, dropout: float = 0.2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=d_model, out_channels=num_filters, kernel_size=k)
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_labels)

    def forward(self, input_ids, attention_mask=None):
        # input_ids: [B,L]
        x = self.emb(input_ids)          # [B,L,D]
        x = x.transpose(1, 2)            # [B,D,L] for Conv1d

        feats = []
        for conv in self.convs:
            h = F.relu(conv(x))          # [B,F,L-k+1]
            h = F.max_pool1d(h, kernel_size=h.size(2)).squeeze(2)  # [B,F]
            feats.append(h)

        out = torch.cat(feats, dim=1)    # [B, F*K]
        out = self.dropout(out)
        logits = self.fc(out)            # [B,C]
        return logits
