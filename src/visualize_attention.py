# src/visualize_attention.py
import os
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt

from src.config import CFG
from src.models_transformer import TransformerClassifier
from src.data_char import TextClsDataset
from src.utils import load_json


def setup_chinese_font():
    """Fix Chinese glyphs in matplotlib on Windows/mac/Linux."""
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


def main():
    setup_chinese_font()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    out_dir = CFG.out_dir  # outputs\transformer
    vocab = load_json(os.path.join(out_dir, "vocab.json"))         # token -> id
    label2id = load_json(os.path.join(out_dir, "label2id.json"))   # label -> id
    id2label = {v: k for k, v in label2id.items()}
    id2tok = {int(v): k for k, v in vocab.items()}                 # id -> token

    # ---------- load model ----------
    model = TransformerClassifier(
        vocab_size=len(vocab),
        num_labels=len(label2id),
        max_len=CFG.max_len,
        d_model=CFG.d_model,
        nhead=CFG.nhead,
        num_layers=CFG.num_layers,
        dim_ff=CFG.dim_ff,
        dropout=CFG.dropout
    ).to(device)

    # 你自己的 best.pt，安全提示可忽略；未来也可改 weights_only=True
    ckpt = torch.load(os.path.join(out_dir, "best.pt"), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # ---------- pick one test sample ----------
    ds = TextClsDataset(
        csv_path=os.path.join(CFG.data_dir, CFG.test_file),
        vocab=vocab,
        label2id=label2id,
        max_len=CFG.max_len
    )

    idx = 0  # 你可以改成别的数字，比如 10、100
    sample = ds[idx]
    input_ids = sample["input_ids"].unsqueeze(0).to(device)         # [1, L]
    attention_mask = sample["attention_mask"].unsqueeze(0).to(device)  # [1, L]
    true_label = id2label[int(sample["label"])]

    # ---------- get attention (via forward hook) ----------
    # 说明：不要求模型实现 forward_with_attn，直接抓最后一层 self_attn 的权重
    last_attn = {"w": None}

    def hook_self_attn(module, inp, out):
        """
        nn.MultiheadAttention forward returns:
        (attn_output, attn_output_weights)
        attn_output_weights shape:
        - average_attn_weights=False: [batch, heads, tgt_len, src_len]
        - average_attn_weights=True:  [batch, tgt_len, src_len]
        """
        if isinstance(out, tuple) and len(out) >= 2:
            last_attn["w"] = out[1]

    # 尝试定位最后一层 encoder layer 的 self-attn
    # 你的 TransformerClassifier 里通常会有 model.encoder.layers[-1].self_attn
    handle = None
    if hasattr(model, "encoder") and hasattr(model.encoder, "layers"):
        handle = model.encoder.layers[-1].self_attn.register_forward_hook(hook_self_attn)
    else:
        raise RuntimeError("Cannot find model.encoder.layers[-1].self_attn, please check models_transformer.py")

        # ---------- forward + manually compute last-layer attention ----------
    with torch.no_grad():
        # 1) 先把输入做 embedding + positional（跟你模型一致）
        x = model.emb(input_ids)              # [1, L, D]
        x = model.pos(x) if hasattr(model, "pos") else x

        # padding mask: True 表示要被 mask
        key_padding_mask = (attention_mask == 0)  # [1, L] bool

        # 2) 逐层跑 encoder，最后一层单独取注意力权重
        num_layers = len(model.encoder.layers)
        attn_w = None

        for li, layer in enumerate(model.encoder.layers):
            if li == num_layers - 1:
                # 最后一层：手动调用 self_attn 来拿权重
                # MultiheadAttention 期望输入形状通常是 (L, N, E)，但如果 batch_first=True 则是 (N, L, E)
                # 你的训练代码里很可能是 batch_first=True（因为你数据是 [B,L,D]）
                # 我们这里优先按 batch_first=True 走；如果报 shape 错，我再给你一行切换方案。
                attn_out, attn_w = layer.self_attn(
                    x, x, x,
                    key_padding_mask=key_padding_mask,
                    need_weights=True,
                    average_attn_weights=False
                )
                # 注意：这里只是取权重，不替代原 layer forward
                x = layer(x, src_key_padding_mask=key_padding_mask)
            else:
                x = layer(x, src_key_padding_mask=key_padding_mask)

        # 3) 分类头（跟你训练时一致：mean pooling）
        pooled = x.mean(dim=1)
        logits = model.fc(pooled) if hasattr(model, "fc") else model.classifier(pooled)
        pred = torch.argmax(logits, dim=1).item()

    pred_label = id2label[pred]
    attn = attn_w  # tensor


    # ---------- prepare tokens ----------
    ids = input_ids[0].detach().cpu().numpy().tolist()
    mask = attention_mask[0].detach().cpu().numpy().tolist()

    tokens = []
    for t, m in zip(ids, mask):
        if m == 0:
            break
        tokens.append(id2tok.get(int(t), "[UNK]"))

    # 只展示前 N 个 token，避免挤爆
    N = min(80, len(tokens))
    tokens = tokens[:N]

    # ---------- normalize / aggregate attn ----------
    # attn -> numpy
    if isinstance(attn, torch.Tensor):
        attn = attn.detach().cpu().numpy()

    # 可能形状：
    # [1, heads, L, L] 或 [1, L, L]
    if attn.ndim == 4:
        attn2d = attn[0].mean(axis=0)  # average heads -> [L, L]
    elif attn.ndim == 3:
        attn2d = attn[0]              # [L, L]
    else:
        raise RuntimeError(f"Unexpected attn shape: {attn.shape}")

    attn2d = attn2d[:N, :N]

    # ---------- CLS -> token attention (best for explanation) ----------
    cls_attn = attn2d[0].astype(np.float64)  # [N]
    cls_attn = cls_attn / (cls_attn.sum() + 1e-9)

    topk = 12
    top_idx = np.argsort(-cls_attn)[:topk]
    print(f"\nSample idx={idx}  true={true_label}  pred={pred_label}")
    print("Top attention tokens (CLS -> token):")
    for j in top_idx:
        print(f"{j:02d}  {tokens[j]}  {cls_attn[j]:.4f}")

    # ---------- 1) Bar plot ----------
    plt.figure(figsize=(12, 4))
    plt.title(f"CLS Attention (true={true_label}, pred={pred_label})")
    x = np.arange(N)
    plt.bar(x, cls_attn)
    plt.xticks(x, tokens, rotation=90)
    plt.tight_layout()
    save_bar = os.path.join(out_dir, "attention_cls_bar.png")
    plt.savefig(save_bar, dpi=200)
    print("✅ Saved:", save_bar)
    plt.show()

    # ---------- 2) Heatmap (CLS row only, clearer than full NxN) ----------
    plt.figure(figsize=(12, 2.2))
    plt.title(f"CLS→Tokens Attention Heatmap (true={true_label}, pred={pred_label})")
    plt.imshow(cls_attn.reshape(1, -1), aspect="auto", cmap="viridis")
    plt.yticks([0], ["[CLS]"])
    plt.xticks(np.arange(N), tokens, rotation=90)
    plt.colorbar()
    plt.tight_layout()
    save_heat = os.path.join(out_dir, "attention_cls_heat.png")
    plt.savefig(save_heat, dpi=200)
    print("✅ Saved:", save_heat)
    plt.show()


if __name__ == "__main__":
    main()
