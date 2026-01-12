import os
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
SRC_CSV = os.path.join("data", "thucnews_all.csv")
OUT_DIR = os.path.join("data", "thucnews_10c_200k")
TARGET_N = 200_000

# 选 10 类（与你之前设定一致）
KEEP = {"科技","股票","体育","娱乐","时政","社会","教育","财经","家居","游戏"}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 只读两列，减少内存压力
    df = pd.read_csv(SRC_CSV, usecols=["text", "label"])
    df["label"] = df["label"].astype(str)

    df = df[df["label"].isin(KEEP)].reset_index(drop=True)
    print("after keep-10c:", len(df))
    print("label counts (top):")
    print(df["label"].value_counts().head(20))

    if len(df) < TARGET_N:
        raise ValueError(f"Not enough data after filtering: {len(df)} < {TARGET_N}")

    # 抽样到 200k（随机但可复现）
    df = df.sample(n=TARGET_N, random_state=SEED).reset_index(drop=True)

    # 分层划分：8/1/1
    train_df, temp_df = train_test_split(
        df, test_size=0.2, random_state=SEED, stratify=df["label"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=SEED, stratify=temp_df["label"]
    )

    train_df.to_csv(os.path.join(OUT_DIR, "train.csv"), index=False, encoding="utf-8")
    val_df.to_csv(os.path.join(OUT_DIR, "val.csv"), index=False, encoding="utf-8")
    test_df.to_csv(os.path.join(OUT_DIR, "test.csv"), index=False, encoding="utf-8")

    print("Saved to:", OUT_DIR)
    print("Sizes:", len(train_df), len(val_df), len(test_df))
    print("Labels:", sorted(train_df["label"].unique().tolist()))

if __name__ == "__main__":
    main()
