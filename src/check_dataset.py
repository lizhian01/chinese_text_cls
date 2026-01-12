# src/check_dataset.py
import os
import pandas as pd

DATA_DIR = r"data\thucnews_10c_200k"
FILES = ["train.csv", "val.csv", "test.csv"]

def main():
    print("DATA_DIR =", DATA_DIR)
    for fn in FILES:
        path = os.path.join(DATA_DIR, fn)
        assert os.path.exists(path), f"Missing: {path}"
        df = pd.read_csv(path)
        print(f"\n== {fn} ==")
        print("shape:", df.shape)
        print("columns:", list(df.columns))
        print(df.head(3))

        # 你脚本里通常会用列名 text / label 或 content / label
        # 我们先自动猜一下
        possible_text_cols = [c for c in df.columns if c.lower() in ["text", "content", "sentence", "title"]]
        possible_label_cols = [c for c in df.columns if c.lower() in ["label", "y", "category"]]

        assert len(possible_text_cols) >= 1, "Cannot find text column (expected text/content/...)"
        assert len(possible_label_cols) >= 1, "Cannot find label column (expected label/...)"

        text_col = possible_text_cols[0]
        label_col = possible_label_cols[0]

        # 基本质量检查
        assert df[text_col].isna().sum() == 0, f"{fn}: NaN in text"
        assert df[label_col].isna().sum() == 0, f"{fn}: NaN in label"
        assert df[text_col].astype(str).str.len().min() > 0, f"{fn}: empty text exists"

        # 标签分布
        vc = df[label_col].value_counts().head(15)
        print("\nlabel top counts:\n", vc)

    print("\n✅ Dataset sanity check passed.")

if __name__ == "__main__":
    main()
