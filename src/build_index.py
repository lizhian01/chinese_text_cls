import os
import csv

# ====== 按你的实际路径确认这里 ======
SRC_ROOT = r"E:\datasets\THUCNews"   # 原始解压目录
OUT_CSV = os.path.join("data", "thucnews_all.csv")
# ===================================

def read_txt(fp):
    with open(fp, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().strip()

def main():
    os.makedirs("data", exist_ok=True)
    rows = 0

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as wf:
        writer = csv.writer(wf)
        writer.writerow(["text", "label"])

        for label in sorted(os.listdir(SRC_ROOT)):
            label_dir = os.path.join(SRC_ROOT, label)
            if not os.path.isdir(label_dir):
                continue

            for name in os.listdir(label_dir):
                if not name.endswith(".txt"):
                    continue
                path = os.path.join(label_dir, name)
                text = read_txt(path)
                if not text:
                    continue
                writer.writerow([text, label])
                rows += 1

    print("Saved:", OUT_CSV)
    print("Total rows:", rows)

if __name__ == "__main__":
    main()
