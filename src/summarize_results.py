# src/summarize_results.py
import os
import re
import shutil
from pathlib import Path

MODELS = ["transformer", "cnn", "lstm"]

def parse_report(path: str):
    """
    Parse sklearn classification_report txt to extract:
    accuracy, macro avg f1, weighted avg f1
    """
    txt = Path(path).read_text(encoding="utf-8", errors="ignore")

    # accuracy line: "accuracy                          0.9378     20000"
    acc = None
    m = re.search(r"accuracy\s+([0-9.]+)\s+\d+", txt)
    if m:
        acc = float(m.group(1))

    # macro avg line: "macro avg       0.93      0.93      0.93     20000"
    macro_f1 = None
    m = re.search(r"macro avg\s+[0-9.]+\s+[0-9.]+\s+([0-9.]+)\s+\d+", txt)
    if m:
        macro_f1 = float(m.group(1))

    # weighted avg line
    weighted_f1 = None
    m = re.search(r"weighted avg\s+[0-9.]+\s+[0-9.]+\s+([0-9.]+)\s+\d+", txt)
    if m:
        weighted_f1 = float(m.group(1))

    return acc, macro_f1, weighted_f1

def main():
    base_out = Path("outputs")
    summary_dir = base_out / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for model in MODELS:
        report_path = base_out / model / "test_report.txt"
        if not report_path.exists():
            print(f"Missing: {report_path}")
            continue

        acc, macro_f1, weighted_f1 = parse_report(str(report_path))
        rows.append((model, acc, macro_f1, weighted_f1))

        # copy confusion matrix
        cm = base_out / model / "confusion_matrix.png"
        if cm.exists():
            shutil.copy(cm, summary_dir / f"confusion_matrix_{model}.png")

    # write csv
    csv_path = summary_dir / "metrics.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("model,accuracy,macro_f1,weighted_f1\n")
        for model, acc, macro_f1, weighted_f1 in rows:
            f.write(f"{model},{acc},{macro_f1},{weighted_f1}\n")

    # write markdown
    md_path = summary_dir / "metrics.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Model Comparison (Test Set)\n\n")
        f.write("| Model | Accuracy | Macro F1 | Weighted F1 |\n")
        f.write("|---|---:|---:|---:|\n")
        for model, acc, macro_f1, weighted_f1 in rows:
            f.write(f"| {model} | {acc:.4f} | {macro_f1:.4f} | {weighted_f1:.4f} |\n")

        f.write("\n## Confusion Matrices\n\n")
        for model in MODELS:
            img = f"confusion_matrix_{model}.png"
            if (summary_dir / img).exists():
                f.write(f"### {model}\n\n")
                f.write(f"![{model}]({img})\n\n")

    print("✅ Saved:", csv_path)
    print("✅ Saved:", md_path)
    print("✅ Confusion matrices copied to:", summary_dir)

if __name__ == "__main__":
    main()
