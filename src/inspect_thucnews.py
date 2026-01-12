from datasets import load_dataset

def main():
    ds = load_dataset("SirlyDreamer/THUCNews", split="train")
    print("rows:", len(ds))
    print("columns:", ds.column_names)
    print("sample[0]:", ds[0])

if __name__ == "__main__":
    main()
