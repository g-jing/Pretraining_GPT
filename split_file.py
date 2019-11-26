train_dir = "small_reddit.jsonl"
length = 50000 // 8
with open(train_dir, "r") as f:
    text = f.readlines()

for idx in range(8):
    split_file_dir = f"train_ids_{idx}.jsonl"
    with open(split_file_dir, "w") as fin:
        start = length*idx
        end = length*(idx+1)
        fin.writelines(text[start:end])

for idx in range(8):
    split_file_dir = f"train_ids_{idx}.jsonl"
    with open(split_file_dir, "r") as f:
        text = f.readlines()
        assert len(text) == length
