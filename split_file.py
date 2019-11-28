original_length = 146790380
#with open(train_dir, "r") as f:
#    text = f.readlines()

import json
import tqdm

import os
os.rename(r"train_ids.jsonl", r"original_train_ids.jsonl")

short_length = 0
with open("original_train_ids.jsonl") as f:
    with open("train_ids.jsonl", mode='w') as fin:
        for i in tqdm.tqdm(f, total=original_length):
            a = json.loads(i)
            if len(a)<800:
                short_length += 1
                fin.write(i)

length = short_length // 8

for idx in tqdm.tqdm(range(8)):
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
