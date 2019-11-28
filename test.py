from transformers import GPT2Tokenizer, GPT2LMHeadModel
# model = GPT2LMHeadModel.from_pretrained("gpt")

import json
import tqdm

import os
os.rename(r"temp_in.jsonl", r"temp_out.jsonl")

with open("original_train_ids.jsonl") as f:
    with open("train_ids.jsonl", mode='w') as fin:
        for i in f:
            a = json.loads(i)
            if len(a)<800:
                fin.write(i)








import gdown

filepath = "/home/jinggu/united-medium"
url = "https://drive.google.com/file/d/1QAqXk5LLaDbccrqfwL6wZYjZ4UHdpxMr/view?usp=sharing"
gdown.cached_download(url, filepath, quiet=False)
breakpoint()

loc = "unified-gpt2-medium.pth"
import torch
#torch.save(model.state_dict(), loc)
torch.load(loc)
#torch.load("/home/jinggu/.cache/torchfly/models/unified-gpt2-medium.pth")



exit(0)

from torchfly.transformers import UnifiedTokenizer
from torchfly.transformers import GPT2SimpleLM

with open("GPTsmall_cached_lm_train.jsonl", "rb") as f:
    import pickle
    example = pickle.load(f)
    breakpoint() 


import jsonlines
import json
import torch
a = torch.distributed.get_world_size()
print(a)
with open("../train.jsonl") as reader:
    example = []
    i = 0
    for obj in reader:
        i += 1
        example.append(json.loads(obj))
        if i==5:
            break
breakpoint()
