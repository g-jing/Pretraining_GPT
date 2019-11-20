from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torchfly.transformers import UnifiedTokenizer
from torchfly.transformers import GPT2SimpleLM

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
