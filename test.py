from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torchfly.transformers import UnifiedTokenizer
from torchfly.transformers import GPT2SimpleLM

import jsonlines

with jsonlines.open('train.jsonl', mode='w') as writer:
    obj = ["hello", "hi", "how are you", "I am good", "do you like movies", "No, I don't"]
    writer.write(obj)

with open("examples.jsonl") as reader:
    example = []
    for obj in reader:
        example.append(obj)

assert len(example) == 1
train_set = [example[0] for i in range(1000)]
eval_set = [example[0] for i in range(1000)]

with jsonlines.open('train.jsonl', mode='w') as writer:
    for obj in train_set:
        writer.write(obj)
        

with jsonlines.open('eval.jsonl', mode='w') as writer:
    for obj in eval_set:
        writer.write(obj)