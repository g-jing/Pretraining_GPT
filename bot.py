
import pickle
import random


import numpy as np
import torch
import torch.nn as nn


from torchfly.utils import init_logging, get_pretrained_states
from torchfly.utils.model_utils import get_pretrained_states
from torchfly.text.tokenizers import UnifiedBPETokenizer
# using tokenizer and gpt-small from torchfly
from torchfly.modules.transformers import GPT2SimpleLM, UnifiedGPT2SmallConfig
from torchfly.decode import top_filtering


device = torch.device("cuda")
model = model.to(device)

tokenizer = UnifiedBPETokenizer()
ending = [tokenizer.encoder["\n\n\n"]]


model = GPT2SimpleLM(config=UnifiedGPT2SmallConfig)
model.load_state_dict(get_pretrained_states("unified-gpt2-small"))

model_dir = ""
model.load_state_dict(torch.load(model_dir))

past = None
temperature = 0.9
top_k = 0.0
top_p = 0.9

model.eval()
prev_input = None

while True:
    with torch.no_grad():
        # input and update B's utterance
        user = input("A:")

        if user == "quit":
            break
            
        user = tokenizer.encode(user)
        prev_input = user
        prev_input = torch.LongTensor(prev_input).unsqueeze(0).to(device)
        _, past = model(prev_input, past=past)

        prev_input = torch.LongTensor([ending]).to(device)

        sent = []
        for i in range(200):
            logits, past = model(prev_input, past=past)
            logits = logits[:, -1, :] / temperature
            logits = top_filtering(logits, top_p=top_p)

            probs = torch.softmax(logits, dim=-1)

            prev_input = torch.multinomial(probs, num_samples=1)
            prev_word = prev_input.item()

            if prev_word == ending[0]:
                break
            sent.append(prev_word)

        print(tokenizer.decode(sent))
        prev_input = torch.LongTensor([ending]).to(device)
        _, past = model(prev_input, past=past)
