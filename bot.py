
import pickle
import random


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from torchfly.utils import init_logging, get_pretrained_states
from torchfly.utils.model_utils import get_pretrained_states
from torchfly.text.tokenizers import UnifiedBPETokenizer
# using tokenizer and gpt-small from torchfly
from torchfly.modules.transformers import GPT2SimpleLM, UnifiedGPT2SmallConfig

def top_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):

    # batch support!
    if top_k > 0:
        values, _ = torch.topk(logits, top_k)
        min_values = values[:, -1].unsqueeze(1).repeat(1, logits.shape[-1])
        logits = torch.where(logits < min_values, 
                             torch.ones_like(logits, dtype=logits.dtype) * -float('Inf'), 
                             logits)
    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        sorted_logits = sorted_logits.masked_fill_(sorted_indices_to_remove, filter_value)
        logits = torch.zeros_like(logits).scatter(1, sorted_indices, sorted_logits)
    
    return logits


model = GPT2SimpleLM(config=UnifiedGPT2SmallConfig)
model.load_state_dict(get_pretrained_states("unified-gpt2-small"))
device = torch.device("cuda")
model = model.to(device)


tokenizer = UnifiedBPETokenizer()
ending = tokenizer.encode("\n\n\n")

model_dir = "Checkpoint/model_state_epoch_160000.th"
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

        if user == "restart":
            past = None
            user = input("A:")
        if user == "quit":
            break

        user = "A:" + user
        user = tokenizer.encode(user)
        prev_input = user + ending

        prev_input = torch.LongTensor(prev_input).unsqueeze(0).to(device)
        _, past = model(prev_input, past=past)
        #prev_input = torch.LongTensor([ending]).to(device)

        sys = "B:"
        sys = tokenizer.encode(sys)
        prev_input = torch.LongTensor([sys]).to(device)
        sent = []
        for i in range(200):
            # breakpoint()
            logits, past = model(prev_input, past=past)
            logits = logits[:, -1, :] / temperature
            logits = top_filtering(logits, top_p=top_p)

            probs = torch.softmax(logits, dim=-1)

            prev_input = torch.multinomial(probs, num_samples=1)
            prev_word = prev_input.item()

            if prev_word == ending[0]:
                break
            sent.append(prev_word)

        print("B:" + tokenizer.decode(sent))
        prev_input = torch.LongTensor([ending]).to(device)
        _, past = model(prev_input, past=past)
