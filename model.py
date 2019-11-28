from utils import parse_args, freeze_model, get_transformer_optim_params, sequence_ce_lm_loss, batch_to_device
from utils import SequenceCrossEntropyLoss
import utils

import numpy as np
import torch
import torch.nn as nn

from torchfly.modules.transformers import UnifiedGPT2SmallConfig, UnifiedGPT2MediumConfig, GPT2SimpleLM
from torchfly.modules.transformers import UnifiedGPT2LargeConfig, UnifiedGPT2XLConfig
from torchfly.utils import get_pretrained_states
from GPTRole import GPT2SimpleLMRole

class HalfARDM(nn.Module):
    def __init__(self, args):
        super(HalfARDM, self).__init__()
        self.args = args

        self.criterion = SequenceCrossEntropyLoss()

        if args.model_size == "small":
            UnifiedGPT2SmallConfig.gradient_checkpointing = True
            self.model = GPT2SimpleLMRole(config=UnifiedGPT2SmallConfig)
            self.model.load_state_dict(get_pretrained_states("unified-gpt2-small"), strict=False)
            
        elif args.model_size == "medium":
            self.UnifiedGPT2MediumConfig.gradient_checkpointing = True
            self.model = GPT2SimpleLMRole(config=UnifiedGPT2MediumConfig)
            self.model.load_state_dict(get_pretrained_states("unified-gpt2-medium-fp16"), strict=False)

        else:
            raise ValueError(args.model_size, " is not correct, use small or medium")  
        
        if args.kl_model_size == "small":
            self.original_model = GPT2SimpleLM(UnifiedGPT2SmallConfig)
            self.original_model.load_state_dict(get_pretrained_states("unified-gpt2-small"), strict=False)
            utils.freeze_model(self.original_model)
        elif args.kl_model_size == "medium":
            self.original_model = GPT2SimpleLM(UnifiedGPT2MediumConfig)
            self.original_model.load_state_dict(get_pretrained_states("unified-gpt2-medium-fp16"), strict=False)
            utils.freeze_model(self.original_model)
        elif args.kl_model_size == "large":
            self.original_model = GPT2SimpleLM(UnifiedGPT2LargeConfig)
            self.original_model.load_state_dict(get_pretrained_states("unified-gpt2-large-fp16"), strict=False)
            utils.freeze_model(self.original_model)
        elif args.kl_model_size == "xlarge":
            self.original_model = GPT2SimpleLM(UnifiedGPT2LargeConfig)
            self.original_model.load_state_dict(get_pretrained_states("unified-gpt2-xl-fp16"), strict=False)
            utils.freeze_model(self.original_model)

        # process "transformers.wre.weight"
        model_state = self.model.state_dict()
        wre_weight = torch.empty(2, self.model.config.n_embd).normal_(mean=0,std=0.01)
        model_state['transformer.wre.weight'] = wre_weight
        self.model.load_state_dict(model_state)


    def forward(self, dialog):
        raise NotImplementedError

    def train_one_step(self, batch):
           
        batch = batch_to_device(batch, self.args.device)
        inputs = batch["input_ids"]
        positions = batch["position_ids"]
        pad_mask = batch["pad_mask"]
        AB_mask = batch["AB_mask"]
        
        batch_size = inputs.shape[0]

        outputs = self.model(inputs, role_id=AB_mask, position_ids=positions, mask=pad_mask)
        # model outputs are always tuple in transformers (see doc)
        logit = outputs[0]

        # change pad_mask to AB_mask
        loss = self.criterion(logit[:, :-1, :].contiguous(), inputs[:, 1:].contiguous(),
                            mask=pad_mask[:, 1:].contiguous().float(), reduce="batch") / self.args.gradient_accumulation_steps 
        
        lm_outputs = self.original_model(inputs, position_ids=positions, mask=pad_mask)
        lm_logits = lm_outputs[0]
        kl = sequence_ce_lm_loss(logits=logit, lm_logits=lm_logits, mask=pad_mask)

        return loss, kl
