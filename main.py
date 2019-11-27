"""
This python script is used to finetune GPT
"""

from __future__ import absolute_import, division, print_function

import os
import glob
import logging
import pickle
import random
import re
import shutil
import tqdm
from tqdm import trange
import jsonlines
import time
import json

from allennlp.training.checkpointer import Checkpointer
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence

from torchfly.utils import init_logging
# from torchfly.modules.losses import SequenceCrossEntropyLoss
from utils import SequenceCrossEntropyLoss
# using tokenizer and gpt-small from torchfly
from torchfly.modules.transformers import GPT2SimpleLM, UnifiedGPT2SmallConfig, UnifiedGPT2MediumConfig
from torchfly.text.tokenizers import UnifiedBPETokenizer
from torchfly.utils import get_pretrained_states

from distributed_utils import DistributedManager
from utils import parse_args, freeze_model, get_transformer_optim_params

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup


init_logging()
logger = logging.getLogger(__name__)

pad_index = 1


# tokenizer.encode("\n\n\n") [50140, 50118]
# tokenizer.encode("A:")
# [250, 35]
# tokenizer.encode("B:")
# [387, 35]


def batch_to_device(batch, device):
    new_batch = {}
    for key, value in batch.items():
        if isinstance(value, list):
            new_batch[key] = [tensor.to(device) for tensor in value]
        else:
            new_batch[key] = value.to(device)

    return new_batch


class TextDataset(Dataset):

    def __init__(self, args, manager, file_path):
        assert os.path.isfile(file_path)
        logger.info("Loading features from %s",
                    file_path)
        
        self.args = args
        self.dataset = []

        with open(file_path, 'r') as handle:
            total_dialog = 146790395 // 8 # only for full reddit data

            if manager.is_main_rank():
                for one in tqdm.tqdm(handle, total=total_dialog):
                    self.dataset.append(json.loads(one))
            else:
                for one in handle:
                    self.dataset.append(json.loads(one))

        self.ending = [50140, 50118]

    def collate(self, inputs):

        inputs, AB_mask =  zip(*inputs)

        batch_size = len(inputs)
        # sort by length
        inputs = sorted(inputs, key=len, reverse=True)

        batch_len = [len(one) for one in inputs]

        batch_input = []
        batch_start_position = []
        for idx in range(batch_size):
            # make random positions
            start_position = random.randint(0, 1024 - batch_len[idx])
            pos = [pos_idx for pos_idx in range(start_position, start_position+batch_len[idx])]
            batch_start_position.append(torch.LongTensor(pos))
        
        inputs_tensor = pad_sequence(inputs, batch_first=True, padding_value=pad_index)
        pos_tensor = pad_sequence(batch_start_position, batch_first=True, padding_value=0)

        pad_mask = inputs_tensor != pad_index

        #AB_mask = get_AB_mask(inputs_tensor)
        AB_mask = pad_sequence(AB_mask, batch_first=True, padding_value=0)

        batch = {
            "input_ids": inputs_tensor,
            "position_ids": pos_tensor,
            "pad_mask": pad_mask,
            "AB_mask": AB_mask
        }

        return batch

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        
        example = self.dataset[item]

        # mask B

        if self.args.loss_type == "all":
            AB_mask = []
            if random.random()>0.1:
                flag = True
            else:
                flag = False
                
            AB_mask.append(flag)
            
            for i in range(1, len(example)):
                AB_mask.append(flag)

                if example[i] == self.ending[1]:
                    if example[i - 1] == self.ending[0]:
                        flag = not flag
        elif self.args.loss_type == "last":
            AB_mask = [False] * (len(example) - 2) + [True] * 2

            for i in range(len(example)-3, 0, -1):
                if example[i] == self.ending[1]:
                    if example[i - 1] == self.ending[0]:
                        break
                AB_mask[i] = True
        else:
            raise ValueError(self.args.loss_type, " is not correct, use all or last")     
            
        return torch.LongTensor(self.dataset[item]), torch.FloatTensor(AB_mask)


def load_dataset(args):
    train_file = f"train_ids_{args.local_rank}.jsonl"
    dataset = TextDataset(args, args.manager, file_path=train_file)
    return dataset


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def main():

    args = parse_args()
    args.checkpoint_dir_constant_time = args.checkpoint_dir + "_constant_time"
    manager = DistributedManager(args)
    args.manager = manager

    # define the tokenizer
    tokenizer = UnifiedBPETokenizer()

    train_dataset = load_dataset(args)

    torch.distributed.barrier()

    if manager.is_main_rank():
        print("Load Finished")

    # if args.local_rank == -1:
    #     train_sampler = RandomSampler(train_dataset)
    # else:
    #     train_sampler = DistributedSampler(train_dataset)

    train_sampler = RandomSampler(train_dataset)

    train_dataloader = DataLoader(dataset=train_dataset,
                                  sampler=train_sampler,
                                  batch_size=args.batch_size,
                                  collate_fn=train_dataset.collate)

    # define the model

    if args.model_size == "small":
        UnifiedGPT2SmallConfig.gradient_checkpointing = True
        model = GPT2SimpleLM(config=UnifiedGPT2SmallConfig)
        model.load_state_dict(get_pretrained_states("unified-gpt2-small-fp16"), strict=False)

        original_model = GPT2SimpleLM(UnifiedGPT2SmallConfig)
        original_model.load_state_dict(get_pretrained_states("unified-gpt2-small-fp16"), strict=False)
        print(original_model)
        utils.freeze_model(original_model)
        print(original_model)
        exit(0)

    elif args.model_size == "medium":
        UnifiedGPT2MediumConfig.gradient_checkpointing = True
        model = GPT2SimpleLM(config=UnifiedGPT2MediumConfig)
        model.load_state_dict(get_pretrained_states("unified-gpt2-medium-fp16"), strict=False)

        original_model = GPT2SimpleLM(config=UnifiedGPT2MediumConfig)
        original_model.load_state_dict(get_pretrained_states("unified-gpt2-medium-fp16"), strict=False)
        print(original_model)
        utils.freeze_model(original_model)
        print(original_model)
        exit(0)

    else:
        raise ValueError(args.model_size, " is not correct, use small or medium")    

    total_optimization_step = (len(train_dataset) *
                              args.num_train_epochs // args.batch_size //
                              args.gradient_accumulation_steps //
                              args.n_gpu)

    optimizer_parameters = get_transformer_optim_params(args, model)
    optimizer = AdamW(optimizer_parameters,
                      lr=args.learning_rate, eps=1e-06)

    if args.warmup_steps < 0:
        args.warmup_steps = int(
            args.warmup_ratio * total_optimization_step)

    scheduler = WarmupLinearSchedule(optimizer,
                                     num_warmup_steps=args.warmup_steps,
                                     num_training_steps=total_optimization_step)
    manager.init_training(model, optimizer)

    update_count = 0
    if manager.is_main_rank():
        progress_bar = tqdm.tqdm
    else:
        progress_bar = iter

    if manager.is_main_rank():

        if not os.path.isdir(args.checkpoint_dir):
            os.mkdir(args.checkpoint_dir)
        checkpointer = Checkpointer(
            args.checkpoint_dir,
            keep_serialized_model_every_num_seconds=None,
            num_serialized_models_to_keep=10)

        if not os.path.isdir(args.checkpoint_dir_constant_time):
            os.mkdir(args.checkpoint_dir_constant_time)
        checkpointer_constant_time = Checkpointer(
            args.checkpoint_dir_constant_time,
            keep_serialized_model_every_num_seconds=None,
            num_serialized_models_to_keep=-1)

        writer = SummaryWriter()
        start = time.time()
        constant_start = time.time()

    model.train()
    criterion = SequenceCrossEntropyLoss()
    for ep in range(args.num_train_epochs):
        pbar = progress_bar(train_dataloader)
        for batch in pbar:
            
            batch = batch_to_device(batch, args.device)
            inputs = batch["input_ids"]
            positions = batch["position_ids"]
            pad_mask = batch["pad_mask"]
            AB_mask = batch["AB_mask"]
            
            batch_size = inputs.shape[0]

            outputs = model(inputs, position_ids=positions, mask=pad_mask)
            # model outputs are always tuple in transformers (see doc)
            logit = outputs[0]

            # change pad_mask to AB_mask
            loss = criterion(logit[:, :-1, :].contiguous(), inputs[:, 1:].contiguous(),
                             mask=AB_mask[:, 1:].contiguous().float(), reduce="batch") / args.gradient_accumulation_steps 

            manager.backward_loss(loss, model, optimizer)
            update_count += 1

            if update_count % args.gradient_accumulation_steps == args.gradient_accumulation_steps - 1:
                manager.clip_grad_norm(model, optimizer)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # timer
                if manager.is_main_rank():
                    end = time.time()
                    speed = args.batch_size * args.n_gpu * args.gradient_accumulation_steps / (
                        end - start)
                    start = end
                    # show progress
                    pbar.set_postfix(loss=loss.item(),
                                     speed=speed)

            # post-processing
            if manager.is_main_rank():
                if update_count % args.logging_steps == 0:
                    writer.add_scalar('loss', loss.item(), update_count)
                    #writer.add_scalar('loss', update_count)

                # saving models

                if update_count % args.save_steps == 0:
                    checkpointer.save_checkpoint(update_count,
                                                 model.state_dict(),
                                                 optimizer.state_dict(),
                                                 is_best_so_far=True)

                if time.time() - constant_start > args.constant_save_time:
                    constant_start = time.time()
                    checkpointer_constant_time.save_checkpoint(update_count,
                                                               model.state_dict(),
                                                               optimizer.state_dict(),
                                                               is_best_so_far=True)


if __name__ == "__main__":
    main()