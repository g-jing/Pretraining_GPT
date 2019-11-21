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

from torchfly.utils import init_logging
from torchfly.modules.losses import SequenceCrossEntropyLoss
from torchfly.utils.model_utils import get_pretrained_states
from torchfly.text.tokenizers import UnifiedBPETokenizer

from distributed_utils import DistributedManager
from utils import parse_args, freeze_model, get_transformer_optim_params

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from transformers import AdamW
from transformers import WarmupLinearSchedule

# using tokenizer and gpt-small from torchfly
from torchfly.modules.transformers import GPT2SimpleLM, UnifiedGPT2SmallConfig
from torchfly.text.tokenizers import UnifiedBPETokenizer
from torchfly.utils import get_pretrained_states

init_logging()
logger = logging.getLogger(__name__)

pad_index = 1

def batch_to_device(batch, device):
    new_batch = {}
    for key, value in batch.items():
        if isinstance(value, list):
            new_batch[key] = [tensor.to(device) for tensor in value]
        else:
            new_batch[key] = value.to(device)

    return new_batch


class TextDataset(Dataset):
    def __init__(self, file_path):
        assert os.path.isfile(file_path)
        logger.info("Loading features from %s",
                    file_path)
        self.dataset = []
        with open(file_path, 'r') as handle:
            for one in handle:
                self.dataset.append(json.loads(one))

    def collate(self, inputs):

        batch_size = len(inputs)
        # sort by length
        inputs = sorted(inputs, key=len, reverse=True)

        batch_len = [len(one) for one in inputs]
        max_len = max(batch_len)

        batch_input = []
        batch_start_position = []
        for idx in range(batch_size):
            # make random positions
            start_position = random.randint(0, 1024 - batch_len[idx])
            pos = [pos_idx for pos_idx in range(
                start_position, start_position+batch_len[idx])]

            pad_tail = torch.LongTensor([pad_index]*max_len)
            # pad input to max_len
            padded_inputs = torch.cat([inputs[idx], pad_tail], dim=0)
            padded_inputs = padded_inputs[:max_len]
            # pad position to max_len
            padded_pos = torch.cat([torch.LongTensor(pos), pad_tail], dim=0)
            padded_pos = padded_pos[:max_len]

            # append
            batch_input.append(padded_inputs)
            batch_start_position.append(padded_pos)

        inputs_tensor = torch.stack(batch_input)
        pos_tensor = torch.stack(batch_start_position)

        batch = {
            "input_ids": inputs_tensor,
            "position_ids": pos_tensor,
        }

        return batch

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return torch.tensor(self.dataset[item])


def load_dataset(args):
    dataset = TextDataset(file_path=args.train_data_file)
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

    # define the tokenizer
    tokenizer = UnifiedBPETokenizer()

    train_dataset = load_dataset(args)

    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)

    train_dataloader = DataLoader(dataset=train_dataset,
                                  sampler=train_sampler,
                                  batch_size=args.batch_size,
                                  collate_fn=train_dataset.collate)

    # define the model

    model = GPT2SimpleLM(config=UnifiedGPT2SmallConfig)
    model.load_state_dict(get_pretrained_states("unified-gpt2-small"))

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
                                     warmup_steps=args.warmup_steps,
                                     t_total=total_optimization_step)

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
    for ep in range(args.early_stop_num_train_epochs):
        pbar = progress_bar(train_dataloader)

        for batch in pbar:
            inputs = batch["input_ids"]
            positions = batch["position_ids"]
            batch_size = inputs.shape[0]

            mask = inputs != pad_index

            inputs = inputs.to(args.device)
            positions = positions.to(args.device)
            mask = mask.to(args.device)

            outputs = model(inputs, position_ids=positions, mask=mask)
            # model outputs are always tuple in transformers (see doc)
            logit = outputs[0]
            loss = criterion(logit[:, :-1, :].contiguous(), inputs[:, 1:].contiguous(),
                             mask=mask[:, 1:].contiguous().float(), reduce="batch")

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
                    pbar.set_postfix(loss=loss,
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