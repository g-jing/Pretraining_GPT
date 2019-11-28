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
# using tokenizer and gpt-small from torchfly
from torchfly.modules.transformers import UnifiedGPT2SmallConfig, UnifiedGPT2MediumConfig
from torchfly.text.tokenizers import UnifiedBPETokenizer
from torchfly.utils import get_pretrained_states

from distributed_utils import DistributedManager
from utils import parse_args, freeze_model, get_transformer_optim_params, sequence_ce_lm_loss
from utils import SequenceCrossEntropyLoss, TextDataset, load_dataset, set_seed
import utils

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

from model import HalfARDM


init_logging()
logger = logging.getLogger(__name__)

pad_index = 1

# tokenizer.encode("\n\n\n") [50140, 50118]
# tokenizer.encode("A:")
# [250, 35]
# tokenizer.encode("B:")
# [387, 35]


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
    model = HalfARDM(args)
 
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

    scheduler = get_linear_schedule_with_warmup(optimizer,
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
    
    for ep in range(args.num_train_epochs):
        pbar = progress_bar(train_dataloader)
        for batch in pbar:
            
            loss, kl = model.train_one_step(batch)
            total_loss = loss + 0.01 * kl

            manager.backward_loss(total_loss, model, optimizer)
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
                                    kl=kl.item(),
                                     speed=speed)

            # post-processing
            if manager.is_main_rank():
                if update_count % args.logging_steps == 0:
                    writer.add_scalar('loss', loss.item(), update_count)
                    writer.add_scalar('kl', kl.item(), update_count)
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