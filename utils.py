import argparse
import regex as re
import numpy as np
import torch
import torch.nn as nn
import random


def parse_args(argv=None):
    """
    Parse the command line arguments
    """
    parser = argparse.ArgumentParser(description="")

    # add all arguments
    parser.add_argument(
        "--train_data_file",
        default="train.jsonl",
        type=str,       
        help="The input training data file (a text file). It should be a jsonlines file"
    )

    parser.add_argument(
        "--block_size", 
        default=1024, 
        type=int,                        
        help="Optional input sequence length after tokenization."
    )
    parser.add_argument(
        "--learning_rate",
        default=1e-5,
        type=float,
        help="The initial learning rate for AdamW."
    )
    parser.add_argument(
        "--batch_size", default=12, type=int, help="Set the batch size"
    )
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=16,
        help="this is calculated by 10000/n_gpu/batch_size  Number of updates steps to accumulate before performing a backward/update pass."
    )
    parser.add_argument(
        "--num_train_epochs",
        default=6,
        type=int,
        help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--early_stop_num_train_epochs",
        default=6,
        type=int,
        help="early stop epoches."
    )
    parser.add_argument(
        "--constant_save_time",
        default=3600 *4,
        type=int,
        help="time interval to save extra model"
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    # warmup settings
    parser.add_argument(
        "--warmup_steps",
        default=50000,
        type=int,
        help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--warmup_ratio",
        default=1/24,
        type=float,
        help="Ratio of warmup steps in terms of the training set"
    )
    # logging
    parser.add_argument(
        '--logging_steps',
        type=int,
        default=50,
        help="Log every X updates steps."
    )
    parser.add_argument(
        '--save_steps',
        type=int,
        default=160000,
        help="Save checkpoint every X updates steps."
    )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default="Checkpoint",
        help="Set checkpoint directory."
    )
    # fp 16 training
    parser.add_argument(
        '--fp16',
        action='store_true',
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex)"
    )
    parser.add_argument(
        '--fp16_opt_level',
        type=str,
        default='O1',
        help=
        "For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
    )
    # distributed training
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank"
    )

    # process all arguments
    args = parser.parse_args()
    return args


def freeze_model(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = False


def get_transformer_optim_params(args, model: nn.Module):
    param_optimizer = model.named_parameters()

    no_decay = ["bias", "ln", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params":
                [
                    p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)
                ],
            "weight_decay": 0.01,
        },
        {
            "params":
                [
                    p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)
                ],
            "weight_decay": 0.0,
        },
    ]

    return optimizer_grouped_parameters