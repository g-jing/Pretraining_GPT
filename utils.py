import argparse
import regex as re
import numpy as np
import torch
import torch.nn as nn
import random
import torch.jit
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
import tqdm
import json
from torch.nn.utils.rnn import pad_sequence

pad_index = 1

class TextDataset(Dataset):

    def __init__(self, args, manager, file_path):
        assert os.path.isfile(file_path)
        print("Loading features from %s",
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
        # if self.args.loss_type == "all":
        AB_mask = []
        flag = True  
        AB_mask.append(flag)
        
        for i in range(1, len(example)):
            AB_mask.append(flag)

            if example[i] == self.ending[1]:
                if example[i - 1] == self.ending[0]:
                    flag = not flag
            
        return torch.LongTensor(self.dataset[item]), torch.LongTensor(AB_mask)


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



def batch_to_device(batch, device):
    new_batch = {}
    for key, value in batch.items():
        if isinstance(value, list):
            new_batch[key] = [tensor.to(device) for tensor in value]
        else:
            new_batch[key] = value.to(device)

    return new_batch

def sequence_ce_lm_loss(
    logits: torch.FloatTensor,
    lm_logits: torch.FloatTensor,
    mask: torch.FloatTensor
):
    """
    Sequence Cross Entropy with Language Model KL
    """

    # shape : (batch, sequence_length, num_classes)
    log_probs = torch.log_softmax(logits, dim=-1)
    lm_probs = torch.softmax(lm_logits, dim=-1)

    # ignored mask and normalized by length
    lm_kl = (
        torch.kl_div(input=log_probs, target=lm_probs, reduction=2) /
        log_probs.shape[1]
    )

    return lm_kl


def parse_args(argv=None):
    """
    Parse the command line arguments
    """
    parser = argparse.ArgumentParser(description="")

    # add all arguments
    parser.add_argument(
        "--kl_model_size",
        default="small",
        type=str,       
        help="You could choose small or medium, large, or xlarge"
    )
    parser.add_argument(
        "--model_size",
        default="small",
        type=str,       
        help="You could choose small or medium"
    )
    parser.add_argument(
        "--train_data_file",
        default="train_ids.jsonl",
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
        default=5e-5,
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
    # last utterances loss or all losses
    parser.add_argument(
        "--loss_type",
        default="all",
        type=str,       
        help="The loss type, last or all" 
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


class SequenceCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, logits, targets, mask, label_smoothing=-1, reduce=None):
        """
        reduce: None, "batch", "sentence"
        """
        return sequence_cross_entropy_with_logits(logits, targets, mask, label_smoothing, reduce)


def sequence_cross_entropy_with_logits(logits, targets, mask, label_smoothing, reduce):
    # type: (Tensor, Tensor, Tensor, float, bool)-> Tensor
    """
    label_smoothing : ``float``, optional (default = 0.0)
        It should be smaller than 1.
    """
    # shape : (batch * sequence_length, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # shape : (batch * sequence_length, num_classes)
    log_probs_flat = F.log_softmax(logits_flat, dim=-1)
    # shape : (batch * max_len, 1)
    targets_flat = targets.view(-1, 1).long()

    if label_smoothing > 0.0:
        num_classes = logits.size(-1)
        smoothing_value = label_smoothing / float(num_classes)
        # Fill all the correct indices with 1 - smoothing value.
        one_hot_targets = torch.zeros_like(log_probs_flat).scatter_(-1, targets_flat, 1.0 - label_smoothing)
        smoothed_targets = one_hot_targets + smoothing_value
        negative_log_likelihood_flat = -log_probs_flat * smoothed_targets
        negative_log_likelihood_flat = negative_log_likelihood_flat.sum(-1, keepdim=True)
    else:
        # shape : (batch * sequence_length, 1)
        negative_log_likelihood_flat = - torch.gather(log_probs_flat, dim=1, index=targets_flat)
                                       
    # shape : (batch, sequence_length)
    negative_log_likelihood = negative_log_likelihood_flat.view(-1, logits.shape[1])
    
    # shape : (batch, sequence_length)
    loss = negative_log_likelihood * mask

    if reduce:
        # shape : (batch,)
        loss = loss.sum(1) / (mask.sum(1) + 1e-13)
        
        if reduce is "batch":
            # shape : scalar
            loss = loss.mean()

    return loss