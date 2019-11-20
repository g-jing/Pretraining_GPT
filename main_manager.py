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
import torch
import torch.nn as nn
from torchfly.utils import get_pretrained, init_logging
 
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
import jsonlines
from torchfly.criterions import SequenceCrossEntropyLoss
from distributed_utils import DistributedManager
from utils import parse_args, freeze_model, get_transformer_optim_params
from allennlp.training.checkpointer import Checkpointer
import time
 
try:
 from torch.utils.tensorboard import SummaryWriter
except:
 from tensorboardX import SummaryWriter
 
import tqdm
from tqdm import trange
 
from transformers import (WEIGHTS_NAME, AdamW,
                               GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                               OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer)
 
from transformers import WarmupLinearSchedule
 
 
# using tokenizer and gpt-small from torchfly
from torchfly.transformers import UnifiedTokenizer, GPT2SimpleLM, UnifiedGPT2SmallConfig
 
logger = logging.getLogger(__name__)
 
 
def batch_to_device(batch, device):
   new_batch = {}
   for key, value in batch.items():
       if isinstance(value, list):
           new_batch[key] = [tensor.to(device) for tensor in value]
       else:
           new_batch[key] = value.to(device)
 
   return new_batch
 
class TextDataset(Dataset):
 def __init__(self, tokenizer, args, file_path='train', block_size=1024):
     assert os.path.isfile(file_path)
     directory, filename = os.path.split(file_path)
     cached_features_file = os.path.join(
         directory, "GPTsmall" + '_cached_lm_' + filename)
 
     if os.path.exists(cached_features_file):
         logger.info("Loading features from cached file %s",
                     cached_features_file)
         with open(cached_features_file, 'rb') as handle:
             self.examples = pickle.load(handle)
     else:
         logger.info("Creating features from dataset file at %s", directory)
 
         self.examples = []
 
         # read date
         with jsonlines.open(file_path) as reader:
             for obj in reader:
                 one_ABrole_dialogue = ["A:"+obj[idx]+"\n\n\n" if idx % 2 ==
                     0 else "B:"+obj[idx]+"\n\n\n" for idx in range(len(obj))]
                 # join all utterances in one dialogue
                 one_ABrole_dialogue = "".join(one_ABrole_dialogue)
                 one_ABrole_dialogue = tokenizer.encode(one_ABrole_dialogue)
                 self.examples.append(one_ABrole_dialogue)
 
         logger.info("Saving features into cached file %s",
                     cached_features_file)
         with open(cached_features_file, 'wb') as handle:
             pickle.dump(self.examples, handle,
                         protocol=pickle.HIGHEST_PROTOCOL)
 
         # breakpoint()
 
 def collate(self, inputs):
 
     batch_size = len(inputs)
     # sort by length
     inputs = sorted(inputs, key=len, reverse=True)
 
     batch_len = [len(one) for one in inputs]
 
     batch_input = []
     batch_start_position = []
     for idx in range(batch_size):
         # make random positions
         start_position = random.randint(0, 1024 - batch_len[idx])
         pos = [pos_idx for pos_idx in range(
             start_position, start_position+batch_len[idx])]
         zero_pad = torch.LongTensor([0]*1024)
         padded_inputs = torch.cat([inputs[idx], zero_pad], dim=0)
         padded_inputs = padded_inputs[:1024]
         padded_pos = torch.cat([torch.LongTensor(pos), zero_pad], dim=0)
         padded_pos = padded_pos[:1024]
         batch_input.append(padded_inputs)
         batch_start_position.append(padded_pos)
 
     inputs_tensor = torch.stack(batch_input)
     pos_tensor = torch.stack(batch_start_position)
     len_tensor = torch.LongTensor(batch_len)
     batch = {
         "input_ids": inputs_tensor,
         "position_ids": pos_tensor,
         "length": len_tensor
     }
 
     return batch
 
 def __len__(self):
     return len(self.examples)
 
 def __getitem__(self, item):
     return torch.tensor(self.examples[item])
 
 
def load_and_cache_examples(args, tokenizer, evaluate=False):
 dataset = TextDataset(tokenizer, args,
                       file_path=args.eval_data_file if evaluate else args.train_data_file, block_size=args.block_size)
 return dataset
 
 
def set_seed(args):
 random.seed(args.seed)
 np.random.seed(args.seed)
 torch.manual_seed(args.seed)
 
 torch.backends.cudnn.deterministic = True
 if args.n_gpu > 0:
     torch.cuda.manual_seed_all(args.seed)


def main():
 
   #from config import Config
   #args = Config()
 
   args = parse_args()

   args.checkpoint_dir_constant_time = args.checkpoint_dir + "_constant_time" 
 
   manager = DistributedManager(args)
 
   print("local rank:", args.local_rank)
 
   # define the tokenizer
   tokenizer = UnifiedTokenizer()
 
   train_dataset = load_and_cache_examples(
       args, tokenizer, evaluate=False)
 
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
 
   num_train_optimization_steps = (len(train_dataset) *
                                   args.num_train_epochs //
                                   args.batch_size //
                                   args.gradient_accumulation_steps //
                                   args.n_gpu)
 
   # dialog = dialog_to_tensor(tokenizer, dialog, device)
   optimizer_parameters = get_transformer_optim_params(args, model)
   optimizer = AdamW(optimizer_parameters,
                       lr=args.learning_rate, eps=1e-06)
 
   if args.warmup_steps < 0:
       args.warmup_steps = int(
           args.warmup_ratio * len(train_dataset))
 
   scheduler = WarmupLinearSchedule(optimizer,
                                   warmup_steps=args.warmup_steps,
                                   t_total=num_train_optimization_steps)
 
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
           os.mkdir(args.checkpoint_dir_constant_time )
       checkpointer_constant_time = Checkpointer(
           args.checkpoint_dir_constant_time,
           keep_serialized_model_every_num_seconds=None,
           num_serialized_models_to_keep=-1)
       writer = SummaryWriter()
       start = time.time()
       loss = 0.0

       constant_start = time.time()

 
   model.train()
   criterion = SequenceCrossEntropyLoss()
   for ep in range(args.num_train_epochs):
       pbar = progress_bar(train_dataloader)
 
       for batch in pbar:
           inputs = batch["input_ids"]
           positions  = batch["position_ids"]
           lengths = batch["length"]
           batch_size = inputs.shape[0]
           batch_max_length = torch.max(lengths).item()
           inputs = inputs[:, :batch_max_length]
           positions = positions[:, :batch_max_length]
           mask = torch.arange(batch_max_length).expand(batch_size, batch_max_length)
 
           inputs = inputs.to(args.device)
           positions = positions.to(args.device)
           mask = mask.to(args.device)
          
           outputs = model(inputs, position_ids=positions, mask=mask)
           logit = outputs[0]  # model outputs are always tuple in transformers (see doc)
           logit = logit.contiguous()
           loss = criterion(logit[:, :-1, :], inputs[:, 1:], mask=mask[:, 1:], reduce="batch")
 
 
           manager.backward_loss(loss, model, optimizer)
           update_count += 1
 
           if update_count % args.gradient_accumulation_steps == args.gradient_accumulation_steps - 1:
               manager.clip_grad_norm(model, optimizer)
               optimizer.step()
               scheduler.step()
               optimizer.zero_grad()
 
               #print(update_count)
 
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
 
 
 

