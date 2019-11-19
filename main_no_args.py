"""
This python script is used to finetune GPT
"""
 
from __future__ import absolute_import, division, print_function
 
import glob
import logging
import os
import pickle
import random
import re
import shutil
 
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
import jsonlines
from torchfly.criterions import SequenceCrossEntropyLoss
 
os.environ["CUDA_VISIBLE_DEVICES"]="4"
try:
   from torch.utils.tensorboard import SummaryWriter
except:
   from tensorboardX import SummaryWriter
 
from tqdm import tqdm, trange
 
from transformers import (WEIGHTS_NAME, AdamW,
                                 GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                                 OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer)
 
from transformers import WarmupLinearSchedule
 
 
# using tokenizer and gpt-small from torchfly
from torchfly.transformers import UnifiedTokenizer, GPT2SimpleLM, UnifiedGPT2SmallConfig
 
logger = logging.getLogger(__name__)
 
 
class TextDataset(Dataset):
   def __init__(self, tokenizer, configuration, file_path='train', block_size=1024):
       assert os.path.isfile(file_path)
       directory, filename = os.path.split(file_path)
       cached_features_file = os.path.join(directory, "GPTsmall" + '_cached_lm_' + str(block_size) + '_' + filename)
 
       if os.path.exists(cached_features_file):
           logger.info("Loading features from cached file %s", cached_features_file)
           with open(cached_features_file, 'rb') as handle:
               self.examples = pickle.load(handle)
       else:
           logger.info("Creating features from dataset file at %s", directory)
 
           self.examples = []
          
           # read date
           with jsonlines.open(file_path) as reader:
               for obj in reader:
                   one_ABrole_dialogue = ["A:"+obj[idx]+"\n\n\n" if idx%2==0 else "B:"+obj[idx]+"\n\n\n" for idx in range(len(obj))]
                  
                   one_ABrole_dialogue = "".join(one_ABrole_dialogue)  # join all utterances in one dialogue
                   one_ABrole_dialogue = tokenizer.encode(one_ABrole_dialogue)
                   self.examples.append(one_ABrole_dialogue)
 
           #breakpoint()
                  
 
   def collate(self, inputs):
 
       batch_size = len(inputs)
       # sort by length
       inputs = sorted(inputs, key=len, reverse=True)
 
       # total_len = sum([len(item) for item in inputs[0]])
       batch_len = [len(one) for one in inputs]
 
       batch_input = []
       batch_start_position = []
       for idx in range(batch_size):
           # make random positions
          
           start_position = random.randint(0, 1024 - batch_len[idx])
 
           pos = [pos_idx for pos_idx in range(start_position, start_position+batch_len[idx])]
  
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
 
 
           #for i in range(0, len(tokenized_text)-block_size+1, block_size): # Truncate in block of block_size
           #    self.examples.append(tokenizer.build_inputs_with_special_tokens(tokenized_text[i:i+block_size]))
           # Note that we are loosing the last truncated example here for the sake of simplicity (no padding)
           # If your dataset is small, first you should look for a bigger one :-) and second you
           # can change this behavior by adding (model specific) padding.
 
           #logger.info("Saving features into cached file %s", cached_features_file)
           #with open(cached_features_file, 'wb') as handle:
           #    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
 
   def __len__(self):
       return len(self.examples)
 
   def __getitem__(self, item):
       return torch.tensor(self.examples[item])
 
 
def load_and_cache_examples(configuration, tokenizer, evaluate=False):
   dataset = TextDataset(tokenizer, configuration, file_path=configuration.eval_data_file if evaluate else configuration.train_data_file, block_size=configuration.block_size)
   return dataset
 
 
def set_seed(configuration):
   random.seed(configuration.seed)
   np.random.seed(configuration.seed)
   torch.manual_seed(configuration.seed)
   if configuration.n_gpu > 0:
       torch.cuda.manual_seed_all(configuration.seed)
 
 
def _rotate_checkpoints(configuration, checkpoint_prefix, use_mtime=False):
   if not configuration.save_total_limit:
       return
   if configuration.save_total_limit <= 0:
       return
 
   # Check if we should delete older checkpoint(s)
   glob_checkpoints = glob.glob(os.path.join(configuration.output_dir, '{}-*'.format(checkpoint_prefix)))
   if len(glob_checkpoints) <= configuration.save_total_limit:
       return
 
   ordering_and_checkpoint_path = []
   for path in glob_checkpoints:
       if use_mtime:
           ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
       else:
           regex_match = re.match('.*{}-([0-9]+)'.format(checkpoint_prefix), path)
           if regex_match and regex_match.groups():
               ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))
 
   checkpoints_sorted = sorted(ordering_and_checkpoint_path)
   checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
   number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - configuration.save_total_limit)
   checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
   for checkpoint in checkpoints_to_be_deleted:
       logger.info("Deleting older checkpoint [{}] due to configuration.save_total_limit".format(checkpoint))
       shutil.rmtree(checkpoint)
 
def train(configuration, train_dataset, model, tokenizer):
   """ Train the model """
   if configuration.local_rank in [-1, 0]:
       tb_writer = SummaryWriter()
 
   configuration.train_batch_size = configuration.per_gpu_train_batch_size * max(1, configuration.n_gpu)
   train_sampler = RandomSampler(train_dataset) if configuration.local_rank == -1 else DistributedSampler(train_dataset)
   train_dataloader = DataLoader(train_dataset, collate_fn=train_dataset.collate, sampler=train_sampler, batch_size=configuration.train_batch_size)
 
   if configuration.max_steps > 0:
       t_total = configuration.max_steps
       configuration.num_train_epochs = configuration.max_steps // (len(train_dataloader) // configuration.gradient_accumulation_steps) + 1
   else:
       t_total = len(train_dataloader) // configuration.gradient_accumulation_steps * configuration.num_train_epochs
 
   # Prepare optimizer and schedule (linear warmup and decay)
   no_decay = ['bias', 'LayerNorm.weight']
   optimizer_grouped_parameters = [
       {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': configuration.weight_decay},
       {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
       ]
   optimizer = AdamW(optimizer_grouped_parameters, lr=configuration.learning_rate, eps=configuration.adam_epsilon)
   scheduler = WarmupLinearSchedule(optimizer, warmup_steps=configuration.warmup_steps, t_total=t_total)
   if configuration.fp16:
       try:
           from apex import amp
       except ImportError:
           raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
       model, optimizer = amp.initialize(model, optimizer, opt_level=configuration.fp16_opt_level)
 
   # multi-gpu training (should be after apex fp16 initialization)
   if configuration.n_gpu > 1:
       model = torch.nn.DataParallel(model)
 
   # Distributed training (should be after apex fp16 initialization)
   if configuration.local_rank != -1:
       model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[configuration.local_rank],
                                                         output_device=configuration.local_rank,
                                                         find_unused_parameters=True)
 
   # Train!
   logger.info("***** Running training *****")
   logger.info("  Num examples = %d", len(train_dataset))
   logger.info("  Num Epochs = %d", configuration.num_train_epochs)
   logger.info("  Instantaneous batch size per GPU = %d", configuration.per_gpu_train_batch_size)
   logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                  configuration.train_batch_size * configuration.gradient_accumulation_steps * (torch.distributed.get_world_size() if configuration.local_rank != -1 else 1))
   logger.info("  Gradient Accumulation steps = %d", configuration.gradient_accumulation_steps)
   logger.info("  Total optimization steps = %d", t_total)
 
   global_step = 0
   tr_loss, logging_loss = 0.0, 0.0
   model.zero_grad()
   train_iterator = trange(int(configuration.num_train_epochs), desc="Epoch", disable=configuration.local_rank not in [-1, 0])
   set_seed(configuration)  # Added here for reproducibility (even between python 2 and 3)
 
   criterion = SequenceCrossEntropyLoss()
   for _ in train_iterator:
       epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=configuration.local_rank not in [-1, 0])
 
       for step, batch in enumerate(epoch_iterator):
 
           inputs = batch["input_ids"]
           positions  = batch["position_ids"]
           lengths = batch["length"]
 
           batch_max_length = torch.max(lengths).item()
 
           inputs = inputs[:, :batch_max_length]
           positions = positions[:, :batch_max_length]
 
           mask = torch.arange(batch_max_length).expand(configuration.train_batch_size, batch_max_length) < lengths.unsqueeze(1)
 
           inputs = inputs.to(configuration.device)
           positions = positions.to(configuration.device)
           mask = mask.to(configuration.device)
           #labels = labels.to(configuration.device)
           model.train()
 
           outputs = model(inputs, position_ids=positions, mask=mask)
 
           logit = outputs[0]  # model outputs are always tuple in transformers (see doc)
 
           logit = logit.contiguous()
           loss = criterion(logit[:, :-1, :], inputs[:, 1:], mask=mask[:, 1:], reduce="batch")
                      
           if configuration.n_gpu > 1:
               loss = loss.mean()  # mean() to average on multi-gpu parallel training
           if configuration.gradient_accumulation_steps > 1:
               loss = loss / configuration.gradient_accumulation_steps
 
           if configuration.fp16:
               with amp.scale_loss(loss, optimizer) as scaled_loss:
                   scaled_loss.backward()
           else:
 
               loss.backward()
 
           tr_loss += loss.item()
           if (step + 1) % configuration.gradient_accumulation_steps == 0:
               if configuration.fp16:
                   torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), configuration.max_grad_norm)
               else:
                   torch.nn.utils.clip_grad_norm_(model.parameters(), configuration.max_grad_norm)
               optimizer.step()
               scheduler.step()  # Update learning rate schedule
               model.zero_grad()
               global_step += 1
 
               if configuration.local_rank in [-1, 0] and configuration.logging_steps > 0 and global_step % configuration.logging_steps == 0:
                   # Log metrics
                   if configuration.local_rank == -1 and configuration.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                       results = evaluate(configuration, model, tokenizer)
                       for key, value in results.items():
                           tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                   tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                   tb_writer.add_scalar('loss', (tr_loss - logging_loss)/configuration.logging_steps, global_step)
                   logging_loss = tr_loss
 
               if configuration.local_rank in [-1, 0] and configuration.save_steps > 0 and global_step % configuration.save_steps == 0:
                   checkpoint_prefix = 'checkpoint'
                   # Save model checkpoint
                   output_dir = os.path.join(configuration.output_dir, '{}-{}'.format(checkpoint_prefix, global_step))
                   if not os.path.exists(output_dir):
                       os.makedirs(output_dir)
                   model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                  
                   # TODO find a better save way.
                   #model_to_save.save_pretrained(output_dir)
                   torch.save(configuration, os.path.join(output_dir, 'training_configuration.bin'))
                   logger.info("Saving model checkpoint to %s", output_dir)
 
                   _rotate_checkpoints(configuration, checkpoint_prefix)
 
           if configuration.max_steps > 0 and global_step > configuration.max_steps:
               epoch_iterator.close()
               break
       if configuration.max_steps > 0 and global_step > configuration.max_steps:
           train_iterator.close()
           break
 
   if configuration.local_rank in [-1, 0]:
       tb_writer.close()
 
   return global_step, tr_loss / global_step
 
 
def evaluate(configuration, model, tokenizer, prefix=""):
   # Loop to handle MNLI double evaluation (matched, mis-matched)
   eval_output_dir = configuration.output_dir
 
   eval_dataset = load_and_cache_examples(configuration, tokenizer, evaluate=True)
 
   if not os.path.exists(eval_output_dir) and configuration.local_rank in [-1, 0]:
       os.makedirs(eval_output_dir)
 
   configuration.eval_batch_size = configuration.per_gpu_eval_batch_size * max(1, configuration.n_gpu)
   # Note that DistributedSampler samples randomly
   eval_sampler = SequentialSampler(eval_dataset) if configuration.local_rank == -1 else DistributedSampler(eval_dataset)
   eval_dataloader = DataLoader(eval_dataset, collate_fn=eval_dataset.collate, sampler=eval_sampler, batch_size=configuration.eval_batch_size)
 
   # multi-gpu evaluate
   if configuration.n_gpu > 1:
       model = torch.nn.DataParallel(model)
 
   # Eval!
   logger.info("***** Running evaluation {} *****".format(prefix))
   logger.info("  Num examples = %d", len(eval_dataset))
   logger.info("  Batch size = %d", configuration.eval_batch_size)
   eval_loss = 0.0
   nb_eval_steps = 0
   model.eval()
 
   criterion = SequenceCrossEntropyLoss()
   for batch in tqdm(eval_dataloader, desc="Evaluating"):
       inputs = batch["input_ids"]
       positions  = batch["position_ids"]
       lengths = batch["length"]
 
       batch_max_length = torch.max(lengths).item()
 
       inputs = inputs[:, :batch_max_length]
       positions = positions[:, :batch_max_length]
 
       mask = torch.arange(batch_max_length).expand(configuration.train_batch_size, batch_max_length) < lengths.unsqueeze(1)
 
       inputs = inputs.to(configuration.device)
       positions = positions.to(configuration.device)
       mask = mask.to(configuration.device)
       #labels = labels.to(configuration.device)
 
       with torch.no_grad():
           outputs = model(inputs, position_ids=positions, mask=mask)
           logit = outputs[0]  # model outputs are always tuple in transformers (see doc)
 
           logit = logit.contiguous()
           loss = criterion(logit[:, :-1, :], inputs[:, 1:], mask=mask[:, 1:], reduce="batch")
 
           eval_loss += loss.mean().item()
       nb_eval_steps += 1
 
   eval_loss = eval_loss / nb_eval_steps
   perplexity = torch.exp(torch.tensor(eval_loss))
 
   result = {
       "perplexity": perplexity
   }
 
   output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
   with open(output_eval_file, "w") as writer:
       logger.info("***** Eval results {} *****".format(prefix))
       for key in sorted(result.keys()):
           logger.info("  %s = %s", key, str(result[key]))
           writer.write("%s = %s\n" % (key, str(result[key])))
 
   return result
 
 
def main():
 
   from config import Config
   configuration = Config()
 
   if configuration.eval_data_file is None and configuration.do_eval:
       raise ValueError("Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
                        "or remove the --do_eval argument.")
 
   if os.path.exists(configuration.output_dir) and os.listdir(configuration.output_dir) and configuration.do_train and not configuration.overwrite_output_dir:
      
       raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(configuration.output_dir))
 
   # Setup distant debugging if needed
   if configuration.server_ip and configuration.server_port:
       # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
       import ptvsd
       print("Waiting for debugger attach")
       ptvsd.enable_attach(address=(configuration.server_ip, configuration.server_port), redirect_output=True)
       ptvsd.wait_for_attach()
 
   # Setup CUDA, GPU & distributed training
   if configuration.local_rank == -1 or configuration.no_cuda:
       device = torch.device("cuda" if torch.cuda.is_available() and not configuration.no_cuda else "cpu")
       configuration.n_gpu = torch.cuda.device_count()
   else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
       torch.cuda.set_device(configuration.local_rank)
       device = torch.device("cuda", configuration.local_rank)
       torch.distributed.init_process_group(backend='nccl')
       configuration.n_gpu = 1
   configuration.device = device
 
   # Setup logging
   logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                       datefmt = '%m/%d/%Y %H:%M:%S',
                       level = logging.INFO if configuration.local_rank in [-1, 0] else logging.WARN)
   logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   configuration.local_rank, device, configuration.n_gpu, bool(configuration.local_rank != -1), configuration.fp16)
 
   # Set seed
   set_seed(configuration)
 
   # Load pretrained model and tokenizer
   if configuration.local_rank not in [-1, 0]:
       torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab
 
   # torchfly
   tokenizer = UnifiedTokenizer()
   model = GPT2SimpleLM(config=UnifiedGPT2SmallConfig)
 
   if configuration.block_size <= 0:
       configuration.block_size = 1024 # tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
   configuration.block_size = min(configuration.block_size, 1024) # tokenizer.max_len_single_sentence
 
   model.to(configuration.device)
 
   if configuration.local_rank == 0:
       torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab
 
   logger.info("Training/evaluation parameters %s", configuration)
 
   # Training
   if configuration.do_train:
       if configuration.local_rank not in [-1, 0]:
           torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache
 
       train_dataset = load_and_cache_examples(configuration, tokenizer, evaluate=False)
 
       if configuration.local_rank == 0:
           torch.distributed.barrier()
 
       global_step, tr_loss = train(configuration, train_dataset, model, tokenizer)
       logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
 
 
   # Saving best-practices: if you use save_pretrained for the model and tokenizer, you can reload them using from_pretrained()
   if configuration.do_train and (configuration.local_rank == -1 or torch.distributed.get_rank() == 0):
       # Create output directory if needed
       if not os.path.exists(configuration.output_dir) and configuration.local_rank in [-1, 0]:
           os.makedirs(configuration.output_dir)
 
       logger.info("Saving model checkpoint to %s", configuration.output_dir)
       # Save a trained model, configuration and tokenizer using `save_pretrained()`.
       # They can then be reloaded using `from_pretrained()`
       model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
       #model_to_save.save_pretrained(configuration.output_dir)
       #tokenizer.save_pretrained(configuration.output_dir)
 
       # Good practice: save your training arguments together with the trained model
       torch.save(configuration, os.path.join(configuration.output_dir, 'training_configuration.bin'))
 
       # Load a trained model and vocabulary that you have fine-tuned
       model = model_class.from_pretrained(configuration.output_dir)
       tokenizer = tokenizer
       model.to(configuration.device)
 
 
   # Evaluation
   results = {}
   if configuration.do_eval and configuration.local_rank in [-1, 0]:
       checkpoints = [configuration.output_dir]
       if configuration.eval_all_checkpoints:
           checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(configuration.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
           logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
       logger.info("Evaluate the following checkpoints: %s", checkpoints)
       for checkpoint in checkpoints:
           global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
           prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""
          
           # model = model_class.from_pretrained(checkpoint)
           model.to(configuration.device)
           result = evaluate(configuration, model, tokenizer, prefix=prefix)
           result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
           results.update(result)
 
   return results
 
 
if __name__ == "__main__":
   main()