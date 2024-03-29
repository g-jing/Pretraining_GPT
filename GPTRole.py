
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import math
from torchfly.modules.transformers.gpt_model import Conv1D, Attention, MLP, Block, GPT2LMHead
# from ...utils.file_utils import gdrive_download
# from ..cuda import gpt_gelu as gelu
# assert installed
from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm
# from cudatest import GPT_GELU
# pylint:disable=no-member


@torch.jit.script
def gelu(x):
    """ GELU Activation Function
        math.sqrt(2 / math.pi) = 0.7978845608028654
    """
    return 0.5 * x * (
        1 + torch.tanh(0.7978845608028654 * (x + 0.044715 * torch.pow(x, 3)))
    )

class GPT2ModelRole(nn.Module):
    """OpenAI GPT-2 model ("Language Models are Unsupervised Multitask Learners").
    """
    def __init__(self, config):
        super(GPT2ModelRole, self).__init__()
        self.gradient_checkpointing = config.gradient_checkpointing
        self.config = config
        self.dropout = nn.Dropout(config.embd_pdrop)

        # word embedding
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        # position embedding
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        # role embedding

        self.h = nn.ModuleList(
            [
                Block(config.n_ctx, config, scale=True)
                for _ in range(config.n_layer)
            ]
        )
        self.ln_f = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.apply(self.init_weights)

        self.wre = nn.Embedding(2, config.n_embd)

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range
            )
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_ids, role_ids, position_ids=None, past=None, mask=None):

        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = past[0][0].size(-2)

        # One-Hot
        if input_ids.dtype == torch.float32:
            input_shape = input_ids.shape[:-1]
            inputs_embeds = input_ids.matmul(self.wte.weight).unsqueeze(1)
        # Long Index
        else:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_ids.size(-1))
            inputs_embeds = self.wte(input_ids)

        if position_ids is None:
            position_ids = torch.arange(
                past_length,
                input_shape[-1] + past_length,
                dtype=torch.long,
                device=input_ids.device
            )
            position_ids = position_ids.unsqueeze(0).expand(input_shape)

        # position embeddings
        position_ids = position_ids.view(-1, position_ids.size(-1))
        position_embeds = self.wpe(position_ids)

        # role embedding
        role_embedding = self.wre(role_ids)

        hidden_states = inputs_embeds + position_embeds + role_embedding
        #print(role_embedding)

        hidden_states = self.dropout(hidden_states)
        presents = []

        for block, layer_past in zip(self.h, past):
            # added gradient checkpointing
            if self.gradient_checkpointing:
                hidden_states, present = torch.utils.checkpoint.checkpoint(
                    block, hidden_states, layer_past, mask
                )
            else:
                hidden_states, present = block(hidden_states, layer_past, mask)
            presents.append(present)

        hidden_states = self.ln_f(hidden_states)
        output_shape = position_ids.shape + (hidden_states.size(-1), )
        return hidden_states.view(*output_shape), presents


class GPT2SimpleLMRole(nn.Module):
    """OpenAI GPT-2 model with a Language Modeling head ("Language Models are Unsupervised Multitask Learners").
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = GPT2ModelRole(config)
        self.lm_head = GPT2LMHead(self.transformer.wte.weight, config)
        self.apply(self.init_weights)

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range
            )
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def set_tied(self):
        """ Make sure we are sharing the embeddings
        """
        self.lm_head.set_embeddings_weights(self.transformer.wte.weight)

    def forward(self, input_ids, role_id, position_ids=None, past=None, mask=None):

        if past is None:
            past_length = input_ids.shape[1]
        else:
            # count self
            past_length = past[0].shape[3] + input_ids.shape[1]

        if mask is None:
            # print("mask is not provided")
            mask = torch.ones(
                input_ids.shape[0],
                past_length,
                dtype=torch.bool,
                device=input_ids.device
            )

        # Fast way to compute lower triangle attention mask
        # shape: (batch, num_head, key_length, query_length/seq_length)
        mask = mask.view(input_ids.shape[0], 1, 1, mask.shape[1]).repeat(
            1, self.config.n_head, mask.shape[1], 1
        )
        mask = mask & mask.permute(0, 1, 3, 2)
        mask = torch.tril(mask.byte())
        mask = mask.bool()
        mask = mask[:, :, -input_ids.shape[1]:, :]

        hidden_states, presents = self.transformer(
            input_ids, role_id, position_ids, past, mask
        )
        lm_logits = self.lm_head(hidden_states)
        return lm_logits, presents