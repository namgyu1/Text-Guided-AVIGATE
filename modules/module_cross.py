from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy
import json
import math
import logging
import tarfile
import tempfile
import shutil
from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F
from .file_utils import cached_path
from .until_config import PretrainedConfig
from .until_module import PreTrainedModel, LayerNorm, ACT2FN
from collections import OrderedDict
from torch import Tensor

def drop_path(x: Tensor, keep_prob: float = 1.0, inplace: bool = False) -> Tensor:
    mask_shape: Tuple[int] = (x.shape[0],) + (1,) * (x.ndim - 1)
    # remember tuples have the * operator -> (1,) * 3 = (1,1,1)
    mask: Tensor = x.new_empty(mask_shape).bernoulli_(keep_prob)
    mask.div_(keep_prob)
    if inplace:
        x.mul_(mask)
    else:
        x = x * mask
    return x
class DropPath(nn.Module):
    def __init__(self, p: float = 0.5, inplace: bool = False):
        super().__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        if self.training and self.p > 0:
            x = drop_path(x, self.p, self.inplace)
        return x

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p})"

logger = logging.getLogger(__name__)

PRETRAINED_MODEL_ARCHIVE_MAP = {}
CONFIG_NAME = 'cross_config.json'
WEIGHTS_NAME = 'cross_pytorch_model.bin'


class CrossConfig(PretrainedConfig):
    """Configuration class to store the configuration of a `CrossModel`.
    """
    pretrained_model_archive_map = PRETRAINED_MODEL_ARCHIVE_MAP
    config_name = CONFIG_NAME
    weights_name = WEIGHTS_NAME
    def __init__(self,
                 vocab_size_or_config_json_file,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02):
        """Constructs CrossConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `CrossModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `CrossModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        if isinstance(vocab_size_or_config_json_file, str):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.n_head = n_head

    def attention(self, x: torch.Tensor, attn_mask: torch.Tensor):
        attn_mask_ = attn_mask.repeat_interleave(self.n_head, dim=0)
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask_)[0]

    def forward(self, para_tuple: tuple):
        # x: torch.Tensor, attn_mask: torch.Tensor
        # print(para_tuple)
        x, attn_mask = para_tuple
        x = x + self.attention(self.ln_1(x), attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return (x, attn_mask)

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads) for _ in range(layers)])

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor):
        return self.resblocks((x, attn_mask))[0]


class ResidualAttentionBlock_Text(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.cross_attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.ln_3 = LayerNorm(d_model)
        self.n_head = n_head
    def attention(self, x: torch.Tensor, attn_mask=None):
        if attn_mask is not None:
            attn_mask_ = attn_mask.repeat_interleave(self.n_head, dim=0)
        else:
            attn_mask_ = attn_mask
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask_)[0]

    def cross_attention(self, query: torch.Tensor, mem: torch.Tensor, attn_mask=None):
        if attn_mask is not None:
            attn_mask_ = attn_mask.repeat_interleave(self.n_head, dim=0)
        else:
            attn_mask_ = attn_mask
        return self.cross_attn(query, mem, mem, need_weights=False, attn_mask=attn_mask_)[0]


    def forward(self, para_tuple: tuple):
        x, v, attn_mask= para_tuple
        x = x + self.attention(self.ln_1(x), attn_mask=None)
        x = x + self.cross_attention(self.ln_3(x), self.ln_3(v), attn_mask)
        x = x + self.mlp(self.ln_2(x))

        return (x, v, attn_mask)
### Transformer_Text denotes Audio_Resampler in the paper. ###
class Transformer_Text(nn.Module):
    def __init__(self, width: int, layers: int, heads: int):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock_Text(width, heads) for _ in range(layers)])

    def forward(self, q: torch.Tensor, v: torch.Tensor,  attn_mask=None):
        return self.resblocks((q, v, attn_mask))[0]

class ResidualAttentionBlock_Gate(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_head)
        self.cross_ff = nn.Sequential(OrderedDict([
            ("cross_fc", nn.Linear(d_model, d_model * 4, bias = False)),
            ("cross_gelu", QuickGELU()),
            ("cross_proj", nn.Linear(d_model * 4, d_model, bias = False))
        ]))
        
        # Modified: Input dimension changed from d_model*2 to d_model*3 (video + audio + text)
        self.attn_gate = nn.Sequential(OrderedDict([
            ("ag_fc", nn.Linear(int(d_model * 3), int(d_model * 0.5), bias = False)),
            ("ag_gelu", QuickGELU()),
            ("ag_proj", nn.Linear(int(d_model * 0.5), 1, bias = False))
        ]))
        self.ff_gate = nn.Sequential(OrderedDict([
            ("fg_fc", nn.Linear(int(d_model * 3), int(d_model * 0.5), bias = False)),
            ("fg_gelu", QuickGELU()),
            ("fg_proj", nn.Linear(int(d_model * 0.5), 1, bias = False))
        ]))
        
        # New: Query fusion gate for multi-modal query (video + text fusion)
        self.query_gate = nn.Sequential(OrderedDict([
            ("qg_fc", nn.Linear(int(d_model * 3), int(d_model * 0.5), bias = False)),
            ("qg_gelu", QuickGELU()),
            ("qg_proj", nn.Linear(int(d_model * 0.5), 1, bias = False))
        ]))
        
        self.ln_3 = LayerNorm(d_model)
        self.ln_4 = LayerNorm(d_model)        

        self.n_head = n_head

    def attention(self, x: torch.Tensor, attn_mask=None):
        if attn_mask is not None:
            attn_mask_ = attn_mask.repeat_interleave(self.n_head, dim=0)
        else:
            attn_mask_ = attn_mask
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask_)[0]
    
    def cross_attention(self, query: torch.Tensor, mem: torch.Tensor, attn_mask=None):
        if attn_mask is not None:
            attn_mask_ = attn_mask.repeat_interleave(self.n_head, dim=0)
        else:
            attn_mask_ = attn_mask
        return self.cross_attn(query, mem, mem, need_weights=False, attn_mask=attn_mask_)[0]

    
    def forward(self, para_tuple: tuple):
        # Modified: Added text embedding (t) to the input tuple
        x, v, t, attn_mask, attn_gate_list, ff_gate_list, query_gate_list = para_tuple
        
        # Pooling: Get mean representations for gating functions
        x_mean = x.mean(dim=0)
        v_mean = v.mean(dim=0)
        t_mean = t.mean(dim=0)
        
        # --- Multi-Modal Query Fusion ---
        # Create fused query by combining video and text embeddings
        # query_gate_weight: 0~1 value (sigmoid activation)
        query_gate_weight = self.query_gate(torch.cat((x_mean, v_mean, t_mean), dim=1)).sigmoid()
        # Fused query = video * query_gate + text * (1 - query_gate)
        fused_query = x * query_gate_weight + t * (1 - query_gate_weight)
        
        # --- Gating functions for audio fusion ---
        # Gating functions now use video (x), audio (v), and text (t) embeddings
        attn_gate = self.attn_gate(torch.cat((x_mean, v_mean, t_mean), dim=1)).tanh()
        ff_gate = self.ff_gate(torch.cat((x_mean, v_mean, t_mean), dim=1)).tanh()        

        # --- Cross-modal fusion with audio ---
        # Use fused_query instead of original video embedding for cross attention
        x = x + self.cross_attention(self.ln_3(fused_query), v, attn_mask/100) * attn_gate
        x = x + self.cross_ff(self.ln_4(x)) * ff_gate
        
        # --- Self-attention and feed-forward ---
        x = x + self.attention(self.ln_1(x), attn_mask=None)
        x = x + self.mlp(self.ln_2(x))
        
        # Store gate values for analysis
        attn_gate_list.append(attn_gate.view(-1))
        ff_gate_list.append(ff_gate.view(-1))
        query_gate_list.append(query_gate_weight.view(-1))
        
        # DEBUG: Print gate statistics periodically (enable during debugging)
        # if self.training and len(attn_gate_list) % 100 == 0:
        #     print(f"\n[Layer {len(attn_gate_list)}] Gate Statistics:")
        #     print(f"  Query Gate: mean={query_gate_weight.mean():.4f}, std={query_gate_weight.std():.4f}, "
        #           f"min={query_gate_weight.min():.4f}, max={query_gate_weight.max():.4f}")
        #     print(f"  Attn Gate:  mean={attn_gate.mean():.4f}, std={attn_gate.std():.4f}, "
        #           f"min={attn_gate.min():.4f}, max={attn_gate.max():.4f}")
        #     print(f"  FF Gate:    mean={ff_gate.mean():.4f}, std={ff_gate.std():.4f}, "
        #           f"min={ff_gate.min():.4f}, max={ff_gate.max():.4f}")
        
        return (x, v, t, attn_mask, attn_gate_list, ff_gate_list, query_gate_list)

### Transformer_Gate denotes Gated_Fusion_Transformer in the paper. ###
class Transformer_Gate(nn.Module):
    def __init__(self, width: int, layers: int, heads: int):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock_Gate(width, heads) for _ in range(layers)])
        
    def forward(self, q: torch.Tensor, v: torch.Tensor, t: torch.Tensor, attn_mask=None):
        """
        Args:
            q: video embedding (query)
            v: audio embedding (value)
            t: text embedding (for gating function)
            attn_mask: attention mask
        Returns:
            Tuple of (output, audio, text, attn_mask, attn_gate_list, ff_gate_list, query_gate_list)
        """
        attn_gate_list = []
        ff_gate_list = []
        query_gate_list = []
        return self.resblocks((q, v, t, attn_mask, attn_gate_list, ff_gate_list, query_gate_list)) 

class CrossEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(CrossEmbeddings, self).__init__()

        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, concat_embeddings, concat_type=None):

        batch_size, seq_length = concat_embeddings.size(0), concat_embeddings.size(1)

        position_ids = torch.arange(seq_length, dtype=torch.long, device=concat_embeddings.device)
        position_ids = position_ids.unsqueeze(0).expand(concat_embeddings.size(0), -1)

        position_embeddings = self.position_embeddings(position_ids)

        embeddings = concat_embeddings + position_embeddings 
        embeddings = self.dropout(embeddings)
        return embeddings

class CrossPooler(nn.Module):
    def __init__(self, config):
        super(CrossPooler, self).__init__()
        self.ln_pool = LayerNorm(config.hidden_size)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = QuickGELU()

    def forward(self, hidden_states, hidden_mask):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        hidden_states = self.ln_pool(hidden_states)
        pooled_output = hidden_states[:, 0]
        pooled_output = self.dense(pooled_output)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class CrossModel(PreTrainedModel):

    def initialize_parameters(self):
        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

    def __init__(self, config):
        super(CrossModel, self).__init__(config)

        self.embeddings = CrossEmbeddings(config)

        transformer_width = config.hidden_size
        transformer_layers = config.num_hidden_layers
        transformer_heads = config.num_attention_heads
        self.transformer = Transformer(width=transformer_width, layers=transformer_layers, heads=transformer_heads,)
        self.pooler = CrossPooler(config)
        self.apply(self.init_weights)

    def build_attention_mask(self, attention_mask):
        extended_attention_mask = attention_mask.unsqueeze(1)
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -1000000.0
        extended_attention_mask = extended_attention_mask.expand(-1, attention_mask.size(1), -1)
        return extended_attention_mask

    def forward(self, concat_input, concat_type=None, attention_mask=None, output_all_encoded_layers=True):

        if attention_mask is None:
            attention_mask = torch.ones(concat_input.size(0), concat_input.size(1))
        if concat_type is None:
            concat_type = torch.zeros_like(attention_mask)

        extended_attention_mask = self.build_attention_mask(attention_mask)

        embedding_output = self.embeddings(concat_input, concat_type)
        embedding_output = embedding_output.permute(1, 0, 2)  # NLD -> LND
        embedding_output = self.transformer(embedding_output, extended_attention_mask)
        embedding_output = embedding_output.permute(1, 0, 2)  # LND -> NLD

        pooled_output = self.pooler(embedding_output, hidden_mask=attention_mask)

        return embedding_output, pooled_output

