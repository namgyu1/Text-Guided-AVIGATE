# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model."""

import logging
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import math
from modules.until_config import PretrainedConfig

logger = logging.getLogger(__name__)

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}

def k_recip(sim_matrix, topk=1):

    topk_index = torch.topk(sim_matrix, topk)[1]
    nn_matrix = torch.zeros_like(sim_matrix).scatter_(1, topk_index, torch.ones_like(sim_matrix))
    mask = ((nn_matrix + nn_matrix.t())/2).float()

    mask = mask.fill_diagonal_(0)

    new_mask = (mask == 1).detach().clone()
#         with torch.no_grad():
#             sim_matrix_ = sim_matrix.detach().clone()
#             # sim_matrix_ = sim_matrix_.fill_diagonal_(-1)
#             topk = 1
#             topk_index = torch.topk(sim_matrix_, topk)[1]
#             nn_matrix = torch.zeros_like(sim_matrix_).scatter_(1, topk_index, torch.ones_like(sim_matrix_))
#             mask = ((nn_matrix + nn_matrix.t())/2).float()


#             # Iden_mask = torch.eye(sim_matrix_.shape[0]) > 0.5
#             # Iden_mask = torch.autograd.Variable(Iden_mask.cuda())
#             # mask = mask.masked_fill(Iden_mask, 0)
#             mask = mask.fill_diagonal_(1)

#             new_mask = (mask == 1).detach().clone()
#             new_mask = torch.autograd.Variable(new_mask)

    return new_mask

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class PreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, config, *inputs, **kwargs):
        super(PreTrainedModel, self).__init__()
        if not isinstance(config, PretrainedConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `PretrainedConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = config

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, LayerNorm):
            if 'beta' in dir(module) and 'gamma' in dir(module):
                module.beta.data.zero_()
                module.gamma.data.fill_(1.0)
            else:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def resize_token_embeddings(self, new_num_tokens=None):
        raise NotImplementedError

    @classmethod
    def init_preweight(cls, model, state_dict, prefix=None, task_config=None):
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        if prefix is not None:
            old_keys = []
            new_keys = []
            for key in state_dict.keys():
                old_keys.append(key)
                new_keys.append(prefix + key)
            for old_key, new_key in zip(old_keys, new_keys):
                state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(model, prefix='')

        if prefix is None and (task_config is None or task_config.local_rank == 0):
            logger.info("-" * 20)
            # if len(missing_keys) > 0:
            #     logger.info("Weights of {} not initialized from pretrained model: {}"
            #                 .format(model.__class__.__name__, "\n   " + "\n   ".join(missing_keys)))
            # if len(unexpected_keys) > 0:
            #     logger.info("Weights from pretrained model not used in {}: {}"
            #                 .format(model.__class__.__name__, "\n   " + "\n   ".join(unexpected_keys)))
            # if len(error_msgs) > 0:
            #     logger.error("Weights from pretrained model cause errors in {}: {}"
            #                  .format(model.__class__.__name__, "\n   " + "\n   ".join(error_msgs)))

        return model

    @property
    def dtype(self):
        """
        :obj:`torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            # For nn.DataParallel compatibility in PyTorch 1.5
            def find_tensor_attributes(module: nn.Module):
                tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
                return tuples

            gen = self._named_members(get_members_fn=find_tensor_attributes)
            first_tuple = next(gen)
            return first_tuple[1].dtype

    @classmethod
    def from_pretrained(cls, config, state_dict=None,  *inputs, **kwargs):
        """
        Instantiate a PreTrainedModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.
        """
        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        if state_dict is None:
            return model
        model = cls.init_preweight(model, state_dict)

        return model

    
##################################
###### LOSS FUNCTION #############
##################################
class CrossEn(nn.Module):
    def __init__(self,):
        super(CrossEn, self).__init__()

    def forward(self, sim_matrix, uni_modal_sim):
        # import pdb; pdb.set_trace()
        # K-reciprocal nearest neighbor positive
        # mask = k_recip(sim_matrix, topk=2)
        # sim_matrix = sim_matrix / 2.0
        # uni_mask = k_recip(uni_mod_sim, topk=2)
        # new_mask = mask
        # import pdb; pdb.set_trace()
        # sim_matrix = sim_matrix.masked_fill(new_mask, float("-Inf"))
        
        # with torch.no_grad():
        #     rel_mg = uni_modal_sim*50#*sim_matrix.shape[0]
        #     rel_mg = rel_mg.fill_diagonal_(0)
        with torch.no_grad():
            rel_mg =  0.05 * 100 *(1-uni_modal_sim).fill_diagonal_(0)
            # rel_mg = 0.1 * 100* uni_modal_sim.fill_diagonal_(0)
        sim_matrix = sim_matrix + rel_mg
        
        logpt = F.log_softmax(sim_matrix, dim=-1)
        logpt = torch.diag(logpt)
        nce_loss = -logpt
        sim_loss = nce_loss.mean()       
        
        
        
#         mask = torch.eye(sim_matrix.shape[0])>0.5
#         mask = mask.cuda()
#         new_mask = torch.cat((mask,mask),1)
        
#         sim_matrix = torch.cat((sim_matrix, uni_modal_sim*10),1)
        
#         # new_sim = sim_matrix.masked_fill(new_mask, 0)
#         # new_sf = F.softmax(sim_matrix, dim=0)
#         # new_sim = sim_matrix * new_sf * len(sim_matrix)/100.0
#         # new_sim = (new_sim+sim_matrix)/2.0
        
        
#         pt = F.softmax(sim_matrix, dim=-1)
#         # # logpt = F.log_softmax(sim_matrix, dim=-1)
#         # # logpt = logpt.masked_fill(new_mask, 0)
#         # # import pdb; pdb.set_trace()
#         sum_pt = (pt*(new_mask.int())).sum(dim=-1)
#         # # logpt = torch.diag(logpt)
#         # # nce_loss = -logpt
#         nce_loss = -torch.log(sum_pt)
#         sim_loss = nce_loss.mean()
#         # # sim_loss = nce_loss/sim_matrix.shape[0]        
        
#         # logpt = F.log_softmax(new_sim, dim=-1)
#         # logpt = torch.diag(logpt)
#         # nce_loss = -logpt
#         # sim_loss = nce_loss.mean()       
        
        return sim_loss#, HN_rate/sim_matrix.shape[0]
    
class CrossEn_Ori(nn.Module):
    def __init__(self,):
        super(CrossEn_Ori, self).__init__()

    def forward(self, sim_matrix):
        logpt = F.log_softmax(sim_matrix, dim=-1)
        logpt = torch.diag(logpt)
        nce_loss = -logpt
        sim_loss = nce_loss.mean()       
        return sim_loss

    
        # logpt = F.log_softmax(sim_matrix, dim=-1)
        # logpt = torch.diag(logpt)
        # nce_loss = -logpt
        # sim_loss = nce_loss.mean()
        
        # sim_loss=0
        # import pdb; pdb.set_trace()
        # sim_matrix = sim_matrix / 100.0
#         margin = m
#         Pos_sim = sim_matrix.diag().view(sim_matrix.size(0), 1)
#         expanded_Pos_sim = Pos_sim.expand_as(sim_matrix)
#         cost = (margin + sim_matrix - expanded_Pos_sim).clamp(min=0)
        
#         mask = torch.eye(sim_matrix.size(0)) > .5
#         I = torch.autograd.Variable(mask)
#         if torch.cuda.is_available():
#             I = mask.cuda()
#         cost = cost.masked_fill_(I, 0)
#         cost = cost.max(1)[0]
#         sim_loss = cost.sum()
        
        
        
        # mask = torch.eye(sim_matrix.size(0)) > .5
        # I = torch.autograd.Variable(mask)
        # if torch.cuda.is_available():
        #     I = mask.cuda()
        # Pos = sim_matrix.diag().detach().clone()
        # Neg = sim_matrix.detach().clone()
        # Neg = Neg.masked_fill_(I,0).max(1)[0]
        # HardNeg = (Neg-Pos).abs()
        # # import pdb; pdb.set_trace()
        # cost = cost.masked_fill_(I, 0)
        # HN_rate = 0
        # for i in range(cost.shape[0]):
        #     if HardNeg[i] > 0.01:
        #         sim_loss += cost.max(1)[0][i]
        #         HN_rate +=1
        #     else:
        #         sim_loss += cost[i].mean()

        # margin_mat = torch.eye(sim_matrix.shape[0]).to(device=sim_matrix.device)*0.1*50
        # sim_matrix = sim_matrix - margin_mat
        
        #Original
    
class CrossEnVis(nn.Module):
    def __init__(self,):
        super(CrossEnVis, self).__init__()

    def forward(self, sim_matrix, visual_guide):
        # with torch.no_grad():
        visual_guide_row = torch.arange(visual_guide.shape[0]).unsqueeze(1)
        visual_guide_col = torch.arange(visual_guide.shape[0]).unsqueeze(0)
        mask =  visual_guide_row != visual_guide_col
        mask = mask.cuda()
        masked_visual_guide = visual_guide[mask].view(visual_guide.shape[0],visual_guide.shape[0]-1)
        # import pdb; pdb.set_trace()
        sim_matrix = torch.cat((sim_matrix, masked_visual_guide),1)
        logpt = F.log_softmax(sim_matrix, dim=-1)
        logpt = torch.diag(logpt)
        nce_loss = -logpt
        sim_loss = nce_loss.mean()
        return sim_loss
    
class CrossEnTxt(nn.Module):
    def __init__(self,):
        super(CrossEnTxt, self).__init__()

    def forward(self, sim_matrix, textual_guide):
        sim_matrix = sim_matrix * 0.01
#         with torch.no_grad():
#             P = textual_guide.detach().clone()
#             mask_ = torch.eye(sim_matrix.shape[0], sim_matrix.shape[0]).cuda()
#         # pos_weight = P
#         neg_weight = (1-P)*(1-mask_) 
        
#         pos_tmp = torch.relu(sim_matrix) * mask_
#         pull_loss = -torch.log(pos_tmp[pos_tmp>0])
        
#         tmp =(1-torch.relu(sim_matrix-0.1)) * neg_weight
#         push_loss = -torch.log(tmp[tmp>0])
        # sim_loss = pull_loss/sim_matrix.shape[0] + 0.0*push_loss.mean()
        with torch.no_grad():
            margin = 0.2 * (1 + textual_guide)
        Pos_sim = sim_matrix.diag().view(sim_matrix.size(0), 1)
        expanded_Pos_sim = Pos_sim.expand_as(sim_matrix)
        cost = (margin + sim_matrix - expanded_Pos_sim).clamp(min=0)
        
        mask = torch.eye(sim_matrix.size(0)) > .5
        I = torch.autograd.Variable(mask)
        if torch.cuda.is_available():
            I = mask.cuda()
        cost = cost.masked_fill_(I, 0)
        cost = cost.max(1)[0]
        sim_loss = cost.sum()        
        
#         logpt = F.log_softmax(sim_matrix, dim=-1)
#         pos_logpt = torch.diag(logpt)
        # mask_ = torch.eye(sim_matrix.shape[0], sim_matrix.shape[0]).cuda()
#         neg_logpt = logpt*(1-mask_)
#         with torch.no_grad():
#             P = visual_guide.detach().clone()
#         import pdb; pdb.set_trace()
#         neg_logpt =torch.relu(neg_logpt-0.5)*(1-P)
        
#         nce_loss = -pos_logpt.mean()
#         # neg_loss = -neg_logpt.sum()/(pos_logpt.shape[0])
#         neg_loss = neg_logpt.mean()
        return sim_loss    
    
class RelCont(nn.Module):
    def __init__(self,):
        super(RelCont, self).__init__()

    def forward(self, sim_matrix, visual_guide):
        T_dist = F.log_softmax(sim_matrix, dim=-1)
        with torch.no_grad():
            P = visual_guide
        pos_weight = P    
        neg_weight = (1-P)
        pull_losses = T_dist * pos_weight
        push_losses = torch.relu(0.5 - T_dist) * neg_weight
        

        return sim_loss
    
class MILNCELoss(nn.Module):
    def __init__(self, batch_size=1, n_pair=1,):
        super(MILNCELoss, self).__init__()
        self.batch_size = batch_size
        self.n_pair = n_pair
        torch_v = float(".".join(torch.__version__.split(".")[:2]))
        self.bool_dtype = torch.bool if torch_v >= 1.3 else torch.uint8

    def forward(self, sim_matrix):
        mm_mask = np.eye(self.batch_size)
        mm_mask = np.kron(mm_mask, np.ones((self.n_pair, self.n_pair)))
        mm_mask = torch.tensor(mm_mask).float().to(sim_matrix.device)

        from_text_matrix = sim_matrix + mm_mask * -1e12
        from_video_matrix = sim_matrix.transpose(1, 0)

        new_sim_matrix = torch.cat([from_video_matrix, from_text_matrix], dim=-1)
        logpt = F.log_softmax(new_sim_matrix, dim=-1)

        mm_mask_logpt = torch.cat([mm_mask, torch.zeros_like(mm_mask)], dim=-1)
        masked_logpt = logpt + (torch.ones_like(mm_mask_logpt) - mm_mask_logpt) * -1e12

        new_logpt = -torch.logsumexp(masked_logpt, dim=-1)

        logpt_choice = torch.zeros_like(new_logpt)
        mark_ind = torch.arange(self.batch_size).to(sim_matrix.device) * self.n_pair + (self.n_pair//2)
        logpt_choice[mark_ind] = 1
        sim_loss = new_logpt.masked_select(logpt_choice.to(dtype=self.bool_dtype)).mean()
        return sim_loss

class MaxMarginRankingLoss(nn.Module):
    def __init__(self,
                 margin=1.0,
                 negative_weighting=False,
                 batch_size=1,
                 n_pair=1,
                 hard_negative_rate=0.5,
        ):
        super(MaxMarginRankingLoss, self).__init__()
        self.margin = margin
        self.n_pair = n_pair
        self.batch_size = batch_size
        easy_negative_rate = 1 - hard_negative_rate
        self.easy_negative_rate = easy_negative_rate
        self.negative_weighting = negative_weighting
        if n_pair > 1 and batch_size > 1:
            alpha = easy_negative_rate / ((batch_size - 1) * (1 - easy_negative_rate))
            mm_mask = (1 - alpha) * np.eye(self.batch_size) + alpha
            mm_mask = np.kron(mm_mask, np.ones((n_pair, n_pair)))
            mm_mask = torch.tensor(mm_mask) * (batch_size * (1 - easy_negative_rate))
            self.mm_mask = mm_mask.float()

    def forward(self, x):
        d = torch.diag(x)
        max_margin = F.relu(self.margin + x - d.view(-1, 1)) + \
                     F.relu(self.margin + x - d.view(1, -1))
        if self.negative_weighting and self.n_pair > 1 and self.batch_size > 1:
            max_margin = max_margin * self.mm_mask.to(max_margin.device)
        return max_margin.mean()

class TripletLoss(nn.Module):
    """
    Compute Triplet loss
    """
    def __init__(self, margin=0.1, max_violation=True):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.max_violation = max_violation

    def forward(self, scores):
        # compute image-sentence score matrix
        diagonal = scores.diag().view(scores.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        if torch.cuda.is_available():
            I = mask.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]
        return (cost_s.sum() + cost_im.sum())/2.0    
    
class AllGather(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor, args):
        output = [torch.empty_like(tensor) for _ in range(args.world_size)]
        torch.distributed.all_gather(output, tensor)
        ctx.rank = args.rank
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank : ctx.batch_size * (ctx.rank + 1)],
            None,
        )
