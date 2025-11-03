# from fvcore.nn import FlopCountAnalysis

import os
import glob
import numpy as np
import torch
from tqdm import tqdm
import time
import torch.nn as nn
import torch.nn.functional as F
import math

device_id = 9
device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

class MultiHeadedAttention(nn.Module):
    def __init__(self):
        super(MultiHeadedAttention, self).__init__()
        self.embed_dim = 512
        self.num_heads = 8
        assert self.embed_dim % self.num_heads == 0
        self.head_dim = self.embed_dim // self.num_heads
        
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    
    def forward(self, text_embeds, video_embeds):
        """
        Input
            text_embeds: num_texts x embed_dim
            video_embeds: num_vids x num_frames x embed_dim
        Output
            o: num_vids x num_texts x embed_dim
        """
        num_texts, _ = text_embeds.shape
        # num_texts x embed_dim
        q = self.q_proj(text_embeds)
        q = q.reshape(num_texts, self.num_heads, self.head_dim)
        # num_heads x head_dim x num_texts
        q = q.permute(1,2,0)

        num_vids, num_frames, _ = video_embeds.shape
        # num_vids x num_frames x embed_dim
        k = self.k_proj(video_embeds)
        k = k.reshape(num_vids, num_frames, self.num_heads, self.head_dim)
        # num_vids x num_heads x num_frames x head_dim
        k = k.permute(0,2,1,3)

        # num_vids x num_frames x embed_dim
        v = self.v_proj(video_embeds)
        v = v.reshape(num_vids, num_frames, self.num_heads, self.head_dim)
        # num_vids x num_heads x head_dim x num_frames
        v = v.permute(0,2,3,1)

        # num_vids x num_heads x num_frames x num_texts
        attention_logits = k @ q
        attention_logits = attention_logits / math.sqrt(self.head_dim)
        attention_weights = F.softmax(attention_logits, dim=2)

        # num_vids x num_heads x head_dim x num_texts
        attention = v @ attention_weights
        # num_vids x num_texts x num_heads x head_dim
        attention = attention.permute(0,3,1,2)
        attention = attention.reshape(num_vids, num_texts, self.embed_dim)

        # num_vids x num_texts x embed_dim
        o = self.out_proj(attention)
        return o
class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.embed_dim = 512
        dropout = 0.1

        self.cross_attn = MultiHeadedAttention()

        self.linear_proj = nn.Linear(self.embed_dim, self.embed_dim)
            
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)
        self.layer_norm3 = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(dropout)

        self._init_parameters()

    
    def _init_parameters(self):
        for name, param in self.named_parameters():
            if 'linear' in name or 'proj' in name:
                if 'weight' in name:
                    nn.init.eye_(param)
                elif 'bias' in name:
                    param.data.fill_(0.)


    def forward(self, text_embeds, video_embeds):
        """
        Input
            text_embeds: num_texts x embed_dim
            video_embeds: num_vids x num_frames x embed_dim
        Output
            out: num_vids x num_texts x embed_dim
        """
        text_embeds = self.layer_norm1(text_embeds)
        video_embeds = self.layer_norm1(video_embeds)

        # num_vids x num_texts x embed_dim
        attn_out = self.cross_attn(text_embeds, video_embeds)
        attn_out = self.layer_norm2(attn_out)

        linear_out = self.linear_proj(attn_out)
        out = attn_out + self.dropout(linear_out)
        out = self.layer_norm3(out)

        return out
    
def _LSE_real(x, lambda_, d=1):
    x_=torch.exp(x*lambda_)
    x_ = x_.sum(dim=d)
    out = torch.log(x_)
    return out


def calculate_sim(v,t):
    global_v = v.mean(dim=1)
#     global_sim = F.linear(t,global_v)
    
#     local_sim = _LSE_real(torch.matmul(F.normalize(t,dim=-1).unsqueeze(1).unsqueeze(1), F.normalize(v,dim=-1).transpose(1,2).unsqueeze(0)).squeeze(2) * 100, 1, mask=None, d=-1)/1
    
    return torch.matmul(t,global_v.t())+_LSE_real(torch.matmul(t.unsqueeze(1).unsqueeze(1), v.transpose(1,2).unsqueeze(0)).squeeze(2) , 1, d=-1)

def calculate_sim_clip4clip(v,t):
    global_v = v.mean(dim=1)
    global_sim = F.linear(global_v,t)
    
    return global_sim

def calculate_xpool(t,v,m1):
    v_cxtn =m1(t,v) 
    # print(v_cxtn.shape)
    return torch.matmul(v_cxtn.permute(1,0,2), t.unsqueeze(-1)).squeeze(-1)

def calculate_tefal(t,v, a,m1,m2):
    v_cxtn =m1(t,v) 
    a_cxtn =m2(t,a)
    
    f_cxtn = v_cxtn + a_cxtn
    return torch.matmul(f_cxtn.permute(1,0,2), t.unsqueeze(-1)).squeeze(-1)
    
text = torch.rand(1,512, device=device)
videos = torch.rand(1000,12,512, device=device)
audios = torch.rand(1,1212,512,device=device)
# cxtn1 = Transformer().to(device)
# cxtn2 = Transformer().to(device)

# start_time = time.time()

# sim = calculate_sim_clip4clip(videos, text)
# end_time = time.time()
# latency_clip4clip = (end_time-start_time)

latency_clip4clip = 0
latency_ours =0
for i in range(1000):
    start_time = time.time()
    sim = calculate_sim_clip4clip(videos, text)
    end_time = time.time()
    if i> 200:
        latency_clip4clip += (end_time-start_time)
latency_clip4clip /= 800

for i in range(1000):
    start_time = time.time()
    sim = calculate_sim(videos, text)
    end_time = time.time()
    if i> 200:
        latency_ours += (end_time-start_time)
latency_ours /=800

# start_time = time.time()
# sim = calculate_sim(videos, text)
# end_time = time.time()
# latency_ours = (end_time-start_time)

# start_time = time.time()
# sim = calculate_sim_clip4clip(videos, text)
# torch.cuda.synchronize(device)
# end_time = time.time()
# latency_clip4clip = (end_time-start_time)



latency_xpool = 0
latency_tefal =0
# for i in range(1000):
#     start_time = time.time()
#     sim = calculate_xpool(text, videos,cxtn1)
#     end_time = time.time()
#     if i> 200:
#         latency_xpool += (end_time-start_time)
# latency_xpool /= 800
# for i in range(1000):
#     start_time = time.time()
#     sim = calculate_tefal(text, videos, audios,cxtn1,cxtn2)
#     end_time = time.time()
#     if i> 200:
#         latency_tefal += (end_time-start_time)
# latency_tefal /= 800


# # Calculate FLOPs for `calculate_sim_clip4clip`
# def calculate_sim_clip4clip_wrapper():
#     return calculate_sim_clip4clip(videos, text)

# # Calculate FLOPs for `calculate_sim`
# def calculate_sim_wrapper():
#     return calculate_sim(videos, text)

# # Calculate FLOPs for `calculate_xpool`
# def calculate_xpool_wrapper():
#     cxtn = Transformer().to(device)
#     v_cxtn = cxtn(videos, text)
#     return F.linear(text, v_cxtn)

# FLOPs estimation
# flops_clip4clip = FlopCountAnalysis(calculate_sim_clip4clip_wrapper(), (videos, text)).total()
# flops_ours = FlopCountAnalysis(calculate_sim_wrapper(), (videos, text)).total()
# flops_xpool = FlopCountAnalysis(Transformer().to(device), (text,videos)).total()

# print(f"FLOPs of CLIP4Clip: {flops_clip4clip:.4f} FLOPs.")
# print(f"FLOPs of our method: {flops_ours:.4f} FLOPs.")
# print(f"FLOPs of X-Pool: {flops_xpool:.4f} FLOPs.")

# print(f"Latency of X-Pool: {latency_xpool*1000*100:.7f} ms.")
# print(f"Latency of TEFAL: {latency_tefal*1000*100:.7f} ms.")
print(f"Latency of CLIP4Clip: {latency_clip4clip*1000:.7f} ms.")
print(f"Latency of ours: {latency_ours*1000:.7f} ms.")

