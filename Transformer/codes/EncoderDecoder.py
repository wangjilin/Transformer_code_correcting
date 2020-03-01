#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
import math

def clones(module, N):
    # "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    # "Core encoder is a stack of N layers"

    def __init__(self, d_model=512, d_k=64, d_v=64, d_ff=2048, h=8, N=6):
        super(Encoder, self).__init__()
        self.testhead = MultiHeadedAttention(h, d_model, d_k, d_v)

    def forward(self, X):
        self.testhead(X)
# 重写这一整个类！！！！！！！！！！！！！！！！！！！！！！！！
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, d_k, d_v, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k, self.d_v, self.h = d_k, d_v, h
        self.WQ = clones(nn.Linear(d_model, d_k), h)
        self.WK = clones(nn.Linear(d_model, d_k), h)
        self.WV = clones(nn.Linear(d_model, d_v), h)
        self.WO = nn.Linear(h*d_v, d_model)

    def forward(self, X, mask=None):
        "Implements Figure 2"
        # X (src_len: sentence max  # words in a batch, b: batch size, d_model: embedding size)

        tensors = []
        for i in range(self.h):
            Q = self.WQ[i](X).transpose(0, 1)  # Q (src_len, b, d_k)-> (b, src_len, d_k)
            K = self.WK[i](X).transpose(0, 1)  # K (src_len, b, d_k) -> (b, src_len, d_k)
            V = self.WV[i](X).transpose(0, 1)  # V (src_len, b, d_v) -> (b, src_len, d_v)
            # print(Q.shape, K.shape, V.shape)
            tensors.append(self.self_attention(Q, K.transpose(1, 2), V)) # h * (b, src_len, d_v)
        Zs = torch.cat(tensors, dim=2) # (b, src_len, d_v*h) -> (b, src_len, d_model)
        return self.WO(Zs).transpose(0,1) # (b, src_len, d_model) -> (src_len, b, d_model) same as input

    def self_attention(self, Q, KT, V):
        scores = F.softmax(torch.matmul(Q, KT)/math.sqrt(self.d_k)) #(src)
        p_attn = F.softmax(scores, dim=-1) # (b, src_len, src_len)
        return torch.matmul(p_attn, V) # (b, src_len, d_v)

'''
#######################################################################
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)





class Encoder(nn.Module):
    # "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        # "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

'''