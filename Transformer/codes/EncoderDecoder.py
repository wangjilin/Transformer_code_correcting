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
        self.encoderlayer = clones(Encoderlayer(h, d_model, d_k, d_v, d_ff, dropout=0.1, eps=1e-6), N)
        self.N = N
        self.h = h
        self.WK = clones(nn.Linear(d_model, d_k), h)
        self.WV = clones(nn.Linear(d_model, d_v), h)

    def forward(self, x):
        for i in range(self.N):
            x = self.encoderlayer[i](x)
        K = []
        V = []
        for i in range(self.h):
            K.append(self.WK[i](x).transpose(0, 1))  # K[i] (src_len, b, d_k) -> (b, src_len, d_k)
            V.append(self.WV[i](x).transpose(0, 1))  # V[i] (src_len, b, d_v) -> (b, src_len, d_v)

        return x, K, V

class Encoderlayer(nn.Module):
    def __init__(self, h, d_model, d_k, d_v, d_ff, dropout=0.1, eps=1e-6):
        super(Encoderlayer, self).__init__()
        self.multihead = MultiHeadedAttention(h, d_model, d_k, d_v)
        self.addnorm1 = AddNormLayer(d_model, eps, dropout)
        self.feedforward1 = FeedForward(d_model, d_ff, dropout)
        self.addnorm2 = AddNormLayer(d_model, eps, dropout)

    def forward(self, x):
        # X (src_len: sentence max # words in a batch, b: batch size, d_model: embedding size)
        x1 = self.multihead(x)
        x = self.addnorm1(x,x1)
        x2 = self.feedforward1(x)
        x = self.addnorm2(x,x2)
        # print(x.shape)
        return x  # (src_len: sentence max # words in a batch, b: batch size, d_model: embedding size)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, d_k, d_v):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k, self.d_v, self.h = d_k, d_v, h
        self.WQ = clones(nn.Linear(d_model, d_k), h)
        self.WK = clones(nn.Linear(d_model, d_k), h)
        self.WV = clones(nn.Linear(d_model, d_v), h)
        self.WO = nn.Linear(h * d_v, d_model)

    def forward(self, X, KK=None, VV=None, mask=None, KV=False):
        "Implements Figure 2"
        # X (src_len: sentence max  # words in a batch, b: batch size, d_model: embedding size)

        tensors = []
        for i in range(self.h):
            Q = self.WQ[i](X).transpose(0, 1)  # Q (src_len, b, d_k)-> (b, src_len, d_k)
            if KV:
                K = KK[i]
                V = VV[i]
            else:
                K = self.WK[i](X).transpose(0, 1)  # K (src_len, b, d_k) -> (b, src_len, d_k)
                V = self.WV[i](X).transpose(0, 1)  # V (src_len, b, d_v) -> (b, src_len, d_v)
            # print(Q.shape, K.shape, V.shape)
            tensors.append(self.self_attention(Q, K.transpose(1, 2), V, mask))  # h * (b, src_len, d_v)
        Zs = torch.cat(tensors, dim=2)  # (b, src_len, d_v*h) -> (b, src_len, d_model)
        return self.WO(Zs).transpose(0, 1)  # (b, src_len, d_model) -> (src_len, b, d_model) same as input

    def self_attention(self, Q, KT, V, mask=None):
        scores = torch.matmul(Q, KT) / math.sqrt(self.d_k)
        if mask != None:
            scores.masked_fill_(mask,-float('inf'))
        p_attn = F.softmax(scores, dim=-1)  # (b, src_len, src_len)
        return torch.matmul(p_attn, V)  # (b, src_len, d_v)

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, featuresize, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(featuresize))  # featuresize == d_model
        self.beta = nn.Parameter(torch.zeros(featuresize))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)  # only average and std the embedding for each word!
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class AddNormLayer(nn.Module):
    def __init__(self, d_model, eps=1e-6, dropout=0.1):
        super(AddNormLayer, self).__init__()
        self.norm = LayerNorm(d_model, eps)
        # self.norm = nn.LayerNorm(d_model, eps=1e-05, elementwise_affine=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x1, x2):
        # import data before/after attention/feed (src_len, b, d_model)
        return x1 + self.dropout(self.norm(x2))  # (src_len, b, d_model) same as input


class FeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))



class Decoder(nn.Module):
    # "Core encoder is a stack of N layers"
    def __init__(self, d_model=512, d_k=64, d_v=64, d_ff=2048, h=8, N=6):
        super(Decoder, self).__init__()
        self.decoderlayer = clones(Decoderlayer(h, d_model, d_k, d_v, d_ff, dropout=0.1, eps=1e-6), N)
        self.N = N

    def forward(self, x, K, V, mask):

        for i in range(self.N):
            x = self.decoderlayer[i](x, K, V, mask)
        return x

class Decoderlayer(nn.Module):
    def __init__(self, h, d_model, d_k, d_v, d_ff, dropout, eps):
        super(Decoderlayer, self).__init__()
        self.multihead1 = MultiHeadedAttention(h, d_model, d_k, d_v)
        self.addnorm1 = AddNormLayer(d_model, eps, dropout)
        self.multihead2 = MultiHeadedAttention(h, d_model, d_k, d_v)
        self.addnorm2 = AddNormLayer(d_model, eps, dropout)
        self.feedforward1 = FeedForward(d_model, d_ff, dropout)
        self.addnorm3 = AddNormLayer(d_model, eps, dropout)

    def forward(self, x, K, V, mask):
        # X (src_len: sentence max # words in a batch, b: batch size, d_model: embedding size)
        x1 = self.multihead1(x, mask=mask)
        x = self.addnorm1(x,x1)
        x2 = self.multihead2(x, KK=K, VV=V, KV=True)
        x = self.addnorm2(x,x2)
        x3 = self.feedforward1(x)
        x = self.addnorm2(x, x3)
        # print(x.shape)
        return x  # (src_len: sentence max # words in a batch, b: batch size, d_model: embedding size)
