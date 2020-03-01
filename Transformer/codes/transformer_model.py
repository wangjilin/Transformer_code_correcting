#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.utils
import math
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from model_embeddings import ModelEmbeddings
from EncoderDecoder import Encoder

class TransformerModel(nn.Module):
    """
    Transformer model from paper
    'Attention is all you need'
    """

    def __init__(self, vocab, d_model=512, d_k=64, d_v=64, d_ff=2048, dropout_rate=0.1):
        """ Init TransformerModel .
        @param vocab (Vocab): Vocabulary object containing src and tgt languages
                              See vocab.py for documentation.
        @param d_model (int): Embedding size (dimensionality)
        @param d_k (int):     Query & Key size (dimensionality)
        @param d_v (int):     Value size (dimensionality)
        @param d_ff (int):    Feed-Forward Layer size (dimensionality)
        @param dropout_rate (float): Dropout probability, for attention
        """

        super(TransformerModel, self).__init__()
        self.model_embeddings = ModelEmbeddings(d_model, vocab)
        self.vocab = vocab
        self.dropout = nn.Dropout(dropout_rate)
        self.encoder = Encoder()


    def forward(self, source, target):
        """ Take a mini-batch of source and target sentences, compute the log-likelihood of
        target sentences under the language models learned by the transformer system.

        @param source (List[List[str]]): list of source sentence tokens
        @param target (List[List[str]]): list of target sentence tokens, wrapped by `<s>` and `</s>`

        @returns scores (Tensor): a variable/tensor of shape (b, ) representing the
                                    log-likelihood of generating the gold-standard target sentence for
                                    each example in the input batch. Here b = batch size.
        """

        # Compute sentence lengths
        source_lengths = [len(s) for s in source]

        # Convert list of lists into tensors with padded to the same length src_len (max sentence length: # words) in the batch
        source_padded = self.vocab.src.to_input_tensor(source, device=self.device)   # Tensor: (src_len, b)
        target_padded = self.vocab.tgt.to_input_tensor(target, device=self.device)   # Tensor: (tgt_len, b)

        # (src_len: sentence max # words in a batch, b: batch size, d_model: embedding size)
        X = self.model_embeddings.source.forward(source_padded)
        X = self.PositionalEncoding(X)
        self.encoder(X)
        # print((X[:,1,:]).shape)





    def PositionalEncoding(self,embeded_words):
        max_len, b, d_model = embeded_words.shape
        pe = torch.zeros(max_len, b, d_model, requires_grad=False, device=self.device)
        position = torch.arange(0, max_len).unsqueeze(1).unsqueeze(2)

        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))

        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0)
        return self.dropout(embeded_words + pe)

    @property
    def device(self) -> torch.device:
        """ Determine which device to place the Tensors upon, CPU or GPU.
        """
        # print(self.model_embeddings.source.weight.device)
        return self.model_embeddings.source.weight.device