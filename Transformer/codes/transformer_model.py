#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils
import math
import sys
import torch.nn.functional as F
Hypothesis = namedtuple('Hypothesis', ['value', 'score'])

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from model_embeddings import ModelEmbeddings
from EncoderDecoder import Encoder, Decoder

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
        self.decoder = Decoder()
        self.outputlayer = Generator(d_model, len(vocab.tgt))
        self.crit = LabelSmoothing(size=len(vocab.tgt), smoothing=0.1)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate

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
        # tensor of int indicating words index in vocab
        source_padded = self.vocab.src.to_input_tensor(source, device=self.device)   # Tensor int: (src_len, b)
        target_padded = self.vocab.tgt.to_input_tensor(target, device=self.device)   # Tensor int: (tgt_len, b)

        target_masks = (target_padded != self.vocab.tgt['<pad>']).float()


        # (src_len: sentence max # words in a batch, b: batch size, d_model: embedding size)
        X = self.model_embeddings.source.forward(source_padded)
        X = self.PositionalEncoding(X)
        X, K, V = self.encoder(X)

        # Chop of the <END> token for max length sentences.
        Y = self.model_embeddings.target.forward(target_padded[:-1])
        Y = self.PositionalEncoding(Y)
        mask = self.subsequent_mask(Y.size()[0]) # (1, tgt_len, tgt_len)
        Y = self.decoder(Y, K, V, mask) # (tgt_len, batch size, d_model: embedding size)

        P = self.outputlayer(Y) # (tgt_len, batch size, vocab size)

        # # Compute log probability of generating true target words
        ## target_gold_words_log_prob = torch.gather(P, index=target_padded[1:].unsqueeze(-1), dim=-1).squeeze(-1) * target_masks[1:]
        ## scores = target_gold_words_log_prob.sum(dim=0)
        # print(target_gold_words_log_prob.sum(dim=0))

        # Label Smoothing
        scores = self.crit(P,target_padded[1:])
        return scores


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

    def subsequent_mask(self,size):
        "Mask out subsequent positions."
        # size is the length of the longest sentence in a batch
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        # label all position in QxK^T needed to mask to be True
        mask = torch.from_numpy(subsequent_mask).to(self.device) == 1
        return mask

    def beam_search(self, src_sent, beam_size, max_decoding_time_step):
        """ Given a single source sentence, perform beam search, yielding translations in the target language.
        @param src_sent (List[str]): a single source sentence (words)
        @param beam_size (int): beam size
        @param max_decoding_time_step (int): maximum number of time steps to unroll the decoding RNN
        @returns hypotheses (List[Hypothesis]): a list of hypothesis, each hypothesis has two fields:
                value: List[str]: the decoded target sentence, represented as a list of words
                score: float: the log-likelihood of the target sentence
        """

        src_sents_var = self.vocab.src.to_input_tensor([src_sent], self.device)

        X = self.model_embeddings.source.forward(src_sents_var)
        X = self.PositionalEncoding(X)
        X, K, V = self.encoder(X)

        hypotheses = [['<s>']]
        hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=self.device)
        completed_hypotheses = []

        t = 0
        while len(completed_hypotheses) < beam_size and t < max_decoding_time_step:
            t += 1
            hyp_num = len(hypotheses)

            tgt_sents_var = self.vocab.src.to_input_tensor(hypotheses, self.device)
            Y = self.model_embeddings.target.forward(tgt_sents_var)
            Y = self.PositionalEncoding(Y)
            mask = self.subsequent_mask(Y.size()[0])  # (1, tgt_len, tgt_len)
            Y = self.decoder(Y, K, V, mask)  # (tgt_len, batch size, d_model: embedding size)
            P = self.outputlayer(Y)  # (tgt_len, batch size, vocab size)

            live_hyp_num = beam_size - len(completed_hypotheses)
            log_p_t = P[-1,:,:] # only check for the last word
            contiuating_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(log_p_t) + log_p_t).view(-1)
            top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(contiuating_hyp_scores, k=live_hyp_num)

            prev_hyp_ids = top_cand_hyp_pos / len(self.vocab.tgt)
            hyp_word_ids = top_cand_hyp_pos % len(self.vocab.tgt)

            new_hypotheses = []
            live_hyp_ids = []
            new_hyp_scores = []

            for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores):
                prev_hyp_id = prev_hyp_id.item()
                hyp_word_id = hyp_word_id.item()
                cand_new_hyp_score = cand_new_hyp_score.item()

                hyp_word = self.vocab.tgt.id2word[hyp_word_id]
                new_hyp_sent = hypotheses[prev_hyp_id] + [hyp_word]

                if hyp_word == '</s>':
                    completed_hypotheses.append(Hypothesis(value=new_hyp_sent[1:-1],
                                                           score=cand_new_hyp_score))
                else:
                    new_hypotheses.append(new_hyp_sent)
                    live_hyp_ids.append(prev_hyp_id)
                    new_hyp_scores.append(cand_new_hyp_score)

            if len(completed_hypotheses) == beam_size:
                break

            live_hyp_ids = torch.tensor(live_hyp_ids, dtype=torch.long, device=self.device)
            hypotheses = new_hypotheses
            hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=self.device)

        if len(completed_hypotheses) == 0:
            completed_hypotheses.append(Hypothesis(value=hypotheses[0][1:],
                                                   score=hyp_scores[0].item()))

        completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)
        return completed_hypotheses

    @property
    def device(self) -> torch.device:
        """ Determine which device to place the Tensors upon, CPU or GPU.
        """
        # print(self.model_embeddings.source.weight.device)
        return self.model_embeddings.source.weight.device

    @staticmethod
    def load(model_path: str):
        """ Load the model from a file.
        @param model_path (str): path to model
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = TransformerModel(vocab=params['vocab'], **args)
        model.load_state_dict(params['state_dict'])

        return model

    def save(self, path: str):
        """ Save the odel to a file.
        @param path (str): path to the model
        """
        print('save model parameters to [%s]' % path, file=sys.stderr)

        params = {
            'args': dict(d_model=self.d_model, d_k=self.d_k, d_v=self.d_v, d_ff = self.d_ff, dropout_rate=self.dropout_rate),
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }

        torch.save(params, path)


class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

class LabelSmoothing(nn.Module):
    "Implement label smoothing. size表示类别总数 "

    def __init__(self, size, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        #self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing #if i=y的公式
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        """
        x表示输入 (N，M)N个样本，M表示总类数，每一个类的概率log P
        target表示label（M，）
        """
        assert x.size(-1) == self.size
        true_dist = x.data.clone()#先深复制过来以借用他的形状
        #print true_dist
        true_dist.fill_(self.smoothing / (self.size - 1))#otherwise的公式
        #print true_dist
        #变成one-hot编码，1表示按列填充，
        #target.data.unsqueeze(1)表示索引,confidence表示填充的数字
        true_dist.scatter_(2, target.data.unsqueeze(2), self.confidence)
        self.true_dist = true_dist
        return self.criterion(x, torch.autograd.Variable(true_dist, requires_grad=False))