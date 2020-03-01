#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
from utils import read_corpus, pad_sents
import torch
import torch.nn as nn

class VocabEntry(object):
    """ Vocabulary Entry, i.e. structure containing either
    src or tgt language terms.
    """
    def __init__(self, word2id=None):
        """ Init VocabEntry Instance.
        @param word2id (dict): dictionary mapping words 2 indices
        """
        if word2id:
            self.word2id = word2id
        else:
            self.word2id = dict()
            self.word2id['<pad>'] = 0   # Pad Token
            self.word2id['<s>'] = 1 # Start Token
            self.word2id['</s>'] = 2    # End Token
            self.word2id['<unk>'] = 3   # Unknown Token
        self.unk_id = self.word2id['<unk>']
        self.id2word = {v: k for k, v in self.word2id.items()}

    def __contains__(self, word):
        """ Check if word is captured by VocabEntry.
        @param word (str): word to look up
        @returns contains (bool): whether word is contained
        """
        return word in self.word2id

    def __getitem__(self, word):
        """ Retrieve word's index. Return the index for the unk
        token if the word is out of vocabulary.
        @param word (str): word to look up.
        @returns index (int): index of word
        """
        return self.word2id.get(word, self.unk_id)

    def __len__(self):
        """ Compute number of words in VocabEntry.
        @returns len (int): number of words in VocabEntry
        """
        return len(self.word2id)

    def id2word(self, wid):
        """ Return mapping of index to word.
        @param wid (int): word index
        @returns word (str): word corresponding to index
        """
        return self.id2word[wid]

    def words2indices(self, sents):
        """ Convert list of words or list of sentences of words
        into list or list of list of indices.
        @param sents (list[str] or list[list[str]]): sentence(s) in words
        @return word_ids (list[int] or list[list[int]]): sentence(s) in indices
        """
        if type(sents[0]) == list:
            return [[self[w] for w in s] for s in sents]
        else:
            return [self[w] for w in sents]

    def to_input_tensor(self, sents, device):
        """ Convert list of sentences (words) into tensor with necessary padding for
        shorter sentences.

        @param sents (List[List[str]]): list of sentences (words)
        @param device: device on which to load the tesnor, i.e. CPU or GPU

        @returns sents_var: tensor of (max_sentence_length, batch_size)
        """
        word_ids = self.words2indices(sents)
        sents_t = pad_sents(word_ids, self['<pad>'])
        sents_var = torch.tensor(sents_t, dtype=torch.long, device=device)
        return torch.t(sents_var)

class Vocab(object):
    """
        Vocab encapsulating src and target langauges.
    """
    def __init__(self, src_vocab: VocabEntry, tgt_vocab: VocabEntry):
        """ Init Vocab.
        @param src_vocab (VocabEntry): VocabEntry for source language
        @param tgt_vocab (VocabEntry): VocabEntry for target language
        """
        self.src = src_vocab
        self.tgt = tgt_vocab
        # print(self.src['<pad>'])

    @staticmethod # 使用@staticmethod或@classmethod，就可以不需要实例化，直接类名.方法名()来调用。
    def load(file_path):
        """ Load vocabulary from txt file.
        @param file_path (str): file path to vocab file
        @returns Vocab object loaded from txt file
        """
        f = open(file_path, "r") # in our model, src_vocab == tgt_vocab
        word2id = dict()
        word2id['<pad>'] = 0  # Pad Token
        word2id['<s>'] = 1  # Start Token
        word2id['</s>'] = 2  # End Token
        word2id['<unk>'] = 3  # Unknown Token
        for i, x in enumerate(f):
            word2id[x[:-1]] = i+4
        # print(word2id)
        src_word2id = copy.deepcopy(word2id)
        tgt_word2id = copy.deepcopy(word2id)
        return Vocab(VocabEntry(src_word2id), VocabEntry(tgt_word2id))

