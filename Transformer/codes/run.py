#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Naval Fate.

Usage:
    run.py
    run.py train --train-src=<file> --train-tgt=<file> --dev-src=<file> --dev-tgt=<file> --vocab=<file> [options]
    run.py decode [options] MODEL_PATH TEST_SOURCE_FILE OUTPUT_FILE
    run.py decode [options] MODEL_PATH TEST_SOURCE_FILE TEST_TARGET_FILE OUTPUT_FILE

Options:
    -h --help                               show this screen.
    --cuda                                  use GPU
    --train-src=<file>                      train source file
    --train-tgt=<file>                      train target file
    --dev-src=<file>                        dev source file
    --dev-tgt=<file>                        dev target file
    --vocab=<file>                          vocab file
    --seed=<int>                            seed [default: 42]
    --batch-size=<int>                      batch size [default: 32]
    --embed-size=<int>                      embedding size [default: 256]
    --d_model=<int>                         d_model size [default: 512]
    --d_k=<int>                             d_k size [default: 64]
    --d_v=<int>                             d_v size [default: 64]
    --d_ff=<int>                            d_ff size [default: 2048]
    --check-src=<file>                      coding check source file
    --check-tgt=<file>                      coding check target file
    --clip-grad=<float>                     gradient clipping [default: 5.0]
    --save-to=<file>                        model save path [default: ../model.bin]
    --dropout=<float>                       dropout [default: 0.1]
"""
from docopt import docopt
import torch
import torch.nn as nn
import numpy as np
from vocab import Vocab, VocabEntry
from transformer_model import TransformerModel

import utils

def train(args):
    """
        Train the NMT Model.
        @param args (Dict): args from cmd line
    """

    ### tokenize each sentence, (# sentences, max_num_words), each item is a word or a punctuation
    # train_data_src = utils.read_corpus(args['--train-src'], source='src')
    # train_data_tgt = utils.read_corpus(args['--train-tgt'], source='tgt')
    # dev_data_src = utils.read_corpus(args['--dev-src'], source='src')
    # dev_data_tgt = utils.read_corpus(args['--dev-tgt'], source='tgt')
    print(args['--check-src'])
    checking_data_src = utils.read_corpus(args['--check-src'], source='src')
    checking_data_tgt = utils.read_corpus(args['--check-tgt'], source='tgt')

    ### [(train_data[0],dev_data[0]),(train_data[1],dev_data[1]),...]
    # train_data = list(zip(train_data_src, train_data_tgt))
    # dev_data = list(zip(dev_data_src, dev_data_tgt))
    checking_data = list(zip(checking_data_src, checking_data_tgt))

    train_batch_size = int(args['--batch-size'])
    model_save_path = args['--save-to']
    vocab = Vocab.load(args['--vocab'])

    # model = NMT(embed_size=int(args['--embed-size']),
    #             hidden_size=int(args['--hidden-size']),
    #             dropout_rate=float(args['--dropout']),
    #             vocab=vocab)
    device = torch.device("cuda:0" if args['--cuda'] else "cpu")
    model = TransformerModel(vocab=vocab,
                             d_model=int(args['--d_model']),
                             d_k=int(args['--d_k']),
                             d_v=int(args['--d_v']),
                             d_ff=int(args['--d_ff']),
                             dropout_rate=float(args['--dropout']))
    # print(model.model_embeddings.source)


    model = model.to(device)

    model.forward(checking_data_src,checking_data_tgt)

    epoch = 0
    while True:
        epoch += 1
        for src_sents, tgt_sents in utils.batch_iter(checking_data, batch_size=train_batch_size, shuffle=True):
            print('Test result')
            print(len(src_sents),len(src_sents[0]))
            example_losses = model(src_sents, tgt_sents)  # (batch_size,)
        break
    pass

def main():
    """
        Main func.
    """
    args = docopt(__doc__) # explaination on docopt: https://wp-lai.gitbooks.io/learn-python/content/0MOOC/docopt.html
    # print(args)
    seed = int(args['--seed']) # 42 by default, which is the answer to the Universe
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed * 13 // 7)

    train(args)

if __name__ == '__main__':
    main()