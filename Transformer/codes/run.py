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
    --clip-grad=<float>                     gradient clipping [default: 5.0]
    --save-to=<file>                        model save path [default: ../model.bin]
    --dropout=<float>                       dropout [default: 0.1]
    --valid-niter=<int>                     perform validation after how many iterations [default: 2000]
    --max-num-trial=<int>                   terminate training after how many trials [default: 5]
    --lr-decay=<float>                      learning rate decay [default: 0.5]
    --beam-size=<int>                       beam size [default: 5]
    --patience=<int>                        wait for how many iterations to decay learning rate [default: 5]
    --lr=<float>                            learning rate [default: 0.001]
    --log-every=<int>                       log every [default: 10130]
    --max-epoch=<int>                       max epoch [default: 30]
    --max-decoding-time-step=<int>          maximum number of decoding time steps [default: 200]
"""
from docopt import docopt
import torch
import torch.nn as nn
import numpy as np
from vocab import Vocab, VocabEntry
from nltk.translate.bleu_score import corpus_bleu
from transformer_model import TransformerModel
import time
import math
import utils
import sys
from tqdm import tqdm


def train(args):
    """
        Train the NMT Model.
        @param args (Dict): args from cmd line
    """

    ### tokenize each sentence, (# sentences, max_num_words), each item is a word or a punctuation
    train_data_src = utils.read_corpus(args['--train-src'], source='src')
    train_data_tgt = utils.read_corpus(args['--train-tgt'], source='tgt')
    dev_data_src = utils.read_corpus(args['--dev-src'], source='src')
    dev_data_tgt = utils.read_corpus(args['--dev-tgt'], source='tgt')
    # print(args['--check-src'])
    # checking_data_src = utils.read_corpus(args['--check-src'], source='src')
    # checking_data_tgt = utils.read_corpus(args['--check-tgt'], source='tgt')

    ### [(train_data[0],dev_data[0]),(train_data[1],dev_data[1]),...]
    train_data = list(zip(train_data_src, train_data_tgt))
    dev_data = list(zip(dev_data_src, dev_data_tgt))
    # checking_data = list(zip(checking_data_src, checking_data_tgt))

    train_batch_size = int(args['--batch-size'])
    clip_grad = float(args['--clip-grad'])
    valid_niter = int(args['--valid-niter'])
    log_every = int(args['--log-every'])
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

    model.train()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(args['--lr']))

    num_trial = 0
    train_iter = patience = cum_loss = report_loss = cum_tgt_words = report_tgt_words = 0
    cum_examples = report_examples = epoch = valid_num = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()
    print('begin Maximum Likelihood training')
    while True:
        epoch += 1
        # remenber to change back to train_data
        for src_sents, tgt_sents in utils.batch_iter(train_data, batch_size=train_batch_size, shuffle=True):
            train_iter += 1
            batch_size = len(src_sents)
            batch_loss = model(src_sents, tgt_sents)  # (batch_size,)
            # batch_loss = example_losses.sum()
            loss = batch_loss / batch_size

            loss.backward()

            # clip gradient
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            optimizer.step()

            batch_losses_val = batch_loss.item()
            report_loss += batch_losses_val
            cum_loss += batch_losses_val

            tgt_words_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
            report_tgt_words += tgt_words_num_to_predict
            cum_tgt_words += tgt_words_num_to_predict
            report_examples += batch_size
            cum_examples += batch_size

            if train_iter % log_every == 0:
                print('epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f ' \
                      'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' % (epoch, train_iter,
                                                                                         report_loss / report_examples,
                                                                                         math.exp(report_loss / report_tgt_words),
                                                                                         cum_examples,
                                                                                         report_tgt_words / (time.time() - train_time),
                                                                                         time.time() - begin_time), file=sys.stderr)

                train_time = time.time()
                report_loss = report_tgt_words = report_examples = 0.

                # perform validation
                if train_iter % valid_niter == 0:
                # if train_iter % 50 == 0:
                    print('epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (epoch, train_iter,
                                                                                                 cum_loss / cum_examples,
                                                                                                 np.exp(
                                                                                                     cum_loss / cum_tgt_words),
                                                                                                 cum_examples),
                          file=sys.stderr)

                    cum_loss = cum_examples = cum_tgt_words = 0.
                    valid_num += 1

                    print('begin validation ...', file=sys.stderr)

                    # compute dev. ppl and bleu
                    dev_ppl = evaluate_ppl(model, dev_data, batch_size=128)  # dev batch size can be a bit larger
                    valid_metric = -dev_ppl

                    print('validation: iter %d, dev. ppl %f' % (train_iter, dev_ppl), file=sys.stderr)

                    is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
                    hist_valid_scores.append(valid_metric)

                    if is_better:
                        patience = 0
                        print('save currently the best model to [%s]' % model_save_path, file=sys.stderr)
                        model.save(model_save_path)

                        # also save the optimizers' state
                        torch.save(optimizer22222222222t(), model_save_path + '.optim')
                    elif patience < int(args['--patience']):
                        patience += 1
                        print('hit patience %d' % patience, file=sys.stderr)

                        if patience == int(args['--patience']):
                            num_trial += 1
                            print('hit #%d trial' % num_trial, file=sys.stderr)
                            if num_trial == int(args['--max-num-trial']):
                                print('early stop!', file=sys.stderr)
                                exit(0)

                            # decay lr, and restore from previously best checkpoint
                            lr = optimizer.param_groups[0]['lr'] * float(args['--lr-decay'])
                            print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)

                            # load model
                            params = torch.load(model_save_path, map_location=lambda storage, loc: storage)
                            model.load_state_dict(params['state_dict'])
                            model = model.to(device)

                            print('restore parameters of the optimizers', file=sys.stderr)
                            optimizer.load_state_dict(torch.load(model_save_path + '.optim'))

                            # set new lr
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = lr

                            # reset patience
                            patience = 0

                    if epoch == int(args['--max-epoch']):
                        print('reached maximum number of epochs!', file=sys.stderr)
                        exit(0)


def evaluate_ppl(model, dev_data, batch_size=32):
    """ Evaluate perplexity on dev sentences
    @param model (NMT): NMT Model
    @param dev_data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
    @param batch_size (batch size)
    @returns ppl (perplixty on dev sentences)
    """
    was_training = model.training
    model.eval()

    cum_loss = 0.
    cum_tgt_words = 0.

    # no_grad() signals backend to throw away all gradients
    with torch.no_grad():
        for src_sents, tgt_sents in utils.batch_iter(dev_data, batch_size):
            loss = -model(src_sents, tgt_sents).sum()
            cum_loss += loss.item()
            tgt_word_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # count total # words in target batch, omitting leading `<s>`
            cum_tgt_words += tgt_word_num_to_predict

        ppl = np.exp(cum_loss / cum_tgt_words) #average loss over each words in the target

    if was_training:
        model.train()

    return ppl

def compute_corpus_level_bleu_score(references, hypotheses):
    """ Given decoding results and reference sentences, compute corpus-level BLEU score.
    @param references (List[List[str]]): a list of gold-standard reference target sentences
    @param hypotheses (List[Hypothesis]): a list of hypotheses, one for each reference
    @returns bleu_score: corpus-level BLEU score
    """
    if references[0][0] == '<s>':
        references = [ref[1:-1] for ref in references]
    bleu_score = corpus_bleu([[ref] for ref in references],
                             [hyp.value for hyp in hypotheses])
    return bleu_score

def decode(args):
    """ Performs decoding on a test set, and save the best-scoring decoding results.
    If the target gold-standard sentences are given, the function also computes
    corpus-level BLEU score.
    @param args (Dict): args from cmd line
    """

    print("load test source sentences from [{}]".format(args['TEST_SOURCE_FILE']), file=sys.stderr)
    test_data_src = utils.read_corpus(args['TEST_SOURCE_FILE'], source='src')
    if args['TEST_TARGET_FILE']:
        print("load test target sentences from [{}]".format(args['TEST_TARGET_FILE']), file=sys.stderr)
        test_data_tgt = utils.read_corpus(args['TEST_TARGET_FILE'], source='tgt')

    print("load model from {}".format(args['MODEL_PATH']), file=sys.stderr)
    model = TransformerModel.load(args['MODEL_PATH'])

    if args['--cuda']:
        model = model.to(torch.device("cuda:0"))

    hypotheses = beam_search(model, test_data_src,
                             beam_size=int(args['--beam-size']),
                             max_decoding_time_step=int(args['--max-decoding-time-step']))

    if args['TEST_TARGET_FILE']:
        top_hypotheses = [hyps[0] for hyps in hypotheses]
        bleu_score = compute_corpus_level_bleu_score(test_data_tgt, top_hypotheses)
        print('Corpus BLEU: {}'.format(bleu_score * 100), file=sys.stderr)

    with open(args['OUTPUT_FILE'], 'w') as f:
        for src_sent, hyps in zip(test_data_src, hypotheses):
            top_hyp = hyps[0]
            hyp_sent = ' '.join(top_hyp.value)
            f.write(hyp_sent + '\n')

def beam_search(model, test_data_src, beam_size, max_decoding_time_step):
    """ Run beam search to construct hypotheses for a list of src-language sentences.
    @param model (NMT): NMT Model
    @param test_data_src (List[List[str]]): List of sentences (words) in source language, from test set.
    @param beam_size (int): beam_size (# of hypotheses to hold for a translation at every step)
    @param max_decoding_time_step (int): maximum sentence length that Beam search can produce
    @returns hypotheses (List[List[Hypothesis]]): List of Hypothesis translations for every source sentence.
    """
    was_training = model.training
    model.eval()

    hypotheses = []
    with torch.no_grad():
        for src_sent in tqdm(test_data_src, desc='Decoding', file=sys.stdout):
            example_hyps = model.beam_search(src_sent, beam_size=beam_size,
                                             max_decoding_time_step=max_decoding_time_step)

            hypotheses.append(example_hyps)

    if was_training: model.train(was_training)

    return hypotheses

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

    if args['train']:
        train(args)
    elif args['decode']:
        decode(args)
    else:
        raise RuntimeError('invalid run mode')

if __name__ == '__main__':
    main()