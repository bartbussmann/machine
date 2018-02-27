import os
import argparse
import logging

import torch
import torchtext

import seq2seq
from seq2seq.loss import Perplexity
from seq2seq.dataset import SourceField, TargetField
from seq2seq.evaluator import Evaluator
from seq2seq.util.checkpoint import Checkpoint

import matplotlib.pyplot as plt

for max_train_length in [1,2,3,4,5,6]:
    top_checkpoints_dir = os.path.join('checkpoints', 'checkpoints_experiment_2c_{}'.format(max_train_length))
    checkpoint_dirs = [os.path.join(top_checkpoints_dir, subdir) for subdir in os.listdir(
        top_checkpoints_dir) if os.path.isdir(os.path.join(top_checkpoints_dir, subdir))]

    test_lengths = [tl for tl in [9, 10, 19, 20, 30, 32, 33, 36, 40, 48] if tl > max_train_length]
    checkpoint_i = 0
    sum_test_acc = {}

    for checkpoint_dir in checkpoint_dirs:
        title = 'experiment_2c_{}'.format(max_train_length, checkpoint_i)

        logging.info("loading checkpoint from {}".format(os.path.join(checkpoint_dir)))
        checkpoint = Checkpoint.load(checkpoint_dir)
        seq2seq = checkpoint.model
        input_vocab = checkpoint.input_vocab
        output_vocab = checkpoint.output_vocab

        for test_length in test_lengths:
            ############################################################################
            # Prepare dataset and loss
            src = SourceField()
            tgt = TargetField()
            src.vocab = input_vocab
            tgt.vocab = output_vocab
            max_len = 50


            def len_filter(example):
                return len(example.src) <= max_len and len(example.tgt) <= max_len


            # generate test set
            test = torchtext.data.TabularDataset(
                path=os.path.join('data', 'CLEANED-SCAN', 'length_split', 'single_lengths', str(test_length),
                                  'tasks_test.txt'), format='tsv',
                fields=[('src', src), ('tgt', tgt)],
                filter_pred=len_filter
            )

            # Prepare loss
            weight = torch.ones(len(output_vocab))
            pad = output_vocab.stoi[tgt.pad_token]
            loss = Perplexity(weight, pad)
            if torch.cuda.is_available():
                loss.cuda()

            #################################################################################
            # Evaluate model on test set

            evaluator = Evaluator(loss=loss, batch_size=128)
            loss, accuracy, seq_accuracy = evaluator.evaluate(seq2seq, test)

            print("Loss: %f, Word accuracy: %f, Sequence accuracy: %f" % (loss, accuracy, seq_accuracy))

            plt.bar(str(test_length), seq_accuracy, color='blue')

            if test_length not in sum_test_acc:
                sum_test_acc[test_length] = seq_accuracy
            else:
                sum_test_acc[test_length] += seq_accuracy

        plt.ylim(0, 1)
        #plt.xticks(range(len(test_lengths)))
        plt.title(title)
        plt.savefig(title)
        plt.clf()
        checkpoint_i += 1

    title = 'avg_seq_accs_run_{}'.format(max_train_length)
    for key, val in sum_test_acc.items():
        plt.bar(str(key), float(val) / checkpoint_i, color='blue')
    plt.ylim(0, 1)
    plt.xticks(range(len(test_lengths)))
    plt.title(title)
    plt.savefig(title)
    plt.clf()