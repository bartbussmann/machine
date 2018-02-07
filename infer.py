import os
import argparse
import logging

import torch

import seq2seq
from seq2seq.evaluator import Predictor
from seq2seq.util.checkpoint import Checkpoint
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

try:
    raw_input          # Python 2
except NameError:
    raw_input = input  # Python 3

parser = argparse.ArgumentParser()

parser.add_argument('--checkpoint_path', help='Give the checkpoint path from which to load the model')
parser.add_argument('--cuda_device', default=0, type=int, help='set cuda device to use')

opt = parser.parse_args()

if torch.cuda.is_available():
        print("Cuda device set to %i" % opt.cuda_device)
        torch.cuda.set_device(opt.cuda_device)

#################################################################################
# load model

logging.info("loading checkpoint from {}".format(os.path.join(opt.checkpoint_path)))
checkpoint = Checkpoint.load(opt.checkpoint_path)
seq2seq = checkpoint.model
input_vocab = checkpoint.input_vocab
output_vocab = checkpoint.output_vocab

#################################################################################
# Generate predictor

predictor = Predictor(seq2seq, input_vocab, output_vocab)

while True:
        seq_str = raw_input("Type in a source sequence:")
        seq = seq_str.strip().split()
        prediction, attentions = predictor.predict(seq)
        print(prediction)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(attentions.data.numpy(), cmap='bone')
        fig.colorbar(cax)

        # Set up axes
        ax.set_xticklabels([''] + seq +
                           ['<EOS>'], rotation=90)
        ax.set_yticklabels([''] + prediction)

        # Show label at every tick
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
        plt.show()
