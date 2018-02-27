import os
import argparse
import logging

import torch

import seq2seq
import scipy
from scipy import stats
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

att_dict = {}

input_file = open('dev.txt', 'r')

for q, line in enumerate(input_file):
    if q > 1500:
        break

    line_stripped = line.strip()
    input_seq, _ = line_stripped.split('\t')
    input_seq = input_seq.strip().split()
    prediction, attentions = predictor.predict(input_seq)

    for i, output_word in enumerate(prediction):
        if output_word in att_dict:
            output_word_dict = att_dict[output_word]
        else:
            output_word_dict = {}
            att_dict[output_word] = output_word_dict
        for j, input_word in enumerate(input_seq):
            if input_word in output_word_dict:
                output_word_dict[input_word].append(attentions.data[i][j])
            else:
                output_word_dict[input_word] = [attentions.data[i][j]]

    coocurrences = np.array(
        [[np.mean(att_dict[output_word][input_word]) for input_word in sorted(att_dict[output_word])] for output_word in
         sorted(att_dict)])

#coocurrences_normed = coocurrences / coocurrences.sum(axis=0)
print(np.var(coocurrences, axis=0))
print(sum(np.var(coocurrences, axis=0)))

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(coocurrences, cmap='bone')
fig.colorbar(cax)

# Set up axes
ax.set_yticklabels([''] + sorted(list(att_dict.keys())))
ax.set_xticklabels([''] + sorted(list(att_dict[list(att_dict.keys())[0]].keys())))

# Show label at every tick
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
plt.show()

