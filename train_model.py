import os
import argparse
import logging

import torch
from torch.optim.lr_scheduler import StepLR
import torchtext

import seq2seq
from seq2seq.trainer import SupervisedTrainer
from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq
from seq2seq.loss import Perplexity
from seq2seq.optim import Optimizer
from seq2seq.dataset import SourceField, TargetField
from seq2seq.evaluator import Predictor, Evaluator
from seq2seq.util.checkpoint import Checkpoint

try:
    raw_input          # Python 2
except NameError:
    raw_input = input  # Python 3

parser = argparse.ArgumentParser()
parser.add_argument('--train', help='Training data')
parser.add_argument('--dev', help='Development data')
parser.add_argument('--output_dir', default='../models', help='Path to model directory. If load_checkpoint is True, then path to checkpoint directory has to be provided')
parser.add_argument('--epochs', type=int, help='Number of epochs', default=6)
parser.add_argument('--optim', type=str, help='Choose optimizer', choices=['adam', 'adadelta', 'adagrad', 'adamax', 'rmsprop', 'sgd'])
parser.add_argument('--max_len', type=int, help='Maximum sequence length', default=50)
parser.add_argument('--rnn_cell', help="Chose type of rnn cell", default='lstm')
parser.add_argument('--bidirectional', action='store_true', help="Flag for bidirectional encoder")
parser.add_argument('--embedding_size', type=int, help='Embedding size', default=128)
parser.add_argument('--hidden_size', type=int, help='Hidden layer size', default=128)
parser.add_argument('--n_layers', type=int, help='Number of RNN layers in both encoder and decoder', default=1)
parser.add_argument('--src_vocab', type=int, help='source vocabulary size', default=50000)
parser.add_argument('--tgt_vocab', type=int, help='target vocabulary size', default=50000)
parser.add_argument('--dropout_p_encoder', type=float, help='Dropout probability for the encoder', default=0.2)
parser.add_argument('--dropout_p_decoder', type=float, help='Dropout probability for the decoder', default=0.2)
parser.add_argument('--teacher_forcing_ratio', type=float, help='Teacher forcing ratio', default=0.2)
parser.add_argument('--attention', choices=['pre-rnn', 'post-rnn'], default=False)
parser.add_argument('--attention_method', choices=['dot', 'mlp', 'concat'], default=None)
parser.add_argument('--batch_size', type=int, help='Batch size', default=32)
parser.add_argument('--lr', type=float, help='Learning rate, recommended settings.\nrecommended settings: adam=0.001 adadelta=1.0 adamax=0.002 rmsprop=0.01 sgd=0.1', default=0.001)
parser.add_argument('--reg_scale', type=int, help='Scaling factor for regularization', default=1000) 

parser.add_argument('--load_checkpoint', help='The name of the checkpoint to load, usually an encoded time string')
parser.add_argument('--save_every', type=int, help='Every how many batches the model should be saved', default=100)
parser.add_argument('--print_every', type=int, help='Every how many batches to print results', default=100)
parser.add_argument('--resume', action='store_true', help='Indicates if training has to be resumed from the latest checkpoint')
parser.add_argument('--log-level', default='info', help='Logging level.')
parser.add_argument('--cuda_device', default=0, type=int, help='set cuda device to use')

opt = parser.parse_args()

if opt.resume and not opt.load_checkpoint:
    parser.error('load_checkpoint argument is required to resume training from checkpoint')

LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, opt.log_level.upper()))
logging.info(opt)

if torch.cuda.is_available():
        print("Cuda device set to %i" % opt.cuda_device)
        torch.cuda.set_device(opt.cuda_device)

if opt.attention:
    if not opt.attention_method:
        opt.attention_method = 'dot'

############################################################################
# Prepare dataset
src = SourceField()
tgt = TargetField()
max_len = opt.max_len

def len_filter(example):
    return len(example.src) <= max_len and len(example.tgt) <= max_len

# generate training and testing data
train = torchtext.data.TabularDataset(
    path=opt.train, format='tsv',
    fields=[('src', src), ('tgt', tgt)],
    filter_pred=len_filter
)

if opt.dev:
    dev = torchtext.data.TabularDataset(
        path=opt.dev, format='tsv',
        fields=[('src', src), ('tgt', tgt)],
        filter_pred=len_filter
    )
else:
    dev = None

#################################################################################
# prepare model

if opt.load_checkpoint is not None:
    logging.info("loading checkpoint from {}".format(os.path.join(opt.output_dir, opt.load_checkpoint)))
    checkpoint_path = os.path.join(opt.output_dir, opt.load_checkpoint)
    checkpoint = Checkpoint.load(checkpoint_path)
    seq2seq = checkpoint.model
    input_vocab = checkpoint.input_vocab
    output_vocab = checkpoint.output_vocab
    src.vocab = input_vocab
    tgt.vocab = output_vocab
else:
    # build vocabulary
    src.build_vocab(train, max_size=opt.src_vocab)
    tgt.build_vocab(train, max_size=opt.tgt_vocab)
    input_vocab = src.vocab
    output_vocab = tgt.vocab

    # Initialize model
    hidden_size = opt.hidden_size
    decoder_hidden_size = hidden_size*2 if opt.bidirectional else hidden_size
    encoder = EncoderRNN(len(src.vocab), max_len, hidden_size,
                         opt.embedding_size,
                         dropout_p=opt.dropout_p_encoder,
                         n_layers=opt.n_layers,
                         bidirectional=opt.bidirectional,
                         rnn_cell=opt.rnn_cell,
                         variable_lengths=True)
    decoder = DecoderRNN(len(tgt.vocab), max_len, decoder_hidden_size,
                         dropout_p=opt.dropout_p_decoder,
                         n_layers=opt.n_layers,
                         use_attention=opt.attention,
                         attention_method=opt.attention_method,
                         bidirectional=opt.bidirectional,
                         rnn_cell=opt.rnn_cell,
                         eos_id=tgt.eos_id, sos_id=tgt.sos_id)
    seq2seq = Seq2seq(encoder, decoder)
    if torch.cuda.is_available():
        seq2seq.cuda()

    for param in seq2seq.parameters():
        param.data.uniform_(-0.08, 0.08)

##############################################################################
# train model

# Prepare loss
weight = torch.ones(len(output_vocab))
pad = output_vocab.stoi[tgt.pad_token]
loss = Perplexity(weight, pad)
if torch.cuda.is_available():
    loss.cuda()

# create trainer
t = SupervisedTrainer(loss=loss, batch_size=opt.batch_size,
                      checkpoint_every=opt.save_every,
                      print_every=opt.print_every, expt_dir=opt.output_dir)

checkpoint_path = os.path.join(opt.output_dir, opt.load_checkpoint) if opt.resume else None

seq2seq = t.train(seq2seq, train,
                  num_epochs=opt.epochs, dev_data=dev,
                  optimizer=opt.optim,
                  teacher_forcing_ratio=opt.teacher_forcing_ratio,
                  learning_rate=opt.lr,
                  resume=opt.resume,
		  reg_scale=opt.reg_scale,
                  checkpoint_path=checkpoint_path,
	          top_k=1)

# evaluator = Evaluator(loss=loss, batch_size=opt.batch_size)
# dev_loss, accuracy = evaluator.evaluate(seq2seq, dev)
