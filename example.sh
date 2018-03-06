#! /bin/sh

nvidia-smi -ac 3004,875 -i 0
nvidia-smi -ac 3004,875 -i 1

TRAIN_PATH=/home/bart/Documents/thesis/machine/data/CLEANED-SCAN/length_split/experiment5/train.txt
DEV_PATH=/home/bart/Documents/thesis/machine/data/CLEANED-SCAN/length_split/experiment5/test.txt
EXPT_DIR=experiments/variance_experiment_short_only_all_words_bidirectional
CUDA=0

# set values
EPOCHS=70
OPTIMIZER='adam'
LR=0.001
RNN_CELL='lstm'
EMB_SIZE=200
H_SIZE=200
N_LAYERS=2
DROPOUT_ENCODER=0.5
DROPOUT_DECODER=0.5
TF=0.5
BATCH_SIZE=128
PRINT_EVERY=20
SAVE_EVERY=100 #Batches per epoch (print steps_per_epoch in supervised_trainer.py to find out)
REG_SCALE=1000
ATTENTION='pre-rnn'
ATTENTION_METHOD='mlp'

# Start training
echo "Train model on example data"
python train_model.py \
    --train $TRAIN_PATH \
    --dev $DEV_PATH \
    --output_dir $EXPT_DIR \
    --epochs $EPOCHS \
    --optim $OPTIMIZER \
    --lr $LR \
    --rnn_cell $RNN_CELL \
    --embedding_size $EMB_SIZE \
    --hidden_size $H_SIZE \
    --n_layers $N_LAYERS \
    --dropout_p_encoder $DROPOUT_ENCODER \
    --dropout_p_decoder $DROPOUT_DECODER \
    --teacher_forcing_ratio $TF \
    --batch_size $BATCH_SIZE \
    --attention $ATTENTION\
    --attention_method $ATTENTION_METHOD \
    --bidirectional \
    --print_every $PRINT_EVERY \
    --save_every $SAVE_EVERY \
    --log-level 'debug' \
    --cuda_device $CUDA \
    --reg_scale $REG_SCALE\
