#! /bin/sh

TRAIN_PATH=data/CLEANED-SCAN/simple_split/train.txt
DEV_PATH=data/CLEANED-SCAN/simple_split/dev.txt
EXPT_DIR=variance_experiment1_gru
ATTENTION=true
CUDA=0

# set values
EPOCHS=70
OPTIMIZER='adam'
LR=0.001
RNN_CELL='gru'
EMB_SIZE=200
H_SIZE=200
N_LAYERS=2
DROPOUT_ENCODER=0.5
DROPOUT_DECODER=0.5
TF=0.5
BATCH_SIZE=128
BIDIRECTIONAL=false
PRINT_EVERY=20
SAVE_EVERY=133 #Batches per epoch (print steps_per_epoch in supervised_trainer.py to find out)

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
    $( (( $BIDIRECTIONAL )) && echo '--bidirectional' ) \
    $( (( $ATTENTION )) && echo '--attention' ) \
    --print_every $PRINT_EVERY \
    --save_every $SAVE_EVERY \
    --log-level 'debug' \
--cuda_device $CUDA
