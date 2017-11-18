#! /bin/sh

TRAIN_PATH=data/toy_reverse/train/data.txt
DEV_PATH=data/toy_reverse/dev/data.txt

# Start training
python ibm-seq2seq/scripts/integration_test.py --train_path $TRAIN_PATH --dev_path $DEV_PATH
# Resume training
python ibm-seq2seq/scripts/integration_test.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --resume
# Load checkpoint
python ibm-seq2seq/scripts/integration_test.py --train_path $TRAIN_PATH --dev_path $DEV_PATH \
	--load_checkpoint $(ls -t experiment/checkpoints/ | head -1)