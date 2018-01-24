#!/bin/bash
cd /home/ebruni/metaphora-detection/OpenNMT
python train.py -data ../new-data/demo-train.pt -save_model model -cuda -rnn_size 200 -word_vec_size 200
echo Done

