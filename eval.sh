#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=3 python train_hdn.py \
    --resume_training --resume_model ~/tmp/MSDN/pretrained_models/h.h5 \
    --dataset_option=normal  --MPS_iter=1 \
    --caption_use_bias --caption_use_dropout \
    --rnn_type LSTM_normal --evaluate
