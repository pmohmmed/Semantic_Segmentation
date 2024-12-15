#!/bin/sh

python3 -m train\
        --lr 0.005 \
        --epochs 1 \
        --enc False \
        --res 256 \
        --data_path ../data_ip/ \
        --model_path pre_trained/unet \
        --pre_path data/pre.pkl


