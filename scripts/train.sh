#!/bin/sh

python3 -m train\
        --lr 0.005 \
        --epochs 1 \
        --enc False \
        --res 256 \
        --data_path ../aug_pre/ \
        --model_path pre_trained/unet.keras \
        --pre_path data/pre.pkl


