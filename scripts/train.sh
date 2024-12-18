#!/bin/sh

## set --enc True if the data labels are RGB
python3 -m train \
        --lr 0.001 \
        --epochs 1 \
        --batch_size 16 \
        --enc False \
        --res 256 \
        --data_path ../aug_pre \
        --model_path pre_trained/unet.keras \
