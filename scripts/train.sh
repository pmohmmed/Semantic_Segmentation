#!/bin/sh

python3 -m train \
        --lr 0.0005 \
        --epochs 1 \
        --enc False \
        --res 256 \
        --data_path ../aug_pre \
        --model_path ./unet.keras \
        --pre_obj_path ./data/pre.pkl
