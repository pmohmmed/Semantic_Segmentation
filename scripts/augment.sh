#!/bin/sh

python3 -m augment \
        --data_path '../dataset/IP-Dataset/train_data' \
        --save_path data_aug/ \
        --pre True \
        --res 256 \


