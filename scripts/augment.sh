#!/bin/sh

python3 -m augment \
        --data_path ../uavid_512/train_data \
        --aug_data_path ./aug_enc_256/ \
        --pre True \
        --res 256 \


