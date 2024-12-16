#!/bin/sh

python3 -m augment \
        --data_path ../aug_pre/ \
        --aug_data_path ./data_aug/ \
        --pre_obj_path ./data/pre.pkl \
        --pre True \
        --res 256 \


