#!/bin/sh

python3 -m test \
        --data_path ../aug_pre/ \
        --results_path results/\
        --model_path pre_trained/unet.keras \
        --pre_path data/pre.pkl



