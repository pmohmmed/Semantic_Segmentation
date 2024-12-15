#!/bin/sh

python3 -m test \
        --data_path ../data_ip/ \
        --results_path results/\
        --model_path pre_trained/unet \
        --pre_path data/pre.pkl



