#!/bin/sh

python3 -m test \
        --data_path ../aug_pre/ \
        --results_path ./results/\
        --model_path ./unet.keras \
        --pre_obj_path data/pre.pkl \
        --show_results False



