#!/bin/sh

python3 -m test \
        --data_path ../inf_data \
        --results_path ../results\
        --model_path pre_trained/incep_mod.keras \
        --show_results False

