#!/bin/sh

python3 -m test \
        --data_path ../aug_pre/test_data \
        --results_path ../results\
        --model_path incep_t5.keras \
        --show_results False



