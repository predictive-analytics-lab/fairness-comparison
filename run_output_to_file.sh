#!/bin/bash

# this command puts the output into the file "log.txt"
python -u benchmark.py 0 \
       --num-trials 4 \
       --datasets '["adult"]' \
	> log_adult_4_trials_vary_both_precision_targets_no_lr_drop_1.0-0.2.txt 2>&1 &
#        --datasets '["propublica-recidivism"]' \
# python -u benchmark.py 0 --num-trials 5  --datasets '["two-gaussians"]'  > log_two_gauss.txt 2>&1 &
# python -u benchmark.py 1 --num-trials 5  --datasets '["ricci"]'  > log_ricci.txt 2>&1 &
# python -u benchmark.py 2 --num-trials 5  --datasets '["adult"]'  > log_adult.txt 2>&1 &
# python -u benchmark.py 3 --num-trials 5  --datasets '["german"]'  > log_german.txt 2>&1 &
# python -u benchmark.py 4 --num-trials 5  --datasets '["propublica-recidivism"]'  > log_pp.txt 2>&1 &
# python -u benchmark.py 5 --num-trials 5  --datasets '["propublica-violent-recidivism"]'  > log_pp_violent.txt 2>&1 &
# python -u benchmark.py 6 --num-trials 5  --datasets '["flipped-labels"]'  > log_flipped.txt 2>&1 &
