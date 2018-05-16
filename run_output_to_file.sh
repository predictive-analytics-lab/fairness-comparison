#!/bin/bash

# this command puts the output into the file "log.txt"
python -u benchmark.py 0 \
	--num-trials 5 \
	--datasets '["two-gaussians", "ricci", "adult", "german", "propublica-recidivism", "propublica-violent-recidivism"]' \
	--algorithms '["UGP_in_True"]' \
	> log.txt 2>&1 &
# python -u benchmark.py 0 --num-trials 5  --datasets '["two-gaussians"]'  > log_two_gauss.txt 2>&1 &
# python -u benchmark.py 1 --num-trials 5  --datasets '["ricci"]'  > log_ricci.txt 2>&1 &
# python -u benchmark.py 2 --num-trials 5  --datasets '["adult"]'  > log_adult.txt 2>&1 &
# python -u benchmark.py 3 --num-trials 5  --datasets '["german"]'  > log_german.txt 2>&1 &
# python -u benchmark.py 4 --num-trials 5  --datasets '["propublica-recidivism"]'  > log_pp.txt 2>&1 &
# python -u benchmark.py 5 --num-trials 5  --datasets '["propublica-violent-recidivism"]'  > log_pp_violent.txt 2>&1 &
# python -u benchmark.py 6 --num-trials 5  --datasets '["flipped-labels"]'  > log_flipped.txt 2>&1 &
