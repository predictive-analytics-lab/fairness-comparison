#!/bin/bash

# this command puts the output into the file "log.txt"
python -u benchmark.py 0 \
	--num-trials 5 \
	--datasets '["two-gaussians", "ricci", "adult", "german", "propublica-recidivism", "propublica-violent-recidivism"]' \
	--algorithms '["UGP_in_True"]' \
	> log.txt 2>&1 &
# python -u benchmark.py 0 --num-trials 5  --datasets '["two-gaussians"]'  > log0.txt 2>&1 &
# python -u benchmark.py 1 --num-trials 5  --datasets '["ricci"]'  > log1.txt 2>&1 &
# python -u benchmark.py 2 --num-trials 5  --datasets '["adult"]'  > log2.txt 2>&1 &
# python -u benchmark.py 3 --num-trials 5  --datasets '["german"]'  > log3.txt 2>&1 &
# python -u benchmark.py 4 --num-trials 5  --datasets '["propublica-recidivism"]'  > log4.txt 2>&1 &
# python -u benchmark.py 5 --num-trials 5  --datasets '["propublica-violent-recidivism"]'  > log5.txt 2>&1 &
# python -u benchmark.py 6 --num-trials 5  --datasets '["flipped-labels"]'  > log6.txt 2>&1 &
