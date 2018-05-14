#!/bin/bash

# this command puts the output into the file "log.txt"
python -u benchmark.py 0 \
	--num-trials 5 \
	--datasets '["two-gaussians", "ricci", "adult", "german", "propublica-recidivism", "propublica-violent-recidivism"]' \
	--algorithms '["UGP_in_True"]' \
	> log.txt 2>&1 &
