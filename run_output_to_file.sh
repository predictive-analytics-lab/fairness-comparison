#!/bin/bash

# this command puts the output into the file "log.txt"
python benchmark.py 0 \
	--num-trials 0 \
	--datasets '["two-gaussians", "ricci", "adult", "german", "propublica-recidivism", "propublica-violent-recidivism"]' \
	--algorithms '["UGP_in_True"]' \
	> log.txt 2>&1 &
