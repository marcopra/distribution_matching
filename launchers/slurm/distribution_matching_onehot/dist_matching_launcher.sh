#!/bin/bash

horizon="200 400 600"
eps_greedy="0.4 0.6 0.8"
sink_schedule="linear(0.0,1, 200000)" #0.00179856115
n_subsamples="2500 5000"




for h in $horizon; do
    for e in $eps_greedy; do
        for s in $sink_schedule; do
            for n in $n_subsamples; do
                sbatch --export=HORIZON=${h},EPS_GREEDY=${e},SINK_SCHEDULE=${s},N_SUBSAMPLES=${n} dist_matching_launcher.sh
            done
        done
    done
done