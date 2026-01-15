#!/bin/bash

horizon="110 200 300"
eps_greedy="0.4 0.6 0.8"
sink_schedule="linear(0.0,0.1,1000000)" #0.00179856115




for h in $horizon; do
    for e in $eps_greedy; do
        for s in $sink_schedule; do
            sbatch --export=ALL,HORIZON="${h}",EPS_GREEDY="${e}",SINK_SCHEDULE="${s}" launchers/slurm/distribution_matching_onehot/dist_matching.sh
               
        done
    done
done