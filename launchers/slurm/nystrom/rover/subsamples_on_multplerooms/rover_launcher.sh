#!/bin/bash

seeds="1"

subsamples=(
"100"
"500"
"1000"
"2000"
"none"
)


for seed in $seeds; do   
    for subsamples in "${subsamples[@]}"; do
        sbatch --export=SEED="${seed}",SUBSAMPLES="${subsamples}" launchers/slurm/nystrom/rover/subsamples_on_multplerooms/rover_base.sh
    done
done    
       