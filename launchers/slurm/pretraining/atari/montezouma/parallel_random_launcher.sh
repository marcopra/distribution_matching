#!/bin/bash

seeds="1 2 3"

for seed in $seeds; do
    sbatch \
        --export=ALL,SEED="$seed" \
        launchers/slurm/pretraining/atari/montezouma/parallel_random_base.sh
done
