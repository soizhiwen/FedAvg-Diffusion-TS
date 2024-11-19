#!/bin/bash

# Hyperparameters
CONFIGS=("stocks" "etth" "mujoco" "energy" "fmri")
NUM_CLIENTS=(10 20 40)
PUB_RATIOS=(0.5)
SPLIT_MODES=("iid_random")

# Assign number of CPUs to each client
declare -A NUM_CPUS
NUM_CPUS[10]=5
NUM_CPUS[20]=2
NUM_CPUS[40]=2

# Assign number of GPUs to each client
declare -A NUM_GPUS
NUM_GPUS[10]=0.18
NUM_GPUS[20]=0.09
NUM_GPUS[40]=0.09

for CONFIG in "${CONFIGS[@]}"; do
    for N_CLIENTS in "${NUM_CLIENTS[@]}"; do
        for PUB_RATIO in "${PUB_RATIOS[@]}"; do
            for SPLIT_MODE in "${SPLIT_MODES[@]}"; do

                # Run FedAvg
                ./run_sim.sh --name "${CONFIG}_NC_${N_CLIENTS}_${SPLIT_MODE}_PR_${PUB_RATIO}" \
                    --config $CONFIG --num_clients $N_CLIENTS --split_mode $SPLIT_MODE --pub_ratio $PUB_RATIO \
                    --num_cpus ${NUM_CPUS[$N_CLIENTS]} --num_gpus ${NUM_GPUS[$N_CLIENTS]}

            done
        done
    done
done
