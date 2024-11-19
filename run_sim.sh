#!/bin/bash
# chmod +x run_sim.sh

usage() {
    echo "Usage: $0 --config config --num_clients num_clients --split_mode split_mode --pub_ratio pub_ratio --num_cpus num_cpus --num_gpus num_gpus [--name name]"
    exit 1
}

NAME=""

# Parse command
while [[ "$#" -gt 0 ]]; do
    case $1 in
    --config)
        CONFIG="$2"
        shift 2
        ;;
    --num_clients)
        NUM_CLIENTS="$2"
        shift 2
        ;;
    --split_mode)
        SPLIT_MODE="$2"
        shift 2
        ;;
    --pub_ratio)
        PUB_RATIO="$2"
        shift 2
        ;;
    --num_cpus)
        NUM_CPUS="$2"
        shift 2
        ;;
    --num_gpus)
        NUM_GPUS="$2"
        shift 2
        ;;
    --name)
        NAME="--name $2"
        shift 2
        ;;
    *)
        usage
        ;;
    esac
done

# Check if all arguments are provided
if [ -z "$CONFIG" ] || [ -z "$NUM_CLIENTS" ] || [ -z "$SPLIT_MODE" ] || [ -z "$PUB_RATIO" ] || [ -z "$NUM_CPUS" ] || [ -z "$NUM_GPUS" ]; then
    usage
fi

python sim.py $NAME --num_clients $NUM_CLIENTS --num_rounds 5 \
    --config_file ./Config/$CONFIG.yaml --split_mode $SPLIT_MODE \
    --pub_ratio $PUB_RATIO --cudnn_deterministic \
    --num_cpus $NUM_CPUS --num_gpus $NUM_GPUS
