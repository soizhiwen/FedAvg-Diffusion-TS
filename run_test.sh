python sim.py --name test --num_clients 3 --num_rounds 2 \
    --config_file ./Config/stocks.yaml --split_mode iid_random \
    --pub_ratio 0.5 --cudnn_deterministic \
    --num_cpus 6 --num_gpus 0.3
