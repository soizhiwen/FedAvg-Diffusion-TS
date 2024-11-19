import copy

import torch
from Utils.io_utils import instantiate_from_config


def build_dataloader_fed(config, name, data, seed, save_dir=""):
    config = copy.deepcopy(config)
    batch_size = config["dataloader"]["batch_size"]
    jud = config["dataloader"]["shuffle"]

    config["dataloader"]["params"]["name"] += f"_{name}"
    config["dataloader"]["params"]["dataset"] = data
    config["dataloader"]["params"]["seed"] = seed
    config["dataloader"]["params"]["output_dir"] = save_dir

    dataset = instantiate_from_config(config["dataloader"])

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=jud,
        num_workers=0,
        pin_memory=True,
        sampler=None,
        drop_last=jud,
    )

    dataload_info = {"dataloader": dataloader, "dataset": dataset}

    return dataload_info
