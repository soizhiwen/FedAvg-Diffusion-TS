import os
import csv
from pathlib import Path
from collections import defaultdict

import torch
import pandas as pd
import numpy as np
import numpy.random as npr
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

from Utils.context_fid import Context_FID
from Utils.cross_correlation import CrossCorrelLoss
from Utils.discriminative_metric import discriminative_score_metrics
from Utils.predictive_metric import predictive_score_metrics
from Utils.metric_utils import display_scores

plt.rcParams.update({"font.family": "serif"})

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def partition_data(dataset, num_clients, pub_ratio, split_mode, seed):
    st0 = npr.get_state()
    npr.seed(seed)

    labels = dataset[..., -1]
    num_pub = int(len(dataset) * pub_ratio)

    if split_mode == "iid_even":
        unique_labels, label_counts = np.unique(labels, return_counts=True)
        label_ratio = label_counts / label_counts.sum()
        pub_per_label = np.array(label_ratio * num_pub, dtype=int)

        pub_idxs = []
        splits = [[] for _ in range(num_clients)]
        for i, label in enumerate(unique_labels):
            idxs = np.where(labels == label)[0]
            npr.shuffle(idxs)
            pub_idxs.extend(idxs[: pub_per_label[i]])
            pvt_idxs = idxs[pub_per_label[i] :]
            label_splits = np.array_split(pvt_idxs, num_clients)

            for j in range(num_clients):
                splits[j].extend(label_splits[j])

        for split in splits:
            npr.shuffle(split)

    elif split_mode == "iid_random":
        idxs = npr.permutation(len(dataset))
        pub_idxs = idxs[:num_pub]
        pvt_idxs = idxs[num_pub:]
        splits = np.array_split(pvt_idxs, num_clients)

    elif split_mode == "non_iid_order":
        idxs = np.argsort(labels)
        pub_idxs = idxs[:num_pub]
        pvt_idxs = idxs[num_pub:]
        splits = np.array_split(pvt_idxs, num_clients)

    elif split_mode == "non_iid_unorder":
        idxs = np.argsort(labels)
        half = num_pub // 2
        pub_idxs = np.concatenate([idxs[:half], idxs[-half:]])
        pvt_idxs = idxs[half:-half]
        shards = np.array_split(pvt_idxs, 2 * num_clients)
        npr.shuffle(shards)
        splits = [np.concatenate(shards[i * 2 : i * 2 + 2]) for i in range(num_clients)]

    else:
        raise NotImplementedError()

    pub_dataset = dataset[pub_idxs]
    pvt_datasets = [dataset[split] for split in splits]
    npr.set_state(st0)

    return pub_dataset, pvt_datasets


def load_data_partitions(
    path,
    num_clients,
    partition_id=None,
    pub_ratio=0.5,
    split_mode="iid_even",
    seed=42,
):
    ds = np.load(path, allow_pickle=True)
    pub_dataset, pvt_datasets = partition_data(
        ds, num_clients, pub_ratio, split_mode, seed
    )
    if partition_id is not None:
        part_ds = pvt_datasets[partition_id]
    else:
        part_ds = pub_dataset
    return np.concatenate(part_ds[:, 0])


def context_fid(ori_data, fake_data, iterations=5):
    context_fid_score = []
    for _ in range(iterations):
        context_fid = Context_FID(ori_data[:], fake_data[: ori_data.shape[0]])
        context_fid_score.append(context_fid)

    mean, sigma = display_scores(context_fid_score)
    return mean, sigma


def cross_corr(ori_data, fake_data, iterations=5):
    def random_choice(size, num_select=100):
        select_idx = npr.randint(low=0, high=size, size=(num_select,))
        return select_idx

    x_real = torch.from_numpy(ori_data)
    x_fake = torch.from_numpy(fake_data)

    correlational_score = []
    size = int(x_real.shape[0] / iterations)

    for _ in range(iterations):
        real_idx = random_choice(x_real.shape[0], size)
        fake_idx = random_choice(x_fake.shape[0], size)
        corr = CrossCorrelLoss(x_real[real_idx, :, :], name="CrossCorrelLoss")
        loss = corr.compute(x_fake[fake_idx, :, :])
        correlational_score.append(loss.item())

    mean, sigma = display_scores(correlational_score)
    return mean, sigma


def discriminative(ori_data, fake_data, iterations=5):
    discriminative_score = []
    for _ in range(iterations):
        temp_disc, *_ = discriminative_score_metrics(
            ori_data[:], fake_data[: ori_data.shape[0]]
        )
        discriminative_score.append(temp_disc)

    mean, sigma = display_scores(discriminative_score)
    return mean, sigma


def predictive(ori_data, fake_data, iterations=5):
    predictive_score = []
    for _ in range(iterations):
        temp_pred = predictive_score_metrics(ori_data, fake_data[: ori_data.shape[0]])
        predictive_score.append(temp_pred)

    mean, sigma = display_scores(predictive_score)
    return mean, sigma


def write_csv(fields, name, save_dir):
    with open(f"{save_dir}/{name}.csv", "a") as f:
        writer = csv.writer(f)
        writer.writerow(fields)


def plot_metrics(history, save_dir):
    m_name = {
        "train_loss": "Train Loss",
        "all_context_fid": "Context-FID Score",
        "all_context_fid_sigma": "Context-FID Sigma",
        "all_cross_corr": "Correlational Score",
        "all_cross_corr_sigma": "Correlational Sigma",
        "all_discriminative": "Discriminative Score",
        "all_discriminative_sigma": "Discriminative Sigma",
        "all_predictive": "Predictive Score",
        "all_predictive_sigma": "Predictive Sigma",
    }
    metrics = defaultdict(list)

    for m, values in history.metrics_distributed_fit.items():
        for r, v in values:
            metrics[m].append((r, v))

    for m, values in history.metrics_distributed.items():
        for r, v in values:
            metrics[m].append((r, v))

    for k, v in metrics.items():
        os.makedirs(f"{save_dir}/{k}", exist_ok=True)
        df = pd.DataFrame(v, columns=["Round", m_name[k]])
        df.sort_values(by=["Round"], inplace=True)
        df.to_csv(f"{save_dir}/{k}/{k}.csv", index=False)

        ax = sns.lineplot(data=df, x="Round", y=m_name[k], marker="o")
        _ = ax.set_xticks(df["Round"].unique())
        _ = ax.set_xlabel("Round")
        _ = ax.set_ylabel(m_name[k])
        plt.savefig(f"{save_dir}/{k}/{k}.pdf", bbox_inches="tight")
        plt.close()

    clients_csv = [(f"clients_{k}", k) for k in metrics]

    for csv, k in clients_csv:
        df = pd.read_csv(f"{save_dir}/{csv}.csv", header=None)
        df.columns = ["Round", m_name[k], "Client ID"]
        df.sort_values(by=["Round", "Client ID"], inplace=True)
        df.to_csv(f"{save_dir}/{k}/{csv}.csv", index=False)
        os.remove(f"{save_dir}/{csv}.csv")

        ax = sns.lineplot(data=df, x="Round", y=m_name[k], hue="Client ID", marker="o")
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        _ = ax.set_xticks(df["Round"].unique())
        _ = ax.set_xlabel("Round")
        _ = ax.set_ylabel(m_name[k])
        plt.savefig(f"{save_dir}/{k}/{csv}.pdf", bbox_inches="tight")
        plt.close()


def increment_path(path, exist_ok=False, sep="", mkdir=False):
    """
    Generates an incremented file or directory path if it exists, with optional mkdir; args: path, exist_ok=False,
    sep="", mkdir=False.

    Example: runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc
    """
    path = Path(path)
    if path.exists() and not exist_ok:
        path, suffix = (
            (path.with_suffix(""), path.suffix) if path.is_file() else (path, "")
        )

        for n in range(2, 9999):
            p = f"{path}{sep}{n}{suffix}"
            if not os.path.exists(p):
                break
        path = Path(p)

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)

    return path.as_posix()
