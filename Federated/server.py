import flwr as fl


def fit_config(server_round: int):
    """Return training configuration dict for each round."""
    config = {
        "server_round": server_round,
        "local_epochs": 5000,
    }
    return config


def evaluate_config(server_round: int):
    """Return evaluation configuration dict for each round."""
    config = {
        "server_round": server_round,
        "size_every": 2001,
        "metric_iterations": 5,
    }
    return config


def fit_weighted_average(metrics):
    # Multiply context fid of each client by number of examples used
    train_loss = [num * m["train_loss"] for num, m in metrics]
    examples = [num for num, _ in metrics]

    agg_metrics = {"train_loss": sum(train_loss) / sum(examples)}

    # Aggregate and return custom metric (weighted average)
    return agg_metrics


def evaluate_weighted_average(metrics):
    # Multiply context fid of each client by number of examples used
    all_context_fids, all_context_fid_sigmas = [], []
    all_cross_corrs, all_cross_corr_sigmas = [], []
    all_discriminatives, all_discriminative_sigmas = [], []
    all_predictives, all_predictive_sigmas = [], []
    examples = []

    for num, m in metrics:
        if m:
            all_context_fids.append(num * m["all_context_fid"])
            all_context_fid_sigmas.append(num * m["all_context_fid_sigma"])
            all_cross_corrs.append(num * m["all_cross_corr"])
            all_cross_corr_sigmas.append(num * m["all_cross_corr_sigma"])
            all_discriminatives.append(num * m["all_discriminative"])
            all_discriminative_sigmas.append(num * m["all_discriminative_sigma"])
            all_predictives.append(num * m["all_predictive"])
            all_predictive_sigmas.append(num * m["all_predictive_sigma"])
            examples.append(num)

    if not examples:
        return {}

    agg_metrics = {
        "all_context_fid": sum(all_context_fids) / sum(examples),
        "all_context_fid_sigma": sum(all_context_fid_sigmas) / sum(examples),
        "all_cross_corr": sum(all_cross_corrs) / sum(examples),
        "all_cross_corr_sigma": sum(all_cross_corr_sigmas) / sum(examples),
        "all_discriminative": sum(all_discriminatives) / sum(examples),
        "all_discriminative_sigma": sum(all_discriminative_sigmas) / sum(examples),
        "all_predictive": sum(all_predictives) / sum(examples),
        "all_predictive_sigma": sum(all_predictive_sigmas) / sum(examples),
    }

    # Aggregate and return custom metric (weighted average)
    return agg_metrics


def get_fedavg_fn(num_clients, model_parameters):
    return fl.server.strategy.FedAvg(
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.ndarrays_to_parameters(model_parameters),
        fit_metrics_aggregation_fn=fit_weighted_average,
        evaluate_metrics_aggregation_fn=evaluate_weighted_average,
    )
