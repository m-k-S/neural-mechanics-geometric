import torch
from order_parameters import ActivationProbe, ConvolutionProbe
from data import InitializationMetric, TrainingMetric
from train import train_regression, train_classification

def initial_order_params(model, dataloader, criterion, optimizer, device):
    model.train() # Set model to train for proper BatchNorm behavior

    if model.lin.out_features == 2:
        train_classification(dataloader, model, criterion, optimizer, device)
    else:
        train_regression(dataloader, model, criterion, optimizer, device)

    metrics = []
    probe_layers = model.activ_probes + model.conv_probes

    for idx, probe in enumerate(probe_layers):
        # Set layer type name
        layer_type="Activation" if isinstance(probe, ActivationProbe) else "Convolution"

        # Get all metrics tracked by probing layer
        for k, v in probe.__dict__:

            # Only look for properties that are lists (these are the metrics)
            if type(v) == list:

                # Feature ranks have additional subindices
                if k == "feature_ranks":
                    # v is a training_iter x hidden_channels size matrix
                    v = torch.tensor(v).T
                    for jdx, feat in enumerate(v):
                        val = torch.tensor(feat).mean()
                        metric = InitializationMetric(
                            layer_type=layer_type,
                            layer_index=idx,
                            depth=model.num_layers,
                            normalization=str(model.norm),
                            name=k,
                            value=val,
                            feature=jdx
                        )
                        metrics.append(metric)
                else:
                    val = torch.tensor(v).mean()
                    metric = InitializationMetric(
                        layer_type=layer_type,
                        layer_index=idx,
                        depth=model.num_layers,
                        normalization=str(model.norm),
                        name=k,
                        value=val
                    )
                    metrics.append(metric)

    return metrics

def save_order_params(model, optimizer):
    metrics = []
    probe_layers = model.activ_probes + model.conv_probes
    optim = "Adam" if isinstance(optimizer, torch.optim.Adam) else "None"

    for idx, probe in enumerate(probe_layers):
        # Set layer type name
        layer_type="Activation" if probe.__str__() = "ActivationProbe()" else "Convolution"

        # Get all metrics tracked by probing layer
        for k, v in probe.__dict__:

            # Only look for properties that are lists (these are the metrics)
            if type(v) == list:

                # Feature ranks have additional subindices
                if k == "feature_ranks":
                    # v is a training_iter x hidden_channels size matrix
                    v = torch.tensor(v).T
                    for jdx, feat in enumerate(v):
                        metric = TrainingMetric(
                            layer_type=layer_type,
                            layer_index=idx,
                            depth=model.num_layers,
                            normalization=str(model.norm),
                            name=k,
                            values=feat,
                            optimizer=optim,
                            feature=jdx
                        )
                        metrics.append(metric)
                else:
                    metric = TrainingMetric(
                        layer_type=layer_type,
                        layer_index=idx,
                        depth=model.num_layers,
                        normalization=str(model.norm),
                        name=k,
                        values=v,
                        optimizer=optim
                    )
                    metrics.append(metric)

    return metrics

def clear_order_params(model):
    for idx, mod in enumerate(model.activ_probes):
        mod.activs_norms = []
        mod.full_ranks = []
        mod.graph_mean_ranks = []
        mod.feature_ranks = []

    for idx, mod in enumerate(model.conv_probes):
        mod.grads_norms = []
        mod.full_ranks = []
        mod.graph_mean_ranks = []
        mod.feature_ranks = []
