import torch
from order_parameters import Activs_prober, Conv_prober

def initial_order_params(model, dataloader, criterion, optimizer, device):
    model.train() # Set model to train for proper BatchNorm behavior

    for data in dataloader:  # Iterate in batches over the training dataset.
        x = data.x.type(torch.FloatTensor).to(device)
        edge_index = data.edge_index.to(device)
        batch = data.batch.to(device)

        out = model(x, edge_index, batch).flatten()  # Perform a single forward pass.
        y = data.y[:, 0].flatten().to(device)

        loss = criterion(out, y)  # Compute the loss.
        loss.backward()  # Derive gradients.

        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

    activs = {}
    grads = {}
    activs_full_ranks = {}
    activs_graph_mean_ranks = {}
    activs_feature_ranks = {}
    conv_full_ranks = {}
    conv_graph_mean_ranks = {}
    conv_feature_ranks = {}

    for idx, mod in enumerate(model.activ_probes):
        if idx not in activs:
            activs[idx] = mod.activs_norms
            activs_full_ranks[idx] = mod.full_ranks
            activs_graph_mean_ranks[idx] = mod.graph_mean_ranks
            activs_feature_ranks[idx] = mod.feature_ranks
        else:
            activs[idx] += mod.activs_norms
            activs_full_ranks[idx] += mod.full_ranks
            activs_graph_mean_ranks[idx] += mod.graph_mean_ranks
            activs_feature_ranks[idx] += mod.feature_ranks

    for idx, mod in enumerate(model.conv_probes):
        if idx not in grads:
            grads[idx] = mod.grads_norms
            conv_full_ranks[idx] = mod.full_ranks
            conv_graph_mean_ranks[idx] = mod.graph_mean_ranks
            conv_feature_ranks[idx] = mod.feature_ranks
        else:
            grads[idx] += mod.grads_norms
            conv_full_ranks[idx] += mod.full_ranks
            conv_graph_mean_ranks[idx] += mod.graph_mean_ranks
            conv_feature_ranks[idx] += mod.feature_ranks

    # Aggregate across a full training epoch
    activs = {k: torch.tensor(activs[k]).mean() for k in activs.keys()}
    grads = {k: torch.tensor(grads[k]).mean() for k in grads.keys()}

    activs_full_ranks = {k: torch.tensor(activs_full_ranks[k]).mean() for k in activs_full_ranks.keys()}
    activs_graph_mean_ranks = {k: torch.tensor(activs_graph_mean_ranks[k]).mean() for k in activs_graph_mean_ranks.keys()}
    activs_feature_ranks = {k: [torch.tensor(feature_rank).mean() for feature_rank in activs_feature_ranks[k]] for k in activs_feature_ranks.keys()}

    conv_full_ranks = {k: torch.tensor(conv_full_ranks[k]).mean() for k in conv_full_ranks.keys()}
    conv_graph_mean_ranks = {k: torch.tensor(conv_graph_mean_ranks[k]).mean() for k in conv_graph_mean_ranks.keys()}
    conv_feature_ranks = {k: [torch.tensor(feature_rank).mean() for feature_rank in conv_feature_ranks[k]] for k in feature_ranks.keys()}

    return activs, grads, activs_full_ranks, activs_graph_mean_ranks, activs_feature_ranks, conv_full_ranks, conv_graph_mean_ranks, conv_feature_ranks

def save_order_params(model):
    activs = {}
    grads = {}
    full_ranks = {}
    graph_mean_ranks = {}
    feature_ranks = {}

    for idx, mod in enumerate(model.activ_probes):
        if idx not in activs:
            activs[idx] = mod.activs_norms
            full_ranks[idx] = mod.full_ranks
            graph_mean_ranks[idx] = mod.graph_mean_ranks
            feature_ranks[idx] = mod.feature_ranks
        else:
            activs[idx] += mod.activs_norms
            full_ranks[idx] += mod.full_ranks
            graph_mean_ranks[idx] += mod.graph_mean_ranks
            feature_ranks[idx] += mod.feature_ranks

    for idx, mod in enumerate(model.conv_probes):
        if idx not in grads:
            grads[idx] = mod.grads_norms
        else:
            grads[idx] += mod.grads_norms

    return activs, grads, full_ranks, graph_mean_ranks, feature_ranks

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
