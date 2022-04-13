import torch
from order_parameters import Activs_prober, Conv_prober

def initial_order_params(model, dataloader, criterion, optimizer, device):
    model.train() # Set model to train for proper BatchNorm behavior

    for data in dataloader:  # Iterate in batches over the training dataset.
        x = data.x.type(torch.FloatTensor).to(device)
        edge_index = data.edge_index.to(device)
        batch = data.batch.to(device)

        out = model(x, edge_index, batch)  # Perform a single forward pass.

        y = data.y.flatten().to(device)

        loss = criterion(out, y)  # Compute the loss.
        loss.backward()  # Derive gradients.

        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

    activs = {}
    grads = {}
    ranks = {}

    for idx, mod in enumerate(model.activ_probes):
        if idx not in activs:
            activs[idx] = mod.activs_norms
            ranks[idx] = mod.activs_ranks
        else:
            activs[idx] += mod.activs_norms
            ranks[idx] += mod.activs_ranks

    for idx, mod in enumerate(model.conv_probes):
        if idx not in grads:
            grads[idx] = mod.grads_norms
        else:
            grads[idx] += mod.grads_norms

    # Aggregate across a full training epoch
    activs = {k: torch.tensor(activs[k]).mean() for k in activs.keys()}
    grads = {k: torch.tensor(grads[k]).mean() for k in grads.keys()}
    ranks = {k: torch.tensor(ranks[k]).mean() for k in ranks.keys()}

    return activs, grads, ranks

def save_order_params(model):
    activs = {}
    grads = {}
    ranks = {}

    for idx, mod in enumerate(model.activ_probes):
        if idx not in activs:
            activs[idx] = mod.activs_norms
            ranks[idx] = mod.activs_ranks
        else:
            activs[idx] += mod.activs_norms
            ranks[idx] += mod.activs_ranks

    for idx, mod in enumerate(model.conv_probes):
        if idx not in grads:
            grads[idx] = mod.grads_norms
        else:
            grads[idx] += mod.grads_norms

    return activs, grads, ranks

def clear_order_params(model):
    for idx, mod in enumerate(model.activ_probes):
        mod.activs_norms = []
        mod.activs_ranks = []

    for idx, mod in enumerate(model.conv_probes):
        mod.grads_norms = []
