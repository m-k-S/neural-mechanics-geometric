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

        optimizer.zero_grad()  # Clear gradients.

    activs = {}
    grads = {}
    a_idx = g_idx = 1
    for mod in model.modules():
        alayer = 'Layer {}'.format(a_idx)
        glayer = 'Layer {}'.format(g_idx)
        if isinstance(mod, Activs_prober):
            if alayer not in activs:
                activs[alayer] = mod.activs_norms
            else:
                activs[alayer] += mod.activs_norms
            a_idx += 1
        elif isinstance(mod, Conv_prober):
            if glayer not in grads:
                grads[glayer] = mod.grads_norms
            else:
                grads[glayer] +=  mod.grads_norms
            g_idx += 1

    activs = {k: torch.tensor(activs[k]).mean() for k in activs.keys()}
    grads = {k: torch.tensor(grads[k]).mean() for k in grads.keys()}

    return activs, grads

def save_order_params(model):
    activs = {}
    grads = {}
    a_idx = g_idx = 1
    for mod in model.modules():
        alayer = 'Layer {}'.format(a_idx)
        glayer = 'Layer {}'.format(g_idx)
        if isinstance(mod, Activs_prober):
            if alayer not in activs:
                activs[alayer] = mod.activs_norms
            else:
                activs[alayer] += mod.activs_norms
            a_idx += 1
        elif isinstance(mod, Conv_prober):
            if glayer not in grads:
                grads[glayer] = mod.grads_norms
            else:
                grads[glayer] +=  mod.grads_norms
            g_idx += 1

    return activs, grads

def clear_order_params(model):
    for mod in model.modules():
        if isinstance(mod, Activs_prober):
            mod.activs_norms = []
        elif isinstance(mod, Conv_prober):
            mod.grads_norms = []
