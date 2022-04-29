# Training Functions

import torch
from torch.optim.lr_scheduler import LambdaLR

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

def train_classification(train_loader, model, criterion, optimizer, device):
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
        x = data.x.type(torch.FloatTensor).to(device)
        edge_index = data.edge_index.to(device)
        batch = data.batch.to(device)

        out = model(x, edge_index, batch)  # Perform a single forward pass.
        y = data.y.flatten().to(device)

        loss = criterion(out, y)  # Compute the loss.
        loss.backward()  # Derive gradients.

        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

def test_classification(loader, model, device):
    model.eval()

    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        x = data.x.type(torch.FloatTensor).to(device)
        edge_index = data.edge_index.to(device)
        batch = data.batch.to(device)

        y = data.y.to(device)
        out = model(x, edge_index, batch)

        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == y).sum())  # Check against ground-truth labels.

        # roc_y = y.clone().detach().cpu()
        # roc_out = pred.clone().detach().cpu()
        # # roc_auc = roc_auc_score(roc_y, roc_out)

    return correct / len(loader.dataset)  # Derive ratio of correct predictions.

def train_regression(train_loader, model, criterion, optimizer, device):
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
        x = data.x.type(torch.FloatTensor).to(device)
        edge_index = data.edge_index.to(device)
        batch = data.batch.to(device)

        out = model(x, edge_index, batch).flatten() # Perform a single forward pass.
        y = data.y[:, 0].flatten().to(device)

        loss = criterion(out, y)  # Compute the loss.
        loss.backward()  # Derive gradients.

        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

def test_regression(loader, model, criterion, device):
    model.eval()

    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        x = data.x.type(torch.FloatTensor).to(device)
        edge_index = data.edge_index.to(device)
        batch = data.batch.to(device)

        y = data.y[:, 0].flatten().to(device)
        out = model(x, edge_index, batch).flatten()

    return criterion(out, y)  # MSE

class LinearSchedule(LambdaLR):
    # Linear warmup and then linear decay learning rate scheduling
    # Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
    # Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    def __init__(self, optimizer, t_total, warmup_steps=0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(LinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))
