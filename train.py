# Training Functions

import torch
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_classification(train_loader, model, criterion, optimizer):
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

def test_classification(loader, model):
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

def train_regression(train_loader, model, criterion, optimizer):
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

def test_regression(loader, model, criterion):
    model.eval()

    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        x = data.x.type(torch.FloatTensor).to(device)
        edge_index = data.edge_index.to(device)
        batch = data.batch.to(device)

        y = data.y[:, 0].flatten().to(device)
        out = model(x, edge_index, batch).flatten()

    return criterion(out, y)  # MSE
