import torch
import json
import optuna
from functools import partial

from torch_geometric.utils import from_networkx
import pandas as pd
import pickle

from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score

from backend.utils.threshold import tune_thresholds

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class GCN(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, dropout=0.0):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.lin = nn.Linear(hidden, out_channels)
        self.dropout = dropout
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.lin(x)

def create_trained_model(in_channels, hidden_dim, out_channels, dropout, lr, epochs, train_loader):
    model = GCN(in_channels=in_channels, hidden=hidden_dim, out_channels=out_channels, dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y.float())
            loss.backward()
            optimizer.step()
        print(f'Epoch: {epoch+1}/{epochs}, Loss: {loss.item():.4f}')

    return model

def val_model(model, test_loader, thresholds=None):
    model.eval()
    y_true, y_prob = [], []
    has_labels = True

    with torch.no_grad():
        for batch in test_loader:
            out = model(batch)
            probs = torch.sigmoid(out).cpu()

            if hasattr(batch, 'y') and batch.y is not None:
                y_true.append(batch.y.cpu())
            else:
                has_labels = False

            y_prob.append(probs)

    y_prob = torch.cat(y_prob).numpy()

    if has_labels:
        y_true = torch.cat(y_true).numpy()

        # Auto-tune thresholds only if not provided
        if thresholds is None:
            from backend.utils.threshold import tune_thresholds
            thresholds, _ = tune_thresholds(y_true, y_prob)

        y_pred = (y_prob > thresholds).astype(int)
        return y_true, y_pred, thresholds

    else:
        # If no labels, return just the probabilities
        return None, y_prob, thresholds




def objective(trial, dataset, in_channels, out_channels):
    hidden_dim = trial.suggest_int("hidden_dim", 32, 256)
    lr = trial.suggest_float("lr", 1e-6, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.8)
    batch_size = trial.suggest_int("batch_size", 16, 256, log=True)
    epochs = trial.suggest_int("epochs", 5, 50)

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = create_trained_model(in_channels, hidden_dim, out_channels, dropout, lr, epochs, train_loader)
    y_true, y_pred, _ = val_model(model, test_loader)

    return f1_score(y_true, y_pred, average='macro')

def form_data(graph_data, feature, y=None):
    data = from_networkx(graph_data)

    data.x = torch.tensor(feature, dtype=torch.float32).repeat(data.num_nodes, 1) # Repeat features for each node

    if y is not None:
        data.y = torch.tensor(y, dtype=torch.float32).unsqueeze(0) # Add a batch dimension

    return data

def form_graph_dataset(graphs, graph_feature, y=None):
    dataset = []
    for i, (address, graph_data) in enumerate(graphs.items()):
        feature = graph_feature.loc[address]
        y_data = None
        if y is not None:
            y_data = y.loc[address][label_cols].values
        data = form_data(graph_data, feature, y_data)

        dataset.append(data)
    return dataset

def load_data(mode='txn'):
    graph_feature = pd.read_csv(f'{mode}_graph_features.csv', index_col=0)
    graph_feature.index = graph_feature.index.str.lower()
    graphs = pickle.load(open('txn.pkl', 'rb'))
    y = pd.read_csv('groundtruth.csv', index_col=0)

    label_cols = y.columns.tolist()

    dataset = form_graph_dataset(graphs, graph_feature, y)

    return dataset, graph_feature.shape[1], label_cols

def get_trained_gcn_model(mode='txn', n_trials=100, save_path=None):
    dataset, in_channel, label_cols = load_data(mode)
    study = optuna.create_study(direction="maximize")
    objective_partial = partial(objective, dataset=dataset, in_channels=in_channel, out_channels=len(label_cols))
    study.optimize(objective_partial, n_trials=n_trials)
    best_params = study.best_params

    print("Best Params:", study.best_params)
    print("Best Score:", study.best_value)

    best_model = create_trained_model(in_channel, best_params['hidden_dim'], len(label_cols), best_params['dropout'], best_params['lr'], best_params['epochs'], DataLoader(dataset, batch_size=best_params['batch_size'], shuffle=True))

    y_test, y_pred, thresholds = val_model(best_model, DataLoader(dataset, batch_size=best_params['batch_size'], shuffle=False))

    weights = f1_score(y_test, y_pred, average=None)
    if save_path is not None:
        filename = f'{mode}_model.pth'
        torch.save(best_model.state_dict(), os.path.join(save_path, filename))
        print(f"Model saved to {filename}")

        filename = f'{mode}_best_params.json'
        with open(os.path.join(save_path, filename), 'w') as f:
            data = {
                **best_params,
                'in_channels':in_channel,
                'out_channels':len(label_cols),
                'thresholds': thresholds,
                'weights': weights.tolist()
            }
            json.dump(data, f, indent=4)
        print(f"Best parameters saved to {filename}")


    return best_model, thresholds
