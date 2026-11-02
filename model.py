"""
The neural network shared by FL and gossip learning:
model, local training, evaluation (accuracy, SPD, EOD) and data loading.

Used by fed.py (federated learning) and gossip.py (gossip learning), so both
settings train and score models in exactly the same way.

Fairness metrics, where PA = 0 is female and PA = 1 is male:
  SPD (statistical parity difference)  P(Ŷ=1 | PA=0) - P(Ŷ=1 | PA=1)
  EOD (equal opportunity difference)   the same gap among rows with Y = 1
Both are 0 for a perfectly fair model and negative when men are predicted
positive more often. The results report their absolute value.

Run directly (python model.py) for a centralised sanity check.
"""

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    # input (106) -> Linear(106 -> 32) -> ReLU -> Linear(32 -> 1)
    # Outputs logits; the sigmoid is applied in the loss and in predict()
    # (BCEWithLogitsLoss combines sigmoid and log-loss in a numerically
    # stable way, which is why the network itself has no sigmoid layer)
    def __init__(self, n_features=106, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        # (batch, 1) -> (batch,), so outputs line up with the label vector y
        return self.net(x).squeeze(-1)


def load_split(seed, processed_path="data/processed/adult.npz",
               splits_dir="data/splits"):
# Returns the test set and the client datasets for one seed,
# each as a tuple of tensors (X, y, pa)
#
# The data comes from adult.npz and the row indices from seed_XXX.npz (both
# made by data_prepration.py). Return value: (test, clients), where clients
# is a list with one (X, y, pa) tuple per client.
# y is float because the loss expects float targets; pa is an integer group id.
    data = np.load(processed_path)
    split = np.load(Path(splits_dir) / f"seed_{seed:03d}.npz")
    X = torch.from_numpy(data["X"])
    y = torch.from_numpy(data["y"]).float()
    pa = torch.from_numpy(data["pa"]).long()

    # Selects the given rows from all three arrays
    def take(idx):
        idx = torch.from_numpy(idx)
        return X[idx], y[idx], pa[idx]

    # The number of clients is read from the file (keys client_0, client_1, ...)
    n_clients = sum(k.startswith("client_") for k in split.files)
    test = take(split["test_idx"])
    clients = [take(split[f"client_{k}"]) for k in range(n_clients)]
    return test, clients


def train_one_epoch(model, X, y, lr=0.05, batch_size=128, generator=None):
# One pass of mini-batch SGD over the local data
#
# Updates `model` in place and returns nothing. A fresh optimiser is created
# on every call: plain SGD keeps no state between steps, so nothing is lost.
# `generator` controls the shuffling; passing a seeded one makes the batch
# order reproducible.
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    # Shuffle the rows, then walk through them batch_size at a time
    # (the last batch may be smaller)
    order = torch.randperm(len(X), generator=generator)
    for start in range(0, len(X), batch_size):
        batch = order[start:start + batch_size]
        optimizer.zero_grad()
        loss_fn(model(X[batch]), y[batch]).backward()
        optimizer.step()


@torch.no_grad()
def predict(model, X, threshold=0.5):
    # Hard 0/1 predictions Ŷ: probability = sigmoid(logit), positive if >= 0.5.
    # Returned as floats so they can be averaged and compared with y directly
    model.eval()
    return (torch.sigmoid(model(X)) >= threshold).float()


def spd(y_hat, pa):
# P(Y_hat = 1 | PA = 0) - P(Y_hat = 1 | PA = 1); None if a group is empty
# (the mean of 0/1 predictions within a group is its positive-prediction rate)
    g0, g1 = pa == 0, pa == 1
    if not g0.any() or not g1.any():
        return None
    return (y_hat[g0].mean() - y_hat[g1].mean()).item()


def eod(y_hat, y, pa):
# Same gap, among the true positives only (Y = 1); None if undefined
# i.e. the difference in true-positive rate between women and men. It is
# undefined more often than SPD: a client may have no positive women at all
    return spd(y_hat[y == 1], pa[y == 1])


def evaluate(model, X, y, pa):
    # Accuracy, SPD and EOD of `model` on one dataset. SPD and EOD are signed
    # here and may be None (see spd / eod); callers take abs() where needed
    y_hat = predict(model, X)
    return {"accuracy": (y_hat == y).float().mean().item(),
            "spd": spd(y_hat, pa),
            "eod": eod(y_hat, y, pa)}


if __name__ == "__main__":
    # Sanity check: centralised training on all client rows of seed 1.
    # Accuracy should land around 0.85.
    # (This is the best case for the FL runs: one model that sees all the
    # training data at once, with the same 25 epochs.)
    torch.manual_seed(1)
    g = torch.Generator().manual_seed(1)
    (X_te, y_te, pa_te), clients = load_split(1)
    # Pool the clients back into one training set
    X_tr = torch.cat([c[0] for c in clients])
    y_tr = torch.cat([c[1] for c in clients])

    model = MLP(n_features=X_tr.shape[1])
    for epoch in range(25):
        train_one_epoch(model, X_tr, y_tr, generator=g)
    print(evaluate(model, X_te, y_te, pa_te))
