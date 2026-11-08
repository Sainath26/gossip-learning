import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    # one-hot inputs -> Linear(-> 32) -> ReLU -> Linear(32 -> 1)
    # Outputs logits; the sigmoid is applied in the loss and in predict()
    # (BCEWithLogitsLoss combines sigmoid and log-loss in a numerically
    # stable way, which is why the network itself has no sigmoid layer)
    def __init__(self, cat_sizes, n_num, hidden=32):
        super().__init__()
        self.cat_sizes = list(cat_sizes)
        self.net = nn.Sequential(
            nn.Linear(sum(self.cat_sizes) + n_num, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, cat, num):
        # Each category code becomes its one-hot block here, one batch at a time, so the
        # full one-hot table never sits in memory. The first layer sees exactly the inputs
        # a one-hot encoded table would give
        onehot = [F.one_hot(cat[:, j], size).to(num.dtype) for j, size in enumerate(self.cat_sizes)]
        # (batch, 1) -> (batch,), so outputs line up with the label vector y
        return self.net(torch.cat(onehot + [num], dim=1)).squeeze(-1)


def to_tensors(arrays):
    # prepare_data's NumPy arrays as tensors. y is float because the loss expects
    # float targets; cat and pa are integer ids
    return {'cat': torch.from_numpy(arrays['cat']).long(),
            'num': torch.from_numpy(arrays['num']).float(),
            'y': torch.from_numpy(arrays['y']).float(),
            'pa': torch.from_numpy(arrays['pa']).long()}


def train_one_epoch(model, data, lr=0.05, batch_size=128, generator=None):
# One pass of mini-batch SGD over the local data (a dict from to_tensors)
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
    order = torch.randperm(len(data['y']), generator=generator)
    for start in range(0, len(order), batch_size):
        batch = order[start:start + batch_size]
        optimizer.zero_grad()
        loss_fn(model(data['cat'][batch], data['num'][batch]), data['y'][batch]).backward()
        optimizer.step()


@torch.no_grad()
def predict(model, cat, num, threshold=0.5):
    # Hard 0/1 predictions Ŷ: probability = sigmoid(logit), positive if >= 0.5.
    # Returned as floats so they can be averaged and compared with y directly
    model.eval()
    return (torch.sigmoid(model(cat, num)) >= threshold).float()


def spd(y_hat, pa):
# P(Y_hat = 1 | PA = 0) - P(Y_hat = 1 | PA = 1); None if a group is empty
# (the mean of 0/1 predictions within a group is its positive-prediction rate)
    g0, g1 = pa == 0, pa == 1
    if not g0.any() or not g1.any():
        return None
    return (y_hat[g0].mean() - y_hat[g1].mean()).item()


def eod(y_hat, y, pa):
# Same gap, among the true positives only (Y = 1); None if undefined,
# i.e. the difference in true-positive rate between the two groups. It is
# undefined more often than SPD: a small set may have no positives in one group
    return spd(y_hat[y == 1], pa[y == 1])


def evaluate(model, data):
    # Accuracy, SPD and EOD of `model` on one dataset (a dict from to_tensors).
    # SPD and EOD are signed here and may be None (see spd / eod); callers take abs()
    y_hat = predict(model, data['cat'], data['num'])
    return {'accuracy': (y_hat == data['y']).float().mean().item(),
            'spd': spd(y_hat, data['pa']),
            'eod': eod(y_hat, data['y'], data['pa'])}


if __name__ == '__main__':
    # Sanity check: centralised training on all states' training rows of seed 1,
    # with the settings in configs.json. This is the best case for the gossip and
    # FedAvg runs: one model that sees all the training data at once.
    # (FairFed reports accuracy 0.800 for FedAvg on ACSIncome)
    import json
    import time
    from pathlib import Path
    import numpy as np
    from prepare_data import prepare

    config = json.loads((Path(__file__).resolve().parent / 'configs.json').read_text())
    tr = config['training']
    out = prepare(config, seed=1)
    # Pool every state's training rows into one training set
    train = to_tensors({k: np.concatenate([n[k] for n in out['nodes'].values()])
                        for k in ('cat', 'num', 'y', 'pa')})
    test = to_tensors(out['test'])

    torch.manual_seed(1)
    g = torch.Generator().manual_seed(1)
    model = MLP(out['cat_sizes'], train['num'].shape[1], hidden=tr['hidden_units'])
    start = time.time()
    epochs = 25  # Fair2Zero's 25 rounds
    for epoch in range(epochs):
        train_one_epoch(model, train, lr=tr['learning_rate'], batch_size=tr['batch_size'], generator=g)
    print(f"{len(train['y']):,} training rows, {time.time() - start:.0f}s for "
          f"{epochs} epochs")
    print(evaluate(model, test))
