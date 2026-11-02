"""
Aggregation rules shared by FL and gossip learning.
Pure functions on weights, fairness values and model parameters;
the training loops live in fed.py and gossip.py.

How the pieces fit together each round (Algorithm 1 of the paper):

  1. Start from size-based weights:   w_i = n_i / sum(n)          fedavg_weights
  2. Measure each client's gap d_i:
       Fair2Zero  d_i = |F_i|            (distance from perfectly fair)
       FairFed    d_i = |F_i - F_global| (distance from the global level)
                                                     fair2zero_gaps / fairfed_gaps
  3. Shift weight away from clients whose gap is above the mean:
       w_i <- w_i - beta * (d_i - mean(d)), floored at eps, renormalised
                                                                      reweight
  4. Average the clients' models with these weights.            average_models

F_i is the chosen fairness metric (SPD or EOD) of the current model on client
i's data. FedAvg skips steps 2-3 and keeps the size-based weights.

Run directly (python aggregation.py) for a hand-checked example.
"""

import numpy as np


def fedavg_weights(sizes):
# Weights proportional to client size (Algorithm 1, line 1)
# e.g. sizes [600, 300, 100] -> weights [0.6, 0.3, 0.1]
    sizes = np.asarray(sizes, dtype=float)
    return sizes / sizes.sum()


def fair2zero_gaps(local_fair, local_acc, global_acc):
# d_i = |F_i|: distance of each client's fairness from zero.
# Falls back to |Acc_i - Acc_global| where F_i is undefined (None)
#
#   local_fair  one fairness value per client (signed SPD or EOD, or None)
#   local_acc   one accuracy per client
#   global_acc  accuracy used as the reference in the fallback
# Returns a numpy array with one gap per client.
    return np.array([abs(f) if f is not None else abs(a - global_acc)
                     for f, a in zip(local_fair, local_acc)])


def fairfed_gaps(local_fair, global_fair, local_acc, global_acc):
# d_i = |F_i - F_global|: distance from the global fairness level.
# Same accuracy fallback as fair2zero_gaps
# (the fallback is also used when the global value itself is undefined).
# Needs F_global, which only a central server can compute, so FairFed has
# no gossip version in this project.
    return np.array([abs(f - global_fair) if f is not None and global_fair is not None
                     else abs(a - global_acc)
                     for f, a in zip(local_fair, local_acc)])


def reweight(weights, gaps, beta, eps=1e-3):
# Algorithm 1, lines 5-8: penalise clients with above-average gaps,
# floor at eps, then renormalise. The result is carried into the next round
#
# Clients with a below-average gap gain weight, those above it lose weight;
# beta controls how far the weights move each round. Because the output is
# fed back in as `weights` next round, the shifts build up over the rounds.
# The eps floor keeps every client in the average with a small positive weight.
    gaps = np.asarray(gaps, dtype=float)
    w = np.asarray(weights, dtype=float) - beta * (gaps - gaps.mean())
    w = np.maximum(w, eps)
    return w / w.sum()


def average_models(states, weights):
# Weighted average of model state_dicts (Algorithm 1, line 10)
# Every parameter tensor (each layer's weights and biases) is averaged
# separately; the result is a new state_dict to load into a model.
# Weights are expected to sum to 1.
    return {k: sum(float(w) * s[k] for w, s in zip(weights, states))
            for k in states[0]}


if __name__ == "__main__":
    # Hand check with 3 clients: sizes 600/300/100, |SPD| 0.30/0.10/0.02.
    # mean gap = 0.14, so with beta = 0.5:
    #   0.6 - 0.5 * (0.30 - 0.14) = 0.52
    #   0.3 - 0.5 * (0.10 - 0.14) = 0.32
    #   0.1 - 0.5 * (0.02 - 0.14) = 0.16   (sums to 1.0 already)
    w = fedavg_weights([600, 300, 100])
    d = fair2zero_gaps([-0.30, 0.10, None], [0.8, 0.8, 0.82], 0.80)
    print("gaps   ", d)              # third falls back to |0.82 - 0.80| = 0.02
    w = reweight(w, d, beta=0.5)
    print("weights", w)              # [0.52 0.32 0.16]
    assert np.allclose(w, [0.52, 0.32, 0.16])

    # Fairness-aware weights shift influence from the unfair big client
    # to the fair small one; FairFed anchors to the global level instead
    print("fairfed", fairfed_gaps([-0.30, 0.10, -0.02], -0.20, [0.8] * 3, 0.8))
