"""
Federated learning loop (Algorithm 1): FedAvg, FairFed and Fair2Zero.
One run = one method, one fairness metric, one seed. Results go to one JSON file.

Every round (25 by default), with all clients taking part:
  1. FairFed / Fair2Zero only, from round 2 on: score the current global model
     on each client's training data and update the aggregation weights
     (see aggregation.py). Round 1 has no metrics yet, so it keeps the
     size-based weights. FedAvg keeps the size-based weights throughout.
  2. Each client copies the global model and trains it for one local epoch.
  3. The server averages the client models with the weights into the new
     global model, which is then scored on the held-out test set.

Example:
    python federated_learning_loop.py --method fair2zero --metric spd --seeds 1 100

Output: <out>/<run name>_sXXX.json per seed, e.g. fair2zero_spd_b0.05_s001.json
"""

import argparse
import copy
import json
from pathlib import Path

import torch

from aggregation import (average_models, fair2zero_gaps, fairfed_gaps,
                         fedavg_weights, reweight)
from model import MLP, evaluate, load_split, train_one_epoch


def run_name(method, metric, beta):
    # FedAvg does not use a fairness metric, so it has one result per seed
    # Others: "<method>_<metric>_b<beta>", e.g. "fairfed_eod_b0.05"
    return method if method == "fedavg" else f"{method}_{metric}_b{beta}"


def run_fl(seed, method="fedavg", metric="spd", beta=0.05, eps=1e-3,
           rounds=25, lr=0.05, batch_size=128, on_round=None):
    # on_round(t, record, models): optional hook for the control panel,
    # called once before training (t = 0) and after every round
    #
    # method   "fedavg", "fairfed" or "fair2zero"
    # metric   fairness metric the weights react to: "spd" or "eod"
    # beta     step size of the weight update; eps: minimum client weight
    #
    # Returns a dict with the run settings, the final test metrics ("final")
    # and one record per round ("history") with test metrics, weights and gaps.
    test, clients = load_split(seed)
    # All training rows pooled together: used to score the global model on
    # the union of the clients' data (FairFed's F_global, and the accuracy
    # used in the fallback)
    X_all = torch.cat([c[0] for c in clients])
    y_all = torch.cat([c[1] for c in clients])
    pa_all = torch.cat([c[2] for c in clients])

    # Same seed -> same starting weights and batch order for every method
    # (so methods are compared on identical conditions: a paired comparison)
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    global_model = MLP(n_features=X_all.shape[1])
    weights = fedavg_weights([len(c[0]) for c in clients])
    if on_round:
        on_round(0, {"client_sizes": [len(c[0]) for c in clients]}, [global_model])

    history = []
    for t in range(1, rounds + 1):
        gaps = None
        # Round 1 has no metrics from a previous round yet, so the weights
        # stay proportional to client size (the authors' correction to
        # Algorithm 1). Reweighting starts in round 2
        if method != "fedavg" and t > 1:
            # Evaluate the current global model on each client's own data
            local = [evaluate(global_model, *c) for c in clients]
            glob = evaluate(global_model, X_all, y_all, pa_all)
            local_fair = [m[metric] for m in local]
            local_acc = [m["accuracy"] for m in local]
            if method == "fair2zero":
                gaps = fair2zero_gaps(local_fair, local_acc, glob["accuracy"])
            elif method == "fairfed":
                gaps = fairfed_gaps(local_fair, glob[metric],
                                    local_acc, glob["accuracy"])
            else:
                raise ValueError(f"unknown method {method}")
            # Weights carry over: this round's update starts from last round's
            weights = reweight(weights, gaps, beta, eps)

        # Each client trains one epoch from the current global model
        # (all clients share the generator g, so the batch order of the whole
        # run is fixed by the seed)
        states = []
        for X, y, _ in clients:
            local_model = copy.deepcopy(global_model)
            train_one_epoch(local_model, X, y, lr=lr, batch_size=batch_size,
                            generator=g)
            states.append(local_model.state_dict())
        # Server step: the weighted average becomes the new global model
        global_model.load_state_dict(average_models(states, weights))

        # Log test-set accuracy/SPD/EOD plus the weights and gaps used this round
        history.append({"round": t, **evaluate(global_model, *test),
                        "weights": weights.tolist(),
                        "gaps": None if gaps is None else gaps.tolist()})
        if on_round:
            on_round(t, history[-1], [global_model])

    return {"seed": seed, "method": method, "metric": metric, "beta": beta,
            "eps": eps, "rounds": rounds,
            "client_sizes": [len(c[0]) for c in clients],
            "final": {k: history[-1][k] for k in ("accuracy", "spd", "eod")},
            "history": history}


def main():
    # Command-line entry point: runs one configuration over a range of seeds
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["fedavg", "fairfed", "fair2zero"],
                   default="fedavg")
    p.add_argument("--metric", choices=["spd", "eod"], default="spd",
                   help="fairness metric used for reweighting (ignored by fedavg)")
    p.add_argument("--beta", type=float, default=0.05)
    p.add_argument("--eps", type=float, default=1e-3)
    p.add_argument("--rounds", type=int, default=25)
    # Inclusive range, e.g. "--seeds 1 100" runs seeds 1, 2, ..., 100
    p.add_argument("--seeds", type=int, nargs=2, default=[1, 100],
                   metavar=("FIRST", "LAST"))
    p.add_argument("--out", default="results/fl")
    args = p.parse_args()

    tag = run_name(args.method, args.metric, args.beta)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for seed in range(args.seeds[0], args.seeds[1] + 1):
        result = run_fl(seed, args.method, args.metric, args.beta, args.eps,
                        args.rounds)
        (out / f"{tag}_s{seed:03d}.json").write_text(json.dumps(result))
        f = result["final"]
        print(f"{tag} seed {seed:3d}: acc {f['accuracy']:.4f}  "
              f"|spd| {abs(f['spd']):.4f}  |eod| {abs(f['eod']):.4f}")


if __name__ == "__main__":
    main()
