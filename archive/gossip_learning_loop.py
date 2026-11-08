"""
Gossip learning loop: no server, each node merges models with its neighbours.
Each node runs Algorithm 1 on its own neighbourhood (itself + neighbours):
weights start from data size and, for Fair2Zero, are updated every round.
With a fully connected topology this is exactly FL.

Every round, all nodes act at the same time (synchronous simulation):
  1. Fair2Zero only, from round 2 on: each node scores its current model on
     its own data. Round 1 has no metrics yet, so weights stay size-based.
     That gives its local unfairness d_j, which it would send along with its
     model (one extra number per message).
  2. Each node trains its own model for one local epoch.
  3. Each node replaces its model with a weighted average of its own model
     and its neighbours' models. Under Fair2Zero the node first updates the
     weights it gives its neighbourhood, using the d_j values it received.

Fair2Zero works here because its target (unfairness = 0) needs only local
values. FairFed would need the global fairness level, which no node knows.

Since nodes no longer share one model, every round logs the test metrics of
each node, their mean and worst case, and a "consensus" model (the plain
average of all nodes, used only for reporting).

Example:
    python gossip_learning_loop.py --method fair2zero --metric spd --topology ring --seeds 1 100

Output: <out>/<run name>_sXXX.json per seed, e.g. ring_avg_r25_s001.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from baseline.aggregation import average_models, fair2zero_gaps, fedavg_weights, reweight
from baseline.model import MLP, evaluate, load_split, train_one_epoch


def _random_regular(n, k, rng):
# Random connected graph where every node has exactly k neighbours.
# Each node gets k link ends; the ends are shuffled and paired up. A draw with
# a self-link, a repeated link or a disconnected graph is redrawn
    while True:
        ends = rng.permutation(np.repeat(np.arange(n), k))
        links = {tuple(sorted((int(a), int(b)))) for a, b in zip(ends[::2], ends[1::2])}
        if len(links) < n * k // 2 or any(a == b for a, b in links):
            continue
        nbrs = [[] for _ in range(n)]
        for a, b in links:
            nbrs[a].append(b)
            nbrs[b].append(a)
        # Walk the graph from node 0: connected if every node is reached
        seen, stack = {0}, [0]
        while stack:
            for j in nbrs[stack.pop()]:
                if j not in seen:
                    seen.add(j)
                    stack.append(j)
        if len(seen) == n:
            return [sorted(x) for x in nbrs]


def build_topology(name, n, rng):
# Neighbour list per node (self excluded). Clients are placed on the
# graph in a random order, so ring neighbours change with the seed
#
#   "ring"   node at position p talks to positions p-1 and p+1 (wrapping
#            around), so every node has exactly 2 neighbours
#   "full"   every node talks to every other node (same as FL)
#   "kreg3"  random connected graph where every node has exactly 3 neighbours;
#            the graph depends on the seed and stays fixed during the run
# Returns a list where entry i is the sorted list of node i's neighbours.
    order = rng.permutation(n)
    if name == "ring":
        nbrs = [[] for _ in range(n)]
        for p in range(n):
            nbrs[order[p]] = sorted({int(order[(p - 1) % n]), int(order[(p + 1) % n])})
        return nbrs
    if name == "full":
        return [[j for j in range(n) if j != i] for i in range(n)]
    if name == "kreg3":
        return _random_regular(n, 3, rng)
    raise ValueError(f"unknown topology {name}")


def _flat(state):
    # All parameters of a state_dict as one long vector (for measuring distances)
    return torch.cat([v.flatten() for v in state.values()])


def run_name(topology, method, metric, beta, rounds):
    # Avg does not use a fairness metric or beta
    # e.g. "ring_avg_r100" or "ring_fair2zero_spd_b0.05_r25"
    tag = f"{topology}_{method}"
    if method != "avg":
        tag += f"_{metric}_b{beta}"
    return tag + f"_r{rounds}"


def run_gossip(seed, method="avg", metric="spd", topology="ring", beta=0.05,
               eps=1e-3, rounds=25, lr=0.05, batch_size=128, on_round=None):
    # on_round(t, record, models): optional hook for the control panel,
    # called once before training (t = 0) and after every round
    #
    # method    "avg" (gossip counterpart of FedAvg) or "fair2zero"
    # metric    fairness metric the weights react to: "spd" or "eod"
    # topology  "ring", "full" or "kreg3"; beta, eps: as in fed.run_fl
    #
    # Returns a dict with the run settings, the graph, the message count, the
    # final mean / worst / consensus metrics, and a per-round "history".
    test, clients = load_split(seed)
    n = len(clients)
    sizes = np.array([len(c[0]) for c in clients])
    # The graph gets its own random stream ([seed, 1]), so it does not disturb
    # the torch randomness that FL also uses
    nbrs = build_topology(topology, n, np.random.default_rng([seed, 1]))
    # hood[i]: node i plus its neighbours, the set of models node i averages
    hood = [sorted([i] + nbrs[i]) for i in range(n)]

    # Same seed -> same starting model (shared by all nodes) and batch order as FL
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    init = MLP(n_features=clients[0][0].shape[1])
    models = [MLP(n_features=init.net[0].in_features) for _ in range(n)]
    for m in models:
        m.load_state_dict(init.state_dict())
    # weights[i]: node i's weights over hood[i], starting proportional to size
    weights = [fedavg_weights(sizes[h]) for h in hood]
    # raw[i]: the same before the floor and renormalising, carried between rounds
    raw = list(weights)
    if on_round:
        on_round(0, {"client_sizes": sizes.tolist(), "neighbours": nbrs,
                     "hood": hood, "weights": [w.tolist() for w in weights]}, models)

    history = []
    for t in range(1, rounds + 1):
        # Round 1 has no metrics from a previous round yet, so the weights
        # stay proportional to size, as in FL. Reweighting starts in round 2
        reweight_now = method == "fair2zero" and t > 1
        if reweight_now:
            # Each node measures its current model on its own data (sent as d_j)
            local = [evaluate(models[i], *clients[i]) for i in range(n)]
            local_fair = [m[metric] for m in local]
            local_acc = np.array([m["accuracy"] for m in local])
        elif method not in ("avg", "fair2zero"):
            raise ValueError(f"unknown method {method}")

        # Local training: every node trains its own model for one epoch
        for i, (X, y, _) in enumerate(clients):
            train_one_epoch(models[i], X, y, lr=lr, batch_size=batch_size,
                            generator=g)
        states = [m.state_dict() for m in models]

        # Every node merges its neighbourhood synchronously
        # (all merges read the same post-training `states`, and the results
        # are only loaded once every node has merged)
        merged = []
        for i, h in enumerate(hood):
            if reweight_now:
                # Fallback reference: size-weighted accuracy of the neighbourhood
                # (the gossip stand-in for FL's global accuracy)
                ref_acc = np.average(local_acc[h], weights=sizes[h])
                gaps = fair2zero_gaps([local_fair[j] for j in h], local_acc[h], ref_acc)
                raw[i], weights[i] = reweight(raw[i], gaps, beta, eps)
            merged.append(average_models([states[j] for j in h], weights[i]))
        for m, s in zip(models, merged):
            m.load_state_dict(s)

        # Per-node test metrics, plus the consensus (parameter-averaged) model
        nodes = [evaluate(m, *test) for m in models]
        consensus = MLP(n_features=init.net[0].in_features)
        mean_state = average_models(merged, np.full(n, 1 / n))
        consensus.load_state_dict(mean_state)
        centre = _flat(mean_state)
        history.append({
            "round": t,
            "nodes": nodes,
            # Mean over nodes; fairness values are made absolute first, so a
            # +0.1 node and a -0.1 node do not cancel out
            "mean": {k: float(np.mean([abs(x[k]) for x in nodes]))
                     for k in ("accuracy", "spd", "eod")},
            # Worst node: lowest accuracy, and largest |SPD| / |EOD|
            "worst": {"accuracy": min(x["accuracy"] for x in nodes),
                      "spd": max(abs(x["spd"]) for x in nodes),
                      "eod": max(abs(x["eod"]) for x in nodes)},
            "consensus": evaluate(consensus, *test),
            # How far the nodes are from agreeing: mean distance (L2 norm of
            # the parameter difference) between each node and the consensus.
            # It is 0 when all nodes hold the same model, as in FL
            "consensus_distance": float(np.mean(
                [(_flat(s) - centre).norm().item() for s in merged])),
            "weights": [w.tolist() for w in weights],
        })
        if on_round:
            on_round(t, history[-1], models)

    # messages: each node sends its model to each neighbour once per round
    # (ring: 10 nodes x 2 x 25 rounds = 500; kreg3: 10 x 3 x 25 = 750;
    # full: 10 x 9 x 25 = 2250)
    return {"seed": seed, "method": method, "metric": metric,
            "topology": topology, "beta": beta, "eps": eps, "rounds": rounds,
            "neighbours": nbrs, "client_sizes": sizes.tolist(),
            "messages": rounds * sum(len(x) for x in nbrs),
            "final": {k: history[-1][k] for k in ("mean", "worst", "consensus")},
            "history": history}


def main():
    # Command-line entry point: runs one configuration over a range of seeds
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["avg", "fair2zero"], default="avg")
    p.add_argument("--metric", choices=["spd", "eod"], default="spd",
                   help="fairness metric used for reweighting (ignored by avg)")
    p.add_argument("--topology", choices=["ring", "full", "kreg3"], default="ring")
    p.add_argument("--beta", type=float, default=0.05)
    p.add_argument("--eps", type=float, default=1e-3)
    p.add_argument("--rounds", type=int, default=25)
    # Inclusive range, e.g. "--seeds 1 100" runs seeds 1, 2, ..., 100
    p.add_argument("--seeds", type=int, nargs=2, default=[1, 100],
                   metavar=("FIRST", "LAST"))
    p.add_argument("--out", default="results/gossip")
    args = p.parse_args()

    tag = run_name(args.topology, args.method, args.metric, args.beta, args.rounds)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for seed in range(args.seeds[0], args.seeds[1] + 1):
        result = run_gossip(seed, args.method, args.metric, args.topology,
                            args.beta, args.eps, args.rounds)
        (out / f"{tag}_s{seed:03d}.json").write_text(json.dumps(result))
        f = result["final"]
        print(f"{tag} seed {seed:3d}: mean acc {f['mean']['accuracy']:.4f}  "
              f"mean |spd| {f['mean']['spd']:.4f}  "
              f"worst |spd| {f['worst']['spd']:.4f}  "
              f"consensus |spd| {abs(f['consensus']['spd']):.4f}")


if __name__ == "__main__":
    main()
