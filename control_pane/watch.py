"""
Runs one experiment slowly for the control panel.

Prints one JSON line per round on stdout (setup, round, end) and reads control
commands on stdin: pause, resume, step, delay <seconds>, stop.
The finished result is saved to results/watch/, never next to the batch results.

Example (normally started by server.py):
    python watch.py --setting gossip --topology ring --method fair2zero --delay 1

How it works: the normal training functions (fed.run_fl / gossip.run_gossip)
are called with an on_round hook. After every round the hook prints the
round's numbers as an event, then blocks in Control.wait() until it is time
for the next round. A background thread reads stdin and changes the Control
state (pause, speed, stop) while training is waiting.

Events printed (one JSON object per line, field "type"):
  setup    before round 1: config, client sizes, graph, initial weights
  round    after each round: test metrics, per-node local metrics, weights
  status   after pause / resume: whether the run is paused
  end      the run finished; "path" is where the result was saved
  stopped  the run was stopped early; nothing is saved

Note: this imports fed, gossip, aggregation and model, so it must run with
the experiment folder (where those files live) as the working directory.
"""

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

import federated_learning_loop
import gossip_learning_loop
from aggregation import fedavg_weights
from model import evaluate, load_split

# Both threads print, so each event line is written under a lock to keep
# lines from interleaving
_print_lock = threading.Lock()


def emit(event):
    # One event as one JSON line, flushed at once so the server sees it now
    with _print_lock:
        print(json.dumps(event), flush=True)


class Stopped(Exception):
    # Raised from inside the on_round hook to abandon the training loop
    pass


class Control:
    # Pause / step / speed state, changed by commands arriving on stdin
    #   delay    seconds to wait between rounds while running
    #   paused   while True, rounds only advance on "step"
    #   steps    "step" commands not used yet (each lets one round through)
    #   stopped  set by "stop" or when stdin closes; ends the run
    #   wake     event that cuts a wait short when a command arrives
    def __init__(self, delay):
        self.delay, self.paused, self.steps, self.stopped = delay, False, 0, False
        self.wake = threading.Event()

    def listen(self):
        # Runs in a background thread: one command per stdin line
        for line in sys.stdin:
            # First word is the command, the rest its arguments; a blank
            # line becomes the empty command and is ignored
            cmd, *arg = line.split() or [""]
            if cmd == "pause":
                self.paused = True
            elif cmd == "resume":
                self.paused = False
            elif cmd == "step":
                self.steps += 1
            elif cmd == "delay" and arg:
                self.delay = max(0.0, float(arg[0]))
            elif cmd == "stop":
                self.stopped = True
            if cmd in ("pause", "resume"):
                emit({"type": "status", "paused": self.paused})
            self.wake.set()
        self.stopped = True  # stdin closed: the server has gone away
        self.wake.set()

    def wait(self):
        # Hold between rounds for the delay, or until resumed / stepped
        # Checks the state every 50 ms (sooner if a command arrives).
        # Raises Stopped if a stop is requested while waiting
        end = time.time() + self.delay
        while not self.stopped:
            if self.paused:
                if self.steps:
                    self.steps -= 1
                    return
            elif time.time() >= end:
                return
            self.wake.wait(0.05)
            self.wake.clear()
        raise Stopped


def _abs(metrics):
    # Metrics in the panel's format: short "acc" key, absolute fairness values
    return {"acc": metrics["accuracy"],
            "spd": None if metrics["spd"] is None else abs(metrics["spd"]),
            "eod": None if metrics["eod"] is None else abs(metrics["eod"])}


def _influence_and_edges(setting, weights, hood, n):
    # FL: a client's influence is its weight; edges join it to the server (-1).
    # Gossip: influence is the average weight other nodes put on it; an edge's
    # weight is the mean of what its two ends give each other
    #
    # Returns (influence: one number per node, edges: [from, to, weight] list),
    # which the panel uses to size nodes and draw links
    if setting == "fl":
        w = [float(x) for x in weights]
        return w, [[k, -1, w[k]] for k in range(n)]
    # W[i, j] = weight node i gives node j's model (0 if j is not in its hood)
    W = np.zeros((n, n))
    for i, h in enumerate(hood):
        W[i, h] = weights[i]
    # One edge per neighbour pair (j > i, so each pair appears once)
    edges = [[i, j, float((W[i, j] + W[j, i]) / 2)]
             for i in range(n) for j in hood[i] if j > i]
    # Column means: the average weight each node receives
    return (W.sum(0) / n).tolist(), edges


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--setting", choices=["fl", "gossip"], default="gossip")
    # fl: fedavg / fairfed / fair2zero; gossip: avg / fair2zero
    p.add_argument("--method", default="avg")
    p.add_argument("--metric", choices=["spd", "eod"], default="spd")
    p.add_argument("--topology", choices=["ring", "full"], default="ring")
    p.add_argument("--beta", type=float, default=0.05)
    p.add_argument("--rounds", type=int, default=25)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--delay", type=float, default=1.0)
    args = p.parse_args()

    control = Control(args.delay)
    # Daemon thread: it does not keep the process alive once main() returns
    threading.Thread(target=control.listen, daemon=True).start()
    # The clients' data is loaded here too, to score models on each client
    _, clients = load_split(args.seed)
    n = len(clients)
    config = {k: getattr(args, k) for k in
              ("setting", "method", "metric", "topology", "beta", "rounds", "seed")}
    # Values from the t = 0 call that later rounds still need
    ctx = {}

    def on_round(t, record, models):
        # Called by run_fl / run_gossip at t = 0 and after every round.
        # `record` is that round's history entry; `models` holds the global
        # model (FL, a list of one) or every node's model (gossip)
        #
        # Each node's model on its own data: the local unfairness d_i that
        # Fair2Zero reacts to (FL: the global model on each client's data)
        local = [_abs(evaluate(models[k if len(models) > 1 else 0], *clients[k]))
                 for k in range(n)]
        if t == 0:
            ctx["sizes"] = record["client_sizes"]
            ctx["hood"] = record.get("hood")  # None in FL
            # run_fl does not report its starting weights, so recompute them
            weights = (fedavg_weights(ctx["sizes"]) if args.setting == "fl"
                       else record["weights"])
            influence, edges = _influence_and_edges(args.setting, weights, ctx["hood"], n)
            emit({"type": "setup", "t": 0, "config": config, "n": n,
                  "sizes": ctx["sizes"], "neighbours": record.get("neighbours"),
                  "local": local, "influence": influence, "edges": edges})
        else:
            influence, edges = _influence_and_edges(args.setting, record["weights"],
                                                    ctx["hood"], n)
            if args.setting == "fl":
                # One global model: it is both the average and the worst case,
                # and all clients agree, so the consensus distance is 0
                test = worst = _abs(record)
                distance = 0.0
            else:
                test = {"acc": record["mean"]["accuracy"], "spd": record["mean"]["spd"],
                        "eod": record["mean"]["eod"]}
                worst = {"acc": record["worst"]["accuracy"],
                         "spd": record["worst"]["spd"], "eod": record["worst"]["eod"]}
                distance = record["consensus_distance"]
            emit({"type": "round", "t": t, "rounds": args.rounds, "test": test,
                  "worst": worst, "local": local, "influence": influence,
                  "edges": edges, "consensus_distance": distance})
        # Block here (inside the training loop) until the next round may start
        control.wait()

    try:
        if args.setting == "fl":
            result = federated_learning_loop.run_fl(args.seed, args.method, args.metric, args.beta,
                                rounds=args.rounds, on_round=on_round)
            name = federated_learning_loop.run_name(args.method, args.metric, args.beta)
        else:
            result = gossip_learning_loop.run_gossip(args.seed, args.method, args.metric,
                                       args.topology, args.beta, rounds=args.rounds,
                                       on_round=on_round)
            name = gossip_learning_loop.run_name(args.topology, args.method, args.metric,
                                   args.beta, args.rounds)
    except Stopped:
        emit({"type": "stopped"})
        return

    # Save with write-then-rename, as in run.py, so the file is never half-written
    path = Path("results/watch") / f"{args.setting}_{name}_s{args.seed:03d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result))
    os.replace(tmp, path)
    emit({"type": "end", "path": str(path)})


if __name__ == "__main__":
    main()
