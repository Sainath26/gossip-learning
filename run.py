"""
Runs every experiment from scratch and summarises the results.

  1. data:    prepare_data + make_splits (skipped if present, unless --fresh)
  2. runs:    every FL and gossip configuration over all seeds, in parallel
              (existing result files are reused, unless --fresh)
  3. summary: <results>/runs.csv     one row per run (per seed)
              <results>/summary.csv  mean ± 95% CI per configuration,
                                     paired against its baseline
              <results>/summary.md   the same as readable tables

Result files are laid out as <results>/<setting>/<config name>_sXXX.json,
where setting is "fl" or "gossip" and the config name comes from
fed.run_name / gossip.run_name.

Examples:
    python run.py                  # only runs what is missing
    python run.py --fresh          # regenerate data and redo every run
    python run.py --summary-only   # just rebuild the tables
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

import federated_learning_loop
import gossip_learning_loop
from data_prepration import make_splits, prepare_data

BETA = 0.05
METRICS = ("spd", "eod")

# Every configuration; each one is run for every seed
# Keys left out of a dict take their default from _full() below.
# An experiment's id is its position in this list (used by --only and the
# control panel), so append new entries at the end to keep ids stable.
EXPERIMENTS = (
    [dict(setting="fl", method="fedavg")]
    + [dict(setting="fl", method=m, metric=k)
       for k in METRICS for m in ("fairfed", "fair2zero")]
    # Fully connected gossip equals FL: kept as a correctness check
    + [dict(setting="gossip", topology="full", method="avg")]
    + [dict(setting="gossip", topology="full", method="fair2zero", metric=k)
       for k in METRICS]
    # Ring gossip at 25 rounds (same as FL) and 100 rounds, since sparse
    # graphs need longer for information to spread
    + [dict(setting="gossip", topology="ring", method="avg", rounds=r)
       for r in (25, 100)]
    + [dict(setting="gossip", topology="ring", method="fair2zero", metric=k, rounds=r)
       for r in (25, 100) for k in METRICS]
    # Random 3-regular gossip, at the same round counts as the ring
    + [dict(setting="gossip", topology="kreg3", method="avg", rounds=r)
       for r in (25, 100)]
    + [dict(setting="gossip", topology="kreg3", method="fair2zero", metric=k, rounds=r)
       for r in (25, 100) for k in METRICS]
)

# Adult results from Table 2 of the paper: (accuracy, |fairness|)
# keyed by (method, metric the method was trained on)
PAPER = {("fedavg", "spd"): (0.8470, 0.1673), ("fairfed", "spd"): (0.8423, 0.1605),
         ("fair2zero", "spd"): (0.8326, 0.0719), ("fedavg", "eod"): (0.8470, 0.1512),
         ("fairfed", "eod"): (0.8452, 0.1358), ("fair2zero", "eod"): (0.8424, 0.1459)}


def _full(exp):
    # Fill in the defaults an EXPERIMENTS entry may leave out. FL has no
    # graph, so its topology is recorded as "server"
    return {"topology": "server", "metric": "spd", "beta": BETA, "rounds": 25, **exp}


def _is_baseline(method):
    # The non-fairness-aware methods that the others are compared against
    return method in ("fedavg", "avg")


def config_name(exp):
    # The file-name tag of an experiment, e.g. "fair2zero_spd_b0.05"
    e = _full(exp)
    if e["setting"] == "fl":
        return federated_learning_loop.run_name(e["method"], e["metric"], e["beta"])
    return gossip_learning_loop.run_name(e["topology"], e["method"], e["metric"],
                           e["beta"], e["rounds"])


def result_path(root, exp, seed):
    # Where the result of one (experiment, seed) run is stored
    return Path(root) / exp["setting"] / f"{config_name(exp)}_s{seed:03d}.json"


def _emit(event):
    # One JSON line on stdout, read by the control panel server
    print(json.dumps(event), flush=True)


def _job(exp, seed, path):
    # Runs one experiment for one seed and saves the result. Executed inside
    # a worker process of the pool in run_all()
    #
    # One thread per worker; the parallelism comes from running seeds side by side
    # (torch is imported here so the setting applies inside the worker process)
    import torch
    torch.set_num_threads(1)
    e = _full(exp)
    if e["setting"] == "fl":
        result = federated_learning_loop.run_fl(seed, e["method"], e["metric"], e["beta"],
                            rounds=e["rounds"])
    else:
        result = gossip_learning_loop.run_gossip(seed, e["method"], e["metric"], e["topology"],
                                   e["beta"], rounds=e["rounds"])
    # Write then rename, so a run killed mid-write never leaves a broken file
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result))
    os.replace(tmp, path)


def run_all(root, seeds, workers, fresh, only=None, events=False):
    # Runs every (experiment, seed) pair whose result file is missing (or
    # every pair, if fresh) on a pool of `workers` processes.
    #   only    experiment ids to restrict to (None = all of EXPERIMENTS)
    #   events  also print JSON progress events for the control panel
    ids = range(len(EXPERIMENTS)) if only is None else only
    # A job is (experiment id, seed, result path)
    jobs = [(i, s, result_path(root, EXPERIMENTS[i], s)) for i in ids for s in seeds]
    todo = [j for j in jobs if fresh or not j[2].exists()]
    print(f"{len(jobs)} runs, {len(jobs) - len(todo)} already done, "
          f"{len(todo)} to run on {workers} workers")
    if events:
        # Tell the panel which configurations are in this batch and how many
        # seeds each one already has on disk
        _emit({"type": "batch_start", "total": len(jobs), "todo": len(todo),
               "configs": [{"id": i, "name": f"{EXPERIMENTS[i]['setting']}: "
                                             f"{config_name(EXPERIMENTS[i])}",
                            "total": len(seeds),
                            "done": sum(not fresh and result_path(root, EXPERIMENTS[i], s).exists()
                                        for s in seeds)} for i in ids]})
    with ProcessPoolExecutor(workers) as pool:
        futures = {pool.submit(_job, EXPERIMENTS[i], s, p): (i, s) for i, s, p in todo}
        # as_completed yields jobs in the order they finish, not submit order
        for n, fut in enumerate(as_completed(futures), 1):
            fut.result()  # re-raises here any exception from the worker
            if events:
                i, s = futures[fut]
                _emit({"type": "run_done", "id": i, "seed": s})
            if n % 50 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)} done")
    if events:
        _emit({"type": "batch_end"})


def _row(exp, seed, r):
    # Flattens one result file into one table row: the configuration, the
    # seed, absolute test metrics for the mean / worst / consensus model
    # (columns "", "worst_", "cons_"), and the number of messages sent
    e = _full(exp)
    base = _is_baseline(e["method"])
    # Baselines ignore metric and beta, so those are blanked ("-" / NaN);
    # metric == "-" is also how summarise() finds the baseline rows
    row = dict(setting=e["setting"], topology=e["topology"], rounds=e["rounds"],
               method=e["method"], metric="-" if base else e["metric"],
               beta=np.nan if base else e["beta"], seed=seed)
    f = r["final"]
    if e["setting"] == "fl":
        # One global model: it is the mean, the worst and the consensus model
        g = {k: abs(f[k]) for k in ("accuracy", "spd", "eod")}
        mean = worst = cons = g
        messages = 2 * len(r["client_sizes"]) * r["rounds"]  # down + up per client
    else:
        # gossip.py already made mean / worst fairness absolute
        mean, worst = f["mean"], f["worst"]
        cons = {k: abs(v) for k, v in f["consensus"].items()}
        messages = r["messages"]
    for prefix, m in (("", mean), ("worst_", worst), ("cons_", cons)):
        row[prefix + "acc"] = m["accuracy"]
        row[prefix + "spd"] = m["spd"]
        row[prefix + "eod"] = m["eod"]
    row["messages"] = messages
    return row


# Columns that identify a configuration (summary rows are grouped by these)
KEYS = ["setting", "topology", "rounds", "method", "metric", "beta"]
# Columns that get a mean and CI in summary.csv. d_* are paired differences
# against the baseline; d_target is d_spd or d_eod, whichever the method used
VALUES = ["acc", "spd", "eod", "worst_acc", "worst_spd", "worst_eod",
          "cons_acc", "cons_spd", "cons_eod", "d_acc", "d_spd", "d_eod", "d_target"]


def _ci(v):
    # Half-width of a 95% confidence interval for the mean (normal
    # approximation: 1.96 x standard error). NaN with fewer than 2 values
    v = v.dropna()
    return 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan


def summarise(root):
    # Covers every result on disk, whatever seeds the last batch ran
    # Writes runs.csv, summary.csv and summary.md into `root`
    root = Path(root)
    # The seed is parsed back out of the file name ("..._s042.json" -> 42)
    rows = [_row(exp, int(p.stem.rsplit("_s", 1)[1]), json.loads(p.read_text()))
            for exp in EXPERIMENTS
            for p in sorted((root / exp["setting"]).glob(f"{config_name(exp)}_s[0-9][0-9][0-9].json"))]
    if not rows:
        print("no results to summarise")
        return
    runs = pd.DataFrame(rows)

    # Paired difference against the baseline on the same seed and setting
    # (FedAvg for FL, gossip Avg on the same topology and rounds)
    # Pairing by seed removes the seed-to-seed variation (different splits
    # and initialisations) and so gives tighter CIs than comparing means
    on = ["setting", "topology", "rounds", "seed"]
    base = runs[runs["metric"] == "-"][on + ["acc", "spd", "eod"]]
    runs = runs.merge(base, on=on, how="left", suffixes=("", "_base"))
    for k in ("acc", "spd", "eod"):
        runs[f"d_{k}"] = runs[k] - runs[f"{k}_base"]
        runs = runs.drop(columns=f"{k}_base")
    # The change in the metric the method actually optimised (NaN for baselines)
    runs["d_target"] = np.select([runs["metric"] == "spd", runs["metric"] == "eod"],
                                 [runs["d_spd"], runs["d_eod"]], np.nan)
    runs.to_csv(root / "runs.csv", index=False)

    # One summary row per configuration. dropna=False keeps the baselines,
    # whose beta is NaN; sort=False keeps the EXPERIMENTS order
    grouped = runs.groupby(KEYS, sort=False, dropna=False)
    summary = grouped.size().rename("n_seeds").to_frame()
    for v in VALUES:
        summary[f"{v}_mean"] = grouped[v].mean()
        summary[f"{v}_ci"] = grouped[v].agg(_ci)
    # Number of seeds where the method was fairer than its baseline
    summary["better_in"] = grouped["d_target"].agg(lambda d: int((d < 0).sum()))
    summary["messages"] = grouped["messages"].first()
    summary = summary.reset_index()
    summary.to_csv(root / "summary.csv", index=False)

    (root / "summary.md").write_text(_markdown(summary), encoding="utf-8")
    print(f"wrote {root / 'runs.csv'}, {root / 'summary.csv'}, {root / 'summary.md'}")


def _markdown(s):
    # Renders the summary DataFrame as summary.md: one combined table for
    # presenting, one table for FL (with the paper's numbers alongside) and
    # one for gossip

    # "mean ± CI" for column k, or a dash when there is no value
    def pm(r, k):
        return "—" if pd.isna(r[f"{k}_mean"]) else f"{r[f'{k}_mean']:.4f} ± {r[f'{k}_ci']:.4f}"

    # The "mean ± CI" cells of column k for one setup, with the best value
    # bold (pick is max or min; values equal to 4 decimals are all bold)
    def cells(g, k, pick):
        v = g[f"{k}_mean"].round(4)
        return [f"**{pm(r, k)}**" if x == pick(v) else pm(r, k)
                for x, (_, r) in zip(v, g.iterrows())]

    # The "Δ target" and "Better in" cells (dashes for baselines)
    def delta(r):
        if r["metric"] == "-":
            return "—", "—"
        return pm(r, "d_target"), f"{r['better_in']}/{r['n_seeds']}"

    out = [f"# Results\n",
           f"Mean ± 95% CI (1.96 · SD / √n) over seeds; β = {BETA}. "
           f"Fairness values are absolute. \"Δ target\" is the paired change in the "
           f"metric the method was trained on, against the baseline on the same seed "
           f"(negative = fairer); \"better in\" counts seeds where it improved.\n"]

    # Combined table: rows are grouped by setup, and methods are compared
    # within a setup (same setting, topology and rounds)
    settings = {"fl": "Federated", "gossip": "Gossip"}
    topologies = {"server": "—", "ring": "Ring", "kreg3": "Random 3-regular", "full": "Full"}
    methods = {"fedavg": "FedAvg", "fairfed": "FairFed", "fair2zero": "Fair2Zero", "avg": "Avg"}
    out += ["## All results\n",
            "Bold marks the best value in each column within a setup (same setting, "
            "topology and rounds): highest accuracy, lowest \\|SPD\\| and \\|EOD\\|. "
            "Gossip values are averaged over the 10 nodes; federated values come from "
            "the one global model.\n",
            "| Setting | Topology | Rounds | Method | Trained on | Accuracy | \\|SPD\\| "
            "| \\|EOD\\| | Messages |",
            "|---|---|---|---|---|---|---|---|---|"]
    for _, g in s.groupby(["setting", "topology", "rounds"], sort=False):
        columns = zip(cells(g, "acc", max), cells(g, "spd", min), cells(g, "eod", min))
        for (_, r), (acc, sp, eo) in zip(g.iterrows(), columns):
            metric = "—" if r["metric"] == "-" else r["metric"].upper()
            out.append(f"| {settings[r['setting']]} | {topologies[r['topology']]} "
                       f"| {r['rounds']} | {methods[r['method']]} | {metric} "
                       f"| {acc} | {sp} | {eo} | {r['messages']} |")
    out.append("")

    fl = s[s["setting"] == "fl"]
    out += ["## Federated learning (Adult, 25 rounds)\n",
            "| Method | Trained on | Accuracy | \\|SPD\\| | \\|EOD\\| | Δ target vs FedAvg "
            "| Better in | Paper acc | Paper \\|target\\| |",
            "|---|---|---|---|---|---|---|---|---|"]
    for _, r in fl.iterrows():
        d, b = delta(r)
        if r["metric"] == "-":
            # FedAvg is not trained on a metric, so show both paper values
            paper = (f"{PAPER[('fedavg', 'spd')][0]:.4f}",
                     f"{PAPER[('fedavg', 'spd')][1]:.4f} / {PAPER[('fedavg', 'eod')][1]:.4f}")
        else:
            a, f = PAPER[(r["method"], r["metric"])]
            paper = (f"{a:.4f}", f"{f:.4f}")
        out.append(f"| {r['method']} | {r['metric']} | {pm(r, 'acc')} | {pm(r, 'spd')} "
                   f"| {pm(r, 'eod')} | {d} | {b} | {paper[0]} | {paper[1]} |")
    out.append("\nFedAvg's paper fairness is shown as SPD / EOD.\n")

    gl = s[s["setting"] == "gossip"]
    out += ["## Gossip learning (Adult)\n",
            "Accuracy and fairness are averaged over the 10 nodes; \"worst\" is the "
            "least fair node; \"consensus\" is the parameter average of all nodes.\n",
            "| Topology | Rounds | Method | Trained on | Mean acc | Mean \\|SPD\\| "
            "| Mean \\|EOD\\| | Worst \\|SPD\\| | Consensus \\|SPD\\| | Δ target vs Avg "
            "| Better in | Messages |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in gl.iterrows():
        d, b = delta(r)
        out.append(f"| {r['topology']} | {r['rounds']} | {r['method']} | {r['metric']} "
                   f"| {pm(r, 'acc')} | {pm(r, 'spd')} | {pm(r, 'eod')} "
                   f"| {pm(r, 'worst_spd')} | {pm(r, 'cons_spd')} | {d} | {b} "
                   f"| {r['messages']} |")
    out.append("\nFully connected gossip is mathematically identical to FL and should "
               "match the FL table exactly.\n")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    # Inclusive range, e.g. "--seeds 1 100" runs seeds 1, 2, ..., 100
    p.add_argument("--seeds", type=int, nargs=2, default=[1, 100],
                   metavar=("FIRST", "LAST"))
    # Default: half the CPU cores (at least 1)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    p.add_argument("--results", default="results")
    p.add_argument("--fresh", action="store_true",
                   help="regenerate data and splits, and redo every run")
    p.add_argument("--summary-only", action="store_true")
    p.add_argument("--only", default=None,
                   help="comma-separated experiment ids (positions in EXPERIMENTS)")
    p.add_argument("--events", action="store_true",
                   help="print JSON progress lines (used by server.py)")
    args = p.parse_args()
    seeds = range(args.seeds[0], args.seeds[1] + 1)
    only = None if args.only is None else [int(i) for i in args.only.split(",")]

    if not args.summary_only:
        # Build the processed data and the splits only if they are missing.
        # Note: make_splits() always writes its default seeds 1-100, whatever
        # --seeds says
        splits = [Path("data/splits") / f"seed_{s:03d}.npz" for s in seeds]
        if args.fresh or not Path("data/processed/adult.npz").exists():
            prepare_data()
        if args.fresh or not all(p.exists() for p in splits):
            make_splits()
        run_all(args.results, seeds, args.workers, args.fresh, only, args.events)
    summarise(args.results)
    if args.events:
        _emit({"type": "summary"})


if __name__ == "__main__":
    main()
