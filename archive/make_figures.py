"""
Two summary figures for a short update email.

  fig1_fl_reproduction.png  FL on Adult: |SPD| of FedAvg / FairFed / Fair2Zero,
                            this reproduction next to the paper's Table 2
  fig2_gossip_ring.png      Gossip on a ring, 100 rounds: |SPD| per round for
                            Avg and Fair2Zero (mean node and worst node),
                            with the FL results at round 25 for reference

Reads results/summary.csv (from run.py) and the ring result files.
Run run.py first; both figures are written to <results>/figures/.

Example:
    python make_figures.py
"""

import argparse
import json
from pathlib import Path

import matplotlib
# Non-interactive backend: figures are only saved to files, no window opens
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Reference palette (light mode): categorical slots 1-2, ink, chrome
# Blue is always "ours" / plain averaging, orange the paper / Fair2Zero
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"

# Paper, Table 2 (Adult, trained on SPD): (accuracy, |SPD|, CI)
PAPER = {"fedavg": (0.8470, 0.1673, 0.0075), "fairfed": (0.8423, 0.1605, 0.0126),
         "fair2zero": (0.8326, 0.0719, 0.0079)}
LABELS = {"fedavg": "FedAvg", "fairfed": "FairFed", "fair2zero": "Fair2Zero"}


def _style(ax):
    # Shared minimal look: no box, light horizontal grid behind the data
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _titles(fig, title, subtitle):
    # Left-aligned title and subtitle above the plot area
    # (positions are fractions of the figure size)
    fig.text(0.02, 0.965, title, fontsize=12, fontweight="bold", color=INK, va="top")
    fig.text(0.02, 0.905, subtitle, fontsize=9, color=INK_2, va="top")


def fl_figure(summary, path):
    # Figure 1: grouped bars, |SPD| per method, ours (blue) next to the
    # paper (orange), with 95% CI error bars and accuracies under the x-axis
    fl = summary[summary["setting"] == "fl"]
    # The fairness-aware methods are shown in their SPD-trained runs
    rows = {"fedavg": fl[fl["method"] == "fedavg"].iloc[0]}
    for m in ("fairfed", "fair2zero"):
        rows[m] = fl[(fl["method"] == m) & (fl["metric"] == "spd")].iloc[0]
    methods = list(rows)
    n_seeds = int(rows["fedavg"]["n_seeds"])

    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=200, facecolor=SURFACE)
    fig.subplots_adjust(top=0.80, bottom=0.2, left=0.09, right=0.98)
    _style(ax)
    # x: one slot per method; w: bar width. The two bars sit either side of
    # the slot centre with a small gap between them
    x, w = np.arange(len(methods)), 0.36
    ours = [rows[m]["spd_mean"] for m in methods]
    ours_ci = [rows[m]["spd_ci"] for m in methods]
    paper = [PAPER[m][1] for m in methods]
    paper_ci = [PAPER[m][2] for m in methods]
    err = dict(ecolor=INK_2, elinewidth=1, capsize=3)
    ax.bar(x - w / 2 - 0.01, ours, w, yerr=ours_ci, color=BLUE,
           label="This reproduction", error_kw=err)
    ax.bar(x + w / 2 + 0.01, paper, w, yerr=paper_ci, color=ORANGE,
           label="Paper (Table 2)", error_kw=err)
    # Value labels just above each error bar
    for xi, (o, oc, p, pc) in enumerate(zip(ours, ours_ci, paper, paper_ci)):
        ax.text(xi - w / 2 - 0.01, o + oc + 0.004, f"{o:.3f}", ha="center",
                fontsize=8.5, color=INK)
        ax.text(xi + w / 2 + 0.01, p + pc + 0.004, f"{p:.3f}", ha="center",
                fontsize=8.5, color=INK)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{LABELS[m]}\nacc {rows[m]['acc_mean']:.3f} vs {PAPER[m][0]:.3f}"
                        for m in methods], fontsize=9, color=INK)
    ax.set_ylabel("|SPD|  (lower is fairer)", fontsize=9, color=INK_2)
    # 25% headroom above the tallest bar for the value labels and legend
    ax.set_ylim(0, max(max(ours), max(paper)) * 1.25)
    ax.legend(frameon=False, fontsize=9, loc="upper right", labelcolor=INK)
    _titles(fig, "Federated learning on Adult: reproduction vs paper",
            f"Fairness-aware methods targeting SPD. Mean ± 95% CI over "
            f"{n_seeds} seeds; accuracy shown as ours vs paper.")
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def _curves(files, key):
    # Per-round mean over seeds and 95% CI, for one statistic ("mean" or "worst")
    # v has shape (n_seeds, n_rounds): the |SPD| of every seed at every round.
    # Returns (mean per round, CI half-width per round, number of seeds)
    v = np.array([[h[key]["spd"] for h in json.loads(f.read_text())["history"]]
                  for f in files])
    return v.mean(0), 1.96 * v.std(0, ddof=1) / np.sqrt(len(v)), len(v)


def gossip_figure(results, summary, path, rounds=100, beta=0.05):
    # Figure 2: |SPD| over the rounds of ring gossip, for Avg (blue) and
    # Fair2Zero (orange). Solid line + band = mean node with 95% CI; dashed
    # line = worst node; hollow circle = the matching FL result at round 25
    g = Path(results) / "gossip"
    # (legend name, colour, result files of all seeds, matching FL method)
    series = [("Gossip Avg", BLUE, sorted(g.glob(f"ring_avg_r{rounds}_s*.json")), "fedavg"),
              ("Gossip Fair2Zero", ORANGE,
               sorted(g.glob(f"ring_fair2zero_spd_b{beta}_r{rounds}_s*.json")), "fair2zero")]
    fl = summary[summary["setting"] == "fl"]

    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=200, facecolor=SURFACE)
    # right=0.80 leaves room for the end-of-line labels
    fig.subplots_adjust(top=0.80, bottom=0.13, left=0.09, right=0.80)
    _style(ax)
    n_seeds = 0
    for name, color, files, fl_method in series:
        mean, mean_ci, n_seeds = _curves(files, "mean")
        worst, _, _ = _curves(files, "worst")
        t = np.arange(1, len(mean) + 1)
        ax.fill_between(t, mean - mean_ci, mean + mean_ci, color=color, alpha=0.15,
                        linewidth=0)
        ax.plot(t, mean, color=color, linewidth=1.6, label=f"{name}, mean node")
        ax.plot(t, worst, color=color, linewidth=1.2, linestyle=(0, (4, 2)),
                label=f"{name}, worst node ({worst[-1]:.3f})")
        # Label with the final value, just right of the line's end
        ax.text(t[-1] + 1.5, mean[-1], f"{name}\n{mean[-1]:.3f}",
                fontsize=8, color=INK, va="center")
        # FL result after its 25 rounds, as a reference point
        # (FedAvg's metric column is "-", the others use their SPD-trained row)
        row = fl[(fl["method"] == fl_method) &
                 (fl["metric"] == ("-" if fl_method == "fedavg" else "spd"))]
        if len(row):
            ax.plot(25, row.iloc[0]["spd_mean"], marker="o", markersize=7,
                    markerfacecolor=SURFACE, markeredgecolor=color,
                    markeredgewidth=1.6, linestyle="none")
    # An empty plot, only so the FL marker gets a (grey) legend entry
    ax.plot([], [], marker="o", markersize=7, markerfacecolor=SURFACE,
            markeredgecolor=MUTED, markeredgewidth=1.6, linestyle="none",
            label="FL result after 25 rounds")

    ax.set_xlim(1, rounds)
    ax.set_ylim(0, None)
    ax.set_xlabel("Round", fontsize=9, color=INK_2)
    ax.set_ylabel("|SPD|  (lower is fairer)", fontsize=9, color=INK_2)
    # Legend in two columns below the plot; the bottom margin grows to fit it
    ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK,
              bbox_to_anchor=(0, -0.13), ncol=2)
    fig.subplots_adjust(bottom=0.30)
    _titles(fig, "Gossip learning on a ring: Fair2Zero vs plain averaging",
            f"10 nodes, 2 neighbours each; targeting SPD. "
            f"Mean over {n_seeds} seeds, band = 95% CI.")
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--out", default=None, help="default: <results>/figures")
    args = p.parse_args()
    out = Path(args.out or Path(args.results) / "figures")
    out.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(Path(args.results) / "summary.csv")
    fl_figure(summary, out / "fig1_fl_reproduction.png")
    gossip_figure(args.results, summary, out / "fig2_gossip_ring.png")
    print(f"wrote {out / 'fig1_fl_reproduction.png'}, {out / 'fig2_gossip_ring.png'}")


if __name__ == "__main__":
    main()
