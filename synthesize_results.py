"""
Tables and figures for the presentation, as PNG files to paste into slides.

  table1_fl_vs_paper.png            FL results next to the paper's Table 2
  table2_all_results.png            every configuration, best values in bold
  table3_fair2zero_by_topology.png  Fair2Zero (trained on SPD) against its
                                    baseline, one row per setup
  fig1_fl_reproduction.png          bars from make_figures.py: ours vs the paper
  fig2_topology_curves.png          |SPD| per round on each sparse topology
  fig3_fairness_vs_messages.png     final |SPD| per setup, ordered by messages

Trains nothing and has no randomness: every result comes from
results/summary.csv and the gossip result files (written by run.py), so the
same results always give the same pictures. The only typed-in numbers are the
paper's, in PAPER_TABLE2. Stops if any configuration has fewer than 100 seeds.

Example:
    python synthesize_results.py
"""

import argparse
import sys
import textwrap
from pathlib import Path

import matplotlib
# Non-interactive backend: figures are only saved to files, no window opens
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import make_figures as mf
import run

N_SEEDS = 100
WHITE = "#ffffff"
# Adult rows of the paper's Table 2, as printed, per metric the method targets:
# method: (accuracy, CI, |fairness|, CI)
PAPER_TABLE2 = {
    "spd": {"fedavg": (0.8470, 0.0010, 0.1673, 0.0075),
            "fairfed": (0.8423, 0.0020, 0.1605, 0.0126),
            "fair2zero": (0.8326, 0.0027, 0.0719, 0.0079)},
    "eod": {"fedavg": (0.8470, 0.0010, 0.1512, 0.0180),
            "fairfed": (0.8452, 0.0015, 0.1358, 0.0190),
            "fair2zero": (0.8424, 0.0016, 0.1459, 0.0230)}}
SETTINGS = {"fl": "Federated", "gossip": "Gossip"}
TOPOLOGIES = {"server": "—", "full": "Full", "ring": "Ring", "kreg3": "Random 3-regular"}
METHODS = {"fedavg": "FedAvg", "fairfed": "FairFed", "fair2zero": "Fair2Zero", "avg": "Avg"}
# (setting, topology, rounds), in table order
SETUPS = [("fl", "server", 25), ("gossip", "full", 25), ("gossip", "ring", 25),
          ("gossip", "ring", 100), ("gossip", "kreg3", 25), ("gossip", "kreg3", 100)]


def find(summary, setting, topology, rounds, method, metric):
    # The summary rows of one configuration (one row, or none if it is missing)
    return summary[(summary["setting"] == setting) & (summary["topology"] == topology)
                   & (summary["rounds"] == rounds) & (summary["method"] == method)
                   & (summary["metric"] == metric)]


def missing_configs(summary):
    # Names of the configurations in run.EXPERIMENTS without all their seeds
    missing = []
    for exp in run.EXPERIMENTS:
        e = run._full(exp)
        metric = "-" if run._is_baseline(e["method"]) else e["metric"]
        row = find(summary, e["setting"], e["topology"], e["rounds"], e["method"], metric)
        if row.empty or row.iloc[0]["n_seeds"] < N_SEEDS:
            missing.append(run.config_name(exp))
    return missing


def pm(r, k):
    # "mean ± CI" of column k
    return f"{r[f'{k}_mean']:.4f} ± {r[f'{k}_ci']:.4f}"


def best_cells(g, k, pick):
    # One (text, bold) cell per row of g for column k, marking the best value
    # (pick is max or min; values equal to 4 decimals are all bold).
    # Same rule as the "All results" table in run._markdown
    v = g[f"{k}_mean"].round(4)
    return [(pm(r, k), x == pick(v)) for x, (_, r) in zip(v, g.iterrows())]


def draw_table(path, title, note, headers, widths, rows, breaks=()):
    # Draws a table with matplotlib and saves it as a PNG
    #
    #   widths  width of each column, in inches
    #   rows    one list per row; a cell is a string, or (text, bold)
    #   breaks  row positions that start a new group (a rule is drawn above them)
    row_h, pad = 0.3, 0.15
    note_lines = textwrap.wrap(note, int(sum(widths) * 15))
    top = 0.42 + 0.2 * len(note_lines)
    width = sum(widths) + 2 * pad
    height = top + row_h * (len(rows) + 1) + pad

    fig = plt.figure(figsize=(width, height), facecolor=WHITE)
    # Axes covering the whole figure, with coordinates in inches from the top left
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")
    ax.text(pad, 0.12, title, fontsize=13, fontweight="bold", color=mf.INK, va="top")
    ax.text(pad, 0.4, "\n".join(note_lines), fontsize=8.5, color=mf.INK_2, va="top",
            linespacing=1.35)

    def line(y, color, lw):
        ax.plot([pad, width - pad], [y, y], color=color, linewidth=lw)

    def draw_row(y, cells, bold_row=False):
        x = pad
        for cell, w in zip(cells, widths):
            text, bold = cell if isinstance(cell, tuple) else (cell, bold_row)
            ax.text(x + 0.05, y + row_h / 2, text, fontsize=10, color=mf.INK, va="center",
                    fontweight="bold" if bold else "normal")
            x += w

    draw_row(top, headers, bold_row=True)
    line(top + row_h, mf.INK, 1.0)
    for i, cells in enumerate(rows):
        y = top + row_h * (i + 1)
        if i in breaks:
            line(y, mf.AXIS, 0.6)
        draw_row(y, cells)
    line(top + row_h * (len(rows) + 1), mf.INK, 1.0)
    fig.savefig(path, facecolor=WHITE)
    plt.close(fig)


def table_fl_vs_paper(summary, path):
    # Table 1: laid out like the paper's Table 2, which reports one fairness
    # value per method: the metric the method targets. FedAvg ignores the
    # metric, so it appears under both, scored on that metric
    fl = summary[summary["setting"] == "fl"]
    rows, breaks = [], []
    for k in run.METRICS:
        breaks.append(len(rows))
        paper = PAPER_TABLE2[k]
        ours = pd.concat([fl[fl["method"] == "fedavg"], fl[fl["metric"] == k]])
        best_acc = max(v[0] for v in paper.values())
        best_fair = min(v[2] for v in paper.values())
        columns = zip(best_cells(ours, "acc", max), best_cells(ours, k, min))
        for (_, r), (acc, fair) in zip(ours.iterrows(), columns):
            a, a_ci, f, f_ci = paper[r["method"]]
            rows.append([k.upper(), METHODS[r["method"]],
                         acc, (f"{a:.4f} ± {a_ci:.4f}", a == best_acc),
                         fair, (f"{f:.4f} ± {f_ci:.4f}", f == best_fair)])
    draw_table(path, "Federated learning on Adult: reproduction vs paper",
               f"Mean ± 95% CI; reproduction over {N_SEEDS} seeds, paper values as printed in its "
               "Table 2. |Fairness| is the metric in the first column. Bold marks the best value "
               "in each column for that metric, as in the paper.",
               ["Metric", "Method", "Reproduction accuracy", "Paper accuracy",
                "Reproduction |Fairness|", "Paper |Fairness|"],
               [0.8, 1.05, 2.0, 1.6, 2.1, 1.6], rows, breaks[1:])


def table_all_results(summary, path):
    # Table 2: every configuration, grouped by setup. Methods are compared
    # within a setup, so that is where the best values are marked
    rows, breaks = [], []
    for setting, topology, rounds in SETUPS:
        g = summary[(summary["setting"] == setting) & (summary["topology"] == topology)
                    & (summary["rounds"] == rounds)]
        if g.empty:
            continue
        breaks.append(len(rows))
        columns = zip(best_cells(g, "acc", max), best_cells(g, "spd", min), best_cells(g, "eod", min))
        for (_, r), (acc, sp, eo) in zip(g.iterrows(), columns):
            rows.append([SETTINGS[setting], TOPOLOGIES[topology], str(rounds),
                         METHODS[r["method"]], "—" if r["metric"] == "-" else r["metric"].upper(),
                         acc, sp, eo, str(r["messages"])])
    draw_table(path, "All results on Adult",
               f"Mean ± 95% CI over {N_SEEDS} seeds. Bold marks the best value in each column "
               "within a setup (same setting, topology and rounds): highest accuracy, lowest "
               "|SPD| and |EOD|. Gossip values are averaged over the 10 nodes; federated values "
               "come from the one global model.",
               ["Setting", "Topology", "Rounds", "Method", "Trained on", "Accuracy",
                "|SPD|", "|EOD|", "Messages"],
               [0.95, 1.45, 0.7, 0.95, 0.95, 1.55, 1.55, 1.55, 0.85], rows, breaks[1:])


def _setup_pairs(summary):
    # For each setup with results: (setup, baseline row, Fair2Zero row trained on SPD)
    pairs = []
    for setup in SETUPS:
        base = find(summary, *setup, "fedavg" if setup[0] == "fl" else "avg", "-")
        f2z = find(summary, *setup, "fair2zero", "spd")
        if len(base) and len(f2z):
            pairs.append((setup, base.iloc[0], f2z.iloc[0]))
    return pairs


def table_fair2zero(summary, path):
    # Table 3: the slide-sized table, Fair2Zero (trained on SPD) in every setup.
    # "Change" is the paired difference against the baseline on the same seeds
    pairs = _setup_pairs(summary)
    best = min(round(f["spd_mean"], 4) for _, _, f in pairs)
    rows = []
    for (setting, topology, rounds), b, f in pairs:
        rows.append([SETTINGS[setting], TOPOLOGIES[topology], str(rounds), str(f["messages"]),
                     pm(b, "spd"), (pm(f, "spd"), round(f["spd_mean"], 4) == best),
                     f"{f['d_target_mean']:+.4f} ± {f['d_target_ci']:.4f}",
                     f"{f['better_in']}/{f['n_seeds']}", pm(f, "acc")])
    draw_table(path, "Fair2Zero across setups (trained on SPD)",
               f"Mean ± 95% CI over {N_SEEDS} seeds. Baseline is FedAvg for federated learning "
               "and Avg for gossip. Change is the paired difference against the baseline "
               "(negative = fairer). Bold marks the lowest Fair2Zero |SPD|.",
               ["Setting", "Topology", "Rounds", "Messages", "Baseline |SPD|", "Fair2Zero |SPD|",
                "Change", "Fairer in", "Fair2Zero acc."],
               [0.95, 1.45, 0.7, 0.85, 1.45, 1.55, 1.55, 0.85, 1.5], rows)


def fig_topology_curves(results, path, rounds=100, beta=run.BETA):
    # Figure 2: one panel per sparse topology, on a shared |SPD| scale.
    # Blue is plain averaging and orange Fair2Zero, as in make_figures.py.
    # Line + band = mean node over the seeds with its 95% CI
    g = Path(results) / "gossip"
    panels = [("ring", "Ring (2 neighbours)"), ("kreg3", "Random 3-regular (3 neighbours)")]
    panels = [p for p in panels if list(g.glob(f"{p[0]}_avg_r{rounds}_s*.json"))]

    fig, axes = plt.subplots(1, len(panels), figsize=(10, 4.6), sharey=True,
                             facecolor=WHITE, squeeze=False)
    fig.subplots_adjust(top=0.78, bottom=0.23, left=0.07, right=0.93, wspace=0.22)
    n_seeds = 0
    for ax, (topology, label) in zip(axes[0], panels):
        mf._style(ax)
        ax.set_title(label, fontsize=10, color=mf.INK, loc="left")
        for name, color, pattern in (
                ("Avg", mf.BLUE, f"{topology}_avg_r{rounds}_s*.json"),
                ("Fair2Zero", mf.ORANGE, f"{topology}_fair2zero_spd_b{beta}_r{rounds}_s*.json")):
            mean, ci, n_seeds = mf._curves(sorted(g.glob(pattern)), "mean")
            t = range(1, len(mean) + 1)
            ax.fill_between(t, mean - ci, mean + ci, color=color, alpha=0.15, linewidth=0)
            ax.plot(t, mean, color=color, linewidth=2, label=name)
            # Final value just right of the line's end
            ax.text(len(mean) + 2, mean[-1], f"{mean[-1]:.3f}", fontsize=9, color=mf.INK,
                    va="center")
        ax.set_xlim(1, rounds)
        ax.set_ylim(0, None)
        ax.set_xlabel("Round", fontsize=9, color=mf.INK_2)
    axes[0][0].set_ylabel("|SPD|  (lower is fairer)", fontsize=9, color=mf.INK_2)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, loc="lower left",
               bbox_to_anchor=(0.06, 0.02), ncol=2, labelcolor=mf.INK)
    mf._titles(fig, "Fairness over the rounds on sparse topologies",
               f"Fair2Zero trained on SPD. Mean node over {n_seeds} seeds, band = 95% CI.")
    fig.savefig(path, facecolor=WHITE)
    plt.close(fig)


def fig_fairness_vs_messages(summary, path):
    # Figure 3: one row per setup, fewest messages at the top. Each row joins
    # the baseline (blue) and Fair2Zero trained on SPD (orange), with 95% CIs,
    # so the length of the join is the fairness gained in that setup
    pairs = sorted(_setup_pairs(summary), key=lambda p: p[2]["messages"])
    fig, ax = plt.subplots(figsize=(9, 4.8), facecolor=WHITE)
    fig.subplots_adjust(top=0.82, bottom=0.21, left=0.27, right=0.96)
    mf._style(ax)
    ax.grid(False)
    ax.grid(axis="x", color=mf.GRID, linewidth=0.6)

    labels = []
    for y, ((setting, topology, rounds), b, f) in enumerate(pairs):
        ax.plot([b["spd_mean"], f["spd_mean"]], [y, y], color=mf.AXIS, linewidth=2, zorder=1)
        for name, color, r in (("Baseline (FedAvg / Avg)", mf.BLUE, b), ("Fair2Zero", mf.ORANGE, f)):
            ax.errorbar(r["spd_mean"], y, xerr=r["spd_ci"], fmt="o", markersize=8, color=color,
                        markeredgecolor=WHITE, markeredgewidth=1.2, ecolor=color, elinewidth=1,
                        capsize=3, zorder=2, label=name if y == 0 else None)
        # Value labels on the outer side of each dot
        lo, hi = sorted((b, f), key=lambda r: r["spd_mean"])
        ax.text(lo["spd_mean"] - lo["spd_ci"] - 0.002, y, f"{lo['spd_mean']:.3f}", fontsize=9,
                color=mf.INK, ha="right", va="center")
        ax.text(hi["spd_mean"] + hi["spd_ci"] + 0.002, y, f"{hi['spd_mean']:.3f}", fontsize=9,
                color=mf.INK, ha="left", va="center")
        name = "Federated" if setting == "fl" else f"Gossip, {TOPOLOGIES[topology]}"
        labels.append(f"{name}\n{rounds} rounds, {f['messages']:,} messages")

    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels(labels, fontsize=9, color=mf.INK)
    # Half a row of space at both ends; the reversed limits put row 0 at the top
    ax.set_ylim(len(pairs) - 0.5, -0.5)
    ax.margins(x=0.15)
    ax.set_xlabel("|SPD|  (lower is fairer)", fontsize=9, color=mf.INK_2)
    fig.legend(*ax.get_legend_handles_labels(), frameon=False, fontsize=9, loc="lower left",
               bbox_to_anchor=(0.26, 0.01), ncol=2, labelcolor=mf.INK)
    mf._titles(fig, "Fairness gained in each setup, ordered by messages sent",
               f"Trained on SPD. Mean ± 95% CI over {N_SEEDS} seeds. The x-axis does not start at 0.")
    fig.savefig(path, facecolor=WHITE)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--out", default=None, help="default: <results>/presentation")
    p.add_argument("--allow-incomplete", action="store_true",
                   help="draw whatever is there; for testing only, not for final slides")
    args = p.parse_args()
    out = Path(args.out or Path(args.results) / "presentation")

    summary = pd.read_csv(Path(args.results) / "summary.csv")
    missing = missing_configs(summary)
    if missing and not args.allow_incomplete:
        sys.exit(f"{len(missing)} configurations are missing or have fewer than {N_SEEDS} "
                 f"seeds, so finish run.py first:\n  " + "\n  ".join(missing))

    out.mkdir(parents=True, exist_ok=True)
    # 300 dpi on a pure white background, also for the figure from make_figures.py
    plt.rcParams["savefig.dpi"] = 300
    mf.SURFACE = WHITE
    table_fl_vs_paper(summary, out / "table1_fl_vs_paper.png")
    table_all_results(summary, out / "table2_all_results.png")
    table_fair2zero(summary, out / "table3_fair2zero_by_topology.png")
    mf.fl_figure(summary, out / "fig1_fl_reproduction.png")
    fig_topology_curves(args.results, out / "fig2_topology_curves.png")
    fig_fairness_vs_messages(summary, out / "fig3_fairness_vs_messages.png")
    print(f"wrote 6 PNG files to {out}")


if __name__ == "__main__":
    main()
