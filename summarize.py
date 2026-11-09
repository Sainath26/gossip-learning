"""
Summarises saved results as tables and figures. Reads results only; nothing is retrained.

    python summarize.py

Folders in results/ named Experiment_<n>_... are grouped by <n>; each group gets
results/summaries/Experiment_<n>/ with:

    table.png, table.csv   final round, mean ± SD over seeds (one row per folder)
    over_rounds.png        accuracy, SPD and EOD over the rounds
    accuracy_vs_spd.png    one dot per state: fairness at equal accuracy
    accuracy_vs_eod.png    the same for EOD
    state_map.png          each state's |SPD| on a tile map of the US

Row names come from each folder's config.json: the settings that differ within the group
(for example "λ = 0.5", or "star, λ = 0.5" once topologies vary), not from the folder name.
"""

import json
import re
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from prepare_data import STATES, _protected, load_acs  # noqa: E402

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
OUT = RESULTS / 'summaries'

# Settings that never change a result, so they never name a row
IGNORED = {'experiment_name', 'seeds', 'compute', 'output'}
# Short names for the settings that can name a row
LABELS = {'gossip.lambda': 'λ', 'graph.topology': '', 'data.sample_fraction': 'sample',
          'gossip.rounds': 'rounds', 'fairness.protected_attribute': ''}

# Categorical colours in fixed order (one per row), then text and surface colours
COLOURS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
INK, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
BLUES = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']

# Each state's square on the tile map: (row, column)
TILES = {'AK': (0, 0), 'ME': (0, 10),
         'WI': (1, 5), 'VT': (1, 9), 'NH': (1, 10),
         'WA': (2, 0), 'ID': (2, 1), 'MT': (2, 2), 'ND': (2, 3), 'MN': (2, 4), 'IL': (2, 5), 'MI': (2, 6),
         'NY': (2, 8), 'MA': (2, 9),
         'OR': (3, 0), 'NV': (3, 1), 'WY': (3, 2), 'SD': (3, 3), 'IA': (3, 4), 'IN': (3, 5), 'OH': (3, 6),
         'PA': (3, 7), 'NJ': (3, 8), 'CT': (3, 9), 'RI': (3, 10),
         'CA': (4, 0), 'UT': (4, 1), 'CO': (4, 2), 'NE': (4, 3), 'MO': (4, 4), 'KY': (4, 5), 'WV': (4, 6),
         'VA': (4, 7), 'MD': (4, 8), 'DE': (4, 9),
         'AZ': (5, 1), 'NM': (5, 2), 'KS': (5, 3), 'AR': (5, 4), 'TN': (5, 5), 'NC': (5, 6), 'SC': (5, 7),
         'OK': (6, 3), 'LA': (6, 4), 'MS': (6, 5), 'AL': (6, 6), 'GA': (6, 7),
         'HI': (7, 0), 'TX': (7, 3), 'FL': (7, 8), 'PR': (7, 10)}
assert set(TILES) == set(STATES)


def load(folder):
    # One experiment: its config and {seed: history}
    config = json.loads((folder / 'config.json').read_text())
    runs = {int(p.stem.split('_')[1]): json.loads(p.read_text())['history']
            for p in folder.glob('seed_*.json')}
    return config, runs


def groups():
    # {experiment number: [folder, ...]} for every results/Experiment_<n>_... folder with results
    found = {}
    for folder in sorted(RESULTS.iterdir()):
        m = re.match(r'Experiment_(\d+)_', folder.name)
        if m and (folder / 'config.json').exists() and any(folder.glob('seed_*.json')):
            found.setdefault(int(m.group(1)), []).append(folder)
    return found


def _flat(config, prefix=''):
    # {'gossip.lambda': 0.5, ...}
    out = {}
    for k, v in config.items():
        if k in IGNORED and not prefix:
            continue
        if isinstance(v, dict) and k != 'island_links':
            out.update(_flat(v, f'{prefix}{k}.'))
        else:
            out[prefix + k] = v
    return out


def row_names(configs):
    # A name for each experiment from the settings that differ between them, e.g. "λ = 0.5"
    flat = [_flat(c) for c in configs]
    differ = [k for k in flat[0] if len({json.dumps(f.get(k)) for f in flat}) > 1]
    names = []
    for f in flat:
        parts = []
        for k in differ:
            short = LABELS.get(k, k.split('.')[-1])
            parts.append(f'{short} = {f[k]:g}' if isinstance(f[k], (int, float)) and short
                         else str(f[k]) if not short else f'{short} = {f[k]}')
        names.append(', '.join(parts) or 'all')
    return names, differ


def node_metrics(checkpoint):
    # Per state at one checkpoint: accuracy, signed SPD and EOD, and whether the model is a
    # constant predictor (the same prediction for everyone: SPD and EOD exactly 0)
    return {s: dict(m, constant=m['spd'] == 0 and m['eod'] == 0) for s, m in checkpoint['nodes'].items()}


def seed_stats(checkpoint):
    # One seed, one checkpoint: summaries across the 51 states
    nodes = node_metrics(checkpoint).values()
    acc = [m['accuracy'] for m in nodes]
    spd = [m['spd'] for m in nodes]
    eod = [m['eod'] for m in nodes]
    return {'accuracy': st.mean(acc), 'worst_accuracy': min(acc),
            'spd': st.mean(spd), 'worst_spd': max(map(abs, spd)),
            'eod': st.mean(eod), 'worst_eod': max(map(abs, eod)),
            'constant': sum(m['constant'] for m in nodes)}


def across_seeds(runs, index=-1):
    # {statistic: (mean, SD over seeds)} at one checkpoint (default: the final one)
    per_seed = [seed_stats(h[index]) for h in runs.values()]
    return {k: (st.mean(s[k] for s in per_seed), st.stdev(s[k] for s in per_seed) if len(per_seed) > 1 else 0.0)
            for k in per_seed[0]}


def data_gap(year, attribute):
    # The gap in the data itself: rate of income > $50K, non-white minus white (all records)
    df = load_acs(year)
    pa = _protected(df, attribute)
    return df['label'][pa == 0].mean() - df['label'][pa == 1].mean()


def per_state(runs, index=-1):
    # {state: mean over seeds of accuracy and SPD} at one checkpoint
    nodes = [h[index]['nodes'] for h in runs.values()]
    return {s: {k: st.mean(n[s][k] for n in nodes) for k in ('accuracy', 'spd', 'eod')} for s in STATES}


# ---------- outputs ----------

COLUMNS = [('Seeds', None), ('Accuracy', 'accuracy'), ('Worst accuracy', 'worst_accuracy'),
           ('SPD', 'spd'), ('Worst |SPD|', 'worst_spd'), ('EOD', 'eod'), ('Worst |EOD|', 'worst_eod'),
           ('Constant states', 'constant')]


def table(names, experiments, gap, out):
    # table.csv and table.png: final round, mean ± SD over seeds
    rows = []
    for name, (_, runs) in zip(names, experiments):
        s = across_seeds(runs)
        row = [name, str(len(runs))]
        for _, key in COLUMNS[1:]:
            mean, sd = s[key]
            row.append(f'{mean:.1f}' if key == 'constant' else f'{mean:+.3f} ± {sd:.3f}' if key in ('spd', 'eod')
                       else f'{mean:.3f} ± {sd:.3f}')
        rows.append(row)
    header = ['Setting'] + [c for c, _ in COLUMNS]
    (out / 'table.csv').write_text('\n'.join(','.join(f'"{c}"' for c in r) for r in [header] + rows), encoding='utf-8')

    fig, ax = plt.subplots(figsize=(14, 0.45 * (len(rows) + 1) + 0.6))
    ax.axis('off')
    t = ax.table(cellText=rows, colLabels=header, cellLoc='center', bbox=[0, 0, 1, 1])
    t.auto_set_font_size(False)
    t.set_fontsize(10)
    t.auto_set_column_width(list(range(len(header))))
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_text_props(color=INK, weight='bold' if r == 0 else 'normal')
        cell.set_facecolor('#f1f0ec' if r == 0 else SURFACE)
    run = experiments[0][0]
    ax.set_title(f"{run['graph']['topology']} graph · protected attribute {run['fairness']['protected_attribute']} · "
                 f"{run['data']['sample_fraction']:.0%} sample · final round, mean ± SD over seeds",
                 color=INK, fontsize=11, loc='left', pad=10)
    fig.text(0.125, -0.02, f'SPD and EOD: non-white minus white (negative = white favoured); gap in the data itself: '
             f'{gap:+.3f}.  Constant states: states per seed whose model predicts the same for everyone.',
             color=MUTED, fontsize=8.5)
    fig.savefig(out / 'table.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.8)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)


def over_rounds(names, experiments, gap, out):
    # Accuracy, SPD and EOD: mean over seeds of the across-state mean, with a ± SD band; one colour per row
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.2))
    for (title, key), ax in zip([('Accuracy (mean across states)', 'accuracy'),
                                 ('SPD (mean across states)', 'spd'),
                                 ('EOD (mean across states)', 'eod')], axes):
        for i, (name, (_, runs)) in enumerate(zip(names, experiments)):
            history = next(iter(runs.values()))
            rounds = [c['round'] for c in history]
            stats = [across_seeds(runs, j)[key] for j in range(len(rounds))]
            mean = [m for m, _ in stats]
            sd = [s for _, s in stats]
            ax.fill_between(rounds, [m - s for m, s in zip(mean, sd)], [m + s for m, s in zip(mean, sd)],
                            color=COLOURS[i], alpha=0.15, linewidth=0)
            ax.plot(rounds, mean, color=COLOURS[i], linewidth=2, label=name)
        if key == 'spd':
            # The gap in the data itself (EOD has none: on the true labels it is always 0)
            ax.axhline(gap, color=MUTED, linestyle=':', linewidth=1.5)
            ax.annotate(f'gap in the data {gap:+.3f}', (rounds[0], gap), xytext=(4, -12), textcoords='offset points',
                        ha='left', color=MUTED, fontsize=8.5)
        _style(ax)
        ax.set_title(title, color=INK, fontsize=11, loc='left')
        ax.set_xlabel('round', color=MUTED)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out / 'over_rounds.png', dpi=200, facecolor='white')
    plt.close(fig)


def accuracy_vs(metric, names, experiments, reference, out):
    # One dot per state (mean over seeds, final round): states that learned little sit at
    # low accuracy and near 0, so fairness is compared at equal accuracy.
    # reference: (value, label) drawn as a dotted line
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for i, (name, (_, runs)) in enumerate(zip(names, experiments)):
        states = per_state(runs)
        ax.scatter([states[s]['accuracy'] for s in STATES], [states[s][metric] for s in STATES], s=36,
                   color=COLOURS[i], edgecolors=SURFACE, linewidths=1, label=name, zorder=3)
        for s in STATES:  # label only the states that stand apart
            if states[s]['accuracy'] < 0.72:
                ax.annotate(s, (states[s]['accuracy'], states[s][metric]), xytext=(4, 3), textcoords='offset points',
                            fontsize=7.5, color=MUTED)
    if reference:
        value, label = reference
        ax.axhline(value, color=MUTED, linestyle=':', linewidth=1.5)
        ax.annotate(label, (ax.get_xlim()[1], value), xytext=(-4, 4), textcoords='offset points', ha='right',
                    color=MUTED, fontsize=8.5)
    _style(ax)
    ax.set_xlabel('accuracy', color=MUTED)
    ax.set_ylabel(f'{metric.upper()} (non-white minus white)', color=MUTED)
    ax.set_title('Each state at the final round (mean over seeds)', color=INK, fontsize=11, loc='left')
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out / f'accuracy_vs_{metric}.png', dpi=200, facecolor='white')
    plt.close(fig)


def state_map(names, experiments, out):
    # |SPD| per state on a tile map, one map per row, on one shared colour scale
    data = [per_state(runs) for _, runs in experiments]
    top = max(abs(d[s]['spd']) for d in data for s in STATES)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('blues', BLUES)
    norm = matplotlib.colors.Normalize(0, top)
    fig, axes = plt.subplots(1, len(names), figsize=(4.6 * len(names), 3.8), squeeze=False)
    for ax, name, d in zip(axes[0], names, data):
        for s, (r, c) in TILES.items():
            v = abs(d[s]['spd'])
            ax.add_patch(plt.Rectangle((c, -r), 0.92, 0.92, color=cmap(norm(v)), linewidth=0))
            ax.text(c + 0.46, -r + 0.46, s, ha='center', va='center', fontsize=7,
                    color='white' if norm(v) > 0.55 else INK)
        ax.set_xlim(-0.2, 11.2)
        ax.set_ylim(-7.3, 1.1)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(name, color=INK, fontsize=11, loc='left')
    bar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes[0].tolist(), shrink=0.7, pad=0.02)
    bar.set_label('|SPD| (mean over seeds, final round)', color=MUTED)
    bar.outline.set_visible(False)
    bar.ax.tick_params(colors=MUTED, labelsize=8)
    fig.savefig(out / 'state_map.png', dpi=200, facecolor='white', bbox_inches='tight')
    plt.close(fig)


def main():
    found = groups()
    if not found:
        raise SystemExit('no results/Experiment_<n>_... folders with results')
    gaps = {}
    for number, folders in found.items():
        experiments = [load(f) for f in folders]
        names, differ = row_names([c for c, _ in experiments])
        # Rows in order of the settings that differ (numbers numerically)
        order = sorted(range(len(names)), key=lambda i: [_flat(experiments[i][0]).get(k) for k in differ])
        names, experiments = [names[i] for i in order], [experiments[i] for i in order]

        cfg = experiments[0][0]
        key = (cfg['data']['survey_year'], cfg['fairness']['protected_attribute'])
        if key not in gaps:
            gaps[key] = data_gap(*key)
        out = OUT / f'Experiment_{number}'
        out.mkdir(parents=True, exist_ok=True)
        table(names, experiments, gaps[key], out)
        over_rounds(names, experiments, gaps[key], out)
        accuracy_vs('spd', names, experiments, (gaps[key], f'gap in the data {gaps[key]:+.3f}'), out)
        accuracy_vs('eod', names, experiments, None, out)
        state_map(names, experiments, out)
        print(f"Experiment_{number}: {', '.join(f'{n} ({len(r)} seeds)' for n, (_, r) in zip(names, experiments))}"
              f' -> {out}')


if __name__ == '__main__':
    main()
