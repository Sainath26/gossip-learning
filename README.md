# Fairness in Gossip Learning

Reproduces [Fair2Zero: Fairness-Aware Aggregation in Federated Learning](https://link.springer.com/chapter/10.1007/978-3-032-37976-4_26) on the Adult dataset, then extends it from federated learning (FL) to gossip learning, where there is no central server.

## Pipeline

```
data_prepration.py → model.py + aggregation.py → training loops → run.py → synthesize_results.py
```

| File | What it does |
|---|---|
| `data_prepration.py` | Prepares Adult, then creates a test set and 10 non-IID clients for each seed |
| `model.py` | Neural network, local training, and accuracy, SPD and EOD scoring |
| `aggregation.py` | Client weights (FedAvg, FairFed, Fair2Zero) and model averaging |
| `federated_learning_loop.py` | One FL run: FedAvg, FairFed or Fair2Zero |
| `gossip_learning_loop.py` | One gossip run: plain averaging or Fair2Zero on a `ring`, `full` or `kreg3` (random 3-regular) graph |
| `run.py` | Runs every configuration over 100 seeds and writes the summaries to `results/` |
| `make_figures.py` | Figures comparing FL with the paper, and ring gossip over the rounds |
| `synthesize_results.py` | Presentation tables and figures as PNGs |
| `control_pane/` | Live viewer for watching runs |

## Running

Place the UCI Adult zip at `data/adult.zip`. Install `torch`, `numpy`, `pandas` and `matplotlib`, then run:

```
python run.py                 # prepares missing data; runs only what is missing
python synthesize_results.py  # writes PNGs to results/presentation/
```

Use `python run.py --fresh` to regenerate the data and redo every run, or `python run.py --summary-only` to rebuild just the tables.

`data/`, `results/` and `artifacts/` are not tracked by git.

## Assumptions

The paper does not specify these, so they are our own choices:

| Detail | Value |
|---|---|
| Client perturbation | `delta_pa = 0.1` (share of women), `delta_fair = 0.5` (positive rates) |
| Minimum client weight | `eps = 0.001` |
| Loss function | `BCEWithLogitsLoss` |
| Weight carried between rounds | The normalised weight |
| Fallback when fairness is undefined | `\|Acc_i - Acc_global\|` |

The authors confirmed that round 1 uses size-based weights, and the code does this.
