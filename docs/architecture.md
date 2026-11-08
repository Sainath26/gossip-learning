# Architecture

How the code for the gossip learning experiments is organised. The experiment itself is defined in `experimental_design.md`.

## Principles

- **One place for the knobs.** Every setting lives in `configs.json`. The code reads it and never changes between experiments; a new experiment means changing values in `configs.json`.
- **Minimal and simple.** Four Python files, each with one job.
- **Ready for the console.** Learning code reports each step through an optional callback, so a live view can plug in later without rewriting anything.
- **Reproducible.** One seed sets all randomness in a run, and the config used is saved next to its results.

## Files

| File              | Job             | Contains                                                                                                                                                                                        |
| ----------------- | --------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `configs.json`    | The knobs       | All settings in one file, edited for each run (pilot values first)                                                                                                                              |
| `prepare_data.py` | Data            | Load ACSIncome once and cache it compactly. For a seed: draw the stratified sample (state × sex × income, minimum per state) and the shared test split                                          |
| `model.py`        | The model       | The network, one training epoch, prediction, and accuracy, SPD and EOD                                                                                                                          |
| `learning.py`     | How nodes learn | The state border graph (with the three island links) and the random graph; the random-step gossip protocol with the λ merge; the FedAvg and local-only baselines; the optional on-step callback |
| `run.py`          | Orchestrator    | Read a config; for each seed, get the data, build the graph, run gossip and the baselines, evaluate at checkpoints, save results and the config                                                 |

## Flow

```
configs.json
        │
      run.py ──→ prepare_data.py   sample + split for this seed
        │   ──→ learning.py        graph, gossip, FedAvg, local-only
        │            └──→ model.py  train, evaluate
        ↓
results/<experiment_name>/         per-node metrics per checkpoint + copy of the config
```

## Build order

The plan above is complete, but code is written only as each part is needed, to avoid unused code. `learning.py` starts with the graph and gossip; the FedAvg and local-only baselines are added when the experiments need them.

## Running

- **Batch:** `python run.py` reads `configs.json` and runs every seed at full speed, in parallel across CPU cores. Used for real results. Results go to `results/<experiment_name>/`, so changing `experiment_name` keeps runs apart.
- **Live (later):** one seed, slowed down, with each step streamed to the console. Used for watching and demos.

## Data and memory

- The full ACSIncome data is loaded and cached once (`data/`, not tracked by git).
- Categorical features are stored as integer codes and expanded to one-hot only per mini-batch, so RAM stays low with several parallel workers. The model sees identical inputs.

## Output

- One results file per seed: accuracy, |SPD| and |EOD| for every node at every checkpoint, plus the gossip message count.
- A copy of the config in the same folder.
- Summary tables and plots come later, in a separate `summarize.py`.

## Experiment console (later)

A browser page, built with D3, with three areas:

- **Knobs:** edits the config and starts a run.
- **Live map:** the US with states as nodes and borders as links; models travel between states as each step happens, and each state is coloured by its current fairness or accuracy.
- **Charts:** fairness and accuracy over time for the average node, the worst node and the spread.

It listens to the on-step callback in `learning.py` and uses a locally saved US states map (`us-atlas`), so it works offline.

## Open

- Where the new code lives: a new folder (the earlier Adult code is in `baseline/`) or the project root.
- Where the state border list lives: inside `learning.py` (simplest) or a separate data file.
