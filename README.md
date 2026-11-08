# Fairness in Gossip Learning

## Research question

**How fair is a model trained by asynchronous gossip learning across US states, and how do the merge weight λ and the communication graph affect group fairness by race (white / non-white)?**

- **Nodes:** 51, one per state (50 states plus Puerto Rico). Each holds only its own residents' ACSIncome records.
- **Links:** states that share a border. Alaska, Hawaii and Puerto Rico are linked to their nearest mainland state (WA, CA and FL).
- **Gossip:** at each step, a random state sends its model to a random neighbour. The receiver merges with `new = λ·mine + (1−λ)·received`, then trains one epoch.
- **Fairness:** SPD and EOD between white and non-white people, measured for every state's model on a shared test set.
- **In scope:** consensus gossip, where all nodes move towards one shared model. Personalised models are out of scope.

The full design is in [docs/experimental_design.md](docs/experimental_design.md). How the code fits together is in [docs/architecture.md](docs/architecture.md).

![Experiment pane running pilot 6](docs/images/experiment_pane.png)

*The experiment pane during a run. Each arrow is one model exchange between neighbouring states, and states are coloured by unfairness (|SPD|). The right panel shows what the receiver's model looked like before and after merging, plus the mean and worst-state |SPD| over the rounds.*

## Folder structure

```
experiment/
├── configs.json          every setting (the only file edited between experiments)
├── prepare_data.py       load ACSIncome, group codes, sample, split and encode per seed
├── model.py              the network (Fair2Zero's MLP), training, accuracy, SPD and EOD
├── learning.py           border graph and random-step gossip with the λ merge
├── run.py                runs all seeds in parallel and saves results
├── eda.ipynb             exploration of ACSIncome by state and group
├── experiment_pane/      live browser view of one run on a US map
│   ├── server.py
│   ├── index.html
│   └── vendor/           D3, topojson and US state outlines, saved for offline use
├── reference/            fixed inputs, checked by SHA-256
│   ├── PUMS_Data_Dictionary_2018.txt   for grouping occupation and birthplace
│   └── state_neighbors.csv             state border list
├── docs/
│   ├── experimental_design.md
│   └── architecture.md
├── archive/              earlier work: Fair2Zero reproduction on Adult (FL and gossip)
├── data/                 cached ACS data and per-seed splits (not tracked)
└── results/              one folder per experiment (not tracked)
    └── <experiment_name>/
        ├── config.json   the settings used
        └── seed_XXX.json per-state accuracy, SPD and EOD at each checkpoint
```

## Running

Use the project's virtual environment (`torch`, `numpy`, `pandas`, `folktables`).

```
python run.py                        # all seeds from configs.json, saved to results/<experiment_name>/
python experiment_pane/server.py     # live view at http://localhost:8000 (one seed)
```

- To start a new experiment, change `experiment_name` in `configs.json`.
- If a run stops, restart it: seeds that already have results are skipped.
- A folder that holds results from different settings is never reused.

## Results so far (pilots, seed 1, 20% sample, 50 rounds)

| Pilot | λ                                        | Mean accuracy | Worst state's accuracy | Mean \|SPD\| | Worst state's \|SPD\| |
| ----- | ---------------------------------------- | ------------- | ---------------------- | ------------ | --------------------- |
| 4     | 0.25                                     | 0.786         | 0.753                  | 0.107        | 0.146                 |
| 5     | 0.5                                      | 0.786         | 0.753                  | 0.107        | 0.145                 |
| 6     | 1.0 (no gossip: each state learns alone) | 0.756         | 0.631                  | 0.092        | 0.144                 |

- **Accuracy:** gossip mainly helps small states. Alaska goes from 0.63 alone to 0.79 with gossip, and Maine from 0.64 to 0.79. Large states (CA, TX, NY) do about the same either way.
- **Fairness:** gossip also spreads the large states' bias. Small states end up with an SPD near that of the large states. The lower mean SPD at λ = 1 is mostly an artefact: alone, AK and PR predict "≤ $50K" for everyone, which gives SPD = 0 with accuracy 0.63.
- **λ:** the results for 0.25 and 0.5 are almost identical, and switching gossip off (λ = 1) changes them clearly.
- **Rounds:** results level off by about round 50 (pilot 2, 100 rounds).

## Changes to be made

**Next runs**

- [ ] λ = 0 (the receiver takes the sender's model), to complete the extremes.
- [ ] Optionally λ = 0.9, to see whether a little mixing is enough to rescue the small states.
- [ ] Experiment 1: 100 seeds, 20%, 50 rounds, evaluation every round (`eval_checkpoint_every: 1`), over the chosen λ values.

**Experiment pane**

- [ ] Fix the bugs in `experiment_pane/` (several known issues; to be listed and fixed one by one).

**Federated learning baseline**

- [ ] Write FedAvg in `learning.py`, using the same seeds, samples, splits, model and evaluation as gossip.
- [ ] Run the FedAvg baseline and compare it with gossip (`baselines` settings in `configs.json`).

**Code**

- [ ] `summarize.py`: mean ± 95% CI across seeds; fairness over time (mean, worst and spread); a per-state table.
- [ ] Flag constant predictors (a model that predicts one class for everyone) so their SPD = 0 is not read as fair. Report accuracy alongside SPD.
- [ ] Record each state's number of training turns. With random steps, degree-1 states (AK, HI, ME, PR) train about 3× less often than average.
- [ ] Pane runs: ignore `seeds` when checking a results folder, and record which seeds were actually saved. A pane-saved `config.json` currently shows `seeds: 100` copied from `configs.json`.
- [ ] Add a `.gitattributes` binary rule for `reference/`, so line-ending changes cannot break the SHA-256 checks.
- [ ] Later: the random graph (Experiment 2), the local-only baseline, and the results map.

**Docs**

- [ ] `experimental_design.md`: still describes time in steps, a 10% pilot and evaluation every 255 steps. Update to rounds, the 20% pilot, `eval_checkpoint_every` and the λ = 1 finding.
- [ ] Close the resolved open decisions (grouping, standardisation, run time) and record the choices made.
- [ ] `architecture.md`: the experiment pane now exists; the "Open" questions about code location and the border list are settled.

**Parked**

- Synchronous reference (Experiment 3, not gossip learning) and the message-queue variant.
