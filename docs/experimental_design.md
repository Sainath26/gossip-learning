# Experimental design: group fairness in gossip learning

## Question

How fair is a model trained by asynchronous gossip learning across US states, and how do the merge weight λ and the communication graph affect group fairness by race (white / non-white)?

## Scope

- **In:** consensus gossip learning. Nodes exchange models without a server, so every node converges towards a common model trained on all participants' data.
- **Out:** personalised gossip learning, where each node keeps a model tuned to its own data.

## 1. Data

- ACSIncome (folktables), ACS PUMS 2018 1-year, 50 states plus Puerto Rico (1,664,500 rows, as in FairFed).
- Target: income above $50,000. Protected attribute: race, white / non-white, as in FairFed (RAC1P = 1, "White alone", is white; codes 2–9 are non-white). Set in `configs.json` (`protected_attribute`), so sex can be used instead.
- Inputs: the 10 ACSIncome features minus the protected attribute. The protected attribute, state and label are never model inputs.
- **Sample:** each seed draws its own stratified sample by state × protected attribute × income label: 10% for the pilot, 20% for the experiments. Every state keeps at least 1,000 records (states below it are topped up, still stratified). Across seeds, different samples cover most of the full data.

## 2. Encoding (as in Fair2Zero)

- Categorical features are one-hot encoded; numerical features (age, hours worked) are standardised.
- Categories are stored as compact integer codes and expanded to one-hot only as each mini-batch enters the model. The model sees identical inputs, so results are unchanged; only memory use drops.

## 3. Split

- Shared test set: 20% of each state × protected attribute × income group of the seed's sample, held out before training.
- Each state's remaining 80% is that node's training data.
- Redrawn for each seed.

## 4. Nodes and graph

- 51 nodes, one per state. Each node holds only its own residents' records (a stylised scenario: state agencies that cannot share raw data but can share models).
- **Border graph:** each state is linked to the states it shares a border with. Alaska, Hawaii and Puerto Rico border no state, so each is linked to its nearest mainland state: Alaska to Washington, Hawaii to California, Puerto Rico to Florida. These three links are not real borders.
- **Random graph** (Experiment 2): each state keeps its number of neighbours, but neighbours are random. This isolates the effect of geography.

## 5. Model and training (as in Fair2Zero)

- One hidden layer of 32 units with ReLU, a sigmoid output, binary cross-entropy loss.
- SGD, learning rate 0.05, batch size 128, one local epoch each time a node trains.
- All nodes start from the same initial model within a seed.

## 6. Gossip protocol

Random-step protocol, a standard model of asynchronous gossip (gossip learning as in Hegedűs, Danner and Jelasity, 2019):

1. One state is picked uniformly at random from all 51.
2. It sends its model to one neighbour, picked uniformly at random from its neighbours in the graph only (Texas picks from New Mexico, Oklahoma, Arkansas and Louisiana; never Florida).
3. The receiver merges, new = λ · mine + (1 − λ) · received, then trains one local epoch.

- One exchange at a time and nobody waits: no rounds, clocks or timers.
- Time is counted in steps. 51 steps is about one round, since each node acts about once; 1,275 steps (51 × 25) matches Fair2Zero's 25 rounds.
- No fairness-aware aggregation in these experiments.

## 7. Evaluation

- All 51 models are tested on the shared test set: accuracy, |SPD| and |EOD|.
- When: every 255 steps (about every 5 rounds) in the pilot; every 51 steps (about every round) in the experiments.
- Reported over time: the average node, the worst node, and the spread between nodes.

## 8. Baselines

- FedAvg with a server.
- Each state training alone (λ = 1).

## 9. Runs

| Run | λ | Sample | Seeds | Evaluate every | Purpose |
|---|---|---|---|---|---|
| Pilot | 0.25 | 10% | 5 | 255 steps | Check the pipeline works; measure run times |
| Experiment 1 (λ sweep) | 0.25 first, then more values | 20% | 100 | 51 steps | How λ affects fairness, border graph fixed |
| Experiment 2 (topology) | Fixed | 20% | 100 | 51 steps | Border graph vs random graph |
| Experiment 3 (parked) | Fixed | 20% | 100 | 51 steps | Synchronous vs asynchronous |

## 10. Repeats

- Each seed sets its own sample, split, initial model, mini-batch order and gossip schedule.
- Paired comparisons: within a seed, every setting uses the same sample and randomness, so only the setting under study differs.
- The nodes and the border graph are fixed across seeds.

## 11. Output

- Per seed, per checkpoint, per node: accuracy, |SPD| and |EOD|.
- Summary: mean ± 95% confidence interval across seeds.
- Plots of fairness over time.
- The settings used are saved next to the results.

## Parked

- **Synchronous reference (not gossip learning):** all nodes act in lockstep rounds, each pushing to one random neighbour with the same λ merge, so timing is the only difference. Reported as a reference, never as gossip learning. The earlier all-neighbour loop (decentralized SGD) may be a second reference.
- **Message queue variant:** nodes train in the background for a fixed time; models arriving meanwhile wait, then merge (train → merge, unlike the paper's merge → train). Needs a training duration and a queue policy.

## Open decisions (before the pilot)

| Decision | Options |
|---|---|
| FedAvg comparison | Equal messages (gossip 2,550 steps against 25 FedAvg rounds of 102 messages each) or roughly equal rounds (1,275 steps against 25 rounds) |
| Local-only baseline length | For example 25 epochs per state, matching about 25 gossip training turns per state |
| Random graph | One graph drawn per seed, or the same graph for all seeds |
| Grouping occupation and birthplace | Map to official Census major groups (about 100 input columns) or keep detailed codes (about 800) |
| Standardisation | Mean and spread from training data only (recommended), or from all data |
| Further λ values | For example 0.5, 0.75 and 0.9 |
| Run time | 100 seeds at 20% may exceed the 2-hour budget; check with the pilot's timings |
