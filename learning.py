"""
How nodes learn: the communication graph and the gossip protocol.

Gossip is the random-step protocol (asynchronous gossip learning):
  1. one state is picked at random from all 51
  2. it sends its model to one random neighbour in the graph
  3. the receiver merges, new = λ · mine + (1 − λ) · received, then trains locally
One exchange at a time, nobody waits. Time is counted in rounds of 51 steps:
in one round each state acts about once, on average.

Built in the order the experiments need it (docs/architecture.md): the FedAvg and
local-only baselines are added here when they are needed.
"""

import copy
import csv
import hashlib
from pathlib import Path
import numpy as np
import torch
from model import MLP, evaluate, to_tensors, train_one_epoch
from prepare_data import STATES

# Neighbouring US states, from the "List of neighboring states for each US state" dataset
# (github.com/ubikuity/List-of-neighboring-states-for-each-US-state, commit 38d9ace, 2015),
# saved unchanged. The checksum makes sure the graph always comes from this exact file
NEIGHBORS = Path(__file__).resolve().parent / 'reference' / 'state_neighbors.csv'
NEIGHBORS_SHA256 = '8e4b2a1581f1ff582ec4bad18b69a7cbc1bfdfe669b2c5c6118379ef5541420c'

# States that touch only at a point (the Four Corners): not neighbours here
CORNER_ONLY = {('AZ', 'CO'), ('NM', 'UT')}


def _connected(graph):
    # True if every state can reach every other through the graph
    seen, todo = {STATES[0]}, [STATES[0]]
    while todo:
        for nb in graph[todo.pop()]:
            if nb not in seen:
                seen.add(nb)
                todo.append(nb)
    return len(seen) == len(graph)


def border_graph(island_links):
    # {state: sorted neighbours}: states sharing a land border, plus the island links from the
    # config (e.g. {'AK': 'WA', 'HI': 'CA', 'PR': 'FL'}). From the published list we drop DC
    # (not a node), corner-only touches, and the list's own AK/HI links (the config sets those)
    raw = NEIGHBORS.read_bytes()
    if hashlib.sha256(raw).hexdigest() != NEIGHBORS_SHA256:
        raise ValueError(f'{NEIGHBORS} is not the expected neighbour list')
    pairs = set()
    for a, b in list(csv.reader(raw.decode().splitlines()))[1:]:
        pair = tuple(sorted((a, b)))
        if a in STATES and b in STATES and pair not in CORNER_ONLY and not {'AK', 'HI'} & set(pair):
            pairs.add(pair)
    pairs |= {tuple(sorted(link)) for link in island_links.items()}

    graph = {st: sorted([b for a, b in pairs if a == st] + [a for a, b in pairs if b == st])
             for st in STATES}
    if not _connected(graph):
        raise ValueError('the border graph is not connected')
    return graph


def random_graph(graph, rng):
    # A random graph where every state keeps its number of neighbours (a degree-preserving
    # shuffle: pairs of links swap ends many times). Redrawn until connected
    base = sorted({tuple(sorted((a, b))) for a in graph for b in graph[a]})
    while True:
        edges = list(base)
        linked = set(edges)
        for _ in range(10 * len(edges)):
            i, j = rng.choice(len(edges), size=2, replace=False)
            (a, b), (c, d) = edges[i], edges[j]
            new1, new2 = tuple(sorted((a, d))), tuple(sorted((c, b)))
            # Skip swaps that would link a state to itself or duplicate a link
            if a == d or c == b or new1 == new2 or new1 in linked or new2 in linked:
                continue
            linked -= {edges[i], edges[j]}
            linked |= {new1, new2}
            edges[i], edges[j] = new1, new2
        shuffled = {st: sorted([b for a, b in edges if a == st] + [a for a, b in edges if b == st])
                    for st in STATES}
        if _connected(shuffled):
            return shuffled


def merge(mine, received, lam):
    # The receiver's new parameters: lam * its own + (1 - lam) * the received model's
    return {k: lam * mine[k] + (1 - lam) * received[k] for k in mine}


def evaluate_all(models, test):
    # Accuracy, SPD and EOD of every node's model on the shared test set
    return {st: evaluate(m, test) for st, m in models.items()}


def gossip(data, graph, config, seed, on_step=None):
    # Runs the random-step protocol for config['gossip']['rounds'] rounds (51 steps each) and
    # returns the evaluations: one record every eval_checkpoint_every rounds (and at the start), with every node's
    # accuracy, SPD and EOD on the shared test set and the number of messages sent.
    # on_step(step, sender, receiver, models) is called after each exchange (for the experiment pane)
    g, tr = config['gossip'], config['training']
    nodes = {st: to_tensors(arrays) for st, arrays in data['nodes'].items()}
    test = to_tensors(data['test'])

    # Same seed -> same initial model (shared by all nodes) and same batch order;
    # who acts and who receives has its own random stream
    torch.manual_seed(seed)
    batches = torch.Generator().manual_seed(seed)
    schedule = np.random.default_rng([seed, 2])
    init = MLP(data['cat_sizes'], data['test']['num'].shape[1], hidden=tr['hidden_units'])
    models = {st: copy.deepcopy(init) for st in STATES}

    # A round is one step per state, so each state acts about once per round on average
    steps, eval_every = g['rounds'] * len(STATES), g['eval_checkpoint_every'] * len(STATES)
    history = [{'round': 0, 'step': 0, 'messages': 0, 'nodes': evaluate_all(models, test)}]
    for step in range(1, steps + 1):
        sender = STATES[schedule.integers(len(STATES))]
        receiver = graph[sender][schedule.integers(len(graph[sender]))]
        model = models[receiver]
        model.load_state_dict(merge(model.state_dict(), models[sender].state_dict(), g['lambda']))
        for _ in range(tr['local_epochs']):
            train_one_epoch(model, nodes[receiver], lr=tr['learning_rate'],
                            batch_size=tr['batch_size'], generator=batches)
        if on_step:
            on_step(step, sender, receiver, models)
        if step % eval_every == 0:
            history.append({'round': step // len(STATES), 'step': step, 'messages': step,
                            'nodes': evaluate_all(models, test)})
    return history
