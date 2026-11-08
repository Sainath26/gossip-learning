"""
Runs an experiment from configs.json: every seed, in parallel, saved to
results/<experiment_name>/.

    python run.py

For each seed: prepare the data (prepare_data), build the graph and run gossip
(learning), then save every checkpoint's per-state accuracy, SPD and EOD to
seed_XXX.json. A copy of the config is saved as config.json in the same folder.

Seeds that already have a results file are skipped, so a stopped run can be
restarted. A folder holding results from different settings is never reused:
change experiment_name in configs.json for a new experiment.
"""

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

from learning import border_graph, gossip
from prepare_data import load_acs, prepare

ROOT = Path(__file__).resolve().parent


def load_config(path=ROOT / 'configs.json'):
    return json.loads(Path(path).read_text())


def run_seed(config, seed):
    # One complete run for one seed; executed inside a worker process.
    # One thread per worker: the parallelism comes from running seeds side by side
    torch.set_num_threads(1)
    topology = config['graph']['topology']
    if topology != 'border':
        # The random graph (Experiment 2) is added when that experiment is run
        raise ValueError(f"topology '{topology}' is not supported yet; use 'border'")
    graph = border_graph(config['graph']['island_links'])
    start = time.time()
    history = gossip(prepare(config, seed), graph, config, seed)
    return {'seed': seed, 'topology': topology, 'seconds': round(time.time() - start, 1),
            'history': history}


def summary(result):
    # One line about the last checkpoint: mean and worst over the states
    nodes = result['history'][-1]['nodes'].values()
    acc = [m['accuracy'] for m in nodes]
    spd = [abs(m['spd']) for m in nodes if m['spd'] is not None]
    return (f"mean acc {sum(acc) / len(acc):.3f} (worst {min(acc):.3f}), "
            f"mean |SPD| {sum(spd) / len(spd):.3f} (worst {max(spd):.3f})")


def seed_path(folder, seed):
    return folder / f'seed_{seed:03d}.json'


def claim_folder(config):
    # The experiment's results folder, with a copy of the config saved in it. A folder that
    # already holds results from different settings is never reused (raises ValueError).
    # Shared with the experiment pane, so both save results the same way
    folder = ROOT / config['output']['results_dir'] / config['experiment_name']
    saved = folder / 'config.json'
    if saved.exists() and json.loads(saved.read_text()) != config:
        raise ValueError(f"results/{config['experiment_name']} holds results from different settings; "
                         f"use another experiment name")
    folder.mkdir(parents=True, exist_ok=True)
    saved.write_text(json.dumps(config, indent=2))
    return folder


def save_result(folder, result):
    # Write then rename, so a run killed mid-write never leaves a broken file
    path = seed_path(folder, result['seed'])
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(result))
    os.replace(tmp, path)
    return path


def main():
    config = load_config()
    try:
        folder = claim_folder(config)
    except ValueError as e:
        raise SystemExit(f'{e}: change experiment_name in configs.json for a new experiment.')

    # Build the data cache once here, so the workers only ever read it
    load_acs(config['data']['survey_year'])

    # "seeds": n runs seeds 1 to n
    seeds = list(range(1, config['seeds'] + 1))
    todo = [s for s in seeds if not seed_path(folder, s).exists()]
    workers = config['compute']['workers']
    print(f"{config['experiment_name']}: {len(seeds)} seeds, {len(seeds) - len(todo)} already done, "
          f"{len(todo)} to run on {workers} workers")

    with ProcessPoolExecutor(workers) as pool:
        futures = [pool.submit(run_seed, config, seed) for seed in todo]
        for n, future in enumerate(as_completed(futures), 1):
            result = future.result()  # re-raises here any error from the worker
            save_result(folder, result)
            print(f"  seed {result['seed']} done ({n}/{len(todo)}, {result['seconds']}s): {summary(result)}")
    print(f'results in {folder}')


if __name__ == '__main__':
    main()
