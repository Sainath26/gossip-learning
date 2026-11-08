"""
Local server for the experiment pane (experiment_pane/index.html).

    python experiment_pane/server.py      then open http://localhost:8000

Runs one gossip seed at full speed in the background and streams every exchange
to the page as it happens (Server-Sent Events). The page replays the exchanges at
its own display speed, so watching never slows the run down.

When a run finishes, its result is saved exactly as run.py saves it:
results/<experiment name>/seed_XXX.json, with the config used in config.json. A name
whose folder holds results from different settings is refused.

Settings start from configs.json; the page can override the experiment name, lambda,
rounds, seed and the sample fraction. Standard library only; listens on 127.0.0.1.
"""

import json
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from learning import border_graph, gossip  # noqa: E402
from model import MLP, evaluate, to_tensors  # noqa: E402
from prepare_data import STATES, prepare  # noqa: E402
from run import claim_folder, save_result, seed_path  # noqa: E402

PAGE = Path(__file__).resolve().parent / 'index.html'
# The map library and US state outlines, saved locally so the pane works offline
VENDOR = Path(__file__).resolve().parent / 'vendor'
# Per-exchange metrics are measured on a fixed random subset of the shared test set,
# so measuring every exchange stays quick
PANE_TEST_ROWS = 20000


class Stopped(Exception):
    pass


def build_config(settings):
    # configs.json with the page's overrides; the seed is not part of the config (one file per seed)
    name = str(settings['experiment_name']).strip()
    if not re.fullmatch(r'[A-Za-z0-9 _.-]+', name) or '..' in name:
        raise ValueError('use letters, digits, spaces, dots, dashes or underscores in the experiment name')
    config = json.loads((ROOT / 'configs.json').read_text())
    config['experiment_name'] = name
    config['gossip']['lambda'] = float(settings['lambda'])
    config['gossip']['rounds'] = int(settings['rounds'])
    config['data']['sample_fraction'] = float(settings['sample_fraction'])
    if config['graph']['topology'] != 'border':
        raise ValueError("the pane supports the 'border' topology only")
    return config


class Run:
    # The current run: its events, and the thread producing them. Every event is kept,
    # so a page that connects late still receives the whole run from the start
    def __init__(self):
        self.cond = threading.Condition()
        self.events, self.run_id = [], 0
        self.thread, self.stop = None, False

    def emit(self, event):
        with self.cond:
            self.events.append(event)
            self.cond.notify_all()

    def running(self):
        return self.thread is not None and self.thread.is_alive()

    def start(self, settings):
        if self.running():
            raise ValueError('a run is already in progress; stop it first')
        config, seed = build_config(settings), int(settings['seed'])
        folder = claim_folder(config)  # refuses a name holding different settings
        with self.cond:
            self.events, self.stop = [], False
            self.run_id += 1
            self.cond.notify_all()
        self.thread = threading.Thread(target=self._run, args=(config, seed, folder), daemon=True)
        self.thread.start()

    def _run(self, config, seed, folder):
        try:
            self._gossip(config, seed, folder)
        except Stopped:
            self.emit({'type': 'stopped'})
        except Exception as e:  # shown in the page instead of only in the terminal
            self.emit({'type': 'error', 'message': f'{type(e).__name__}: {e}'})

    def _gossip(self, config, seed, folder):
        g = config['gossip']
        torch.set_num_threads(2)
        start = time.time()

        self.emit({'type': 'status', 'message': 'preparing data'})
        data = prepare(config, seed)
        graph = border_graph(config['graph']['island_links'])
        test = to_tensors(data['test'])
        pick = torch.randperm(len(test['y']), generator=torch.Generator().manual_seed(0))[:PANE_TEST_ROWS]
        test = {k: v[pick] for k, v in test.items()}

        def measure(model):
            r = evaluate(model, test)
            return {'accuracy': r['accuracy'],
                    'spd': None if r['spd'] is None else abs(r['spd']),
                    'eod': None if r['eod'] is None else abs(r['eod'])}

        # Every state starts from the same model: build that model the way gossip does
        torch.manual_seed(seed)
        initial = measure(MLP(data['cat_sizes'], data['test']['num'].shape[1],
                              hidden=config['training']['hidden_units']))
        current = {st: dict(initial) for st in STATES}
        links = sorted({tuple(sorted((a, b))) for a in graph for b in graph[a]})
        self.emit({'type': 'setup', 'name': config['experiment_name'], 'seed': seed,
                   'lambda': g['lambda'], 'rounds': g['rounds'],
                   'steps': g['rounds'] * len(STATES), 'sample_fraction': config['data']['sample_fraction'],
                   'attribute': config['fairness']['protected_attribute'], 'links': links,
                   'states': {st: {'rows': int(len(n['y'])), 'privileged_share': float(n['pa'].mean()),
                                   'neighbours': graph[st], 'metrics': initial}
                              for st, n in data['nodes'].items()},
                   'test_rows': int(len(pick))})

        def summary(round_):
            out = {'type': 'checkpoint', 'round': round_}
            for key in ('accuracy', 'spd', 'eod'):
                values = [m[key] for m in current.values() if m[key] is not None]
                worst = min(values) if key == 'accuracy' else max(values)
                out[key] = {'mean': sum(values) / len(values), 'worst': worst}
            return out

        self.emit(summary(0))

        def on_step(step, sender, receiver, models):
            if self.stop:
                raise Stopped
            before = current[receiver]
            current[receiver] = measure(models[receiver])
            self.emit({'type': 'exchange', 'step': step, 'round': step / len(STATES),
                       'sender': sender, 'receiver': receiver, 'lambda': g['lambda'],
                       'rows': int(len(data['nodes'][receiver]['y'])),
                       'sent': current[sender], 'before': before, 'after': current[receiver]})
            if step % len(STATES) == 0:
                self.emit(summary(step // len(STATES)))

        # The saved result is gossip's own record (every state on the full test set at each
        # checkpoint), the same as run.py saves; the per-exchange numbers above are for display
        history = gossip(data, graph, config, seed, on_step=on_step)
        seconds = round(time.time() - start, 1)
        self.emit({'type': 'done', 'seconds': seconds})
        path = seed_path(folder, seed)
        if path.exists():
            self.emit({'type': 'saved', 'message': f'seed {seed} was already saved in {path.parent.name}; kept the earlier file'})
        else:
            save_result(folder, {'seed': seed, 'topology': 'border', 'seconds': seconds, 'history': history})
            self.emit({'type': 'saved', 'message': f'saved to results/{path.parent.name}/{path.name}'})


RUN = Run()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, code, body, kind='application/json'):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ('/', '/index.html'):
            self._send(200, PAGE.read_bytes(), 'text/html; charset=utf-8')
        elif self.path == '/api/config':
            config = json.loads((ROOT / 'configs.json').read_text())
            self._send(200, {'experiment_name': config['experiment_name'],
                             'lambda': config['gossip']['lambda'], 'rounds': config['gossip']['rounds'],
                             'seed': 1, 'sample_fraction': config['data']['sample_fraction'],
                             'running': RUN.running()})
        elif self.path == '/api/events':
            self._events()
        elif self.path.startswith('/vendor/'):
            # Only files directly inside vendor/ (Path.name drops any '../')
            f = VENDOR / Path(self.path).name
            kind = 'text/javascript' if f.suffix == '.js' else 'application/json'
            self._send(200, f.read_bytes(), kind) if f.is_file() else self._send(404, {'error': 'not found'})
        else:
            self._send(404, {'error': 'not found'})

    def do_POST(self):
        length = int(self.headers.get('Content-Length') or 0)
        try:
            body = json.loads(self.rfile.read(length) or b'{}')
            if self.path == '/api/start':
                RUN.start(body)
            elif self.path == '/api/stop':
                RUN.stop = True
            else:
                return self._send(404, {'error': 'not found'})
            self._send(200, {'ok': True})
        except (ValueError, KeyError, TypeError) as e:
            self._send(409, {'error': str(e)})

    def _events(self):
        # Sends every event of the current run, then new ones as they come; when a new run
        # starts, sends 'reset' and starts again from its first event
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        sent, run_id = 0, None
        try:
            while True:
                with RUN.cond:
                    if run_id == RUN.run_id and sent >= len(RUN.events):
                        RUN.cond.wait(timeout=15)
                    reset = run_id != RUN.run_id
                    if reset:
                        run_id, sent = RUN.run_id, 0
                    batch = RUN.events[sent:]
                    sent += len(batch)
                lines = [{'type': 'reset'}] if reset else []
                if not batch and not reset:
                    self.wfile.write(b': ping\n\n')
                for event in lines + batch:
                    self.wfile.write(f'data: {json.dumps(event)}\n\n'.encode())
                self.wfile.flush()
        except OSError:
            pass  # the page was closed


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    server.daemon_threads = True
    print(f'Experiment pane: http://localhost:{port}  (Ctrl+C to quit)')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        RUN.stop = True


if __name__ == '__main__':
    main()
