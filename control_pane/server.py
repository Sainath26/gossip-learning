"""
Local server for the control panel (artifacts/control_panel.html).

    python server.py            then open http://localhost:8000

Experiments run as subprocesses: watch.py for one slowed-down run, run.py for
the batch. Their JSON lines are relayed to the browser as Server-Sent Events.
Standard library only; listens on 127.0.0.1, so it is reachable from this
machine only.

Pieces:
  Job      one subprocess; turns each line it prints into an event
  State    what the panel shows (watch run and batch), updated from events,
           and the start / stop actions behind the panel's buttons
  Hub      sends every event to every open browser tab
  Handler  the HTTP endpoints:
             GET  /                     the panel page
             GET  /api/events           live event stream (Server-Sent Events)
             GET  /api/state            current State, as JSON
             GET  /api/experiments      run.py's EXPERIMENTS with progress
             GET  /api/summary          results/summary.md and figure names
             GET  /results/figures/X    a figure image
             POST /api/watch/start      start watch.py (JSON body: its options)
             POST /api/watch/control    {"cmd": pause|resume|step|stop|delay}
             POST /api/batch/start      start run.py for {"ids", "seeds", "workers"}
             POST /api/batch/stop       kill the batch
             POST /api/summary/rebuild  run.py --summary-only, then the summary

At most one watch run and one batch run at a time; each panel action returns
409 with an error message if it is not possible right now.
"""

import argparse
import json
import os
import queue
import signal
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

# Every path is relative to the folder holding this file. They assume it sits
# in the experiment folder, next to run.py, watch.py, artifacts/ and results/
ROOT = Path(__file__).resolve().parent
PANEL = ROOT / "artifacts" / "control_panel.html"
RESULTS = ROOT / "results"
SEEDS = range(1, 101)

# Allowed values for the watch-run options that come from the browser; anything
# else is rejected before it reaches a command line
WATCH_CHOICES = {"setting": {"fl", "gossip"},
                 "method": {"fedavg", "fairfed", "fair2zero", "avg"},
                 "metric": {"spd", "eod"}, "topology": {"ring", "full"}}


class Hub:
    # Fans events out to every open browser tab
    # Each tab's event stream has its own queue; publish() puts a copy of the
    # message in every queue. The lock guards the list, since requests run on
    # several threads
    def __init__(self):
        self.lock = threading.Lock()
        self.queues = []

    def subscribe(self):
        q = queue.Queue()
        with self.lock:
            self.queues.append(q)
        return q

    def unsubscribe(self, q):
        with self.lock:
            if q in self.queues:
                self.queues.remove(q)

    def publish(self, msg):
        with self.lock:
            for q in self.queues:
                q.put(msg)


class Job:
    # One experiment subprocess; every stdout line becomes an event
    #   source    "watch" or "batch", which part of the panel it belongs to
    #   on_event  called as on_event(job, event) for every event
    def __init__(self, source, cmd, on_event):
        # stdin is a pipe so commands can be sent (watch.py reads it).
        # stderr is merged into stdout so errors also reach the panel log.
        # bufsize=1 + text mode = line-buffered, so lines arrive as printed
        kwargs = dict(cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                      stderr=subprocess.STDOUT, text=True, bufsize=1,
                      encoding="utf-8", errors="replace",
                      env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        # Start the child in its own process group, so kill() can take down
        # the child together with any processes it starts
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        self.source, self.on_event = source, on_event
        self.proc = subprocess.Popen(cmd, **kwargs)
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        # Background thread: reads output until the process exits.
        # JSON-object lines are passed on as events; any other line (plain
        # prints, tracebacks) becomes a "log" event
        for line in self.proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except ValueError:
                event = None
            if not isinstance(event, dict):
                event = {"type": "log", "text": line}
            self.on_event(self, event)
        # Output closed: the process has ended; report its exit code
        self.on_event(self, {"type": "exit", "code": self.proc.wait()})

    def alive(self):
        return self.proc.poll() is None

    def send(self, line):
        # Write one command line to the process's stdin. Errors are ignored:
        # they only mean the process has already exited
        try:
            self.proc.stdin.write(line + "\n")
            self.proc.stdin.flush()
        except OSError:
            pass

    def kill(self):
        # Kill the whole process tree: run.py's pool workers too
        if not self.alive():
            return
        if os.name == "nt":
            # /T: include child processes, /F: force
            subprocess.run(["taskkill", "/PID", str(self.proc.pid), "/T", "/F"],
                           capture_output=True)
        else:
            os.killpg(self.proc.pid, signal.SIGTERM)


class State:
    # What the panel shows; replayed to a tab when it (re)connects
    #   watch  status ("idle", "starting", "running", "finished", "stopped"),
    #          whether it is paused, the setup event, every round event so far,
    #          and the last log lines
    #   batch  status, per-configuration progress, and the last log lines
    def __init__(self, hub):
        self.hub, self.lock = hub, threading.Lock()
        self.watch_job = self.batch_job = None
        self.watch = {"status": "idle", "paused": False, "setup": None,
                      "rounds": [], "log": []}
        self.batch = {"status": "idle", "configs": [], "log": []}

    def snapshot(self):
        # A deep copy (via a JSON round trip), safe to send while jobs keep
        # updating the original
        with self.lock:
            return json.loads(json.dumps({"watch": self.watch, "batch": self.batch}))

    def on_event(self, job, event):
        # Called from a Job's reader thread for every event: update the stored
        # view, then forward the event (tagged with its source) to all tabs
        with self.lock:
            view = self.watch if job.source == "watch" else self.batch
            kind = event.get("type")
            if kind == "log":
                # Keep only the last 200 log lines
                view["log"] = (view["log"] + [event["text"]])[-200:]
            elif job.source == "watch":
                if kind == "setup":
                    view.update(status="running", paused=False, setup=event, rounds=[])
                elif kind == "round":
                    view["rounds"].append(event)
                elif kind == "status":
                    view["paused"] = event["paused"]
                elif kind in ("end", "stopped"):
                    view["status"] = "finished" if kind == "end" else "stopped"
                # Exited without "end" or "stopped" (e.g. it crashed or was killed)
                elif kind == "exit" and view["status"] in ("running", "starting"):
                    view["status"] = "stopped"
            else:
                if kind == "batch_start":
                    view.update(status="running", configs=event["configs"])
                elif kind == "run_done":
                    for c in view["configs"]:
                        if c["id"] == event["id"]:
                            c["done"] += 1
                elif kind == "exit":
                    view["status"] = "finished" if event["code"] == 0 else "stopped"
        self.hub.publish({"source": job.source, **event})

    def start_watch(self, body):
        # Start watch.py with the options from the request body. Raises
        # ValueError (sent back as 409) if one is already running or an
        # option is invalid
        with self.lock:
            if self.watch_job and self.watch_job.alive():
                raise ValueError("a watch run is already running")
        args = {k: str(body.get(k, "")) for k in WATCH_CHOICES}
        for k, allowed in WATCH_CHOICES.items():
            if args[k] not in allowed:
                raise ValueError(f"invalid {k}: {args[k]}")
        # -u: unbuffered output, so events arrive as soon as they are printed.
        # Numeric options go through float() / int() to validate them
        cmd = [sys.executable, "-u", "watch.py"]
        for k, v in args.items():
            cmd += [f"--{k}", v]
        cmd += ["--beta", str(float(body.get("beta", 0.05))),
                "--rounds", str(int(body.get("rounds", 25))),
                "--seed", str(int(body.get("seed", 1))),
                "--delay", str(float(body.get("delay", 1.0)))]
        with self.lock:
            self.watch.update(status="starting", paused=False, setup=None,
                              rounds=[], log=[])
        self.watch_job = Job("watch", cmd, self.on_event)
        self.hub.publish({"source": "watch", "type": "starting"})

    def control_watch(self, body):
        # Forward a pause / resume / step / stop / delay command to watch.py
        cmd = body.get("cmd")
        if cmd not in ("pause", "resume", "step", "stop", "delay"):
            raise ValueError(f"invalid command: {cmd}")
        job = self.watch_job
        if not job or not job.alive():
            raise ValueError("no watch run is running")
        job.send(f"delay {float(body.get('value', 1.0))}" if cmd == "delay" else cmd)
        if cmd == "stop":
            # Fall back to killing it if it does not stop on its own
            threading.Timer(5, job.kill).start()

    def start_batch(self, body):
        # Start run.py for the selected experiment ids and seed range
        with self.lock:
            if self.batch_job and self.batch_job.alive():
                raise ValueError("the batch is already running")
        ids = [int(i) for i in body.get("ids", [])]
        if not ids:
            raise ValueError("no configurations selected")
        first, last = (int(s) for s in body.get("seeds", [1, 100]))
        cmd = [sys.executable, "-u", "run.py", "--events",
               "--only", ",".join(map(str, ids)), "--seeds", str(first), str(last),
               "--workers", str(int(body.get("workers", 8)))]
        with self.lock:
            self.batch.update(status="starting", configs=[], log=[])
        self.batch_job = Job("batch", cmd, self.on_event)
        self.hub.publish({"source": "batch", "type": "starting"})

    def stop_batch(self):
        # Finished runs keep their result files; a rerun skips them
        if self.batch_job:
            self.batch_job.kill()


# One shared instance of each for the whole server
hub = Hub()
state = State(hub)
_run = None


def run_module():
    # run.py imports torch, so load it only when the panel first asks
    global _run
    if _run is None:
        import run
        _run = run
    return _run


def experiments():
    # Every configuration in run.EXPERIMENTS, with how many of the seeds
    # already have a result file
    run = run_module()
    return [{"id": i, "name": f"{e['setting']}: {run.config_name(e)}",
             "setting": e["setting"],
             "done": sum(run.result_path(RESULTS, e, s).exists() for s in SEEDS),
             "total": len(SEEDS)} for i, e in enumerate(run.EXPERIMENTS)]


def summary():
    # The results table (None if run.py has not summarised yet) and the names
    # of the figure images from make_figures.py
    md = RESULTS / "summary.md"
    figures = sorted(p.name for p in (RESULTS / "figures").glob("*.png"))
    return {"markdown": md.read_text(encoding="utf-8") if md.exists() else None,
            "figures": figures}


class Handler(BaseHTTPRequestHandler):
    # Handles one HTTP request; ThreadingHTTPServer runs each on its own thread

    def log_message(self, *args):
        # Silence the default one-line-per-request log
        pass

    def _send(self, code, body, kind="application/json"):
        # Send a complete response: bytes as they are, anything else as JSON
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send(200, PANEL.read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/events":
            self._events()
        elif path == "/api/state":
            self._send(200, state.snapshot())
        elif path == "/api/experiments":
            self._send(200, experiments())
        elif path == "/api/summary":
            self._send(200, summary())
        elif path.startswith("/results/figures/"):
            # Path(...).name keeps only the file name, so a request cannot
            # reach files outside results/figures (e.g. with "../")
            f = RESULTS / "figures" / Path(path).name
            if f.suffix == ".png" and f.exists():
                self._send(200, f.read_bytes(), "image/png")
            else:
                self._send(404, {"error": "not found"})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        # Every POST takes an optional JSON body. Invalid requests (State
        # raises ValueError, or bad types) get a 409 with the error message
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
            if path == "/api/watch/start":
                state.start_watch(body)
            elif path == "/api/watch/control":
                state.control_watch(body)
            elif path == "/api/batch/start":
                state.start_batch(body)
            elif path == "/api/batch/stop":
                state.stop_batch()
            elif path == "/api/summary/rebuild":
                # Waits for run.py to finish, then returns the new summary
                subprocess.run([sys.executable, "run.py", "--summary-only"], cwd=ROOT,
                               capture_output=True)
                return self._send(200, summary())
            else:
                return self._send(404, {"error": "not found"})
            self._send(200, {"ok": True})
        except (ValueError, TypeError) as e:
            self._send(409, {"error": str(e)})

    def _events(self):
        # Server-Sent Events stream: the connection stays open and each
        # message is sent as a "data: <json>" block. The first message is a
        # snapshot of the full state, so a new or reconnected tab can draw
        # everything; live events follow
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        q = hub.subscribe()
        try:
            self._sse({"source": "server", "type": "snapshot", **state.snapshot()})
            while True:
                try:
                    self._sse(q.get(timeout=15))
                except queue.Empty:
                    # Nothing for 15 s: send an SSE comment line, which keeps
                    # the connection open and shows whether the tab is gone
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
        except OSError:
            pass  # the tab was closed
        finally:
            hub.unsubscribe(q)

    def _sse(self, msg):
        # One SSE message; the blank line ends it
        self.wfile.write(f"data: {json.dumps(msg)}\n\n".encode())
        self.wfile.flush()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    # Request threads do not block shutdown (event streams never end on their own)
    server.daemon_threads = True
    print(f"Control panel: http://localhost:{args.port}  (Ctrl+C to quit)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        # Don't leave experiment processes running after the server exits
        for job in (state.watch_job, state.batch_job):
            if job:
                job.kill()


if __name__ == "__main__":
    main()
