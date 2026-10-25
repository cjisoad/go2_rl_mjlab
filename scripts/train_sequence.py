"""Train Go2 tasks sequentially, keeping a durable summary for remote monitoring."""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import queue
import re
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TASKS = (
    "Unitree-Go2-Flat", "Unitree-Go2-Rough", "Unitree-Go2-Backflip-V1.6",
    "Unitree-Go2-StandUp", "Unitree-Go2-DreamWaQ", "Unitree-Go2-AMP",
    "Unitree-Go2-Quad-Pedipulation", "Unitree-Go2-Biped-Pedipulation",
)
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
ITERATION = re.compile(r"Learning iteration\s+(\d+)/(\d+)")


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save(path, state):
    state["updated_at"] = now()
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)  # Readers always see a complete JSON document.



def task_specs():
    import src.tasks  # noqa: F401 -- register repository tasks
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

    return [{"task": name, "max_iterations": load_rl_cfg(name).max_iterations,
             "num_envs": load_env_cfg(name).scene.num_envs,
             "status": "pending", "completed_iterations": 0} for name in TASKS]


def extend_saved_queue(state, additions):
    names = [task["task"] for task in state["tasks"]]
    if names != list(TASKS[:len(names)]):
        raise ValueError("Saved queue does not match the current task order")
    for task in additions:
        if task["task"] not in names:
            state["tasks"].append(task.copy())
            names.append(task["task"])
    if names != list(TASKS):
        raise ValueError("Queue extension does not provide the remaining tasks")

def read_output(stream, lines):
    for line in stream:
        lines.put(line)
    lines.put(None)


def train(task, state, state_path, run_dir):
    log_path = run_dir / (task["task"] + ".log")
    task.update(status="running", started_at=now(), log_file=str(log_path))
    state.update(current_task=task["task"], status="running")
    task["command"] = [sys.executable, "-u", str(ROOT / "scripts/train.py"),
                       task["task"], "--agent.run-name", state["run_id"],
                       "--agent.max-iterations", str(task["max_iterations"]),
                       "--env.scene.num-envs", str(task["num_envs"])]
    first_rewards = []
    recent_rewards = deque(maxlen=100)
    child = subprocess.Popen(task["command"], cwd=ROOT, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1,
                             start_new_session=True)
    task["pid"] = child.pid
    save(state_path, state)
    lines = queue.Queue()
    threading.Thread(target=read_output, args=(child.stdout, lines), daemon=True).start()
    last_write = time.monotonic()
    try:
        with log_path.open("a", buffering=1) as log:
            while True:
                try:
                    line = lines.get(timeout=10)
                except queue.Empty:
                    save(state_path, state)
                    continue
                if line is None:
                    break
                log.write(line)
                clean = ANSI.sub("", line).strip()
                match = ITERATION.search(clean)
                if match:
                    # RSL-RL reports zero-based iterations after each PPO update.
                    task["completed_iterations"] = int(match[1]) + 1
                if "Logging experiment in directory:" in clean:
                    task["experiment_dir"] = str((ROOT / clean.split("directory:", 1)[1].strip()).resolve())
                for label, key in (("Mean reward:", "reward"),
                                   ("Mean episode length:", "episode_length"),
                                   ("Steps per second:", "steps_per_second")):
                    if clean.startswith(label):
                        task[key] = float(clean.split(":", 1)[1].strip())
                        if key == "reward":
                            value = task[key]
                            recent_rewards.append(value)
                            if len(first_rewards) < 100:
                                first_rewards.append(value)
                            task["reward_first_100_mean"] = sum(first_rewards) / len(first_rewards)
                            task["reward_last_100_mean"] = sum(recent_rewards) / len(recent_rewards)
                            task["reward_best"] = max(task.get("reward_best", value), value)
                if clean.startswith("ETA:"):
                    task["eta"] = clean.split(":", 1)[1].strip()
                if clean:
                    task["last_message"] = clean[-500:]
                if clean.startswith("ETA:") or time.monotonic() - last_write >= 10:
                    save(state_path, state)
                    last_write = time.monotonic()
        code = child.wait()
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
    task.update(return_code=code, ended_at=now())
    # A zero exit without all expected iterations must not silently pass the queue.
    finished = task.get("completed_iterations", 0) >= task["max_iterations"]
    task["status"] = "completed" if code == 0 and finished else "failed"
    experiment = task.get("experiment_dir")
    if experiment:
        checkpoints = list(Path(experiment).glob("model_*.pt"))
        if checkpoints:
            task["checkpoint"] = str(max(checkpoints, key=lambda p: p.stat().st_mtime))
    save(state_path, state)
    return task["status"] == "completed"


def interrupt(signum, frame):
    raise KeyboardInterrupt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, default=ROOT / "logs/sequence")
    parser.add_argument("--resume", action="store_true",
                        help="Keep completed tasks and restart the interrupted task from scratch")
    args = parser.parse_args()
    args.state_dir = args.state_dir.expanduser().resolve()
    args.state_dir.mkdir(parents=True, exist_ok=True)
    lock = (args.state_dir / "queue.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error("Another training queue is already running in this state directory")
    path = args.state_dir / "status.json"
    if args.resume:
        state = json.loads(path.read_text())
        try:
            extend_saved_queue(state, task_specs())
        except ValueError as exc:
            parser.error(str(exc))
    else:
        if path.exists():
            parser.error("Status already exists; use --resume or a different --state-dir")
        state = {"run_id": datetime.now().strftime("queue_%Y%m%d_%H%M%S"),
                 "started_at": now(), "tasks": task_specs()}
    state.update(pid=os.getpid(), status="running", current_task=None)
    run_dir = args.state_dir / state["run_id"]
    run_dir.mkdir(exist_ok=True)
    save(path, state)
    signal.signal(signal.SIGTERM, interrupt)
    try:
        for index, task in enumerate(state["tasks"]):
            if task["status"] == "completed":
                continue
            # Explicit resume keeps finished tasks, but restarts an unfinished job.
            task = {k: task[k] for k in ("task", "max_iterations", "num_envs")}
            task.update(status="pending", completed_iterations=0)
            state["tasks"][index] = task
            print(f"{now()} Starting {task['task']} ({task['max_iterations']} iterations)", flush=True)
            if not train(task, state, path, run_dir):
                state["status"] = "failed"
                print(f"Stopped after failure; inspect {task['log_file']}", flush=True)
                return 1
        state.update(status="completed", current_task=None, ended_at=now())
        return 0
    except KeyboardInterrupt:
        state["status"] = "interrupted"
        for task in state["tasks"]:
            if task["status"] == "running":
                task.update(status="interrupted", ended_at=now())
        return 130
    except Exception as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        for task in state["tasks"]:
            if task["status"] == "running":
                task["status"] = "failed"
        raise
    finally:
        save(path, state)


if __name__ == "__main__":
    raise SystemExit(main())
