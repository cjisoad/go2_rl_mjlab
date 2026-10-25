"""Wait for an existing queue to finish, then run its appended tasks.

A running Python queue already owns its in-memory task list. This process
waits on its lock rather than interrupting the trainer or editing live status.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_sequence import ROOT, save, task_specs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir",type=Path,default=ROOT/"logs/sequence")
    args = parser.parse_args()
    directory = args.state_dir.resolve()
    # Only one handoff process may wait for this queue.
    extension_lock = (directory/"extension.lock").open("w")
    fcntl.flock(extension_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state_path = directory/"status.json"
    state = json.loads(state_path.read_text())
    existing = {task["task"] for task in state["tasks"]}
    pending = [task for task in task_specs() if task["task"] not in existing]
    if not pending:
        print("Queue already includes all tasks",flush=True)
        return 0
    save(directory/"planned_tasks.json",{"tasks":pending,"handoff_pid":os.getpid()})
    print("Waiting for current queue; appended tasks: "+", ".join(task["task"] for task in pending),flush=True)
    with (directory/"queue.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        state = json.loads(state_path.read_text())
        if state["status"] != "completed":
            print("Existing queue did not complete; extension stopped. Inspect status.json.",flush=True)
            return 1
    # The new queue acquires the same lock and skips completed tasks on resume.
    os.execv(sys.executable,[sys.executable,"-u",str(ROOT/"scripts/train_sequence.py"),
                           "--resume","--state-dir",str(directory)])


if __name__ == "__main__":
    raise SystemExit(main())
