# Remote sequential training plan

Approved design: run Flat → Rough → Backflip V1.6 → StandUp → DreamWaQ → AMP using registry defaults. Stop on failure; preserve per-task logs, checkpoints, summary metrics and state. Query over SSH with password or key; never store passwords in the repository.

- Implement scripts/train_sequence.py: exclusive queue lock, atomic status JSON, console metric extraction, signal cleanup, isolated run directories, default configuration snapshot.
- Implement scripts/monitor_remote.py: password prompt, host-key checking, SSH state retrieval, process liveness, summary and optional watch mode.
- Document usage and pinned training dependencies.
- Upload a snapshot of current working code, excluding local logs, caches, Git internals and credentials, to /mjlab/unitree_rl_mjlab.
- Inspect existing Python environment and install compatible versions into persistent /mjlab environment. Verify task defaults and actual GPU startup.
- Start the full queue detached, confirm progress, and exercise the remote monitor.

Default iterations: 2000/3000/1000/6000/5000/10000; train environments: 4096 except AMP 2048. JSON summaries report reward trends without equating reward with task success. Completed jobs are skipped only when resuming the same queue; failed jobs require explicit resume.
